#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda.h>
#include <omp.h>

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o, float scalar);
void UpdateMiLiOi(float *d_mi, float *d_li, float *d_oi, float *d_mij, float *d_lij, float *d_pij, float *d_vj, int br, int bc, int d);

__global__ void FusedQKDotAndRowOpsKernel(
    const float *d_qi,   // q tile: br x d
    const float *d_kj,   // k tile: bc x d
    float *d_sij,        // output: br x bc (QK^T scaled)
    float *d_pij,        // output: br x bc (exp(val - row_max))
    float *d_mij,        // output: br (row-wise max)
    float *d_lij,        // output: br (row-wise sum of exp)
    float *d_mi,         // input/output: br (updated row-wise max)
    float *d_li,         // input/output: br (updated row-wise sum)
    float *d_oi,         // input/output: br x d (updated O)
    const float *d_vj,   // input: bc x d
    int br, int bc, int d,
    float scalar
) {
    int i = blockIdx.y;     // row index within this tile
    int col = threadIdx.x;  // column index for p_ij

    if (i >= br || col >= bc) return;

    // --- QK Dot Product ---
    float sum = 0.0f;
    int q_offset = i * d;
    int k_offset = col * d;

    #pragma unroll
    for (int t = 0; t < d; t++) {
        sum += d_qi[q_offset + t] * d_kj[k_offset + t];
    }

    // scalar here represents 1 / (d^1/2)
    sum *= scalar;
    d_sij[i * bc + col] = sum;

    // --- Warp-Level Reductions for row_max and row_sum ---
    __shared__ float warp_max[4]; 
    __shared__ float warp_sum[4];

    unsigned mask = 0xffffffff;
    int laneId = col & 31;     
    int warpId = col >> 5;     

    // Find row_max via warp reduction
    float val = sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, val, offset);
        val = fmaxf(val, other);
    }

    if (laneId == 0) {
        warp_max[warpId] = val;
    }

    // Reduce warp maxima (assuming bc=128 here; adjust if bc=256)
    if (threadIdx.x == 0) {
        float m = warp_max[0];
        m = fmaxf(m, warp_max[1]);
        m = fmaxf(m, warp_max[2]);
        m = fmaxf(m, warp_max[3]);
        // If bc=256, add more steps here for warp_max[4..7]
        warp_max[0] = m; 
    }
    __syncthreads();

    float row_max = warp_max[0];

    // Compute exp(val - row_max)
    float exp_val = __expf(sum - row_max);
    d_pij[i * bc + col] = exp_val;

    // Find row_sum of exp_val via warp reduction
    val = exp_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, val, offset);
        val += other;
    }

    if (laneId == 0) {
        warp_sum[warpId] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = warp_sum[0] + warp_sum[1] + warp_sum[2] + warp_sum[3];
        // If bc=256, add sums for warp_sum[4..7]
        warp_sum[0] = s;
    }
    __syncthreads();

    float row_sum = warp_sum[0];

    // Write out row_max and row_sum
    if (col == 0) {
        d_mij[i] = row_max;
        d_lij[i] = row_sum;
    }

    __syncthreads();

    // --- Now fuse UpdateMiLiOi logic ---
    // Wait until row_max, row_sum, and p_ij are computed
    // We need one thread per row to compute mi_new_val and li_new_val:
    static __shared__ float mi_new_val;
    static __shared__ float li_new_val;

    if (threadIdx.x == 0) {
        float mi_i = d_mi[i];
        float li_i = d_li[i];
        float mi_val = d_mij[i]; 
        float lij_i = d_lij[i];

        float mx = fmaxf(mi_i, mi_val);
        mi_new_val = mx;
        li_new_val = expf(mi_i - mx) * li_i + expf(mi_val - mx) * lij_i;
    }
    __syncthreads();

    // Update oi: one thread per dimension j (assuming d <= bc)
    if (col < d) {
        // Compute pv = sum_{t=0}^{bc-1} p_ij[i, t]*v_j[t, col]
        float pv = 0.0f;
        for (int t = 0; t < bc; t++) {
            pv += d_pij[i * bc + t] * d_vj[t * d + col];
        }

        float mi_i = d_mi[i];
        float li_i = d_li[i];
        float oi_ij = d_oi[i * d + col];
        float mij_i = d_mij[i]; 

        float numerator = (li_i * expf(mi_i - mi_new_val) * oi_ij)
                        + (expf(mij_i - mi_new_val) * pv);
        d_oi[i * d + col] = numerator / li_new_val;
    }

    __syncthreads();

    // Update mi[i] and li[i]
    if (threadIdx.x == 0) {
        d_mi[i] = mi_new_val;
        d_li[i] = li_new_val;
    }
}

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();
    float scalar = 1.0f / sqrtf(d);
    for (int i = 0; i < B; i++) {
        flash_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d),
            scalar
        );
    }

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    // memset(O, 0x00, B * N * d * sizeof(float));

    size_t total_size = (size_t)B * N * d;

    // Use OpenMP parallel for to speed up O initialization
    // Setting O to zero in parallel
    #pragma omp parallel for
    for (int i = 0; i < (int)total_size; i++) {
        O[i] = 0.0f;
    }

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

void flash_attention(float *q, float *k, float *v, float *o, float scalar) {
    // Dimensions
    int br = N, bc = 128;
    int tr = (N + br - 1) / br; // round up
    int tc = (N + bc - 1) / bc;

    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        l[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    // Device arrays
    float *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, N * d * sizeof(float));
    cudaMalloc(&d_k, N * d * sizeof(float));
    cudaMalloc(&d_v, N * d * sizeof(float));
    cudaMalloc(&d_o, N * d * sizeof(float));

    cudaMemcpy(d_q, q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, o, N * d * sizeof(float), cudaMemcpyHostToDevice);

    float *d_l, *d_m;
    cudaMalloc(&d_l, N * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));
    cudaMemcpy(d_l, l, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_qi, *d_oi, *d_kj, *d_vj;
    float *d_sij, *d_pij, *d_mij, *d_lij;
    float *d_li, *d_mi;

    size_t size_qi = br * d * sizeof(float);
    size_t size_oi = br * d * sizeof(float);
    size_t size_kj = bc * d * sizeof(float);
    size_t size_vj = bc * d * sizeof(float);
    size_t size_sij = br * bc * sizeof(float);
    size_t size_pij = br * bc * sizeof(float);
    size_t size_mij = br * sizeof(float);
    size_t size_lij = br * sizeof(float);
    size_t size_li = br * sizeof(float);
    size_t size_mi = br * sizeof(float);

    cudaMalloc(&d_qi, size_qi);
    cudaMalloc(&d_oi, size_oi);
    cudaMalloc(&d_kj, size_kj);
    cudaMalloc(&d_vj, size_vj);
    cudaMalloc(&d_sij, size_sij);
    cudaMalloc(&d_pij, size_pij);
    cudaMalloc(&d_mij, size_mij);
    cudaMalloc(&d_lij, size_lij);
    cudaMalloc(&d_li, size_li);
    cudaMalloc(&d_mi, size_mi);

    // Kernel configurations for fused kernel
    // dim3 block2(bc, 1);  
    // dim3 grid2(1, br);
    // size_t sharedMemSize = bc * sizeof(float); // for RowMaxMinusMaxExpAndRowSumKernel

    // Grid and block for the fused kernel
    dim3 blockFused(bc, 1);
    dim3 gridFused(1, br);

    for (int j = 0; j < tc; j++) {
        // Directly compute the pointers instead of copying them back and forth
        float* d_kj_ptr = d_k + j * bc * d;
        float* d_vj_ptr = d_v + j * bc * d;

        for (int i = 0; i < tr; i++) {
            float* d_qi_ptr = d_q + i * br * d;
            float* d_oi_ptr = d_o + i * br * d;
            float* d_li_ptr = d_l + i * br;
            float* d_mi_ptr = d_m + i * br;

            FusedQKDotAndRowOpsKernel<<<gridFused, blockFused>>>(
                d_qi_ptr,
                d_kj_ptr,
                d_sij,
                d_pij,
                d_mij,
                d_lij,
                d_mi_ptr,
                d_li_ptr,
                d_oi_ptr,
                d_vj_ptr,
                br, bc, d,
                scalar
            );
        }
    }

    cudaMemcpy(o, d_o, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(d_qi);
    cudaFree(d_oi);
    cudaFree(d_kj);
    cudaFree(d_vj);
    cudaFree(d_sij);
    cudaFree(d_pij);
    cudaFree(d_mij);
    cudaFree(d_lij);
    cudaFree(d_li);
    cudaFree(d_mi);

    free(l);
    free(m);
}