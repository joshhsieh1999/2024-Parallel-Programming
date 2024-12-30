#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include <chrono>

#define DEV_NO 0
#define B 64
#define half_B 32

// Global data
int* D;                  
int V;                   
int V_before_padding;    
int E;                   

void input(const char* filename) {
    // open file
    FILE* file = fopen(filename, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    V_before_padding = V;
    int r = V % B;
    if (r != 0) V += (B - r);

    // allocate
    D = (int *)malloc((size_t)V * V * sizeof(int));

    // init distances
    for (int i = 0; i < V; ++i){
        int rowBase = i * V;
        for (int j = 0; j < V; ++j){
            if (i == j) D[rowBase + j] = 0;
            else        D[rowBase + j] = 1073741823;
        }
    }

    // read edges
    int tmp[300];
    if (E >= 100){
        int j = 0;
        for (; j < E; j += 100) {
            fread(tmp, sizeof(int), 300, file);
            for (int idx = 0; idx < 300; idx += 3){
                D[tmp[idx] * V + tmp[idx+1]] = tmp[idx+2];
            }
        }
        for (int x = j - 100; x < E; ++x) {
            fread(tmp, sizeof(int), 3, file);
            D[tmp[0] * V + tmp[1]] = tmp[2];
        }
    } else {
        for (int x = 0; x < E; ++x) {
            fread(tmp, sizeof(int), 3, file);
            D[tmp[0] * V + tmp[1]] = tmp[2];
        }
    }
    fclose(file);
}

void output(const char* filename) {
    FILE* file = fopen(filename, "wb");
    for (int i = 0; i < V_before_padding; ++i){
        fwrite(&D[i * V], sizeof(int), V_before_padding, file);
    }
    fclose(file);
}

__global__ void phase_1(int *d_D, int round, int myV) {
    // 2D shared memory: pivot tile
    __shared__ int share_D[B][B];

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int gx = round * B + lx;
    int gy = round * B + ly;

    // Load 64x64 pivot block
    share_D[ly][lx]                   = d_D[gy * myV + gx];
    share_D[ly][lx + half_B]          = d_D[gy * myV + (gx + half_B)];
    share_D[ly + half_B][lx]         = d_D[(gy + half_B) * myV + gx];
    share_D[ly + half_B][lx + half_B] = d_D[(gy + half_B) * myV + (gx + half_B)];

    __syncthreads();

    // Floyd-Warshall inside the tile
    #pragma unroll 32
    for (int k = 0; k < B; ++k) {
        share_D[ly][lx] 
            = min(share_D[ly][lx], 
                  share_D[ly][k] + share_D[k][lx]);

        share_D[ly][lx + half_B]
            = min(share_D[ly][lx + half_B],
                  share_D[ly][k] + share_D[k][lx + half_B]);

        share_D[ly + half_B][lx]
            = min(share_D[ly + half_B][lx],
                  share_D[ly + half_B][k] + share_D[k][lx]);

        share_D[ly + half_B][lx + half_B]
            = min(share_D[ly + half_B][lx + half_B],
                  share_D[ly + half_B][k] + share_D[k][lx + half_B]);

        __syncthreads();
    }

    // Store the pivot block back
    d_D[gy * myV + gx] 
        = share_D[ly][lx];
    d_D[gy * myV + (gx + half_B)]
        = share_D[ly][lx + half_B];
    d_D[(gy + half_B) * myV + gx]
        = share_D[ly + half_B][lx];
    d_D[(gy + half_B) * myV + (gx + half_B)]
        = share_D[ly + half_B][lx + half_B];
}

__global__ void phase_2(int *d_D, int round, int myV) {
    if (blockIdx.x == round) return;

    // 2D shared memory
    __shared__ int pivot_D[B][B];
    __shared__ int row_D[B][B];
    __shared__ int col_D[B][B];

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    // 1) Load pivot block
    {
        int gx = round * B + lx;
        int gy = round * B + ly;

        pivot_D[ly][lx]                    = d_D[gy * myV + gx];
        pivot_D[ly][lx + half_B]           = d_D[gy * myV + (gx + half_B)];
        pivot_D[ly + half_B][lx]          = d_D[(gy + half_B) * myV + gx];
        pivot_D[ly + half_B][lx + half_B] = d_D[(gy + half_B) * myV + (gx + half_B)];
    }

    // 2) Load row block
    {
        int gx = blockIdx.x * B + lx;
        int gy = round * B + ly;

        row_D[ly][lx]                    = d_D[gy * myV + gx];
        row_D[ly][lx + half_B]           = d_D[gy * myV + (gx + half_B)];
        row_D[ly + half_B][lx]          = d_D[(gy + half_B) * myV + gx];
        row_D[ly + half_B][lx + half_B] = d_D[(gy + half_B) * myV + (gx + half_B)];
    }

    // 3) Load col block
    {
        int gx = round * B + lx;
        int gy = blockIdx.x * B + ly;

        col_D[ly][lx]                    = d_D[gy * myV + gx];
        col_D[ly][lx + half_B]           = d_D[gy * myV + (gx + half_B)];
        col_D[ly + half_B][lx]          = d_D[(gy + half_B) * myV + gx];
        col_D[ly + half_B][lx + half_B] = d_D[(gy + half_B) * myV + (gx + half_B)];
    }

    __syncthreads();

    // 4) Row & Col updates with pivot
    #pragma unroll 32
    for (int k = 0; k < B; ++k) {
        // row
        row_D[ly][lx] 
            = min(row_D[ly][lx], 
                  pivot_D[ly][k] + row_D[k][lx]);

        row_D[ly][lx + half_B]
            = min(row_D[ly][lx + half_B], 
                  pivot_D[ly][k] + row_D[k][lx + half_B]);

        row_D[ly + half_B][lx]
            = min(row_D[ly + half_B][lx],
                  pivot_D[ly + half_B][k] + row_D[k][lx]);

        row_D[ly + half_B][lx + half_B]
            = min(row_D[ly + half_B][lx + half_B],
                  pivot_D[ly + half_B][k] + row_D[k][lx + half_B]);

        // col
        col_D[ly][lx]
            = min(col_D[ly][lx], 
                  col_D[ly][k] + pivot_D[k][lx]);

        col_D[ly][lx + half_B]
            = min(col_D[ly][lx + half_B], 
                  col_D[ly][k] + pivot_D[k][lx + half_B]);

        col_D[ly + half_B][lx]
            = min(col_D[ly + half_B][lx],
                  col_D[ly + half_B][k] + pivot_D[k][lx]);

        col_D[ly + half_B][lx + half_B]
            = min(col_D[ly + half_B][lx + half_B],
                  col_D[ly + half_B][k] + pivot_D[k][lx + half_B]);
    }

    // 5) Write col block back
    {
        int gx = round * B + lx;
        int gy = blockIdx.x * B + ly;

        d_D[gy * myV + gx]
            = col_D[ly][lx];
        d_D[gy * myV + (gx + half_B)]
            = col_D[ly][lx + half_B];
        d_D[(gy + half_B) * myV + gx]
            = col_D[ly + half_B][lx];
        d_D[(gy + half_B) * myV + (gx + half_B)]
            = col_D[ly + half_B][lx + half_B];
    }

    // 6) Write row block back
    {
        int gx = blockIdx.x * B + lx; 
        int gy = round * B + ly;

        d_D[gy * myV + gx]
            = row_D[ly][lx];
        d_D[gy * myV + (gx + half_B)]
            = row_D[ly][lx + half_B];
        d_D[(gy + half_B) * myV + gx]
            = row_D[ly + half_B][lx];
        d_D[(gy + half_B) * myV + (gx + half_B)]
            = row_D[ly + half_B][lx + half_B];
    }
}

__global__ void phase_3(int *d_D, int pivot, int myV, int y_off) {
    if ((blockIdx.x == pivot) || ((blockIdx.y + y_off) == pivot)) {
        return;
    }

    __shared__ int row_D[B][B];
    __shared__ int col_D[B][B];

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int rowMul = ly * B; 
    int rowMul2 = (ly + half_B) * B;

    // 1) Load row-block
    {
        int gx = blockIdx.x * B + lx; 
        int gy = pivot * B + ly;

        row_D[ly][lx]                = d_D[gy * myV + gx];
        row_D[ly][lx + half_B]       = d_D[gy * myV + (gx + half_B)];
        row_D[ly + half_B][lx]      = d_D[(gy + half_B) * myV + gx];
        row_D[ly + half_B][lx + half_B]
            = d_D[(gy + half_B) * myV + (gx + half_B)];
    }

    // 2) Load col-block
    {
        int gx = pivot * B + lx; 
        int gy = (blockIdx.y + y_off) * B + ly;

        col_D[ly][lx]                = d_D[gy * myV + gx];
        col_D[ly][lx + half_B]       = d_D[gy * myV + (gx + half_B)];
        col_D[ly + half_B][lx]      = d_D[(gy + half_B) * myV + gx];
        col_D[ly + half_B][lx + half_B]
            = d_D[(gy + half_B) * myV + (gx + half_B)];
    }

    // 3) Load the base block
    int baseX = blockIdx.x * B + lx; 
    int baseY = (blockIdx.y + y_off) * B + ly;

    int base0 = d_D[baseY * myV + baseX];
    int base1 = d_D[baseY * myV + (baseX + half_B)];
    int base2 = d_D[(baseY + half_B) * myV + baseX];
    int base3 = d_D[(baseY + half_B) * myV + (baseX + half_B)];

    __syncthreads();

    // 4) partial unrolled FW updates
    #pragma unroll 32
    for (int k = 0; k < B; ++k) {
        base0 = min(base0, col_D[ly][k] + row_D[k][lx]);
        base1 = min(base1, col_D[ly][k] + row_D[k][lx + half_B]);
        base2 = min(base2, col_D[ly + half_B][k] + row_D[k][lx]);
        base3 = min(base3, col_D[ly + half_B][k] + row_D[k][lx + half_B]);
    }

    // 5) Store
    d_D[baseY * myV + baseX]                    = base0;
    d_D[baseY * myV + (baseX + half_B)]         = base1;
    d_D[(baseY + half_B) * myV + baseX]         = base2;
    d_D[(baseY + half_B) * myV + (baseX + half_B)]
        = base3;
}

int main(int argc, char** argv) {

    // read graph
    input(argv[1]);

    int rounds = V / B;
    size_t totalBytes = (size_t)V * V * sizeof(int);

    // Pin memory on host
    cudaHostRegister(D, totalBytes, cudaHostRegisterDefault);

    // We'll store device pointers for 2 GPUs
    int* dArr[2];
    dim3 threads(32, 32);

    #pragma omp parallel num_threads(2)
    {
        int tID = omp_get_thread_num();
        cudaSetDevice(tID);

        cudaMalloc(&dArr[tID], totalBytes);

        // how many blocks in phase_3
        dim3 p3Grid(rounds, rounds / 2);
        if (tID == 1 && (rounds & 1)) {
            p3Grid.y += 1;
        }

        int offsetY = (tID == 0) ? 0 : (rounds / 2);

        // copy portion to device
        cudaMemcpy(
            dArr[tID] + offsetY * B * V,
            D + offsetY * B * V,
            (size_t)p3Grid.y * B * V * sizeof(int),
            cudaMemcpyHostToDevice
        );

        // pivot loop
        for (int pivot = 0; pivot < rounds; pivot++) {
            // if pivot belongs to GPU0, copy pivot => host
            if ((tID == 0) && (pivot < p3Grid.y)) {
                cudaMemcpy(
                    D + pivot * B * V,
                    dArr[tID] + pivot * B * V,
                    (size_t)B * V * sizeof(int),
                    cudaMemcpyDeviceToHost
                );
            }
            // if pivot belongs to GPU1, copy pivot => host
            else if ((tID == 1) && (pivot >= offsetY)) {
                cudaMemcpy(
                    D + pivot * B * V,
                    dArr[tID] + pivot * B * V,
                    (size_t)B * V * sizeof(int),
                    cudaMemcpyDeviceToHost
                );
            }

            #pragma omp barrier

            // if GPU0 doesn't own pivot, fetch from host
            if ((tID == 0) && (pivot >= p3Grid.y)) {
                cudaMemcpy(
                    dArr[tID] + pivot * B * V,
                    D + pivot * B * V,
                    (size_t)B * V * sizeof(int),
                    cudaMemcpyHostToDevice
                );
            }
            // if GPU1 doesn't own pivot, fetch from host
            else if ((tID == 1) && (pivot < offsetY)) {
                cudaMemcpy(
                    dArr[tID] + pivot * B * V,
                    D + pivot * B * V,
                    (size_t)B * V * sizeof(int),
                    cudaMemcpyHostToDevice
                );
            }

            // run phases
            phase_1<<<1, threads>>>(dArr[tID], pivot, V);
            phase_2<<<rounds, threads>>>(dArr[tID], pivot, V);
            phase_3<<<p3Grid, threads>>>(dArr[tID], pivot, V, offsetY);
        }

        // copy final portion back
        cudaMemcpy(
            D + offsetY * B * V,
            dArr[tID] + offsetY * B * V,
            (size_t)p3Grid.y * B * V * sizeof(int),
            cudaMemcpyDeviceToHost
        );

        cudaFree(dArr[tID]);
    }

    // output
    output(argv[2]);

    // cleanup
    cudaHostUnregister(D);
    free(D);

    return 0;
}
