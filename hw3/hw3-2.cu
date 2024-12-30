#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <pthread.h>
#include <iostream>
#include <zlib.h>

using namespace std;

// Slightly different macro names
#define BLOCK_SIZE 64

// Large distance for unreachable vertices
static const int INF_DISTANCE = (1 << 30) - 1;

// Global variables with new names
int* distMatrixGlobal;
int vertexCountGlobal, edgeCountGlobal;
int paddedVertexCount;
int realVertexCount;

static void loadInputData(const char* fname) {
    FILE* fp = fopen(fname, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open input file.\n");
        exit(EXIT_FAILURE);
    }

    fread(&vertexCountGlobal, sizeof(int), 1, fp);
    fread(&edgeCountGlobal,   sizeof(int), 1, fp);
    realVertexCount = vertexCountGlobal;

    // Ensure multiple of BLOCK_SIZE
    int remainder = vertexCountGlobal % BLOCK_SIZE;
    if (remainder != 0) {
        vertexCountGlobal += (BLOCK_SIZE - remainder);
    }
    paddedVertexCount = vertexCountGlobal;

    size_t totalSize = (size_t)vertexCountGlobal * (size_t)vertexCountGlobal;
    distMatrixGlobal = (int*)malloc(totalSize * sizeof(int));
    if (!distMatrixGlobal) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Initialize to INF_DISTANCE
    for (size_t idx = 0; idx < totalSize; idx++) {
        distMatrixGlobal[idx] = INF_DISTANCE;
    }

    // Diagonal = 0
    for (int i = 0; i < vertexCountGlobal; i++) {
        distMatrixGlobal[(size_t)i * vertexCountGlobal + i] = 0;
    }

    // Read edges
    for (int e = 0; e < edgeCountGlobal; e++) {
        int s, t, w;
        fread(&s, sizeof(int), 1, fp);
        fread(&t, sizeof(int), 1, fp);
        fread(&w, sizeof(int), 1, fp);
        distMatrixGlobal[(size_t)s * vertexCountGlobal + t] = w;
    }
    fclose(fp);
}

static void saveOutputData(const char* fname) {
    FILE* outFile = fopen(fname, "wb");
    if (!outFile) {
        fprintf(stderr, "Cannot open output file.\n");
        exit(EXIT_FAILURE);
    }

    // We only write out the real (original) number of rows and columns
    for (int row = 0; row < realVertexCount; row++) {
        int* rowStart = distMatrixGlobal + (size_t)row * vertexCountGlobal;
        if (fwrite(rowStart, sizeof(int), realVertexCount, outFile) != (size_t)realVertexCount) {
            fprintf(stderr, "Write error on row %d.\n", row);
            fclose(outFile);
            exit(EXIT_FAILURE);
        }
    }
    fclose(outFile);
}

//------------------------------------------------------------------------------
// Phase 1 Kernel
//------------------------------------------------------------------------------
__global__ void floydPhase1(int* __restrict__ gDist, int pivotIdx, int dim) {
    __shared__ int pivotData[BLOCK_SIZE][BLOCK_SIZE];

    int lx = threadIdx.x, ly = threadIdx.y;
    int pivotBase = pivotIdx * BLOCK_SIZE;

    int rowGlobal = pivotBase + ly;
    int colGlobal = pivotBase + lx;

    int rowStart    = rowGlobal * dim;
    int rowStart32  = (rowGlobal + 32) * dim;
    int colStart    = colGlobal;
    int colStart32  = colGlobal + 32;

    int idxTL = rowStart + colStart;
    int idxBL = rowStart32 + colStart;
    int idxTR = rowStart + colStart32;
    int idxBR = rowStart32 + colStart32;

    // Load pivot block into shared memory
    pivotData[ly][lx]         = gDist[idxTL];
    pivotData[ly+32][lx]      = gDist[idxBL];
    pivotData[ly][lx+32]      = gDist[idxTR];
    pivotData[ly+32][lx+32]   = gDist[idxBR];

    __syncthreads();

    // Classic Floyd-Warshall on pivot block
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        int viaTop    = pivotData[ly][k];
        int viaBot    = pivotData[ly+32][k];
        int viaLeft   = pivotData[k][lx];
        int viaRight  = pivotData[k][lx+32];

        int oldTL = pivotData[ly][lx];
        int oldBL = pivotData[ly+32][lx];
        int oldTR = pivotData[ly][lx+32];
        int oldBR = pivotData[ly+32][lx+32];

        int possibleTL = viaTop + viaLeft;
        int possibleBL = viaBot + viaLeft;
        int possibleTR = viaTop + viaRight;
        int possibleBR = viaBot + viaRight;

        pivotData[ly][lx]         = min(oldTL, possibleTL);
        pivotData[ly+32][lx]      = min(oldBL, possibleBL);
        pivotData[ly][lx+32]      = min(oldTR, possibleTR);
        pivotData[ly+32][lx+32]   = min(oldBR, possibleBR);

        __syncthreads();
    }

    // Write results back
    gDist[idxTL] = pivotData[ly][lx];
    gDist[idxBL] = pivotData[ly+32][lx];
    gDist[idxTR] = pivotData[ly][lx+32];
    gDist[idxBR] = pivotData[ly+32][lx+32];
}

// //------------------------------------------------------------------------------
// // Phase 2 Kernel
// //------------------------------------------------------------------------------
// __global__ void floydPhase2(int* __restrict__ gMat, int pivotRound, int V) {
//     // Do nothing for the pivot column
//     if (blockIdx.y == pivotRound) return;

//     // 2) Declare shared-memory tiles for:
//     //    - pivotData: the pivot row/column data
//     //    - rowData:   the block in the same pivot row
//     //    - colData:   the block in the same pivot column
//     __shared__ int pivotData[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ int rowData[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ int colData[BLOCK_SIZE][BLOCK_SIZE];

//     int lx = threadIdx.x, ly = threadIdx.y;

//     int pivotBase = pivotRound * BLOCK_SIZE;
//     int blockBase = blockIdx.y * BLOCK_SIZE;

//     // Compute global row addresses
//     int pivotDataBase = (pivotBase + ly) * V;
//     int rowBase      = pivotDataBase; // same row as pivot
//     int colBase      = (blockBase + ly) * V;

//     int pivotCol = pivotBase + lx;
//     int rowCol   = blockBase + lx;
//     int colCol   = pivotBase + lx;

//     // Indices for sub-blocks
//     int pIdx     = pivotDataBase + pivotCol;
//     int pIdxV32  = pIdx + 32*V;
//     int pIdxH32  = pIdx + 32;
//     int pIdxVH32 = pIdxV32 + 32;

//     int rIdx     = rowBase + rowCol;
//     int rIdxV32  = rIdx + 32*V;
//     int rIdxH32  = rIdx + 32;
//     int rIdxVH32 = rIdxV32 + 32;

//     int cIdx     = colBase + colCol;
//     int cIdxV32  = cIdx + 32*V;
//     int cIdxH32  = cIdx + 32;
//     int cIdxVH32 = cIdxV32 + 32;

//     // Load pivot row
//     pivotData[ly][lx]         = gMat[pIdx];
//     pivotData[ly+32][lx]      = gMat[pIdxV32];
//     pivotData[ly][lx+32]      = gMat[pIdxH32];
//     pivotData[ly+32][lx+32]   = gMat[pIdxVH32];

//     // Load rowData
//     rowData[ly][lx]          = gMat[rIdx];
//     rowData[ly+32][lx]       = gMat[rIdxV32];
//     rowData[ly][lx+32]       = gMat[rIdxH32];
//     rowData[ly+32][lx+32]    = gMat[rIdxVH32];

//     // Load colData
//     colData[ly][lx]          = gMat[cIdx];
//     colData[ly+32][lx]       = gMat[cIdxV32];
//     colData[ly][lx+32]       = gMat[cIdxH32];
//     colData[ly+32][lx+32]    = gMat[cIdxVH32];

//     __syncthreads();

//     // Update rowData and colData
//     #pragma unroll
//     for (int k = 0; k < BLOCK_SIZE; k++) {
//         int pRowTop    = pivotData[ly][k];
//         int pRowBot    = pivotData[ly+32][k];
//         int pColLeft   = pivotData[k][lx];
//         int pColRight  = pivotData[k][lx+32];

//         // rowData
//         int oldVal = rowData[ly][lx];
//         rowData[ly][lx] = min(oldVal, pRowTop + rowData[k][lx]);

//         oldVal = rowData[ly+32][lx];
//         rowData[ly+32][lx] = min(oldVal, pRowBot + rowData[k][lx]);

//         oldVal = rowData[ly][lx+32];
//         rowData[ly][lx+32] = min(oldVal, pRowTop + rowData[k][lx+32]);

//         oldVal = rowData[ly+32][lx+32];
//         rowData[ly+32][lx+32] = min(oldVal, pRowBot + rowData[k][lx+32]);

//         // colData
//         oldVal = colData[ly][lx];
//         colData[ly][lx] = min(oldVal, colData[ly][k] + pColLeft);

//         oldVal = colData[ly+32][lx];
//         colData[ly+32][lx] = min(oldVal, colData[ly+32][k] + pColLeft);

//         oldVal = colData[ly][lx+32];
//         colData[ly][lx+32] = min(oldVal, colData[ly][k] + pColRight);

//         oldVal = colData[ly+32][lx+32];
//         colData[ly+32][lx+32] = min(oldVal, colData[ly+32][k] + pColRight);

//         __syncthreads();
//     }

//     // Write updated data
//     gMat[rIdx]       = rowData[ly][lx];
//     gMat[rIdxV32]    = rowData[ly+32][lx];
//     gMat[rIdxH32]    = rowData[ly][lx+32];
//     gMat[rIdxVH32]   = rowData[ly+32][lx+32];

//     gMat[cIdx]       = colData[ly][lx];
//     gMat[cIdxV32]    = colData[ly+32][lx];
//     gMat[cIdxH32]    = colData[ly][lx+32];
//     gMat[cIdxVH32]   = colData[ly+32][lx+32];
// }

__global__ void floydPhase2(int* __restrict__ gMat, int pivotRound, int V) {
    // 1) If this block is the pivot column, skip it.
    if (blockIdx.y == pivotRound) return;

    // 2) Declare shared-memory tiles for:
    //    - pivotData: the pivot row/column data
    //    - rowData:   the block in the same pivot row
    //    - colData:   the block in the same pivot column
    __shared__ int pivotData[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int rowData[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int colData[BLOCK_SIZE][BLOCK_SIZE];

    // 3) Thread indices within a 2D block of (32,32)
    int lx = threadIdx.x, ly = threadIdx.y;

    // 4) pivotBase is the top-left corner of the pivot block along that row/column
    //    blockBase is the top-left corner of the row/column block we want to update
    int pivotBase = pivotRound * BLOCK_SIZE;
    int blockBase = blockIdx.y * BLOCK_SIZE;

    // 5) Compute the row addresses in global memory.
    //    pivotDataBase: points to the pivot row for threads ly = 0..31 or 32..63
    int pivotDataBase = (pivotBase + ly) * V;
    //    rowBase: we are in the same row as pivot (so pivotDataBase),
    //    colBase: a different row, which is blockBase + ly
    int rowBase = pivotDataBase;
    int colBase = (blockBase + ly) * V;

    // 6) Now, compute the columns in global memory.
    //    pivotCol: the pivot column, rowCol: the "row tile" column, colCol: the pivot column again
    int pivotCol = pivotBase + lx;
    int rowCol   = blockBase + lx;
    int colCol   = pivotBase + lx;

    // ------------------- Sub-block indexing -----------------------
    // Because each BLOCK_SIZE=64 is handled as four 32x32 sub-blocks:
    // top-left (TL), bottom-left (BL), top-right (TR), bottom-right (BR).

    // 7) pivotData sub-block indices
    int pIdx     = pivotDataBase + pivotCol;       // top-left
    int pIdxV32  = pIdx + 32*V;                    // bottom-left
    int pIdxH32  = pIdx + 32;                      // top-right
    int pIdxVH32 = pIdxV32 + 32;                   // bottom-right

    // 8) rowData sub-block indices
    int rIdx     = rowBase + rowCol;
    int rIdxV32  = rIdx + 32*V;
    int rIdxH32  = rIdx + 32;
    int rIdxVH32 = rIdxV32 + 32;

    // 9) colData sub-block indices
    int cIdx     = colBase + colCol;
    int cIdxV32  = cIdx + 32*V;
    int cIdxH32  = cIdx + 32;
    int cIdxVH32 = cIdxV32 + 32;

    // ------------------- Load from global to shared memory -------------------
    // pivotData: the pivot row (or column) tile
    pivotData[ly][lx]         = gMat[pIdx];
    pivotData[ly+32][lx]      = gMat[pIdxV32];
    pivotData[ly][lx+32]      = gMat[pIdxH32];
    pivotData[ly+32][lx+32]   = gMat[pIdxVH32];

    // rowData: the block in the pivot row
    rowData[ly][lx]          = gMat[rIdx];
    rowData[ly+32][lx]       = gMat[rIdxV32];
    rowData[ly][lx+32]       = gMat[rIdxH32];
    rowData[ly+32][lx+32]    = gMat[rIdxVH32];

    // colData: the block in the pivot column
    colData[ly][lx]          = gMat[cIdx];
    colData[ly+32][lx]       = gMat[cIdxV32];
    colData[ly][lx+32]       = gMat[cIdxH32];
    colData[ly+32][lx+32]    = gMat[cIdxVH32];

    __syncthreads();

    // ------------------- Update rowData and colData using pivotData -------------------
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        // pivotData is used to combine row data and column data
        int pRowTop    = pivotData[ly][k];
        int pRowBot    = pivotData[ly+32][k];
        int pColLeft   = pivotData[k][lx];
        int pColRight  = pivotData[k][lx+32];

        // ========== Update rowData ===============
        // rowData[ly][lx] = min(rowData[ly][lx], pRowTop + rowData[k][lx]) ...
        int oldVal = rowData[ly][lx];
        rowData[ly][lx] = min(oldVal, pRowTop + rowData[k][lx]);

        oldVal = rowData[ly+32][lx];
        rowData[ly+32][lx] = min(oldVal, pRowBot + rowData[k][lx]);

        oldVal = rowData[ly][lx+32];
        rowData[ly][lx+32] = min(oldVal, pRowTop + rowData[k][lx+32]);

        oldVal = rowData[ly+32][lx+32];
        rowData[ly+32][lx+32] = min(oldVal, pRowBot + rowData[k][lx+32]);

        // ========== Update colData ===============
        // colData[ly][lx] = min(colData[ly][lx], colData[ly][k] + pColLeft) ...
        oldVal = colData[ly][lx];
        colData[ly][lx] = min(oldVal, colData[ly][k] + pColLeft);

        oldVal = colData[ly+32][lx];
        colData[ly+32][lx] = min(oldVal, colData[ly+32][k] + pColLeft);

        oldVal = colData[ly][lx+32];
        colData[ly][lx+32] = min(oldVal, colData[ly][k] + pColRight);

        oldVal = colData[ly+32][lx+32];
        colData[ly+32][lx+32] = min(oldVal, colData[ly+32][k] + pColRight);

        __syncthreads();
    }

    // ------------------- Write updated rowData/colData back to global memory -------------------
    gMat[rIdx]       = rowData[ly][lx];
    gMat[rIdxV32]    = rowData[ly+32][lx];
    gMat[rIdxH32]    = rowData[ly][lx+32];
    gMat[rIdxVH32]   = rowData[ly+32][lx+32];

    gMat[cIdx]       = colData[ly][lx];
    gMat[cIdxV32]    = colData[ly+32][lx];
    gMat[cIdxH32]    = colData[ly][lx+32];
    gMat[cIdxVH32]   = colData[ly+32][lx+32];
}


//------------------------------------------------------------------------------
// Phase 3 Kernel
//------------------------------------------------------------------------------
__global__ void floydPhase3(int* __restrict__ dArr, int diagRound, int vCount) {
    // Skip pivot row/column
    if ((diagRound == blockIdx.x) || (diagRound == blockIdx.y)) {
        return;
    }

    __shared__ int mainTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int rowTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int colTile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;

    int pivotBase = diagRound * BLOCK_SIZE;
    int bX = blockIdx.x * BLOCK_SIZE;
    int bY = blockIdx.y * BLOCK_SIZE;

    int mainIdx       = (bY + ty) * vCount + (bX + tx);
    int mainIdxV32    = mainIdx + 32*vCount;
    int mainIdxH32    = mainIdx + 32;
    int mainIdxVH32   = mainIdxV32 + 32;

    int rowIdx        = (bY + ty) * vCount + (pivotBase + tx);
    int rowIdxV32     = rowIdx + 32*vCount;
    int rowIdxH32     = rowIdx + 32;
    int rowIdxVH32    = rowIdxV32 + 32;

    int colIdx        = (pivotBase + ty) * vCount + (bX + tx);
    int colIdxV32     = colIdx + 32*vCount;
    int colIdxH32     = colIdx + 32;
    int colIdxVH32    = colIdxV32 + 32;

    // Load main tile
    mainTile[ty][tx]         = dArr[mainIdx];
    mainTile[ty+32][tx]      = dArr[mainIdxV32];
    mainTile[ty][tx+32]      = dArr[mainIdxH32];
    mainTile[ty+32][tx+32]   = dArr[mainIdxVH32];

    // Load row tile
    rowTile[ty][tx]          = dArr[rowIdx];
    rowTile[ty+32][tx]       = dArr[rowIdxV32];
    rowTile[ty][tx+32]       = dArr[rowIdxH32];
    rowTile[ty+32][tx+32]    = dArr[rowIdxVH32];

    // Load col tile
    colTile[ty][tx]          = dArr[colIdx];
    colTile[ty+32][tx]       = dArr[colIdxV32];
    colTile[ty][tx+32]       = dArr[colIdxH32];
    colTile[ty+32][tx+32]    = dArr[colIdxVH32];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        mainTile[ty][tx] = min(mainTile[ty][tx], rowTile[ty][k] + colTile[k][tx]);
        mainTile[ty+32][tx] = min(mainTile[ty+32][tx], rowTile[ty+32][k] + colTile[k][tx]);
        mainTile[ty][tx+32] = min(mainTile[ty][tx+32], rowTile[ty][k] + colTile[k][tx+32]);
        mainTile[ty+32][tx+32] = min(mainTile[ty+32][tx+32], rowTile[ty+32][k] + colTile[k][tx+32]);
    }

    // Store updates
    dArr[mainIdx]       = mainTile[ty][tx];
    dArr[mainIdxV32]    = mainTile[ty+32][tx];
    dArr[mainIdxH32]    = mainTile[ty][tx+32];
    dArr[mainIdxVH32]   = mainTile[ty+32][tx+32];
}

//------------------------------------------------------------------------------
// main()
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    // Load input data
    loadInputData(argv[1]);

    size_t arrBytes = (size_t)vertexCountGlobal * vertexCountGlobal * sizeof(int);

    // Page-locked
    cudaHostRegister(distMatrixGlobal, arrBytes, cudaHostRegisterDefault);

    // Allocate device memory
    int* deviceDist;
    cudaMalloc(&deviceDist, arrBytes);
    cudaMemcpy(deviceDist, distMatrixGlobal, arrBytes, cudaMemcpyHostToDevice);

    // Kernel config
    dim3 threads(32, 32);
    int totalRounds = vertexCountGlobal / BLOCK_SIZE;

    dim3 gridPh2(1, totalRounds);
    dim3 gridPh3(totalRounds, totalRounds);

    // Run the 3-phase Floyd-Warshall
    for (int r = 0; r < totalRounds; r++) {
        // Phase 1
        floydPhase1<<<1, threads>>>(deviceDist, r, vertexCountGlobal);

        // Phase 2
        floydPhase2<<<gridPh2, threads>>>(deviceDist, r, vertexCountGlobal);

        // Phase 3
        floydPhase3<<<gridPh3, threads>>>(deviceDist, r, vertexCountGlobal);
    }

    // Copy results back
    cudaMemcpy(distMatrixGlobal, deviceDist, arrBytes, cudaMemcpyDeviceToHost);
    saveOutputData(argv[2]);

    // Cleanup
    cudaFree(deviceDist);
    free(distMatrixGlobal);
    return 0;
}