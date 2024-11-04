#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>  // Use AVX instructions

#define NUM_THREAD 6

typedef struct {
    int start_row;
    int end_row;
    int width;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int* buffer;
} thread_data_t;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    /* Get MPI rank and size */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* divide the rows among MPI processes */
    int rows_per_proc = height / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? height : start_row + rows_per_proc;

    /* gather results at root process -> rank0 */
    int* global_image = NULL;
    int* local_image = NULL;

    
    /* allocate memory for local image */
    local_image = (int*)malloc(width * (end_row - start_row) * sizeof(int));
    assert(local_image);

    if (rank == 0) {
        global_image = (int *)malloc(sizeof(int) * width * height);
    }

    const int simd_width = 8; // AVX512 processes 8 doubles(64-bit each) at a time (512-bit)
    __m512d two_vec = _mm512_set1_pd(2.0);
    __m512d length_squared_limit_vec = _mm512_set1_pd(4.0);

    /* Parallelize Mandelbrot computation with OpenMP */
    #pragma omp parallel for num_threads(NUM_THREAD) schedule(dynamic)
    for (int j = start_row; j < end_row; j++) {        
        double y0 = j * (upper - lower) / height + lower;

        __m512d y0_vec = _mm512_set1_pd(y0);    // Broadcast y0 across all elements of the vector

        int i = 0;
        
        for (; i < width; i += simd_width) {
            if (i + simd_width > width) {
                break;
            }

            double x0 = i * ((right - left) / width) + left;
            int repeat = 0;

            __m512d x_vec = _mm512_setzero_pd();
            __m512d y_vec = _mm512_setzero_pd();
            __m512d length_squared_vec = _mm512_setzero_pd();

            // Load eight pixels into x0_vec
            __m512d x0_vec = _mm512_set_pd(
                (i + 7) * ((right - left) / width) + left,
                (i + 6) * ((right - left) / width) + left,
                (i + 5) * ((right - left) / width) + left,
                (i + 4) * ((right - left) / width) + left,
                (i + 3) * ((right - left) / width) + left,
                (i + 2) * ((right - left) / width) + left,
                (i + 1) * ((right - left) / width) + left,
                i * ((right - left) / width) + left
            );

            __m512i repeat_vec = _mm512_set1_epi64(0); // Store 8 repeat counters for 8 pixels
            __m512i repeat_add = _mm512_set1_epi64(1); // To perform addition to repeat_vec
            __m512i iters_vec = _mm512_set1_epi64(iters);    // To check whether repeat exceeds iteration count

            int mask = 0xFF; // Each bit corresponds to one element within the vector
                            // true if the squared length is smaller than 4
                            // false otherwise

            __mmask8 mask_vec = 0xFF;

            while (mask != 0x0 && repeat < iters) {
                __m512d x_squared_vec = _mm512_mul_pd(x_vec, x_vec);
                __m512d y_squared_vec = _mm512_mul_pd(y_vec, y_vec);
                __m512d temp_vec = _mm512_sub_pd(x_squared_vec, y_squared_vec);
                temp_vec = _mm512_add_pd(temp_vec, x0_vec);

                // y = 2 * x * y + y0;
                y_vec = _mm512_mul_pd(x_vec, y_vec);
                y_vec = _mm512_fmadd_pd(two_vec, y_vec, y0_vec);  // Fused multiply-add

                x_vec = temp_vec;

                x_squared_vec = _mm512_mul_pd(x_vec, x_vec);
                y_squared_vec = _mm512_mul_pd(y_vec, y_vec);
                length_squared_vec = _mm512_add_pd(x_squared_vec, y_squared_vec);

                repeat_vec = _mm512_mask_add_epi64(repeat_vec, mask_vec, repeat_vec, repeat_add);

                // _CMP_LT_OQ -> less than, ordered, quiet
                // ordered -> consider numbers that are not NaN
                // quiet -> No floating point exceptions will be raised during the comparison
                // If an element in length_squared_vec is less than 4, all bits in mask_vec corresponding
                // to that element is set to true, otherwise, false
                mask_vec = _mm512_cmp_pd_mask(length_squared_vec, length_squared_limit_vec, _CMP_LT_OQ);

                mask = (int)mask_vec;

                repeat++;
            }

            // Extract the elements from the vectors into an array
            long long repeat_arr[8];

            _mm512_storeu_epi64(repeat_arr, repeat_vec);

            local_image[(j - start_row) * width + (i + 7)] = (int)repeat_arr[7];
            local_image[(j - start_row) * width + (i + 6)] = (int)repeat_arr[6];
            local_image[(j - start_row) * width + (i + 5)] = (int)repeat_arr[5];
            local_image[(j - start_row) * width + (i + 4)] = (int)repeat_arr[4];
            local_image[(j - start_row) * width + (i + 3)] = (int)repeat_arr[3];
            local_image[(j - start_row) * width + (i + 2)] = (int)repeat_arr[2];
            local_image[(j - start_row) * width + (i + 1)] = (int)repeat_arr[1];
            local_image[(j - start_row) * width + i] = (int)repeat_arr[0];
        }

        // Calculate the remaining pixels
        if (width % simd_width != 0) {
            for (; i < width; i++) {
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0.0;

                while (repeats < iters && length_squared < 4.0) {
                    double x0 = i * ((right - left) / width) + left;
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                local_image[(j - start_row) * width + i] = repeats;
            }
        }
    }
    
    if (rank == 0) {
        // An array storing the pixel counts rank0 should receive from other ranks
        int* recvcounts = (int *)malloc(sizeof(int) * size);
        int* displs = (int *)malloc(sizeof(int) * size);

        for (int i = 0; i < size; i++) {
            if (i == size - 1) {
                recvcounts[i] = width * (height - rows_per_proc * (size - 1));
            } else {
                recvcounts[i] = width * rows_per_proc;
            }
            displs[i] = i * rows_per_proc * width;
        }

        MPI_Gatherv(local_image, width * (end_row - start_row), MPI_INT, global_image, 
               recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
        free(recvcounts);
        free(displs);

    } else {
        MPI_Gatherv(local_image, width * (end_row - start_row), MPI_INT, NULL, 
               NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }

    
    /* draw image at root process */
    if (rank == 0) {
        write_png(filename, iters, width, height, global_image);
        free(global_image);
    }

    free(local_image);

    MPI_Finalize();

    return 0;
}
