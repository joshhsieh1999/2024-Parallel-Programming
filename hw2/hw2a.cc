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
#include <immintrin.h>  // Use AVX instructions
#include <time.h>       // For timing

typedef struct {
    int thread_id;
    int num_threads;
    int start_row;
    int end_row;
    int width;
    int height;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int* image;
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

void* mandelbrot_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    const int simd_width = 8; // AVX512 processes 8 doubles(64-bit each) at a time (512-bit)
    __m512d two_vec = _mm512_set1_pd(2.0);
    __m512d length_squared_limit_vec = _mm512_set1_pd(4.0);

    for (int j = data->thread_id; j < data->height; j += data->num_threads) {
        double y0 = j * ((data->upper - data->lower) / data->height) + data->lower;
        __m512d y0_vec = _mm512_set1_pd(y0);    // Broadcast y0 across all elements of the vector

        int i = 0;
        for (; i < data->width; i += simd_width) {
            if (i + simd_width > data->width) {
                break;
            }

            __m512d x0_vec = _mm512_set_pd(
                (i + 7) * ((data->right - data->left) / data->width) + data->left,
                (i + 6) * ((data->right - data->left) / data->width) + data->left,
                (i + 5) * ((data->right - data->left) / data->width) + data->left,
                (i + 4) * ((data->right - data->left) / data->width) + data->left,
                (i + 3) * ((data->right - data->left) / data->width) + data->left,
                (i + 2) * ((data->right - data->left) / data->width) + data->left,
                (i + 1) * ((data->right - data->left) / data->width) + data->left,
                i * ((data->right - data->left) / data->width) + data->left
            );

            __m512d x_vec = _mm512_setzero_pd();
            __m512d y_vec = _mm512_setzero_pd();
            __m512d length_squared_vec = _mm512_setzero_pd();
            __m512i repeat_vec = _mm512_set1_epi64(0);
            __m512i repeat_add = _mm512_set1_epi64(1);
            __m512i iters_vec = _mm512_set1_epi64(data->iters);
            __mmask8 mask_vec = 0xFF;
            int mask = 0xFF;
            int repeat = 0;

            while (mask != 0x0 && repeat < data->iters) {
                __m512d x_squared_vec = _mm512_mul_pd(x_vec, x_vec);
                __m512d y_squared_vec = _mm512_mul_pd(y_vec, y_vec);
                __m512d temp_vec = _mm512_sub_pd(x_squared_vec, y_squared_vec);
                temp_vec = _mm512_add_pd(temp_vec, x0_vec);

                y_vec = _mm512_mul_pd(x_vec, y_vec);
                y_vec = _mm512_fmadd_pd(two_vec, y_vec, y0_vec);

                x_vec = temp_vec;
                x_squared_vec = _mm512_mul_pd(x_vec, x_vec);
                y_squared_vec = _mm512_mul_pd(y_vec, y_vec);
                length_squared_vec = _mm512_add_pd(x_squared_vec, y_squared_vec);

                repeat_vec = _mm512_mask_add_epi64(repeat_vec, mask_vec, repeat_vec, repeat_add);
                mask_vec = _mm512_cmp_pd_mask(length_squared_vec, length_squared_limit_vec, _CMP_LT_OQ);
                mask = (int)mask_vec;

                repeat++;
            }

            long long repeat_arr[8];
            _mm512_storeu_epi64(repeat_arr, repeat_vec);

            data->image[j * data->width + (i + 7)] = (int)repeat_arr[7];
            data->image[j * data->width + (i + 6)] = (int)repeat_arr[6];
            data->image[j * data->width + (i + 5)] = (int)repeat_arr[5];
            data->image[j * data->width + (i + 4)] = (int)repeat_arr[4];
            data->image[j * data->width + (i + 3)] = (int)repeat_arr[3];
            data->image[j * data->width + (i + 2)] = (int)repeat_arr[2];
            data->image[j * data->width + (i + 1)] = (int)repeat_arr[1];
            data->image[j * data->width + i] = (int)repeat_arr[0];
        }

        if (data->width % 8 != 0) {
            for (; i < data->width; i++) {
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0.0;

                while (repeats < data->iters && length_squared < 4.0) {
                    double x0 = i * ((data->right - data->left) / data->width) + data->left;
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                data->image[j * data->width + i] = repeats;
            }
        }
    }

    return NULL;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_threads = CPU_COUNT(&cpu_set);

    /* argument parsing */
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* start timing */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* create threads */
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];
    int rows_per_thread = height / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        thread_data[t].thread_id = t;
        thread_data[t].num_threads = num_threads;
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t == num_threads - 1) ? height : (t + 1) * rows_per_thread;
        thread_data[t].width = width;
        thread_data[t].height = height;
        thread_data[t].iters = iters;
        thread_data[t].left = left;
        thread_data[t].right = right;
        thread_data[t].lower = lower;
        thread_data[t].upper = upper;
        thread_data[t].image = image;
        pthread_create(&threads[t], NULL, mandelbrot_thread, &thread_data[t]);
    }

    /* wait for threads to finish */
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);

    /* Cleanup */
    free(image);
    return 0;
}
