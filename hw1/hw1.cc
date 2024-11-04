#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>  // Needed for std::sort and std::merge
#include <boost/sort/spreadsort/spreadsort.hpp>  // For Boost Spreadsort

// Function to merge and sort two subarrays and keep their respective halves
void merge_and_split(float *left, int left_size, float *right, int right_size, float *tmp, int isLeftSubarray)
{
    int tmp_head = 0;
    int tmp_tail = 0;

    if (isLeftSubarray) {
        int left_ptr = 0;
        int right_ptr = 0;
        
        while (left_ptr < left_size) {
            // If something is inside tmp array, remember to check it first
            if (tmp_head != tmp_tail) {
                tmp[tmp_tail] = left[left_ptr];
                tmp_tail++;

                if (right_ptr < right_size) {
                    if (tmp[tmp_head] < right[right_ptr]) {
                        left[left_ptr] = tmp[tmp_head];
                        left_ptr++;
                        tmp_head++;
                    } else {
                        left[left_ptr] = right[right_ptr];
                        left_ptr++;
                        right_ptr++;
                    }
                } else {
                    left[left_ptr] = tmp[tmp_head];
                    left_ptr++;
                    tmp_head++;
                }
            // tmp array is empty, compare left and right array directly
            } else {
                if (right_ptr < right_size) {
                    if (left[left_ptr] < right[right_ptr]) {
                        left_ptr++;
                    } else {
                        tmp[tmp_tail] = left[left_ptr];
                        tmp_tail++;
                        left[left_ptr] = right[right_ptr];
                        left_ptr++;
                        right_ptr++;
                    }
                } else {
                    left_ptr++;
                }
            }
        }
    } else {
        int left_ptr = left_size - 1;
        int right_ptr = right_size - 1;

        while (right_ptr >= 0) {
            // If something is inside tmp array, remember to check it first
            if (tmp_head != tmp_tail) {
                tmp[tmp_tail] = right[right_ptr];
                tmp_tail++;

                if (left_ptr >= 0) {
                    if (tmp[tmp_head] > left[left_ptr]) {
                        right[right_ptr] = tmp[tmp_head];
                        right_ptr--;
                        tmp_head++;
                    } else {
                        right[right_ptr] = left[left_ptr];
                        right_ptr--;
                        left_ptr--;
                    }
                } else {
                    right[right_ptr] = tmp[tmp_head];
                    right_ptr--;
                    tmp_head++;
                }
            // tmp array is empty, compare left and right array directly
            } else {
                if (left_ptr >= 0) {
                    if (right[right_ptr] > left[left_ptr]) {
                        right_ptr--;
                    } else {
                        tmp[tmp_tail] = right[right_ptr];
                        tmp_tail++;
                        right[right_ptr] = left[left_ptr];
                        right_ptr--;
                        left_ptr--;
                    }
                } else {
                    right_ptr--;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read command line arguments
    int tot_arr_size = atoi(argv[1]); // Total size of the array
    char *input_filename = argv[2];   // Input file name
    char *output_filename = argv[3];  // Output file name

    // Adjust size if total array size is less than the number of processes
    if (tot_arr_size < size) {
        size = tot_arr_size;
    }

    // Calculate base array size and remainder
    int base_arr_size = tot_arr_size / size;
    int remainder = tot_arr_size % size;

    // Determine local array size for each rank
    int local_arr_size = base_arr_size + (rank < remainder ? 1 : 0);

    // Calculate the displacement for each rank's data in the input file
    int offset = rank * base_arr_size + std::min(rank, remainder);

    // Allocate local subarray
    float *data = (float *)malloc(sizeof(float) * local_arr_size);

    // Allocate memory for neighbor's data (maximum possible size)
    int max_neighbor_size = base_arr_size + (remainder > 0 ? 1 : 0);
    float *neighbor_data = (float *)malloc(sizeof(float) * max_neighbor_size);

    // Allocate tmp subarray to place elements when comparing
    float *tmp = (float *)malloc(sizeof(float) * (local_arr_size + max_neighbor_size));

    // Open the input file and read the subarray for this process
    MPI_File input_file, output_file;

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);

    if (rank < size) {
        // Read data from the file at the calculated offset
        MPI_File_read_at(input_file, sizeof(float) * offset, data, local_arr_size, MPI_FLOAT, MPI_STATUS_IGNORE);

        // Initial local sorting using Boost Spreadsort
        boost::sort::spreadsort::spreadsort(data, data + local_arr_size);
    }

    MPI_File_close(&input_file);

    int global_sorted = (size == 1) ? 1 : 0;

    while (!global_sorted)
    {
        int local_sorted = 1; // Assume local subarray is sorted
        float local_min = data[0];                      // Smallest element in this process
        float local_max = data[local_arr_size - 1];     // Largest element in this process
        float neighbor_min, neighbor_max;
        int neighbor_size;

        if (rank < size) {
            // Odd phase
            if (rank % 2 == 1 && rank - 1 >= 0)
            {
                // Exchange boundary elements with left neighbor
                MPI_Sendrecv(&local_min, 1, MPI_FLOAT, rank - 1, 0,
                            &neighbor_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Check if merging is needed
                if (local_min < neighbor_max)
                {
                    // Send local array size and receive neighbor's array size
                    MPI_Sendrecv(&local_arr_size, 1, MPI_INT, rank - 1, 0,
                                &neighbor_size, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Exchange subarrays
                    MPI_Sendrecv(data, local_arr_size, MPI_FLOAT, rank - 1, 0,
                                neighbor_data, neighbor_size, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Perform merging
                    merge_and_split(neighbor_data, neighbor_size, data, local_arr_size, tmp, 0);
                    local_sorted = 0; // Set flag if merging was done
                }
            }
            else if (rank % 2 == 0 && rank + 1 < size)
            {
                // Exchange boundary elements with right neighbor
                MPI_Sendrecv(&local_max, 1, MPI_FLOAT, rank + 1, 0,
                            &neighbor_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Check if merging is needed
                if (local_max > neighbor_min)
                {
                    // Send local array size and receive neighbor's array size
                    MPI_Sendrecv(&local_arr_size, 1, MPI_INT, rank + 1, 0,
                                &neighbor_size, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Exchange subarrays
                    MPI_Sendrecv(data, local_arr_size, MPI_FLOAT, rank + 1, 0,
                                neighbor_data, neighbor_size, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Perform merging
                    merge_and_split(data, local_arr_size, neighbor_data, neighbor_size, tmp, 1);
                    local_sorted = 0; // Set flag if merging was done
                }
            }

            // Even phase
            if (rank % 2 == 0 && rank - 1 >= 0)
            {
                // Exchange boundary elements with left neighbor
                MPI_Sendrecv(&local_min, 1, MPI_FLOAT, rank - 1, 0,
                            &neighbor_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Check if merging is needed
                if (local_min < neighbor_max)
                {
                    // Send local array size and receive neighbor's array size
                    MPI_Sendrecv(&local_arr_size, 1, MPI_INT, rank - 1, 0,
                                &neighbor_size, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Exchange subarrays
                    MPI_Sendrecv(data, local_arr_size, MPI_FLOAT, rank - 1, 0,
                                neighbor_data, neighbor_size, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Perform merging
                    merge_and_split(neighbor_data, neighbor_size, data, local_arr_size, tmp, 0);
                    local_sorted = 0; // Set flag if merging was done
                }
            }
            else if (rank % 2 == 1 && rank + 1 < size)
            {
                // Exchange boundary elements with right neighbor
                MPI_Sendrecv(&local_max, 1, MPI_FLOAT, rank + 1, 0,
                            &neighbor_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Check if merging is needed
                if (local_max > neighbor_min)
                {
                    // Send local array size and receive neighbor's array size
                    MPI_Sendrecv(&local_arr_size, 1, MPI_INT, rank + 1, 0,
                                &neighbor_size, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Exchange subarrays
                    MPI_Sendrecv(data, local_arr_size, MPI_FLOAT, rank + 1, 0,
                                neighbor_data, neighbor_size, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Perform merging
                    merge_and_split(data, local_arr_size, neighbor_data, neighbor_size, tmp, 1);
                    local_sorted = 0; // Set flag if merging was done
                }
            }
        }

        // Check if all processes are globally sorted
        MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    // Write the sorted subarray back to the output file
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    if (rank < size) {
        // Calculate the displacement for each rank's data in the output file
        MPI_File_write_at(output_file, sizeof(float) * offset, data, local_arr_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&output_file);

    // Free allocated memory
    free(data);
    free(neighbor_data);
    free(tmp);

    MPI_Finalize();

    return 0;
}