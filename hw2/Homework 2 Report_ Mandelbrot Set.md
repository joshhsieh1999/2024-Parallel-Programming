---
title: 'Homework 2 Report: Mandelbrot Set'

---

### CS542200 Parallel Programming
### Homework 2 Report: Mandelbrot Set
**Name:** 謝奕謀  
**Student ID:** 112065520

---

### 1. Implementation

#### 1.1 Pthreads Version (hw2a)

**Versions Implemented**:
1. **Version 1: Baseline Mandelbrot (Single-threaded)**  
   - A simple, single-threaded version that sequentially computes the Mandelbrot set without any parallelism or optimizations.
  
2. **Version 2: Mandelbrot with Pthreads (Multithreaded)**  
   - Uses Pthreads to divide and process the workload across multiple threads, enhancing parallel computation.

3. **Version 3: Mandelbrot with Pthreads and SIMD Vectorization**  
   - Builds on Version 2 by incorporating AVX-512 SIMD instructions to process eight double-precision pixels simultaneously, boosting performance.

4. **Version 4: Mandelbrot with Pthreads, SIMD, and Optimized PNG Output**  
   - Adds PNG output optimization to Version 3 by tweaking `write_png()` to reduce filtering and compression time.

**Implementation Details**:
- **Common Approach**:
  - The Mandelbrot computation involves iterating over each pixel to determine set membership.
  - Image data is represented as a 2D array and processed by multiple threads in parallel.

**Key Enhancements and Modifications**:
- **Multithreading with Pthreads**: Allocates **non-contiguous** rows to each thread to balance workload more evenly (applied in Versions 2–4).
- **AVX-512 SIMD**: Enhances pixel computation speed by processing vectors of data (Versions 3 and 4).
- **Optimized PNG Output**:
  ```c
  png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
  png_set_compression_level(png_ptr, 0);
  ```

**Explanation of PNG Functions**:
- **`png_set_filter(png_ptr, 0, PNG_FILTER_NONE);`**
  - **Purpose**: Sets the filtering strategy for rows in the PNG image.
  - **Details**: The PNG format supports various filters that pre-process image data to reduce size by detecting patterns. By setting the filter to `PNG_FILTER_NONE`, no filtering is applied, and raw pixel data is written directly. This reduces the computational overhead during the image write process, leading to faster image output at the cost of potentially larger file sizes.
- **`png_set_compression_level(png_ptr, 0);`**
  - **Purpose**: Sets the level of compression when writing the PNG file.
  - **Details**: Compression levels in PNG range from `0` (no compression) to `9` (maximum compression). Setting the compression level to `0` ensures that the image is written without compression, significantly speeding up the process as the CPU skips data compression. This change prioritizes performance by reducing the time needed for image output, though it may result in a larger file size.

**Reason for Parameter Changes**:
- The adjustments in PNG output functions are aimed at optimizing performance for high-resolution images. The decision to disable filtering (`PNG_FILTER_NONE`) and set the compression level to `0` was made to minimize the time spent in the I/O phase, ensuring that the program's computational efficiency was not negated by slow image writing.

---

### 2. Experiment & Analysis

#### 2.1 Methodology

**(a) System Specifications(Using QCT-Server):**
- **CPU**: Intel Xeon Platinum 8568Y+, AVX-512 support, 96 physical cores, 192 logical cores.
- **RAM**: 471 GB.
- **Storage**: Multiple NVMe drives (total storage capacity exceeding 42 TB).
- **Network**: Ethernet controllers with up to 25 Gbit/s capabilities.

**(b) Performance Metrics**:
- Execution time was measured using `clock_gettime()` for high precision.

#### 2.2 Experimental Setup

**Test Case**:
- **strict29.txt** with parameters: 10000 iterations, coordinate bounds [0.2936331264625346, 0.2938268308813129] (real axis) and [-0.014948477128821249, -0.015051194365620192] (imaginary axis), image size 7680 x 4320.

**Configurations**:
- Thread counts from 1 to 12 were tested to observe scalability.

#### 2.3 Performance Comparison

**Execution Time Results**:
- Execution times were collected for all versions and plotted for comparison.

**Speedup Analysis**:
- Speedup was calculated relative to the single-core performance of each version:
  $$
  \text{Speedup} = \frac{\text{Execution Time for 1 Core}}{\text{Execution Time for Current Core Count}}
  $$

#### 2.4 Plots

- **Execution Time Comparison**:
![execution_time_comparison_12_cores](https://hackmd.io/_uploads/Bye2EO4Zkx.png)

- **Speedup Comparison**:
![strong_scalability_speedup_comparison_96_cores](https://hackmd.io/_uploads/ryqZB_NWkx.png)

- **Load Balance(Pthread + SIMD + Optimized write_png())**:
![execution_times_12_threads](https://hackmd.io/_uploads/Hkw9K0Vbke.png)
![execution_times_8_threads](https://hackmd.io/_uploads/BJP5YC4Wyl.png)
![execution_times_4_threads](https://hackmd.io/_uploads/r1P5YAEWJe.png)

---

### 3. Discussion & Conclusion

#### 3.1 Key Observations
- **Multithreading Efficiency**: Parallelizing with Pthreads provided substantial speedup up to a saturation point where synchronization and management overheads started limiting performance gains.
- **SIMD Impact**: The use of AVX-512 vectorization significantly reduced pixel computation time, balancing the workload across threads and reducing the overall CV.
- **PNG Output Optimization**: By disabling the filtering and compression, it significantly reduced computation time, resulting in more ideal speedups.

#### 3.2 Insights & Lessons Learned
- Effective parallelization requires consideration of both computation and I/O stages to prevent bottlenecks.
- SIMD vectorization is a powerful tool for enhancing parallel computation and balancing workload distribution across threads.
- Higher thread counts require advanced load balancing techniques to avoid synchronization overheads and uneven workloads.

#### 3.3 Challenges
- Aligning data for SIMD operations and managing vectorized data structures posed difficulties but were essential for optimal performance.
- I/O operations introduced challenges in maintaining balanced thread utilization, particularly in high-resolution image outputs.

#### 3.4 Future Work
- Investigating dynamic load balancing to better distribute work among threads such as thread pool.
- Further optimizing I/O operations by exploring multithreaded or asynchronous I/O strategies, for example, use several threads to write into intermediate kernel buffer.

#### 3.5 Proposed Optimization Strategies:
1. **Dynamic Load Balancing**:
   - Implement adaptive scheduling to distribute workload in real-time, improving load distribution and reducing idle times(such as thread pooling).
2. **Multithreaded I/O**:
   - Implement multithreaded image writing to prevent bottlenecks during the output phase, especially in high-core-count scenarios.
4. **Data Management Optimization**:
   - Align data structures with cache lines for minimized access delays and smoother SIMD operation.

---

### 5. Hybrid Version (hw2b)

---

### 1. Implementation (Hybrid Version)

**Overview**:
- The hybrid version leverages **MPI** for process-level parallelism and **OpenMP** for multithreaded computation within each process, combined with **AVX-512 SIMD** instructions for efficient vectorized operations.

**Key Details**:
- **MPI Integration**:
  - Initializes with `MPI_Init` and finalizes with `MPI_Finalize`.
  - Distributes workload across multiple processes, where each process computes non-contiguous rows to improve load balancing.
- **OpenMP Parallelization**:
  - Utilizes `#pragma omp parallel for` with `dynamic` scheduling for optimal thread workload distribution.
- **Vectorization with AVX-512**:
  - Vector operations are conducted in batches of 8 double-precision pixels, improving the throughput of pixel calculations.

**Implementation Highlights**:
```c
#pragma omp parallel for num_threads(NUM_THREAD) schedule(dynamic)
for (int j = rank; j < total_rows; j += num_procs) {
    // Vectorized computation using AVX-512
    __m512d x_vec = _mm512_setzero_pd();
    __m512d y_vec = _mm512_setzero_pd();
    __m512d length_squared_vec = _mm512_setzero_pd();
    ...
    mask_vec = _mm512_cmp_pd_mask(length_squared_vec, length_squared_limit_vec, _CMP_LT_OQ);
}
```

**I/O Operations**:
- Only the root process writes the final image using `write_png`, avoiding redundant file operations by other processes.

**Strengths**:
- **Efficient Parallelism**: Distributing rows among processes and using threads within each process maximizes parallelism.
- **Enhanced Computation**: Vectorized operations with AVX-512 reduce the time spent per pixel.

**Challenges**:
- **Communication Overhead**: Using `MPI_Gatherv` to collect data can introduce delays at scale.
- **Load Imbalance**: Ensuring even workload distribution across processes and threads can be difficult for non-uniform images.

**Potential Optimizations**:
- **Non-blocking MPI Communication**: Use `MPI_Isend` and `MPI_Irecv` for better overlap between computation and communication.
- **Advanced Scheduling**: Experiment with different OpenMP scheduling policies (e.g., `guided`) for varying image sizes.

---

### 2. Experiment & Analysis (Hybrid Version)

#### 2.1 Methodology

**(a) System Specifications**:
- Same as **hw2a** specifications.

**(b) Performance Metrics**:
- Execution times captured using `MPI_Wtime()` for global process timing.

#### 2.2 Experimental Setup

**Test Case**:
- **strict39.txt** with the same parameters used for **hw2a** for consistent comparisons.

**Configurations**:
- Two separate experiments:
  - **Fixed Threads, Varying Processes**: Number of threads fixed at 6; processes varied (1 to 12).
  - **Fixed Processes, Varying Threads**: Number of processes fixed (e.g., 1, 2, 4, 6); threads varied (1 to 12).

---

#### 2.3 Performance Comparison

**Execution Time and Scalability**:
- Execution times were collected and compared across configurations to evaluate the impact of varying threads and processes.

**Speedup Analysis**:
- Speedup was calculated similarly to **hw2a**:
  $$
  \text{Speedup} = \frac{\text{Execution Time for 1 Process/Thread}}{\text{Execution Time for Current Process/Thread Count}}
  $$

---

### 2.4 Plots

- **Execution Time Comparison (Fixed Threads)**:
![execution_time_comparison_hybrid](https://hackmd.io/_uploads/HJM5odVWJl.png)



- **Speedup Plot (Fixed Threads)**:
![speedup_comparison_hybrid](https://hackmd.io/_uploads/Skn5jO4Wyl.png)



- **Execution Time Comparison (Fixed Processes)**:
![execution_time_comparison_threads_varying_processes](https://hackmd.io/_uploads/H1ddwsEZJx.png)


- **Speedup Plot (Fixed Processes)**:
![speedup_comparison_threads_varying_processes](https://hackmd.io/_uploads/H17tDiVZkl.png)


- **Load Balance Plot(6 Threads Each Process)**:
![execution_times_all_processes](https://hackmd.io/_uploads/SJ-XQpN-kl.png)

---

### 3. Discussion & Conclusion (Hybrid Version)

#### 3.1 Key Observations

- **Enhanced Scalability**: The hybrid approach using MPI for process-level parallelism combined with OpenMP for thread-level parallelism demonstrated better scalability than the Pthreads-only implementation (**hw2a**), particularly in multi-node setups. The addition of AVX-512 SIMD instructions further improved the per-thread computation speed.
- **Load Balancing Issues**: Despite using OpenMP's `dynamic` scheduling for threads, the overall workload distribution was impacted due to the contiguous row distribution strategy across processes. This led to some processes handling rows with significantly more computational load while others processed lighter rows, resulting in imbalances. I have also tried other scheduling techniques but `dynamic` has the best performance.

#### 3.2 Challenges

- **Imbalanced Row Distribution**: The main challenge stemmed from distributing rows contiguously across processes. Rows varied in computational intensity, leading to processes completing at different times, causing a load imbalance. The reason I used contiguous rows is because in order to utilize OpenMP, if I use non-contiguous rows some threads will execute duplicated iterations. 
- **Communication Delays**: Processes that finished computation sooner had to wait for others during MPI data gathering, increasing the overall execution time and amplifying the impact of communication overhead.

#### 3.3 Future Work

- **Adaptive Row Distribution**: Implement non-contiguous or dynamic row allocation strategies to distribute rows more evenly across processes and threads, ensuring that computationally heavy and light rows are balanced more effectively. One method is to put non-contiguous rows into contiguous format, compute, then distribute them back to where they belong.