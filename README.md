# PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES AY 23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3> Pragadeesh M
<h3>ENTER YOUR REGISTER NO</h3>212225040309
<h3>EX. NO</h3> 3
<h3>DATE</h3> 15/11/2026
<h1> <align=center> PARALLEL REDUCTION USING UNROLLING TECHNIQUES </h3>
  Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.</h3>

## AIM:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Initialization and Memory Allocation
2.	Define the input size n.
3.	Allocate host memory (h_idata and h_odata) for input and output data.
Input Data Initialization
4.	Initialize the input data on the host (h_idata) by assigning a value of 1 to each element.
Device Memory Allocation
5.	Allocate device memory (d_idata and d_odata) for input and output data on the GPU.
Data Transfer: Host to Device
6.	Copy the input data from the host (h_idata) to the device (d_idata) using cudaMemcpy.
Grid and Block Configuration
7.	Define the grid and block dimensions for the kernel launch:
8.	Each block consists of 256 threads.
9.	Calculate the grid size based on the input size n and block size.
10.	Start CPU Timer
11.	Initialize a CPU timer to measure the CPU execution time.
12.	Compute CPU Sum
13.	Calculate the sum of the input data on the CPU using a for loop and store the result in sum_cpu.
14.	Stop CPU Timer
15.	Record the elapsed CPU time.
16.	Start GPU Timer
17.	Initialize a GPU timer to measure the GPU execution time.
Kernel Execution
18.	Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.
Data Transfer: Device to Host
19.	Copy the result data from the device (d_odata) to the host (h_odata) using cudaMemcpy.
20.	Compute GPU Sum
21.	Calculate the final sum on the GPU by summing the elements in h_odata and store the result in sum_gpu.
22.	Stop GPU Timer
23.	Record the elapsed GPU time.
24.	Print Results
25.	Display the computed CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.
Memory Deallocation
26.	Free the allocated host and device memory using free and cudaFree.
27.	Exit
28.	Return from the main function.

## PROGRAM:
TYPE YOUR CODE HERE
~~~
%%writefile reduction.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__global__ void reduceUnrolling8(int *input, long long *output, int n)
{
    extern __shared__ long long sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    long long sum = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        unsigned int pos = idx + i * blockDim.x;
        if (pos < n)
            sum += input[pos];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__ void reduceUnrolling16(int *input, long long *output, int n)
{
    extern __shared__ long long sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + tid;

    long long sum = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++)
    {
        unsigned int pos = idx + i * blockDim.x;
        if (pos < n)
            sum += input[pos];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    int size = 1 << 24;
    int blocksize = 256;

    size_t bytes = size * sizeof(int);

    int *h_idata = (int *)malloc(bytes);

    for (int i = 0; i < size; i++)
        h_idata[i] = rand() & 0xFF;

    long long cpu_sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < size; i++)
        cpu_sum += h_idata[i];

    auto end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double>(end - start).count();

    printf("CPU Sum: %lld\n", cpu_sum);
    printf("CPU Time: %f seconds\n\n", cpu_time);

    int *d_idata;
    long long *d_odata;

    CHECK(cudaMalloc(&d_idata, bytes));

    int grid8 = (size + blocksize * 8 - 1) /
                (blocksize * 8);

    int grid16 = (size + blocksize * 16 - 1) /
                 (blocksize * 16);

    int maxGrid = (grid8 > grid16) ? grid8 : grid16;

    CHECK(cudaMalloc(&d_odata,
                     maxGrid * sizeof(long long)));

    long long *h_odata =
        (long long *)malloc(maxGrid * sizeof(long long));

    CHECK(cudaMemcpy(d_idata, h_idata, bytes,
                     cudaMemcpyHostToDevice));

    long long gpu_sum = 0;

    start = std::chrono::high_resolution_clock::now();

    reduceUnrolling8<<<grid8, blocksize,
        blocksize * sizeof(long long)>>>(
        d_idata, d_odata, size);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    end = std::chrono::high_resolution_clock::now();

    double gpu_time8 =
        std::chrono::duration<double>(end - start).count();

    CHECK(cudaMemcpy(h_odata, d_odata,
        grid8 * sizeof(long long), cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid8; i++)
        gpu_sum += h_odata[i];

    printf("GPU Unrolling 8 Sum: %lld\n", gpu_sum);
    printf("GPU Unrolling 8 Time: %f seconds\n\n", gpu_time8);

    bool result8 = (gpu_sum == cpu_sum);

    gpu_sum = 0;

    start = std::chrono::high_resolution_clock::now();

    reduceUnrolling16<<<grid16, blocksize,
        blocksize * sizeof(long long)>>>(
        d_idata, d_odata, size);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    end = std::chrono::high_resolution_clock::now();

    double gpu_time16 =
        std::chrono::duration<double>(end - start).count();

    CHECK(cudaMemcpy(h_odata, d_odata,
        grid16 * sizeof(long long), cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid16; i++)
        gpu_sum += h_odata[i];

    printf("GPU Unrolling 16 Sum: %lld\n", gpu_sum);
    printf("GPU Unrolling 16 Time: %f seconds\n\n", gpu_time16);

    bool result16 = (gpu_sum == cpu_sum);

    if (result8 && result16)
        printf("TEST PASSED!\n");
    else
        printf("TEST FAILED!\n");

    free(h_idata);
    free(h_odata);

    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    CHECK(cudaDeviceReset());

    return 0;
}
~~~

## OUTPUT:



## RESULT:
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that _________ has executed with less elapsed time than _____________ with blocks_____,______.
