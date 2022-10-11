
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>



using namespace std;
using namespace chrono;

//#define BS 4096
#define BS 8




__global__ void kernel(int* a, int* b, int n)
{

    __shared__ int temp[BS]; 
    int thid = threadIdx.x;
    int pout = 0, pin = 1;
    // загружаем вход в память. 
    temp[pout * n + thid] = (thid > 0) ? a[thid - 1] : 0;// первый элемент массива 0
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; 
        pin = 1 - pout;
        if (thid >= offset)
            temp[pout * n + thid] += temp[pin * n + thid - offset];
        else
            temp[pout * n + thid] = temp[pin * n + thid];
        __syncthreads();
    }
    b[thid] = temp[pout * n + thid]; // записываем выход
}


void testCPU(int* a, int* b, int N)
{

    for (int i = 1; i < N; i++)
    {
        b[i] = a[i - 1] + b[i - 1];
    }
}




    
  /*
__global__ void kernel(int* a, int* b, int n)
{
    __shared__ int temp[BS];
	//int idx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    temp[idx] = a[idx];
    temp[idx + BS] = a[idx + BS];

    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (idx < d)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;

            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }
    
    if (idx == 0)
        temp[n - 1] = 0;

    for (int d = 1; d < n; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (idx < d)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            int t =+ temp[ai];

            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    b[2 * idx] = temp[2 * idx];
    b[2 * idx + 1] = temp[2 * idx + 1];
}

void testCPU(int* a, int* b, int N)
{
    
    for (int i = 1; i < N; i++)
    {
        b[i] = a[i - 1] + b[i - 1];
    }
}
*/


int main()
{


    using clock = std::chrono::system_clock;
    using sec = std::chrono::duration<double, std::milli>;




    int a[BS];
    
    for (int z = 0; z < BS; z++)
    {
        a[z] = z*z;
        printf("%i\ ", a[z]);
    }
    printf("\n");

	int b[BS];
	
    int *d_a, *d_b;

    int bytes = BS * sizeof(int);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
   
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < (BS / 2) - 1; i++)
    {
        kernel << < 1, BS >> > (d_a, d_b, BS);
        cudaMemcpy(d_b, d_b, bytes, cudaMemcpyDeviceToHost);
    }
    
    //kernel << < 1, BS >> > (d_a, d_b, BS);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    
    
  
        
    //for (int i = 0; i < BS; i++) printf("b[%i] = %i\n" /* CPU result 2^%d = %lld\n */, i, b[i]/*, i, b_cpu[i]*/);
    
    const auto before = clock::now();
    testCPU(a, b, BS);
    const sec duration = clock::now() - before;

    for (int i = 0; i < BS; i++) printf("b[%i] = %i\n" /* CPU result 2^%d = %lld\n */, i, b[i]/*, i, b_cpu[i]*/);
    printf("Sum = %i\n" /* CPU result 2^%d = %lld\n */, b[BS-1]/*, i, b_cpu[i]*/);
    printf("Time elapsed on GPU: %f ms\n", gpuTime);
    printf("Time elapsed on CPU: %f ms\n", duration.count());

	return 0;
}