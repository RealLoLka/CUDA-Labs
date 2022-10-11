#include<iostream>
#include<math.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
using namespace chrono;

#define degree 50000



__global__ void kernel(long long* a) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < degree) {
        a[idx] = pow(2, idx);
     };
}

void tcpu(long long *a, int N) {
    int i = 0;
    for (i = 0; i < N; i++) {
        a[i] = pow(2, i);
        }
}



int main() {
    int thread = 1024;
    long long b[degree];
    long long b_cpu[degree];
    long long* dev;
    int bytes = degree * sizeof(long long);

    cudaMalloc(&dev, bytes);

    cudaMemcpy(dev, b, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    kernel << < 50, thread >> > (dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(b, dev, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev);

    const auto before = system_clock::now();
    tcpu(b_cpu, degree);
    const duration<double> sec = system_clock::now() - before;

    for (int i = 0; i < degree; i++) printf("GPU result 2^%d = %lld\n CPU result 2^%d = %lld\n ", i, b[i], i, b_cpu[i]);


    printf("Time elapsed on GPU: %f ms\n",gpuTime);
    printf("Time elapsed on CPU: %f ms\n", sec.count());

    return 0;
}