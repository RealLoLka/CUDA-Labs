#include<iostream>
#include<math.h>
#define degree 32
using namespace std;
__global__ void kernel(long long* a) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < degree) {
        a[idx] = pow(2, idx);
     };
}
int main() {
    int thread = 1024;
    long long b[degree];
    long long* dev;
    int bytes = degree * sizeof(long long);

    cudaMalloc(&dev, bytes);

    cudaMemcpy(dev, b, bytes, cudaMemcpyHostToDevice);

    kernel << < 1, thread >> > (dev);

    cudaMemcpy(b, dev, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev);

    for (int i = 0; i < degree; i++) printf("2^%d = %lld\n", i, b[i]);

    return 0;
}












