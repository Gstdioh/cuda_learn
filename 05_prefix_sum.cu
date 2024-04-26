#include <cuda_runtime.h>
#include <iostream>

#include "freshman.h"

bool check(float *in_host, float *out_host, int n, int NUM_PER_BLOCK) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        if (i % NUM_PER_BLOCK == 0) {
            sum = in_host[i];
        } else {
            sum += in_host[i];
        }
        if (fabs(sum - out_host[i]) > 1e-5) {
            return false;
        }
    }

    return true;
}

template <unsigned int NUM_PER_BLOCK> __global__ void PrefixSum(float *d_int, float *d_out) {
    unsigned int data_end = (blockIdx.x + 1) * NUM_PER_BLOCK;
    extern __shared__ float shm[];

    for (unsigned int data_start = blockIdx.x * NUM_PER_BLOCK; data_start < data_end;
         data_start += blockDim.x) {
        unsigned int data_index = data_start + threadIdx.x;
        shm[threadIdx.x] = d_int[data_index];
        for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
            int mod = threadIdx.x & ((s << 1) - 1); // 这样取余更快
            if (mod >= s) {
                shm[threadIdx.x] += shm[threadIdx.x - mod + s - 1];
            }
            __syncthreads();
        }
        d_out[data_index] = shm[threadIdx.x];
        __syncthreads();
    }

    for (unsigned int data_start = blockIdx.x * NUM_PER_BLOCK + blockDim.x; data_start < data_end;
         data_start += blockDim.x) {
        d_out[data_start + threadIdx.x] += d_out[data_start - 1];
        __syncthreads();
    }
}

int main() {
    int size = 32 * 1024 * 1024;
    const unsigned int block_size = 256;
    const unsigned int NUM_PER_THREAD = 4;
    const unsigned int NUM_PER_BLOCK = NUM_PER_THREAD * block_size; // 表达式的两个值都需要是const
    int grid_size = size / NUM_PER_BLOCK;

    unsigned int nBytes = size * sizeof(float);
    float *in_host = (float *)malloc(nBytes);
    float *in_dev = NULL;
    cudaMalloc((float **)&in_dev, nBytes);

    for (int i = 0; i < size; i++) {
        in_host[i] = 1.0f;
        if (i % 100 == 0) {
            in_host[i] = 1.5f;
        }
    }
    cudaMemcpy(in_dev, in_host, nBytes, cudaMemcpyHostToDevice);

    float *out_host = (float *)malloc(nBytes);
    float *out_dev = NULL;
    cudaMalloc(&out_dev, nBytes);

    PrefixSum<NUM_PER_BLOCK><<<grid_size, block_size, block_size>>>(in_dev, out_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(out_host, out_dev, nBytes, cudaMemcpyDeviceToHost);

    printf("PrefixSum0, the result is %s", check(in_host, out_host, size, NUM_PER_BLOCK) ? "true" : "false");

    free(in_host);
    free(out_host);
    cudaFree(in_dev);
    cudaFree(out_dev);

    cudaDeviceReset();

    return 0;
}