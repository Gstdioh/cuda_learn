#include <cuda_runtime.h>
#include <iostream>

const size_t MEMORY_SIZE = 32 * 100;

using Type = uint32_t;
__global__ void kernel(int offset) {
    __shared__ Type sharedMem[MEMORY_SIZE];

    int threadId = threadIdx.x;

    // init shared memory
    if (threadId == 0) {
        for (int i = 0; i < MEMORY_SIZE; i++)
            sharedMem[i] = 0;
    }
    __syncthreads();

    // repeatedly read and write to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++) {
        sharedMem[index] += index * i;
        index += 32;
        index %= MEMORY_SIZE;
    }
}

int main(int argc, char **argv) {
    int offset = 1;

    if (argc > 1) {
        offset = atoi(argv[1]);
    }

    kernel<<<1, 32>>>(offset);
    cudaDeviceSynchronize();
    return 0;
}
