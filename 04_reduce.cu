#include <cuda_runtime.h>
#include <iostream>

#include "freshman.h"

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

double check(float *part_sum_host, int block_num, int size) {
    float sum = 0.0f;
    for (int i = 0; i < block_num; i++) {
        sum += part_sum_host[i];
    }
    return fabs(sum - (float)size);
}

// 0. 单个线程进行一个block内的reduce
__global__ void reduce0(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int total_thread_num = gridDim.x * blockDim.x;

    // 将数据数先缩小到thread数
    float sum = 0.0f;
    for (unsigned int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 存入shared memory
    extern __shared__ float shm[];
    // shm[threadIdx.x] = d_in[gtid];
    shm[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < blockDim.x; i++) {
            sum += shm[i];
        }
        d_out[blockIdx.x] = sum;
    }
}

// 1. 多个线程进行相邻值的归约
__global__ void reduce1(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int total_thread_num = gridDim.x * blockDim.x;

    // 将数据数先缩小到thread数
    float sum = 0.0f;
    for (unsigned int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 存入shared memory
    extern __shared__ float shm[];
    // shm[threadIdx.x] = d_in[gtid];
    shm[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            shm[threadIdx.x] += shm[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

// 2. 多个线程进行相邻值的归约，但是多个线程是相邻的
// 解决warp divergence，即多个线程组成的warp中进入的分支相同
__global__ void reduce2(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int total_thread_num = gridDim.x * blockDim.x;

    // 将数据数先缩小到thread数
    float sum = 0.0f;
    for (unsigned int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 存入shared memory
    extern __shared__ float shm[];
    // shm[threadIdx.x] = d_in[gtid];
    shm[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * threadIdx.x; // 选择哪些线程执行
        if (index < blockDim.x) {
            shm[index] += shm[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

// 3. 交错进行归约，解决bank冲突，即不同的线程访问不同的bank（连续线程访问连续bank）
__global__ void reduce3(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int total_thread_num = gridDim.x * blockDim.x;

    // 将数据数先缩小到thread数
    float sum = 0.0f;
    for (unsigned int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 存入shared memory
    extern __shared__ float shm[];
    // shm[threadIdx.x] = d_in[gtid];
    shm[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shm[threadIdx.x] += shm[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

// 4. 相邻两块的数据先使用一个线程相加到一个块的共享内存中，再进行归约
__global__ void reduce4(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x * 2;
    // unsigned int total_thread_num = gridDim.x * blockDim.x;

    // // 将数据数先缩小到thread数
    // float sum = 0.0f;
    // for (unsigned int i = gtid; i < n; i += total_thread_num) {
    //     sum += d_in[i];
    // }

    // 存入shared memory
    extern __shared__ float shm[];
    shm[threadIdx.x] = d_in[gtid] + d_in[gtid + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shm[threadIdx.x] += shm[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

// 5. 最后一个warp进行展开，最后一个warp不需要用同步了
__global__ void reduce5(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x * 2;
    // unsigned int total_thread_num = gridDim.x * blockDim.x;

    // // 将数据数先缩小到thread数
    // float sum = 0.0f;
    // for (unsigned int i = gtid; i < n; i += total_thread_num) {
    //     sum += d_in[i];
    // }

    // 存入shared memory
    extern __shared__ float shm[];
    shm[threadIdx.x] = d_in[gtid] + d_in[gtid + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shm[threadIdx.x] += shm[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        volatile float *tmp = shm;
        tmp[threadIdx.x] += tmp[threadIdx.x + 32];
        tmp[threadIdx.x] += tmp[threadIdx.x + 16];
        tmp[threadIdx.x] += tmp[threadIdx.x + 8];
        tmp[threadIdx.x] += tmp[threadIdx.x + 4];
        tmp[threadIdx.x] += tmp[threadIdx.x + 2];
        tmp[threadIdx.x] += tmp[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

// 6. 将for循环展开到模板参数中，编译时会根据模板参数将不符合的if删除
template <unsigned int blockSize> __global__ void reduce6(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x * 2;
    // unsigned int total_thread_num = gridDim.x * blockDim.x;

    // // 将数据数先缩小到thread数
    // float sum = 0.0f;
    // for (unsigned int i = gtid; i < n; i += total_thread_num) {
    //     sum += d_in[i];
    // }

    // 存入shared memory
    extern __shared__ float shm[];
    shm[threadIdx.x] = d_in[gtid] + d_in[gtid + blockDim.x];
    __syncthreads();

    if (blockSize >= 1024) {
        if (threadIdx.x < 512) {
            shm[threadIdx.x] += shm[threadIdx.x + 512];
        }
        __syncthreads();
    }

    if (blockSize >= 512) {
        if (threadIdx.x < 256) {
            shm[threadIdx.x] += shm[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (threadIdx.x < 128) {
            shm[threadIdx.x] += shm[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (threadIdx.x < 64) {
            shm[threadIdx.x] += shm[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        volatile float *tmp = shm;
        tmp[threadIdx.x] += tmp[threadIdx.x + 32];
        tmp[threadIdx.x] += tmp[threadIdx.x + 16];
        tmp[threadIdx.x] += tmp[threadIdx.x + 8];
        tmp[threadIdx.x] += tmp[threadIdx.x + 4];
        tmp[threadIdx.x] += tmp[threadIdx.x + 2];
        tmp[threadIdx.x] += tmp[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

// 7. 先将数据缩小到总thread数，再进行归约，即合理设置block数
template <unsigned int blockSize> __global__ void reduce7(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int total_thread_num = gridDim.x * blockDim.x;

    // 将数据数先缩小到thread数
    float sum = 0.0f;
    for (unsigned int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 存入shared memory
    extern __shared__ float shm[];
    // shm[threadIdx.x] = d_in[gtid];
    shm[threadIdx.x] = sum;
    __syncthreads();

    if (blockSize >= 1024) {
        if (threadIdx.x < 512) {
            shm[threadIdx.x] += shm[threadIdx.x + 512];
        }
        __syncthreads();
    }

    if (blockSize >= 512) {
        if (threadIdx.x < 256) {
            shm[threadIdx.x] += shm[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (threadIdx.x < 128) {
            shm[threadIdx.x] += shm[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (threadIdx.x < 64) {
            shm[threadIdx.x] += shm[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        volatile float *tmp = shm;
        tmp[threadIdx.x] += tmp[threadIdx.x + 32];
        tmp[threadIdx.x] += tmp[threadIdx.x + 16];
        tmp[threadIdx.x] += tmp[threadIdx.x + 8];
        tmp[threadIdx.x] += tmp[threadIdx.x + 4];
        tmp[threadIdx.x] += tmp[threadIdx.x + 2];
        tmp[threadIdx.x] += tmp[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shm[0];
    }
}

template <unsigned int blockSize> __device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
    return sum;
}

// 8. 使用shuffle，尽量使用shuffle进行线程之间的通信
template <unsigned int blockSize> __global__ void reduce8(float *d_in, float *d_out, unsigned int n) {
    unsigned int gtid = threadIdx.x + blockIdx.x * blockSize;
    unsigned int total_thread_num = gridDim.x * blockSize;

    // 将数据数先缩小到thread数
    float sum = 0.0f;
#pragma unroll
    for (unsigned int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 存入shared memory
    static __shared__ float shm[blockSize / WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    // 对每一个warp进行reduce，结果一个block中有blockSize / WARP_SIZE个结果
    sum = warpReduceSum<blockSize>(sum);

    if (laneId == 0) {
        shm[warpId] = sum;
    }
    __syncthreads();

    // 从shared memory中读取数据到寄存器中，在一个warp中使用shuffle进行reduce
    // 前提是blockSize / WARP_SIZE的值小于等于32
    // 如blockSize = 256, WARP_SIZE = 32, 则blockSize / WARP_SIZE = 8
    // 即对一个warp中的8个thread使用shuffle进行reduce
    sum = (threadIdx.x < blockSize / WARP_SIZE ? shm[laneId] : 0);

    // 注意，因为只剩一个warp，所以不需要显示同步了，其他warp不会进入分支，直接返回了
    // 即这个block下的活跃线程束只剩一个了
    // 现在要对前8个thread进行reduce，是第一个warp中的8个thread
    if (warpId == 0) {
        sum = warpReduceSum<blockSize / WARP_SIZE>(sum);
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sum;
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 32 * 1024 * 1024;
    int NUM_PER_BLOCK = THREAD_PER_BLOCK;
    int block_num = size / NUM_PER_BLOCK;

    // set up execution configuration
    dim3 block(THREAD_PER_BLOCK, 1);
    dim3 grid(block_num, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, block_num);

    // allocate gpu memory
    size_t nBytes = size * sizeof(float);
    float *in_host = (float *)malloc(nBytes);
    float *in_dev;
    cudaMalloc((float **)&in_dev, nBytes);

    // initialize host data
    for (int i = 0; i < size; i++) {
        in_host[i] = 1.0f;
    }
    cudaMemcpy(in_dev, in_host, nBytes, cudaMemcpyHostToDevice);

    // output memory
    float *part_sum_host;
    float *part_sum_dev;

    // warm up
    double start, elaps;
    int tmp = 8;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce0<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("warmup\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce0
    tmp = 1;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce0<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce0\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce1
    tmp = 1;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce1<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce1\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce2
    tmp = 1;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce2<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce2\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce3
    tmp = 1;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce3<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce3\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce4
    tmp = 2;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce4<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce4\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce5
    tmp = 2;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce5<<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce5\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce6
    tmp = 2;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce6<THREAD_PER_BLOCK><<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce6\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce7
    tmp = 8;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce7<THREAD_PER_BLOCK><<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce7\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    // reduce8
    tmp = 64;
    block_num /= tmp;
    part_sum_host = (float *)malloc(block_num * sizeof(float));
    cudaMalloc((float **)&part_sum_dev, block_num * sizeof(float));
    cudaDeviceSynchronize();
    start = cpuSecond();
    reduce8<THREAD_PER_BLOCK><<<block_num, block, block.x * sizeof(float)>>>(in_dev, part_sum_dev, size);
    cudaDeviceSynchronize();
    elaps = cpuSecond() - start;
    cudaMemcpy(part_sum_host, part_sum_dev, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduce7\t elapsed %f msec, errors is %f\n", elaps * 1e3, check(part_sum_host, block_num, size));
    free(part_sum_host);
    cudaFree(part_sum_dev);
    block_num *= tmp;

    cudaFree(in_dev);
    free(in_host);
    cudaDeviceReset();

    return 0;
}