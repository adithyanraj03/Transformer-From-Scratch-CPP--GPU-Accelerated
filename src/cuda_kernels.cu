#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <map>
#include <vector>
#include <cstring>
#include "cuda_kernels.cuh"

//tile size for shared memory matmul
#define TILE_SIZE 16
#define BLOCK_SIZE 256

static int g_cuda_initialized = 0;

#define CUDA_CHECK(call) {                                                  \
    cudaError_t err = call;                                                 \
    if(err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
    }                                                                       \
}

// ================================================================
//  GPU MEMORY POOL — caching allocator
// ================================================================

static size_t roundToBucket(size_t bytes) {
    if(bytes <= 512) return 512;
    size_t b = 512;
    while(b < bytes) b <<= 1;
    return b;
}

static std::map<size_t, std::vector<void*>> g_pool_free;
static std::map<void*, size_t> g_pool_allocated;
static size_t g_pool_total_bytes = 0;
static size_t g_pool_hits        = 0;
static size_t g_pool_misses      = 0;

static double* pool_alloc(int numElements) {
    size_t bytes  = numElements * sizeof(double);
    size_t bucket = roundToBucket(bytes);

    auto it = g_pool_free.find(bucket);
    if(it != g_pool_free.end() && !it->second.empty()) {
        void *ptr = it->second.back();
        it->second.pop_back();
        g_pool_allocated[ptr] = bucket;
        g_pool_hits++;
        return (double*)ptr;
    }

    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bucket));
    g_pool_allocated[ptr] = bucket;
    g_pool_total_bytes += bucket;
    g_pool_misses++;
    return (double*)ptr;
}

static void pool_free(double *ptr) {
    if(!ptr) return;
    auto it = g_pool_allocated.find(ptr);
    if(it != g_pool_allocated.end()) {
        size_t bucket = it->second;
        g_pool_free[bucket].push_back(ptr);
        g_pool_allocated.erase(it);
    } else {
        cudaFree(ptr);
    }
}

static void pool_destroy() {
    for(auto &[bucket, ptrs] : g_pool_free) {
        for(void *p : ptrs) cudaFree(p);
    }
    g_pool_free.clear();
    for(auto &[ptr, bucket] : g_pool_allocated) {
        cudaFree(ptr);
    }
    g_pool_allocated.clear();
    g_pool_total_bytes = 0;
}

// ================================================================
//  KERNELS
// ================================================================

//tiled matmul ;shared memory for data reuse
__global__ void kernel_matmul(const double *A, const double *B, double *C,
                               int m, int n, int p) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double sum = 0.0;

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for(int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < n) ? A[row * n + aCol] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < p) ? B[bRow * p + col] : 0.0;

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if(row < m && col < p)
        C[row * p + col] = sum;
}

__global__ void kernel_add(const double *A, const double *B, double *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) C[idx] = A[idx] + B[idx];
}

__global__ void kernel_subtract(const double *A, const double *B, double *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) C[idx] = A[idx] - B[idx];
}

__global__ void kernel_scale(const double *A, double *B, double scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) B[idx] = A[idx] * scalar;
}

__global__ void kernel_hadamard(const double *A, const double *B, double *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) C[idx] = A[idx] * B[idx];
}

__global__ void kernel_transpose(const double *A, double *B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        B[c * rows + r] = A[r * cols + c];
    }
}

//row-wise softmax ;shared memory reduction
__global__ void kernel_softmax(const double *input, double *output, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;

    extern __shared__ double sdata[];

    const double *rowIn = input + row * cols;
    double *rowOut = output + row * cols;

    //find max
    double localMax = -1e30;
    for(int j = threadIdx.x; j < cols; j += blockDim.x) {
        if(rowIn[j] > localMax) localMax = rowIn[j];
    }
    sdata[threadIdx.x] = localMax;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride && sdata[threadIdx.x + stride] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + stride];
        __syncthreads();
    }
    double maxVal = sdata[0];
    __syncthreads();

    //exp and sum
    double localSum = 0.0;
    for(int j = threadIdx.x; j < cols; j += blockDim.x) {
        double val = exp(rowIn[j] - maxVal);
        rowOut[j] = val;
        localSum += val;
    }
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    double sumExp = sdata[0];
    __syncthreads();

    //normalize
    for(int j = threadIdx.x; j < cols; j += blockDim.x)
        rowOut[j] /= sumExp;
}

__global__ void kernel_relu(const double *A, double *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) B[idx] = (A[idx] > 0.0) ? A[idx] : 0.0;
}

__global__ void kernel_relu_derivative(const double *A, double *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) B[idx] = (A[idx] > 0.0) ? 1.0 : 0.0;
}

__global__ void kernel_add_inplace(double *A, const double *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) A[idx] += B[idx];
}

__global__ void kernel_scale_inplace(double *A, double scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) A[idx] *= scalar;
}

__global__ void kernel_fill(double *A, double val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) A[idx] = val;
}

__global__ void kernel_clip(double *A, double minVal, double maxVal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        if(A[idx] < minVal) A[idx] = minVal;
        if(A[idx] > maxVal) A[idx] = maxVal;
    }
}

__global__ void kernel_sqrt(const double *A, double *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) B[idx] = sqrt(A[idx]);
}

//fused AdamW kernel ;one thread per parameter element
//computes moment updates + bias correction + weight update in one pass
__global__ void kernel_adamw(double *param, double *grad, double *m, double *v,
                              double lr, double beta1, double beta2,
                              double bc1, double bc2, double eps, double wd,
                              int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        double g = grad[idx];

        //update first moment: m = beta1*m + (1-beta1)*g
        double mi = beta1 * m[idx] + (1.0 - beta1) * g;
        m[idx] = mi;

        //update second moment: v = beta2*v + (1-beta2)*g^2
        double vi = beta2 * v[idx] + (1.0 - beta2) * g * g;
        v[idx] = vi;

        //bias-corrected moments
        double mHat = mi / bc1;
        double vHat = vi / bc2;

        //AdamW update: param -= lr * (mHat / (sqrt(vHat) + eps) + wd * param)
        param[idx] -= lr * (mHat / (sqrt(vHat) + eps) + wd * param[idx]);
    }
}

// ================================================================
//  HOST API — initialization / memory management
// ================================================================

int cuda_init() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if(err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA: no GPU devices found\n");
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA: using %s (%d SMs, %.1f GB VRAM)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    CUDA_CHECK(cudaSetDevice(0));
    g_cuda_initialized = 1;
    return 0;
}

int cuda_is_available() {
    return g_cuda_initialized;
}

void cuda_shutdown() {
    if(!g_cuda_initialized) return;
    size_t freeBufs = 0;
    for(auto &[b, v] : g_pool_free) freeBufs += v.size();
    printf("GPU pool: %zu cached buffers, %.1f MB held, hits=%zu misses=%zu (%.1f%% hit rate)\n",
           freeBufs, g_pool_total_bytes / (1024.0 * 1024.0),
           g_pool_hits, g_pool_misses,
           (g_pool_hits + g_pool_misses > 0)
               ? 100.0 * g_pool_hits / (g_pool_hits + g_pool_misses) : 0.0);
    pool_destroy();
    g_cuda_initialized = 0;
}

double* cuda_alloc(int numElements) {
    return pool_alloc(numElements);
}

void cuda_free(double *ptr) {
    pool_free(ptr);
}

void cuda_copy_to_device(double *dst, const double *src, int size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(double), cudaMemcpyHostToDevice));
}

void cuda_copy_to_host(double *dst, const double *src, int size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(double), cudaMemcpyDeviceToHost));
}

// ================================================================
//  GPU-RESIDENT WRAPPERS — operate directly on device pointers
//  NO host<->device memcpy! Data stays on GPU between ops.
//  Only cudaDeviceSynchronize after each kernel for correctness.
// ================================================================

void cuda_matmul_gpu(double *dA, double *dB, double *dC, int m, int n, int p) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((p + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);
    kernel_matmul<<<grid, block>>>(dA, dB, dC, m, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_add_gpu(const double *dA, const double *dB, double *dC, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add<<<blocks, BLOCK_SIZE>>>(dA, dB, dC, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_subtract_gpu(const double *dA, const double *dB, double *dC, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_subtract<<<blocks, BLOCK_SIZE>>>(dA, dB, dC, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_scale_gpu(const double *dA, double *dB, double scalar, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_scale<<<blocks, BLOCK_SIZE>>>(dA, dB, scalar, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_hadamard_gpu(const double *dA, const double *dB, double *dC, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_hadamard<<<blocks, BLOCK_SIZE>>>(dA, dB, dC, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_transpose_gpu(const double *dA, double *dB, int rows, int cols) {
    int size = rows * cols;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_transpose<<<blocks, BLOCK_SIZE>>>(dA, dB, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_softmax_gpu(const double *dIn, double *dOut, int rows, int cols) {
    int threadsPerBlock = 1;
    while(threadsPerBlock < cols && threadsPerBlock < 256) threadsPerBlock <<= 1;
    kernel_softmax<<<rows, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(
        dIn, dOut, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_relu_gpu(const double *dA, double *dB, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_relu<<<blocks, BLOCK_SIZE>>>(dA, dB, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_relu_derivative_gpu(const double *dA, double *dB, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_relu_derivative<<<blocks, BLOCK_SIZE>>>(dA, dB, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_add_inplace_gpu(double *dA, const double *dB, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add_inplace<<<blocks, BLOCK_SIZE>>>(dA, dB, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_scale_inplace_gpu(double *dA, double scalar, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_scale_inplace<<<blocks, BLOCK_SIZE>>>(dA, scalar, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_fill_gpu(double *dA, double val, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_fill<<<blocks, BLOCK_SIZE>>>(dA, val, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_clip_gpu(double *dA, double minVal, double maxVal, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_clip<<<blocks, BLOCK_SIZE>>>(dA, minVal, maxVal, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_sqrt_gpu(const double *dA, double *dB, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sqrt<<<blocks, BLOCK_SIZE>>>(dA, dB, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_adamw_gpu(double *param, double *grad, double *m, double *v,
                    double lr, double beta1, double beta2,
                    double bc1, double bc2, double eps, double wd,
                    int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_adamw<<<blocks, BLOCK_SIZE>>>(param, grad, m, v,
                                          lr, beta1, beta2,
                                          bc1, bc2, eps, wd, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}
