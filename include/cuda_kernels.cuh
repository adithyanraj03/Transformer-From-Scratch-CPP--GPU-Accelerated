#ifndef _CUDA_KERNELS_CUH_
#define _CUDA_KERNELS_CUH_

// ================================================================
//  GPU-RESIDENT CUDA KERNEL API
//  All _gpu functions operate DIRECTLY on device pointers.
//  No host<->device memcpy ;data stays on GPU between ops.
//  Matrix class manages upload/download via toGpu()/toHost().
// ================================================================

#ifdef __cplusplus
extern "C" {
#endif

// ---- device-pointer ops (GPU-resident, no memcpy) ----

//matmul: C = A x B on device ;[m x n] x [n x p] = [m x p]
void cuda_matmul_gpu(double *dA, double *dB, double *dC, int m, int n, int p);

//element-wise add on device: dC = dA + dB
void cuda_add_gpu(const double *dA, const double *dB, double *dC, int size);

//element-wise subtract on device: dC = dA - dB
void cuda_subtract_gpu(const double *dA, const double *dB, double *dC, int size);

//scalar scale on device: dB = dA * scalar
void cuda_scale_gpu(const double *dA, double *dB, double scalar, int size);

//hadamard on device: dC = dA ⊙ dB
void cuda_hadamard_gpu(const double *dA, const double *dB, double *dC, int size);

//transpose on device: dB = dA^T
void cuda_transpose_gpu(const double *dA, double *dB, int rows, int cols);

//row-wise softmax on device
void cuda_softmax_gpu(const double *dIn, double *dOut, int rows, int cols);

//ReLU on device: dB = max(0, dA)
void cuda_relu_gpu(const double *dA, double *dB, int size);

//ReLU derivative on device: dB = (dA > 0) ? 1 : 0
void cuda_relu_derivative_gpu(const double *dA, double *dB, int size);

//in-place add on device: dA += dB
void cuda_add_inplace_gpu(double *dA, const double *dB, int size);

//in-place scale on device: dA *= scalar
void cuda_scale_inplace_gpu(double *dA, double scalar, int size);

//fill on device: dA[i] = val
void cuda_fill_gpu(double *dA, double val, int size);

//clip on device: dA = clamp(dA, min, max)
void cuda_clip_gpu(double *dA, double minVal, double maxVal, int size);

//element-wise sqrt on device: dB = sqrt(dA)
void cuda_sqrt_gpu(const double *dA, double *dB, int size);

//fused AdamW update on device ;one kernel for all elements
//param -= lr * (mHat / (sqrt(vHat) + eps) + wd * param)
void cuda_adamw_gpu(double *param, double *grad, double *m, double *v,
                    double lr, double beta1, double beta2,
                    double bc1, double bc2, double eps, double wd,
                    int size);

// ---- memory management ----

//initialize CUDA ;returns 0 on success
int cuda_init();

//check if CUDA is available
int cuda_is_available();

//shutdown ;free pools, print stats
void cuda_shutdown();

//allocate device memory from pool
double* cuda_alloc(int numElements);

//free device memory back to pool
void cuda_free(double *ptr);

//host<->device copy (used by Matrix::toGpu/toHost)
void cuda_copy_to_device(double *dst, const double *src, int size);
void cuda_copy_to_host(double *dst, const double *src, int size);

#ifdef __cplusplus
}
#endif

#endif
