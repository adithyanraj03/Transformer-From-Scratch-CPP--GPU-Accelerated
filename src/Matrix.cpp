#include "Matrix.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "cuda_kernels.cuh"

//GPU threshold ;send everything to GPU, pool handles overhead
static const int GPU_MIN_ELEMENTS = 64;

static bool useGpu(int elements) {
    return cuda_is_available() && elements >= GPU_MIN_ELEMENTS;
}
#endif

// ================================================================
//  GPU-RESIDENT TENSOR HELPERS
//  These manage the d_data device pointer and dirty flags.
//  Data stays on GPU between operations ;only syncs on at()/ptr().
// ================================================================

//upload CPU data to GPU ;allocates d_data if needed
void Matrix::toGpu() {
#ifdef USE_CUDA
    if(!cuda_is_available()) return;
    int sz = this->numRows * this->numCols;
    if(!this->d_data) {
        this->d_data = cuda_alloc(sz);
    }
    if(!this->gpuValid) {
        cuda_copy_to_device(this->d_data, this->data.data(), sz);
        this->gpuValid = true;
    }
    this->gpuDirty = false;
#endif
}

//download GPU data to CPU ;only if GPU has newer data
void Matrix::toHost() {
#ifdef USE_CUDA
    if(this->gpuDirty && this->d_data) {
        int sz = this->numRows * this->numCols;
        cuda_copy_to_host(this->data.data(), this->d_data, sz);
        this->gpuDirty = false;
    }
#endif
}

//release GPU memory
void Matrix::freeGpu() {
#ifdef USE_CUDA
    if(this->d_data) {
        if(this->gpuDirty) this->toHost();  //save data first
        cuda_free(this->d_data);
        this->d_data   = nullptr;
        this->gpuValid = false;
        this->gpuDirty = false;
    }
#endif
}

//get device pointer ;uploads if needed
double* Matrix::gpu() {
#ifdef USE_CUDA
    if(!cuda_is_available()) return nullptr;
    int sz = this->numRows * this->numCols;
    if(!this->d_data) {
        this->d_data = cuda_alloc(sz);
        this->gpuValid = false;
    }
    if(!this->gpuValid) {
        cuda_copy_to_device(this->d_data, this->data.data(), sz);
        this->gpuValid = true;
    }
    return this->d_data;
#else
    return nullptr;
#endif
}

// ---- constructors ----

//random init ;uniform distribution [-1,1] scaled
Matrix::Matrix(int numRows, int numCols, bool randomize) {
    this->numRows   = numRows;
    this->numCols   = numCols;
    this->d_data    = nullptr;
    this->gpuValid  = false;
    this->gpuDirty  = false;
    this->data.resize(numRows * numCols, 0.0);

    if(randomize) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-1.0, 1.0);
        for(int i = 0; i < numRows * numCols; i++) {
            this->data[i] = dist(gen);
        }
    }
}

//constant init
Matrix::Matrix(int numRows, int numCols, double initVal) {
    this->numRows   = numRows;
    this->numCols   = numCols;
    this->d_data    = nullptr;
    this->gpuValid  = false;
    this->gpuDirty  = false;
    this->data.resize(numRows * numCols, initVal);
}

//from flat data pointer
Matrix::Matrix(const double* flatData, int numRows, int numCols) {
    this->numRows   = numRows;
    this->numCols   = numCols;
    this->d_data    = nullptr;
    this->gpuValid  = false;
    this->gpuDirty  = false;
    this->data.assign(flatData, flatData + numRows * numCols);
}

//copy constructor ;copies CPU data, does NOT copy GPU data
Matrix::Matrix(const Matrix &m) {
    this->numRows   = m.numRows;
    this->numCols   = m.numCols;
    this->d_data    = nullptr;
    this->gpuValid  = false;
    this->gpuDirty  = false;
    //if source has dirty GPU data, sync it first
    if(m.gpuDirty) {
        const_cast<Matrix&>(m).toHost();
    }
    this->data = m.data;
}

Matrix::~Matrix() {
#ifdef USE_CUDA
    if(this->d_data) {
        cuda_free(this->d_data);
        this->d_data = nullptr;
    }
#endif
}

// ================================================================
//  CORE MATRIX OPS — GPU-RESIDENT VERSION
//  All ops: get GPU pointers for inputs, allocate GPU result,
//  run kernel, mark result as gpuDirty=true (CPU data stale).
//  NO host<->device memcpy between chained operations!
// ================================================================

//matmul: C = A x B ;[m x n] x [n x p] = [m x p]
Matrix* Matrix::multiply(Matrix *a, Matrix *b) {
    assert(a->numCols == b->numRows);
    int m = a->numRows;
    int n = a->numCols;
    int p = b->numCols;

    Matrix *result = new Matrix(m, p, 0.0);

#ifdef USE_CUDA
    if(useGpu(m * p)) {
        double *dA = a->gpu();
        double *dB = b->gpu();
        double *dC = result->gpu();

        cuda_matmul_gpu(dA, dB, dC, m, n, p);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    //CPU fallback ;sync from GPU if needed
    if(a->gpuDirty) a->toHost();
    if(b->gpuDirty) b->toHost();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            double sum = 0.0;
            for(int k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            result->data[i * p + j] = sum;
        }
    }
    return result;
}

//transpose: A^T ;[m x n] -> [n x m]
Matrix* Matrix::transpose(Matrix *m) {
    int rows = m->numRows;
    int cols = m->numCols;
    Matrix *result = new Matrix(cols, rows, 0.0);

#ifdef USE_CUDA
    if(useGpu(rows * cols)) {
        double *dA = m->gpu();
        double *dB = result->gpu();

        cuda_transpose_gpu(dA, dB, rows, cols);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(m->gpuDirty) m->toHost();
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            result->data[j * rows + i] = m->data[i * cols + j];
        }
    }
    return result;
}

//element-wise add: C = A + B
Matrix* Matrix::add(Matrix *a, Matrix *b) {
    assert(a->numRows == b->numRows && a->numCols == b->numCols);
    int sz = a->size();
    Matrix *result = new Matrix(a->numRows, a->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = a->gpu();
        double *dB = b->gpu();
        double *dC = result->gpu();

        cuda_add_gpu(dA, dB, dC, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(a->gpuDirty) a->toHost();
    if(b->gpuDirty) b->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

//element-wise subtract: C = A - B
Matrix* Matrix::subtract(Matrix *a, Matrix *b) {
    assert(a->numRows == b->numRows && a->numCols == b->numCols);
    int sz = a->size();
    Matrix *result = new Matrix(a->numRows, a->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = a->gpu();
        double *dB = b->gpu();
        double *dC = result->gpu();

        cuda_subtract_gpu(dA, dB, dC, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(a->gpuDirty) a->toHost();
    if(b->gpuDirty) b->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

//scalar multiply: C = A * s
Matrix* Matrix::scale(Matrix *m, double scalar) {
    int sz = m->size();
    Matrix *result = new Matrix(m->numRows, m->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = m->gpu();
        double *dB = result->gpu();

        cuda_scale_gpu(dA, dB, scalar, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(m->gpuDirty) m->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = m->data[i] * scalar;
    }
    return result;
}

//element-wise multiply (hadamard product): C = A ⊙ B
Matrix* Matrix::elementWise(Matrix *a, Matrix *b) {
    assert(a->numRows == b->numRows && a->numCols == b->numCols);
    int sz = a->size();
    Matrix *result = new Matrix(a->numRows, a->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = a->gpu();
        double *dB = b->gpu();
        double *dC = result->gpu();

        cuda_hadamard_gpu(dA, dB, dC, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(a->gpuDirty) a->toHost();
    if(b->gpuDirty) b->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

//alias for elementWise
Matrix* Matrix::hadamard(Matrix *a, Matrix *b) {
    return elementWise(a, b);
}

// ---- activation and math ops ----

//row-wise softmax ;numerically stable
Matrix* Matrix::softmax(Matrix *m) {
    int rows = m->numRows;
    int cols = m->numCols;
    int sz = rows * cols;
    Matrix *result = new Matrix(rows, cols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dIn  = m->gpu();
        double *dOut = result->gpu();

        cuda_softmax_gpu(dIn, dOut, rows, cols);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(m->gpuDirty) m->toHost();
    for(int i = 0; i < rows; i++) {
        double maxVal = m->data[i * cols];
        for(int j = 1; j < cols; j++) {
            if(m->data[i * cols + j] > maxVal) maxVal = m->data[i * cols + j];
        }
        double sumExp = 0.0;
        for(int j = 0; j < cols; j++) {
            result->data[i * cols + j] = exp(m->data[i * cols + j] - maxVal);
            sumExp += result->data[i * cols + j];
        }
        for(int j = 0; j < cols; j++) {
            result->data[i * cols + j] /= sumExp;
        }
    }
    return result;
}

//element-wise ReLU: max(0, x)
Matrix* Matrix::relu(Matrix *m) {
    int sz = m->size();
    Matrix *result = new Matrix(m->numRows, m->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = m->gpu();
        double *dB = result->gpu();

        cuda_relu_gpu(dA, dB, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(m->gpuDirty) m->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = max(0.0, m->data[i]);
    }
    return result;
}

//ReLU derivative: 1 if x > 0, else 0
Matrix* Matrix::reluDerivative(Matrix *m) {
    int sz = m->size();
    Matrix *result = new Matrix(m->numRows, m->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = m->gpu();
        double *dB = result->gpu();

        cuda_relu_derivative_gpu(dA, dB, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(m->gpuDirty) m->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = (m->data[i] > 0.0) ? 1.0 : 0.0;
    }
    return result;
}

//element-wise sqrt
Matrix* Matrix::sqrt_m(Matrix *m) {
    int sz = m->size();
    Matrix *result = new Matrix(m->numRows, m->numCols, 0.0);

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = m->gpu();
        double *dB = result->gpu();

        cuda_sqrt_gpu(dA, dB, sz);

        result->gpuValid = true;
        result->gpuDirty = true;
        return result;
    }
#endif

    if(m->gpuDirty) m->toHost();
    for(int i = 0; i < sz; i++) {
        result->data[i] = sqrt(m->data[i]);
    }
    return result;
}

// ---- utility ----

void Matrix::setValue(int r, int c, double val) {
    if(this->gpuDirty) this->toHost();
    this->data[r * this->numCols + c] = val;
    this->gpuValid = false;  //CPU data changed, GPU copy is stale
}

double Matrix::getValue(int r, int c) {
    if(this->gpuDirty) this->toHost();
    return this->data[r * this->numCols + c];
}

void Matrix::printMatrix(string label) {
    if(this->gpuDirty) this->toHost();
    cout << label << " [" << this->numRows << "x" << this->numCols << "]:" << endl;
    for(int i = 0; i < this->numRows; i++) {
        for(int j = 0; j < this->numCols; j++) {
            cout << this->data[i * this->numCols + j] << "\t";
        }
        cout << endl;
    }
}

//slice submatrix ;rows [rowStart, rowEnd), cols [colStart, colEnd)
Matrix* Matrix::slice(int rowStart, int rowEnd, int colStart, int colEnd) {
    if(this->gpuDirty) this->toHost();
    int r = rowEnd - rowStart;
    int c = colEnd - colStart;
    Matrix *result = new Matrix(r, c, 0.0);
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            result->data[i * c + j] = this->data[(rowStart + i) * this->numCols + (colStart + j)];
        }
    }
    return result;
}

//concatenate matrices along axis ;0=rows, 1=cols
Matrix* Matrix::concat(vector<Matrix*> matrices, int axis) {
    if(matrices.empty()) return nullptr;

    for(auto m : matrices) {
        if(m->gpuDirty) m->toHost();
    }

    if(axis == 1) {
        int totalCols = 0;
        int rows = matrices[0]->numRows;
        for(auto m : matrices) totalCols += m->numCols;

        Matrix *result = new Matrix(rows, totalCols, 0.0);
        int colOffset = 0;
        for(auto m : matrices) {
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < m->numCols; j++) {
                    result->data[i * totalCols + colOffset + j] = m->data[i * m->numCols + j];
                }
            }
            colOffset += m->numCols;
        }
        return result;
    } else {
        int totalRows = 0;
        int cols = matrices[0]->numCols;
        for(auto m : matrices) totalRows += m->numRows;

        Matrix *result = new Matrix(totalRows, cols, 0.0);
        int rowOffset = 0;
        for(auto m : matrices) {
            for(int i = 0; i < m->numRows; i++) {
                for(int j = 0; j < cols; j++) {
                    result->data[(rowOffset + i) * cols + j] = m->data[i * cols + j];
                }
            }
            rowOffset += m->numRows;
        }
        return result;
    }
}

Matrix* Matrix::zeros(int rows, int cols) {
    return new Matrix(rows, cols, 0.0);
}

Matrix* Matrix::ones(int rows, int cols) {
    return new Matrix(rows, cols, 1.0);
}

//xavier/glorot init ;scale = sqrt(2 / (fanIn + fanOut))
void Matrix::xavierInit(int fanIn, int fanOut) {
    double scale = sqrt(2.0 / (fanIn + fanOut));
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, scale);
    for(int i = 0; i < this->numRows * this->numCols; i++) {
        this->data[i] = dist(gen);
    }
    this->gpuValid = false;
}

//he init ;scale = sqrt(2 / fanIn)
void Matrix::heInit(int fanIn) {
    double scale = sqrt(2.0 / fanIn);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, scale);
    for(int i = 0; i < this->numRows * this->numCols; i++) {
        this->data[i] = dist(gen);
    }
    this->gpuValid = false;
}

//in-place add ;accumulate gradients
void Matrix::addInPlace(Matrix *other) {
    assert(this->numRows == other->numRows && this->numCols == other->numCols);
    int sz = this->size();

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = this->gpu();
        double *dB = other->gpu();

        cuda_add_inplace_gpu(dA, dB, sz);

        this->gpuValid = true;
        this->gpuDirty = true;
        return;
    }
#endif

    if(this->gpuDirty) this->toHost();
    if(other->gpuDirty) other->toHost();
    for(int i = 0; i < sz; i++) {
        this->data[i] += other->data[i];
    }
    this->gpuValid = false;
}

//in-place scale
void Matrix::scaleInPlace(double scalar) {
    int sz = this->size();

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = this->gpu();

        cuda_scale_inplace_gpu(dA, scalar, sz);

        this->gpuValid = true;
        this->gpuDirty = true;
        return;
    }
#endif

    if(this->gpuDirty) this->toHost();
    for(int i = 0; i < sz; i++) {
        this->data[i] *= scalar;
    }
    this->gpuValid = false;
}

//zero out all values
void Matrix::zeroOut() {
#ifdef USE_CUDA
    if(this->d_data && cuda_is_available()) {
        cuda_fill_gpu(this->d_data, 0.0, this->size());
        this->gpuValid = true;
        this->gpuDirty = true;
        return;
    }
#endif
    fill(this->data.begin(), this->data.end(), 0.0);
    this->gpuValid = false;
}

//reshape ;total elements must match
Matrix* Matrix::reshape(int newRows, int newCols) {
    assert(this->numRows * this->numCols == newRows * newCols);
    if(this->gpuDirty) this->toHost();
    Matrix *result = new Matrix(newRows, newCols, 0.0);
    result->data = this->data;
    return result;
}

//clip values to [minVal, maxVal]
void Matrix::clip(double minVal, double maxVal) {
    int sz = this->size();

#ifdef USE_CUDA
    if(useGpu(sz)) {
        double *dA = this->gpu();

        cuda_clip_gpu(dA, minVal, maxVal, sz);

        this->gpuValid = true;
        this->gpuDirty = true;
        return;
    }
#endif

    if(this->gpuDirty) this->toHost();
    for(int i = 0; i < sz; i++) {
        if(this->data[i] < minVal) this->data[i] = minVal;
        if(this->data[i] > maxVal) this->data[i] = maxVal;
    }
    this->gpuValid = false;
}
