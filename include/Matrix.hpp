#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include <functional>

using namespace std;

// ================================================================
//  GPU-RESIDENT TENSOR SYSTEM
//  Each Matrix has an optional GPU mirror (d_data).
//  Operations keep data on GPU between calls to avoid PCIe
//  round-trips. Data only moves to CPU when at()/ptr() is called.
//
//  Flow: toGpu() uploads, toHost() downloads, gpu() returns device ptr
//  Flag 'gpuDirty' means GPU has newer data than CPU.
//  Flag 'gpuValid' means d_data is allocated and has valid data.
// ================================================================

class Matrix {
public:
    int                 numRows;
    int                 numCols;
    vector<double>      data;       //flat row-major ;data[i*numCols+j]

    //GPU mirror state
    double*             d_data;     //device pointer (nullptr if not on GPU)
    bool                gpuValid;   //true if d_data has valid data
    bool                gpuDirty;   //true if GPU data is newer than CPU data

    //sync helpers ;call before CPU access or GPU access
    void                toGpu();    //upload CPU data to GPU (if not already valid)
    void                toHost();   //download GPU data to CPU (if gpuDirty)
    void                freeGpu();  //release GPU memory
    double*             gpu();      //get device pointer ;uploads if needed

    //inline accessors ;auto-sync from GPU if dirty
    inline double& at(int r, int c) {
        if(gpuDirty) toHost();      //pull latest from GPU
        return data[r * numCols + c];
    }
    inline const double& at(int r, int c) const {
        if(gpuDirty) const_cast<Matrix*>(this)->toHost();
        return data[r * numCols + c];
    }

    //raw pointer to CPU buffer ;auto-sync from GPU if dirty
    inline double* ptr() {
        if(gpuDirty) toHost();
        return data.data();
    }
    inline const double* ptr() const {
        if(gpuDirty) const_cast<Matrix*>(this)->toHost();
        return data.data();
    }
    inline int size() const { return numRows * numCols; }

    //constructors
    Matrix(int numRows, int numCols, bool randomize);
    Matrix(int numRows, int numCols, double initVal);
    Matrix(const double* flatData, int numRows, int numCols);
    Matrix(const Matrix &m);
    ~Matrix();

    //core matrix ops ;matmul, transpose, add, scale etc
    static Matrix* multiply(Matrix *a, Matrix *b);
    static Matrix* transpose(Matrix *m);
    static Matrix* add(Matrix *a, Matrix *b);
    static Matrix* subtract(Matrix *a, Matrix *b);
    static Matrix* scale(Matrix *m, double scalar);
    static Matrix* elementWise(Matrix *a, Matrix *b);
    static Matrix* hadamard(Matrix *a, Matrix *b);

    //activation and math ops
    static Matrix* softmax(Matrix *m);
    static Matrix* relu(Matrix *m);
    static Matrix* reluDerivative(Matrix *m);
    static Matrix* sqrt_m(Matrix *m);

    //utility
    void            setValue(int r, int c, double val);
    double          getValue(int r, int c);
    void            printMatrix(string label);
    Matrix*         slice(int rowStart, int rowEnd, int colStart, int colEnd);
    static Matrix*  concat(vector<Matrix*> matrices, int axis);
    static Matrix*  zeros(int rows, int cols);
    static Matrix*  ones(int rows, int cols);

    //xavier/he init ;scaled random weights for stable training
    void            xavierInit(int fanIn, int fanOut);
    void            heInit(int fanIn);

    //accumulate gradients
    void            addInPlace(Matrix *other);
    void            scaleInPlace(double scalar);
    void            zeroOut();

    //flatten/reshape
    Matrix*         reshape(int newRows, int newCols);

    //clip values ;prevent exploding gradients
    void            clip(double minVal, double maxVal);
};

#endif
