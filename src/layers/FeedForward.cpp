#include "FeedForward.hpp"

//constructor ;init weights with xavier, biases to zero
FeedForward::FeedForward(int dModel, int dFF) {
    this->dModel    = dModel;
    this->dFF       = dFF;

    //W1 [dModel x dFF], W2 [dFF x dModel]
    this->W1        = new Matrix(dModel, dFF, true);
    this->b1        = new Matrix(1, dFF, 0.0);
    this->W2        = new Matrix(dFF, dModel, true);
    this->b2        = new Matrix(1, dModel, 0.0);

    //he init for ReLU layers ;sqrt(2/fanIn)
    this->W1->heInit(dModel);
    this->W2->xavierInit(dFF, dModel);

    //gradient buffers
    this->gradW1    = new Matrix(dModel, dFF, 0.0);
    this->gradB1    = new Matrix(1, dFF, 0.0);
    this->gradW2    = new Matrix(dFF, dModel, 0.0);
    this->gradB2    = new Matrix(1, dModel, 0.0);

    this->lastInput     = nullptr;
    this->lastHidden    = nullptr;
    this->lastPreRelu   = nullptr;
}

FeedForward::~FeedForward() {
    delete this->W1;
    delete this->b1;
    delete this->W2;
    delete this->b2;
    delete this->gradW1;
    delete this->gradB1;
    delete this->gradW2;
    delete this->gradB2;
    if(this->lastInput)     delete this->lastInput;
    if(this->lastHidden)    delete this->lastHidden;
    if(this->lastPreRelu)   delete this->lastPreRelu;
}

//add bias to each row ;broadcasting [1 x cols] across [rows x cols]
Matrix* FeedForward::addBias(Matrix *m, Matrix *b) {
    Matrix *result = new Matrix(m->numRows, m->numCols, 0.0);
    for(int i = 0; i < m->numRows; i++) {
        for(int j = 0; j < m->numCols; j++) {
            result->at(i, j) = m->at(i, j) + b->at(0, j);
        }
    }
    return result;
}

//forward: FFN(x) = ReLU(xW1 + b1)W2 + b2
//x [seqLen x dModel] -> hidden [seqLen x dFF] -> output [seqLen x dModel]
Matrix* FeedForward::forward(Matrix *input) {
    //cache input for backward
    if(this->lastInput)     delete this->lastInput;
    if(this->lastHidden)    delete this->lastHidden;
    if(this->lastPreRelu)   delete this->lastPreRelu;

    this->lastInput = new Matrix(*input);

    //hidden = xW1 + b1
    Matrix *h = Matrix::multiply(input, this->W1);
    Matrix *hBias = this->addBias(h, this->b1);
    delete h;

    //cache pre-relu for backward ;needed for relu derivative
    this->lastPreRelu = new Matrix(*hBias);

    //apply ReLU
    Matrix *hRelu = Matrix::relu(hBias);
    delete hBias;
    this->lastHidden = new Matrix(*hRelu);

    //output = hidden * W2 + b2
    Matrix *out = Matrix::multiply(hRelu, this->W2);
    Matrix *outBias = this->addBias(out, this->b2);
    delete hRelu;
    delete out;

    return outBias;
}

//backward: compute gradients for W1,b1,W2,b2 and return dL/dInput
//given dL/dOutput [seqLen x dModel]:
//  dL/dW2 = hidden^T * dL/dOutput
//  dL/db2 = sum_rows(dL/dOutput)
//  dL/dHidden = dL/dOutput * W2^T
//  dL/dPreRelu = dL/dHidden ⊙ relu'(preRelu)
//  dL/dW1 = input^T * dL/dPreRelu
//  dL/db1 = sum_rows(dL/dPreRelu)
//  dL/dInput = dL/dPreRelu * W1^T
Matrix* FeedForward::backward(Matrix *gradOutput) {
    int seqLen = gradOutput->numRows;

    // ---- output layer gradients ----

    //dL/dW2 = lastHidden^T x gradOutput ;[dFF x seqLen] x [seqLen x dModel] = [dFF x dModel]
    Matrix *hiddenT = Matrix::transpose(this->lastHidden);
    Matrix *dW2 = Matrix::multiply(hiddenT, gradOutput);
    this->gradW2->addInPlace(dW2);
    delete hiddenT;
    delete dW2;

    //dL/db2 = column sums of gradOutput
    for(int j = 0; j < this->dModel; j++) {
        double sum = 0.0;
        for(int i = 0; i < seqLen; i++) {
            sum += gradOutput->at(i, j);
        }
        this->gradB2->at(0, j) += sum;
    }

    //dL/dHidden = gradOutput x W2^T ;[seqLen x dModel] x [dModel x dFF] = [seqLen x dFF]
    Matrix *W2T = Matrix::transpose(this->W2);
    Matrix *gradHidden = Matrix::multiply(gradOutput, W2T);
    delete W2T;

    //dL/dPreRelu = dL/dHidden ⊙ relu'(preRelu)
    Matrix *reluDeriv = Matrix::reluDerivative(this->lastPreRelu);
    Matrix *gradPreRelu = Matrix::hadamard(gradHidden, reluDeriv);
    delete gradHidden;
    delete reluDeriv;

    // ---- hidden layer gradients ----

    //dL/dW1 = lastInput^T x gradPreRelu ;[dModel x seqLen] x [seqLen x dFF] = [dModel x dFF]
    Matrix *inputT = Matrix::transpose(this->lastInput);
    Matrix *dW1 = Matrix::multiply(inputT, gradPreRelu);
    this->gradW1->addInPlace(dW1);
    delete inputT;
    delete dW1;

    //dL/db1 = column sums of gradPreRelu
    for(int j = 0; j < this->dFF; j++) {
        double sum = 0.0;
        for(int i = 0; i < seqLen; i++) {
            sum += gradPreRelu->at(i, j);
        }
        this->gradB1->at(0, j) += sum;
    }

    //dL/dInput = gradPreRelu x W1^T ;[seqLen x dFF] x [dFF x dModel] = [seqLen x dModel]
    Matrix *W1T = Matrix::transpose(this->W1);
    Matrix *gradInput = Matrix::multiply(gradPreRelu, W1T);
    delete W1T;
    delete gradPreRelu;

    return gradInput;
}

void FeedForward::zeroGrad() {
    this->gradW1->zeroOut();
    this->gradB1->zeroOut();
    this->gradW2->zeroOut();
    this->gradB2->zeroOut();
}
