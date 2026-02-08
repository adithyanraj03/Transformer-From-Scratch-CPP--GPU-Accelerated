#include "PositionalEncoding.hpp"

//constructor ;precompute sinusoidal PE table
PositionalEncoding::PositionalEncoding(int maxSeqLen, int dModel) {
    this->maxSeqLen     = maxSeqLen;
    this->dModel        = dModel;
    this->encodingTable = new Matrix(maxSeqLen, dModel, 0.0);
    this->buildTable();
}

PositionalEncoding::~PositionalEncoding() {
    delete this->encodingTable;
}

//build sinusoidal PE table ;called once at init
//PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
void PositionalEncoding::buildTable() {
    for(int pos = 0; pos < this->maxSeqLen; pos++) {
        for(int i = 0; i < this->dModel; i++) {
            //div_term = 10000^(2*(i/2) / d_model)
            double exponent = (2.0 * (i / 2)) / (double)this->dModel;
            double divTerm = pow(10000.0, exponent);

            if(i % 2 == 0) {
                //even index: sin
                this->encodingTable->at(pos, i) = sin((double)pos / divTerm);
            } else {
                //odd index: cos
                this->encodingTable->at(pos, i) = cos((double)pos / divTerm);
            }
        }
    }
}

//forward: add PE to input embeddings
//output[i][j] = input[i][j] + PE[i][j]
Matrix* PositionalEncoding::forward(Matrix *input) {
    int seqLen = input->numRows;
    Matrix *output = new Matrix(seqLen, this->dModel, 0.0);

    for(int i = 0; i < seqLen; i++) {
        for(int j = 0; j < this->dModel; j++) {
            output->at(i, j) = input->at(i, j) + this->encodingTable->at(i, j);
        }
    }

    return output;
}

//backward: PE is constant, gradient passes through unchanged
//dL/dInput = dL/dOutput (identity for the addition)
Matrix* PositionalEncoding::backward(Matrix *gradOutput) {
    //just return a copy ;PE has no learnable params
    return new Matrix(*gradOutput);
}
