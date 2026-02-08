#include "MultiHeadAttention.hpp"
#include <limits>

//compute scaled dot-product attention for a single head
//Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
//Q,K,V [seqLen x dK] -> output [seqLen x dK]
//also stores attention weights for backward pass
Matrix* MultiHeadAttention::computeScaledDotProduct(Matrix *Q, Matrix *K, Matrix *V,
                                                     Matrix **attnWeightsOut) {
    int seqLen = Q->numRows;
    double scale = sqrt((double)this->dK);

    //scores = Q x K^T / sqrt(d_k) ;[seqLen x seqLen]
    Matrix *KT = Matrix::transpose(K);
    Matrix *scores = Matrix::multiply(Q, KT);
    delete KT;

    //scale by 1/sqrt(d_k)
    scores->scaleInPlace(1.0 / scale);

    //apply causal mask ;upper triangle -> -inf (decoder can't look ahead)
    this->applyCausalMask(scores);

    //softmax along last axis ;each row sums to 1
    Matrix *attnW = Matrix::softmax(scores);
    delete scores;

    //store attention weights for backward
    *attnWeightsOut = attnW;

    //output = attnW x V ;[seqLen x seqLen] x [seqLen x dK] = [seqLen x dK]
    Matrix *output = Matrix::multiply(attnW, V);

    return output;
}

//apply causal mask ;set upper triangle to -inf
//position i can only attend to positions j <= i
void MultiHeadAttention::applyCausalMask(Matrix *scores) {
    double negInf = -1e9;  //large negative ;close enough to -inf for softmax
    for(int i = 0; i < scores->numRows; i++) {
        for(int j = i + 1; j < scores->numCols; j++) {
            scores->at(i, j) = negInf;
        }
    }
}
