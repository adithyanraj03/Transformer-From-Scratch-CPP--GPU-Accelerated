#ifndef _EMBEDDING_HPP_
#define _EMBEDDING_HPP_

#include "Matrix.hpp"

//token embedding layer ;lookup table + sqrt(d_model) scaling
//W_emb[vocabSize x dModel] ;each row = embedding vector for that token id
class Embedding {
public:
    int         vocabSize;
    int         dModel;
    Matrix*     weights;        //W_emb [vocabSize x dModel]
    Matrix*     gradWeights;    //dL/dW_emb accumulated

    //cache for backward pass
    vector<int> lastInputIds;

    Embedding(int vocabSize, int dModel);
    ~Embedding();

    //forward: ids -> embedded matrix [seqLen x dModel] ;scaled by sqrt(dModel)
    Matrix*     forward(const vector<int> &inputIds);

    //backward: receive dL/dOutput [seqLen x dModel], accumulate into gradWeights
    void        backward(Matrix *gradOutput);

    //zero gradients
    void        zeroGrad();
};

#endif
