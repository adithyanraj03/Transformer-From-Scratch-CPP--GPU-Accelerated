#ifndef _MULTI_HEAD_ATTENTION_HPP_
#define _MULTI_HEAD_ATTENTION_HPP_

#include "Matrix.hpp"
#include <vector>

//multi-head self-attention (causal/masked for decoder-only GPT)
//Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
//splits into h heads, computes attention per head, concats, projects
class MultiHeadAttention {
public:
    int         dModel;
    int         nHeads;
    int         dK;             //d_model / n_heads

    //linear projection weights ;W_q,W_k,W_v,W_o all [dModel x dModel]
    Matrix*     Wq;
    Matrix*     Wk;
    Matrix*     Wv;
    Matrix*     Wo;

    //gradients for each weight matrix
    Matrix*     gradWq;
    Matrix*     gradWk;
    Matrix*     gradWv;
    Matrix*     gradWo;

    //cached for backward pass
    Matrix*     lastInput;
    Matrix*     lastQ;
    Matrix*     lastK;
    Matrix*     lastV;
    vector<Matrix*>     attentionWeights;  //softmax outputs per head
    vector<Matrix*>     headOutputs;       //attention output per head
    Matrix*     concatHeads;

    MultiHeadAttention(int dModel, int nHeads);
    ~MultiHeadAttention();

    //forward: x [seqLen x dModel] -> output [seqLen x dModel]
    //causal mask applied ;position i can only attend to positions <= i
    Matrix*     forward(Matrix *input);

    //backward: dL/dOutput -> dL/dInput, accumulate weight gradients
    Matrix*     backward(Matrix *gradOutput);

    void        zeroGrad();

private:
    //compute scaled dot-product attention for a single head
    //Q,K,V all [seqLen x dK] -> output [seqLen x dK], stores attn weights
    Matrix*     computeScaledDotProduct(Matrix *Q, Matrix *K, Matrix *V,
                                        Matrix **attnWeightsOut);

    //apply causal mask ;set upper triangle to -inf before softmax
    void        applyCausalMask(Matrix *scores);
};

#endif
