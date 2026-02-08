#ifndef _TRANSFORMER_BLOCK_HPP_
#define _TRANSFORMER_BLOCK_HPP_

#include "MultiHeadAttention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"

//single transformer decoder block (pre-norm variant)
//x -> LayerNorm -> MultiHeadAttention -> +residual -> LayerNorm -> FFN -> +residual
class TransformerBlock {
public:
    int                     dModel;
    int                     nHeads;
    int                     dFF;

    MultiHeadAttention*     attention;
    FeedForward*            ffn;
    LayerNorm*              norm1;      //pre-attention norm
    LayerNorm*              norm2;      //pre-ffn norm

    //cached for backward ;residual connections
    Matrix*                 lastInput;
    Matrix*                 lastNorm1Out;
    Matrix*                 lastAttnOut;
    Matrix*                 lastResidual1;
    Matrix*                 lastNorm2Out;

    TransformerBlock(int dModel, int nHeads, int dFF);
    ~TransformerBlock();

    //forward: x [seqLen x dModel] -> output [seqLen x dModel]
    Matrix*     forward(Matrix *input);

    //backward: dL/dOutput -> dL/dInput
    Matrix*     backward(Matrix *gradOutput);

    void        zeroGrad();
};

#endif
