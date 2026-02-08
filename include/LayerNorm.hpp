#ifndef _LAYER_NORM_HPP_
#define _LAYER_NORM_HPP_

#include "Matrix.hpp"

//layer normalization ;normalizes across feature dim (last axis)
//y = gamma * (x - mean) / sqrt(var + eps) + beta
//learnable params: gamma [1 x dModel], beta [1 x dModel]
class LayerNorm {
public:
    int         dModel;
    double      eps;
    Matrix*     gamma;          //scale [1 x dModel]
    Matrix*     beta;           //shift [1 x dModel]
    Matrix*     gradGamma;
    Matrix*     gradBeta;

    //cached for backward
    Matrix*     normalized;     //x_hat = (x-mean)/std
    Matrix*     stdInv;         //1/sqrt(var+eps) per row
    Matrix*     inputCentered;  //x - mean

    LayerNorm(int dModel);
    ~LayerNorm();

    //forward: x [seqLen x dModel] -> normalized [seqLen x dModel]
    Matrix*     forward(Matrix *input);

    //backward: dL/dy -> dL/dx, accumulate dL/dgamma, dL/dbeta
    Matrix*     backward(Matrix *gradOutput);

    void        zeroGrad();
};

#endif
