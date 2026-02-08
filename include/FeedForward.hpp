#ifndef _FEED_FORWARD_HPP_
#define _FEED_FORWARD_HPP_

#include "Matrix.hpp"

//position-wise feed-forward network
//FFN(x) = max(0, xW1 + b1)W2 + b2
//W1 [dModel x dFF], W2 [dFF x dModel]
class FeedForward {
public:
    int         dModel;
    int         dFF;

    Matrix*     W1;         //[dModel x dFF]
    Matrix*     b1;         //[1 x dFF]
    Matrix*     W2;         //[dFF x dModel]
    Matrix*     b2;         //[1 x dModel]

    //gradients
    Matrix*     gradW1;
    Matrix*     gradW2;
    Matrix*     gradB1;
    Matrix*     gradB2;

    //cached for backward
    Matrix*     lastInput;
    Matrix*     lastHidden;     //after ReLU
    Matrix*     lastPreRelu;    //before ReLU ;needed for relu derivative

    FeedForward(int dModel, int dFF);
    ~FeedForward();

    //forward: x [seqLen x dModel] -> output [seqLen x dModel]
    Matrix*     forward(Matrix *input);

    //backward: dL/dOutput -> dL/dInput, accumulate weight gradients
    Matrix*     backward(Matrix *gradOutput);

    void        zeroGrad();

private:
    //add bias row-wise ;each row of m gets b added
    Matrix*     addBias(Matrix *m, Matrix *b);
};

#endif
