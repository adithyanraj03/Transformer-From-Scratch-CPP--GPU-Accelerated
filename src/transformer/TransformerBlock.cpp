#include "TransformerBlock.hpp"

//constructor ;init sub-layers
TransformerBlock::TransformerBlock(int dModel, int nHeads, int dFF) {
    this->dModel    = dModel;
    this->nHeads    = nHeads;
    this->dFF       = dFF;

    this->attention     = new MultiHeadAttention(dModel, nHeads);
    this->ffn           = new FeedForward(dModel, dFF);
    this->norm1         = new LayerNorm(dModel);
    this->norm2         = new LayerNorm(dModel);

    this->lastInput     = nullptr;
    this->lastNorm1Out  = nullptr;
    this->lastAttnOut   = nullptr;
    this->lastResidual1 = nullptr;
    this->lastNorm2Out  = nullptr;
}

TransformerBlock::~TransformerBlock() {
    delete this->attention;
    delete this->ffn;
    delete this->norm1;
    delete this->norm2;
    if(this->lastInput)     delete this->lastInput;
    if(this->lastNorm1Out)  delete this->lastNorm1Out;
    if(this->lastAttnOut)   delete this->lastAttnOut;
    if(this->lastResidual1) delete this->lastResidual1;
    if(this->lastNorm2Out)  delete this->lastNorm2Out;
}

//forward: pre-norm transformer block
//1. norm1 -> attention -> +residual
//2. norm2 -> ffn -> +residual
//
//x_norm1 = LayerNorm(x)
//attn_out = MultiHeadAttention(x_norm1)
//residual1 = x + attn_out
//x_norm2 = LayerNorm(residual1)
//ffn_out = FFN(x_norm2)
//output = residual1 + ffn_out
Matrix* TransformerBlock::forward(Matrix *input) {
    //cleanup caches
    if(this->lastInput)     delete this->lastInput;
    if(this->lastNorm1Out)  delete this->lastNorm1Out;
    if(this->lastAttnOut)   delete this->lastAttnOut;
    if(this->lastResidual1) delete this->lastResidual1;
    if(this->lastNorm2Out)  delete this->lastNorm2Out;

    this->lastInput = new Matrix(*input);

    //pre-norm -> attention
    this->lastNorm1Out = this->norm1->forward(input);
    this->lastAttnOut  = this->attention->forward(this->lastNorm1Out);

    //residual connection ;x + attention(norm(x))
    this->lastResidual1 = Matrix::add(input, this->lastAttnOut);

    //pre-norm -> ffn
    this->lastNorm2Out = this->norm2->forward(this->lastResidual1);
    Matrix *ffnOut = this->ffn->forward(this->lastNorm2Out);

    //residual connection ;residual1 + ffn(norm(residual1))
    Matrix *output = Matrix::add(this->lastResidual1, ffnOut);
    delete ffnOut;

    return output;
}

//backward: reverse the forward pass
//given dL/dOutput:
//  dL/dResidual1 += dL/dOutput (residual)
//  dL/dFFN_out = dL/dOutput
//  dL/dNorm2 = FFN.backward(dL/dFFN_out)
//  dL/dResidual1 += Norm2.backward(dL/dNorm2)
//  dL/dInput += dL/dResidual1 (residual)
//  dL/dAttn = dL/dResidual1
//  dL/dNorm1 = Attention.backward(dL/dAttn)
//  dL/dInput += Norm1.backward(dL/dNorm1)
Matrix* TransformerBlock::backward(Matrix *gradOutput) {
    //backprop through second residual + FFN
    //dL/dFFN_out = gradOutput (from residual path)
    Matrix *gradNorm2Out = this->ffn->backward(gradOutput);
    Matrix *gradResidual1_ffn = this->norm2->backward(gradNorm2Out);
    delete gradNorm2Out;

    //add residual gradient ;dL/dResidual1 = gradOutput + gradResidual1_ffn
    Matrix *gradResidual1 = Matrix::add(gradOutput, gradResidual1_ffn);
    delete gradResidual1_ffn;

    //backprop through first residual + attention
    Matrix *gradNorm1Out = this->attention->backward(gradResidual1);
    Matrix *gradInput_attn = this->norm1->backward(gradNorm1Out);
    delete gradNorm1Out;

    //add residual gradient ;dL/dInput = gradResidual1 + gradInput_attn
    Matrix *gradInput = Matrix::add(gradResidual1, gradInput_attn);
    delete gradResidual1;
    delete gradInput_attn;

    return gradInput;
}

void TransformerBlock::zeroGrad() {
    this->attention->zeroGrad();
    this->ffn->zeroGrad();
    this->norm1->zeroGrad();
    this->norm2->zeroGrad();
}
