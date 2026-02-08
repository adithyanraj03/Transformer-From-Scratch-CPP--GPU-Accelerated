#include "MultiHeadAttention.hpp"

//constructor ;init projection weights with xavier
MultiHeadAttention::MultiHeadAttention(int dModel, int nHeads) {
    this->dModel    = dModel;
    this->nHeads    = nHeads;
    this->dK        = dModel / nHeads;

    //linear projections ;W_q,W_k,W_v,W_o all [dModel x dModel]
    this->Wq        = new Matrix(dModel, dModel, true);
    this->Wk        = new Matrix(dModel, dModel, true);
    this->Wv        = new Matrix(dModel, dModel, true);
    this->Wo        = new Matrix(dModel, dModel, true);

    this->Wq->xavierInit(dModel, dModel);
    this->Wk->xavierInit(dModel, dModel);
    this->Wv->xavierInit(dModel, dModel);
    this->Wo->xavierInit(dModel, dModel);

    //gradient buffers
    this->gradWq    = new Matrix(dModel, dModel, 0.0);
    this->gradWk    = new Matrix(dModel, dModel, 0.0);
    this->gradWv    = new Matrix(dModel, dModel, 0.0);
    this->gradWo    = new Matrix(dModel, dModel, 0.0);

    this->lastInput     = nullptr;
    this->lastQ         = nullptr;
    this->lastK         = nullptr;
    this->lastV         = nullptr;
    this->concatHeads   = nullptr;
}

MultiHeadAttention::~MultiHeadAttention() {
    delete this->Wq;
    delete this->Wk;
    delete this->Wv;
    delete this->Wo;
    delete this->gradWq;
    delete this->gradWk;
    delete this->gradWv;
    delete this->gradWo;
    if(this->lastInput)     delete this->lastInput;
    if(this->lastQ)         delete this->lastQ;
    if(this->lastK)         delete this->lastK;
    if(this->lastV)         delete this->lastV;
    if(this->concatHeads)   delete this->concatHeads;
    for(auto p : this->attentionWeights) delete p;
    for(auto p : this->headOutputs)     delete p;
}

//forward: x [seqLen x dModel] -> output [seqLen x dModel]
//1. project input to Q,K,V via linear layers
//2. split Q,K,V into h heads
//3. compute attention per head
//4. concat heads, project via Wo
Matrix* MultiHeadAttention::forward(Matrix *input) {
    int seqLen = input->numRows;

    //cleanup caches from previous forward
    if(this->lastInput)     delete this->lastInput;
    if(this->lastQ)         delete this->lastQ;
    if(this->lastK)         delete this->lastK;
    if(this->lastV)         delete this->lastV;
    if(this->concatHeads)   delete this->concatHeads;
    for(auto p : this->attentionWeights) delete p;
    for(auto p : this->headOutputs)     delete p;
    this->attentionWeights.clear();
    this->headOutputs.clear();

    this->lastInput = new Matrix(*input);

    //Q = x * Wq, K = x * Wk, V = x * Wv ;all [seqLen x dModel]
    this->lastQ = Matrix::multiply(input, this->Wq);
    this->lastK = Matrix::multiply(input, this->Wk);
    this->lastV = Matrix::multiply(input, this->Wv);

    //split into h heads and compute attention per head
    //head_i gets columns [i*dK : (i+1)*dK] from Q,K,V
    vector<Matrix*> headResults;
    for(int h = 0; h < this->nHeads; h++) {
        int colStart = h * this->dK;
        int colEnd   = (h + 1) * this->dK;

        Matrix *Qh = this->lastQ->slice(0, seqLen, colStart, colEnd);
        Matrix *Kh = this->lastK->slice(0, seqLen, colStart, colEnd);
        Matrix *Vh = this->lastV->slice(0, seqLen, colStart, colEnd);

        //compute scaled dot-product attention
        Matrix *attnW = nullptr;
        Matrix *headOut = this->computeScaledDotProduct(Qh, Kh, Vh, &attnW);

        this->attentionWeights.push_back(attnW);
        this->headOutputs.push_back(new Matrix(*headOut));  //cache for backward

        headResults.push_back(headOut);

        delete Qh;
        delete Kh;
        delete Vh;
    }

    //concat all heads ;[seqLen x dK] * h -> [seqLen x dModel]
    this->concatHeads = Matrix::concat(headResults, 1);
    for(auto p : headResults) delete p;

    //output projection: concat * Wo ;[seqLen x dModel] x [dModel x dModel] = [seqLen x dModel]
    Matrix *output = Matrix::multiply(this->concatHeads, this->Wo);

    return output;
}

//backward: dL/dOutput [seqLen x dModel] -> dL/dInput [seqLen x dModel]
//backprop through Wo, then through each attention head, then through Wq/Wk/Wv
Matrix* MultiHeadAttention::backward(Matrix *gradOutput) {
    int seqLen = gradOutput->numRows;

    // ---- backprop through output projection Wo ----
    //dL/dWo = concatHeads^T x gradOutput
    Matrix *concatT = Matrix::transpose(this->concatHeads);
    Matrix *dWo = Matrix::multiply(concatT, gradOutput);
    this->gradWo->addInPlace(dWo);
    delete concatT;
    delete dWo;

    //dL/dConcat = gradOutput x Wo^T ;[seqLen x dModel]
    Matrix *WoT = Matrix::transpose(this->Wo);
    Matrix *gradConcat = Matrix::multiply(gradOutput, WoT);
    delete WoT;

    // ---- backprop through each attention head ----
    //split gradConcat into per-head gradients
    Matrix *gradQ_full = Matrix::zeros(seqLen, this->dModel);
    Matrix *gradK_full = Matrix::zeros(seqLen, this->dModel);
    Matrix *gradV_full = Matrix::zeros(seqLen, this->dModel);

    for(int h = 0; h < this->nHeads; h++) {
        int colStart = h * this->dK;
        int colEnd   = (h + 1) * this->dK;

        //gradient for this head's output ;slice from gradConcat
        Matrix *gradHeadOut = gradConcat->slice(0, seqLen, colStart, colEnd);

        //get cached attention weights and head Q,K,V
        Matrix *attnW = this->attentionWeights[h];
        Matrix *Qh = this->lastQ->slice(0, seqLen, colStart, colEnd);
        Matrix *Kh = this->lastK->slice(0, seqLen, colStart, colEnd);
        Matrix *Vh = this->lastV->slice(0, seqLen, colStart, colEnd);

        //backprop through attnW * V:
        //dL/dattnW = gradHeadOut x V^T ;[seqLen x seqLen]
        //dL/dV = attnW^T x gradHeadOut ;[seqLen x dK]
        Matrix *VhT = Matrix::transpose(Vh);
        Matrix *gradAttnW = Matrix::multiply(gradHeadOut, VhT);
        delete VhT;

        Matrix *attnWT = Matrix::transpose(attnW);
        Matrix *gradVh = Matrix::multiply(attnWT, gradHeadOut);
        delete attnWT;

        //backprop through softmax
        //dL/dScores = attnW ⊙ (dL/dattnW - sum_j(dL/dattnW_j * attnW_j)) per row
        Matrix *gradScores = new Matrix(seqLen, seqLen, 0.0);
        for(int i = 0; i < seqLen; i++) {
            //dot product of gradAttnW[i] and attnW[i]
            double dot = 0.0;
            for(int j = 0; j < seqLen; j++) {
                dot += gradAttnW->at(i, j) * attnW->at(i, j);
            }
            for(int j = 0; j < seqLen; j++) {
                gradScores->at(i, j) = attnW->at(i, j) * (gradAttnW->at(i, j) - dot);
            }
        }
        delete gradAttnW;

        //scale by 1/sqrt(dK) ;matching forward pass scaling
        double scale = sqrt((double)this->dK);
        gradScores->scaleInPlace(1.0 / scale);

        //backprop through QK^T:
        //dL/dQ = gradScores x K ;[seqLen x seqLen] x [seqLen x dK] = [seqLen x dK]
        //dL/dK = gradScores^T x Q ;[seqLen x seqLen] x [seqLen x dK] = [seqLen x dK]
        Matrix *gradQh = Matrix::multiply(gradScores, Kh);
        Matrix *gradScoresT = Matrix::transpose(gradScores);
        Matrix *gradKh = Matrix::multiply(gradScoresT, Qh);
        delete gradScores;
        delete gradScoresT;

        //accumulate per-head gradients into full Q,K,V gradients
        for(int i = 0; i < seqLen; i++) {
            for(int j = 0; j < this->dK; j++) {
                gradQ_full->at(i, colStart + j) += gradQh->at(i, j);
                gradK_full->at(i, colStart + j) += gradKh->at(i, j);
                gradV_full->at(i, colStart + j) += gradVh->at(i, j);
            }
        }

        delete gradHeadOut;
        delete gradQh;
        delete gradKh;
        delete gradVh;
        delete Qh;
        delete Kh;
        delete Vh;
    }
    delete gradConcat;

    // ---- backprop through Q,K,V projections ----
    //dL/dWq = input^T x gradQ_full
    Matrix *inputT = Matrix::transpose(this->lastInput);
    Matrix *dWq = Matrix::multiply(inputT, gradQ_full);
    Matrix *dWk = Matrix::multiply(inputT, gradK_full);
    Matrix *dWv = Matrix::multiply(inputT, gradV_full);
    this->gradWq->addInPlace(dWq);
    this->gradWk->addInPlace(dWk);
    this->gradWv->addInPlace(dWv);
    delete dWq;
    delete dWk;
    delete dWv;
    delete inputT;

    //dL/dInput = gradQ*Wq^T + gradK*Wk^T + gradV*Wv^T
    Matrix *WqT = Matrix::transpose(this->Wq);
    Matrix *WkT = Matrix::transpose(this->Wk);
    Matrix *WvT = Matrix::transpose(this->Wv);

    Matrix *gradInput1 = Matrix::multiply(gradQ_full, WqT);
    Matrix *gradInput2 = Matrix::multiply(gradK_full, WkT);
    Matrix *gradInput3 = Matrix::multiply(gradV_full, WvT);

    Matrix *gradInput12 = Matrix::add(gradInput1, gradInput2);
    Matrix *gradInput = Matrix::add(gradInput12, gradInput3);

    delete WqT; delete WkT; delete WvT;
    delete gradInput1; delete gradInput2; delete gradInput3;
    delete gradInput12;
    delete gradQ_full; delete gradK_full; delete gradV_full;

    return gradInput;
}

void MultiHeadAttention::zeroGrad() {
    this->gradWq->zeroOut();
    this->gradWk->zeroOut();
    this->gradWv->zeroOut();
    this->gradWo->zeroOut();
}
