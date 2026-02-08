#include "Embedding.hpp"

//constructor ;init embedding matrix with xavier
Embedding::Embedding(int vocabSize, int dModel) {
    this->vocabSize     = vocabSize;
    this->dModel        = dModel;
    this->weights       = new Matrix(vocabSize, dModel, true);
    this->gradWeights   = new Matrix(vocabSize, dModel, 0.0);

    //xavier init for stable training
    this->weights->xavierInit(vocabSize, dModel);
}

Embedding::~Embedding() {
    delete this->weights;
    delete this->gradWeights;
}

//forward: token ids -> embedded vectors [seqLen x dModel]
//each token id maps to a row in W_emb, scaled by sqrt(dModel)
//scaling prevents embeddings from being too small relative to positional encodings
Matrix* Embedding::forward(const vector<int> &inputIds) {
    int seqLen = inputIds.size();
    this->lastInputIds = inputIds;

    Matrix *output = new Matrix(seqLen, this->dModel, 0.0);
    double scale = sqrt((double)this->dModel);

    for(int i = 0; i < seqLen; i++) {
        int id = inputIds[i];
        //bounds check ;clamp to vocab
        if(id < 0 || id >= this->vocabSize) id = 1; //unk
        for(int j = 0; j < this->dModel; j++) {
            output->at(i, j) = this->weights->at(id, j) * scale;
        }
    }

    return output;
}

//backward: accumulate gradients into embedding rows
//dL/dW_emb[id] += dL/dOutput[i] * sqrt(dModel) for each position i that used token id
void Embedding::backward(Matrix *gradOutput) {
    double scale = sqrt((double)this->dModel);

    for(int i = 0; i < (int)this->lastInputIds.size(); i++) {
        int id = this->lastInputIds[i];
        if(id < 0 || id >= this->vocabSize) id = 1;
        for(int j = 0; j < this->dModel; j++) {
            this->gradWeights->at(id, j) += gradOutput->at(i, j) * scale;
        }
    }
}

void Embedding::zeroGrad() {
    this->gradWeights->zeroOut();
}
