#include "Transformer.hpp"

//constructor
CrossEntropyLoss::CrossEntropyLoss() {
    this->lastProbs = nullptr;
}

CrossEntropyLoss::~CrossEntropyLoss() {
    if(this->lastProbs) delete this->lastProbs;
}

//forward: compute cross-entropy loss from logits and target token ids
//loss = -(1/N) * sum_i(log(softmax(logits[i])[target[i]]))
//returns scalar loss value
double CrossEntropyLoss::forward(Matrix *logits, const vector<int> &targets) {
    if(this->lastProbs) delete this->lastProbs;

    //softmax over logits ;row-wise [seqLen x vocabSize]
    this->lastProbs = Matrix::softmax(logits);

    int seqLen = logits->numRows;
    double loss = 0.0;

    for(int i = 0; i < seqLen; i++) {
        int t = targets[i];
        //clamp probability to avoid log(0) -> -inf
        double p = max(this->lastProbs->at(i, t), 1e-10);
        loss -= log(p);
    }

    //average over sequence length
    loss /= seqLen;
    return loss;
}

//backward: dL/dLogits = (1/N) * (softmax(logits) - one_hot(targets))
//this is the clean derivative of cross-entropy + softmax combined
//dL/dLogits[i][j] = (1/N) * (probs[i][j] - (j == target[i] ? 1 : 0))
Matrix* CrossEntropyLoss::backward(const vector<int> &targets) {
    int seqLen      = this->lastProbs->numRows;
    int vocabSize   = this->lastProbs->numCols;

    Matrix *gradLogits = new Matrix(*this->lastProbs);

    //subtract 1 from the target class probability
    for(int i = 0; i < seqLen; i++) {
        gradLogits->at(i, targets[i]) -= 1.0;
    }

    //average over sequence length
    gradLogits->scaleInPlace(1.0 / seqLen);

    return gradLogits;
}
