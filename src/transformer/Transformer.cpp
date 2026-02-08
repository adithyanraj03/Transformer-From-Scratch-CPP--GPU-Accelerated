#include "Transformer.hpp"
#include <fstream>

//constructor ;builds the full decoder-only transformer stack
Transformer::Transformer(TransformerConfig config) {
    this->config        = config;

    //token embedding ;vocab -> dModel
    this->embedding     = new Embedding(config.vocabSize, config.dModel);

    //sinusoidal positional encoding
    this->posEncoding   = new PositionalEncoding(config.maxSeqLen, config.dModel);

    //N transformer blocks
    for(int i = 0; i < config.nLayers; i++) {
        this->blocks.push_back(new TransformerBlock(config.dModel, config.nHeads, config.dFF));
    }

    //final layer norm (pre-norm architecture needs one at the end)
    this->finalNorm     = new LayerNorm(config.dModel);

    //output projection head ;[dModel x vocabSize] maps to logits
    this->headWeights       = new Matrix(config.dModel, config.vocabSize, true);
    this->headWeights->xavierInit(config.dModel, config.vocabSize);
    this->gradHeadWeights   = new Matrix(config.dModel, config.vocabSize, 0.0);

    this->lastFinalNormOut  = nullptr;
}

Transformer::~Transformer() {
    delete this->embedding;
    delete this->posEncoding;
    for(auto b : this->blocks) delete b;
    delete this->finalNorm;
    delete this->headWeights;
    delete this->gradHeadWeights;
    if(this->lastFinalNormOut) delete this->lastFinalNormOut;
}

//forward: token ids -> logits [seqLen x vocabSize]
//pipeline: embed -> +PE -> N blocks -> final norm -> linear head
Matrix* Transformer::forward(const vector<int> &inputIds) {
    if(this->lastFinalNormOut) delete this->lastFinalNormOut;

    //token embeddings ;[seqLen x dModel] scaled by sqrt(dModel)
    Matrix *embedded = this->embedding->forward(inputIds);

    //add positional encoding
    Matrix *withPos = this->posEncoding->forward(embedded);
    delete embedded;

    //pass through N transformer blocks
    Matrix *hidden = withPos;
    for(int i = 0; i < (int)this->blocks.size(); i++) {
        Matrix *blockOut = this->blocks[i]->forward(hidden);
        if(hidden != withPos || i > 0) {
            //delete intermediate ;but not the first withPos on first iteration
        }
        if(i > 0) delete hidden;
        hidden = blockOut;
    }
    //delete withPos if more than 0 blocks
    if(!this->blocks.empty()) delete withPos;

    //final layer norm
    this->lastFinalNormOut = this->finalNorm->forward(hidden);
    if(!this->blocks.empty()) delete hidden;

    //output projection: [seqLen x dModel] x [dModel x vocabSize] = [seqLen x vocabSize]
    Matrix *logits = Matrix::multiply(this->lastFinalNormOut, this->headWeights);

    return logits;
}

//backward: given dL/dLogits [seqLen x vocabSize], backprop through entire model
void Transformer::backward(Matrix *gradLogits) {
    //backprop through output head
    //dL/dHeadWeights = finalNormOut^T x gradLogits
    Matrix *fnT = Matrix::transpose(this->lastFinalNormOut);
    Matrix *dHead = Matrix::multiply(fnT, gradLogits);
    this->gradHeadWeights->addInPlace(dHead);
    delete fnT;
    delete dHead;

    //dL/dFinalNormOut = gradLogits x headWeights^T
    Matrix *hwT = Matrix::transpose(this->headWeights);
    Matrix *gradFinalNorm = Matrix::multiply(gradLogits, hwT);
    delete hwT;

    //backprop through final layer norm
    Matrix *gradHidden = this->finalNorm->backward(gradFinalNorm);
    delete gradFinalNorm;

    //backprop through transformer blocks in reverse order
    for(int i = (int)this->blocks.size() - 1; i >= 0; i--) {
        Matrix *gradBlockInput = this->blocks[i]->backward(gradHidden);
        delete gradHidden;
        gradHidden = gradBlockInput;
    }

    //backprop through positional encoding (pass-through, no learnable params)
    Matrix *gradEmbedded = this->posEncoding->backward(gradHidden);
    delete gradHidden;

    //backprop through embedding layer
    this->embedding->backward(gradEmbedded);
    delete gradEmbedded;
}

//zero all gradients
void Transformer::zeroGrad() {
    this->embedding->zeroGrad();
    for(auto b : this->blocks) b->zeroGrad();
    this->finalNorm->zeroGrad();
    this->gradHeadWeights->zeroOut();
}

//collect all trainable parameters and their gradient matrices
//used by optimizer to update weights
void Transformer::getParameters(vector<Matrix*> &params, vector<Matrix*> &grads) {
    params.clear();
    grads.clear();

    //embedding weights
    params.push_back(this->embedding->weights);
    grads.push_back(this->embedding->gradWeights);

    //each transformer block
    for(auto b : this->blocks) {
        //attention weights
        params.push_back(b->attention->Wq);     grads.push_back(b->attention->gradWq);
        params.push_back(b->attention->Wk);     grads.push_back(b->attention->gradWk);
        params.push_back(b->attention->Wv);     grads.push_back(b->attention->gradWv);
        params.push_back(b->attention->Wo);     grads.push_back(b->attention->gradWo);

        //ffn weights
        params.push_back(b->ffn->W1);           grads.push_back(b->ffn->gradW1);
        params.push_back(b->ffn->b1);           grads.push_back(b->ffn->gradB1);
        params.push_back(b->ffn->W2);           grads.push_back(b->ffn->gradW2);
        params.push_back(b->ffn->b2);           grads.push_back(b->ffn->gradB2);

        //layer norm params
        params.push_back(b->norm1->gamma);       grads.push_back(b->norm1->gradGamma);
        params.push_back(b->norm1->beta);        grads.push_back(b->norm1->gradBeta);
        params.push_back(b->norm2->gamma);       grads.push_back(b->norm2->gradGamma);
        params.push_back(b->norm2->beta);        grads.push_back(b->norm2->gradBeta);
    }

    //final layer norm
    params.push_back(this->finalNorm->gamma);    grads.push_back(this->finalNorm->gradGamma);
    params.push_back(this->finalNorm->beta);     grads.push_back(this->finalNorm->gradBeta);

    //output head
    params.push_back(this->headWeights);         grads.push_back(this->gradHeadWeights);
}

//save weights to json file
void Transformer::saveWeights(const string &filePath) {
    json j;
    vector<Matrix*> params;
    vector<Matrix*> grads;
    this->getParameters(params, grads);

    //serialize each parameter matrix as flat array
    for(int p = 0; p < (int)params.size(); p++) {
        json paramJson;
        paramJson["rows"] = params[p]->numRows;
        paramJson["cols"] = params[p]->numCols;
        vector<double> flat;
        for(int i = 0; i < params[p]->numRows; i++) {
            for(int j = 0; j < params[p]->numCols; j++) {
                flat.push_back(params[p]->at(i, j));
            }
        }
        paramJson["data"] = flat;
        j["params"].push_back(paramJson);
    }

    //save config alongside weights
    j["config"]["dModel"]       = this->config.dModel;
    j["config"]["nHeads"]       = this->config.nHeads;
    j["config"]["nLayers"]      = this->config.nLayers;
    j["config"]["dFF"]          = this->config.dFF;
    j["config"]["vocabSize"]    = this->config.vocabSize;
    j["config"]["maxSeqLen"]    = this->config.maxSeqLen;

    ofstream f(filePath);
    if(!f.is_open()) {
        cerr << "error: could not open weights file for writing: " << filePath << endl;
        return;
    }
    f << j.dump();
    f.close();
    cout << "model weights saved to " << filePath << " (" << params.size() << " parameter matrices)" << endl;
}

//load weights from json file
void Transformer::loadWeights(const string &filePath) {
    ifstream f(filePath);
    if(!f.is_open()) {
        cerr << "error: could not open weights file: " << filePath << endl;
        return;
    }

    json j;
    f >> j;
    f.close();

    vector<Matrix*> params;
    vector<Matrix*> grads;
    this->getParameters(params, grads);

    if(j["params"].size() != params.size()) {
        cerr << "error: weight file has " << j["params"].size()
             << " params, model expects " << params.size() << endl;
        return;
    }

    for(int p = 0; p < (int)params.size(); p++) {
        int rows = j["params"][p]["rows"].get<int>();
        int cols = j["params"][p]["cols"].get<int>();
        vector<double> flat = j["params"][p]["data"].get<vector<double>>();

        if(rows != params[p]->numRows || cols != params[p]->numCols) {
            cerr << "error: shape mismatch at param " << p << endl;
            return;
        }

        int idx = 0;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                params[p]->at(i, j) = flat[idx++];
            }
        }
    }

    cout << "model weights loaded from " << filePath << endl;
}

//load config from json file
TransformerConfig Transformer::loadConfig(const string &filePath) {
    ifstream f(filePath);
    TransformerConfig config;

    if(!f.is_open()) {
        cerr << "warning: could not open config file: " << filePath
             << " , using defaults" << endl;
        return config;
    }

    json j;
    f >> j;
    f.close();

    //read each field if present ;use defaults otherwise
    if(j.contains("dModel"))        config.dModel       = j["dModel"].get<int>();
    if(j.contains("nHeads"))        config.nHeads       = j["nHeads"].get<int>();
    if(j.contains("nLayers"))       config.nLayers      = j["nLayers"].get<int>();
    if(j.contains("dFF"))           config.dFF          = j["dFF"].get<int>();
    if(j.contains("vocabSize"))     config.vocabSize    = j["vocabSize"].get<int>();
    if(j.contains("maxSeqLen"))     config.maxSeqLen    = j["maxSeqLen"].get<int>();
    if(j.contains("learningRate"))  config.learningRate = j["learningRate"].get<double>();
    if(j.contains("beta1"))         config.beta1        = j["beta1"].get<double>();
    if(j.contains("beta2"))         config.beta2        = j["beta2"].get<double>();
    if(j.contains("epsilon"))       config.epsilon      = j["epsilon"].get<double>();
    if(j.contains("weightDecay"))   config.weightDecay  = j["weightDecay"].get<double>();
    if(j.contains("gradClip"))      config.gradClip     = j["gradClip"].get<double>();
    if(j.contains("epochs"))        config.epochs       = j["epochs"].get<int>();
    if(j.contains("batchSize"))     config.batchSize    = j["batchSize"].get<int>();
    if(j.contains("dataFile"))      config.dataFile     = j["dataFile"].get<string>();
    if(j.contains("vocabFile"))     config.vocabFile    = j["vocabFile"].get<string>();
    if(j.contains("weightsFile"))   config.weightsFile  = j["weightsFile"].get<string>();

    cout << "config loaded from " << filePath << endl;
    return config;
}
