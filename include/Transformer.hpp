#ifndef _TRANSFORMER_HPP_
#define _TRANSFORMER_HPP_

#include "Embedding.hpp"
#include "PositionalEncoding.hpp"
#include "TransformerBlock.hpp"
#include "json.hpp"
#include <vector>
#include <string>

using json = nlohmann::json;

//model config ;loaded from json file
struct TransformerConfig {
    int         dModel       = 64;
    int         nHeads       = 4;
    int         nLayers      = 4;
    int         dFF          = 256;
    int         vocabSize    = 512;
    int         maxSeqLen    = 128;
    double      learningRate = 0.001;
    double      beta1        = 0.9;
    double      beta2        = 0.999;
    double      epsilon      = 1e-8;
    double      weightDecay  = 0.01;
    double      gradClip     = 1.0;
    int         epochs       = 10;
    int         batchSize    = 16;
    string      dataFile     = "data/input.txt";
    string      vocabFile    = "data/vocab.json";
    string      weightsFile  = "data/weights.json";
};

//decoder-only GPT-style transformer
//embedding -> positional encoding -> N x TransformerBlock -> layer norm -> linear head -> softmax
class Transformer {
public:
    TransformerConfig           config;
    Embedding*                  embedding;
    PositionalEncoding*         posEncoding;
    vector<TransformerBlock*>   blocks;
    LayerNorm*                  finalNorm;

    //output projection head ;projects dModel -> vocabSize for next-token prediction
    Matrix*                     headWeights;    //[dModel x vocabSize]
    Matrix*                     gradHeadWeights;

    //cached for backward
    Matrix*                     lastFinalNormOut;

    Transformer(TransformerConfig config);
    ~Transformer();

    //forward: token ids -> logits [seqLen x vocabSize]
    Matrix*     forward(const vector<int> &inputIds);

    //backward: given loss gradient [seqLen x vocabSize], backprop through entire model
    void        backward(Matrix *gradLogits);

    //zero all gradients across all layers
    void        zeroGrad();

    //get all trainable parameters and their gradients ;for optimizer
    void        getParameters(vector<Matrix*> &params, vector<Matrix*> &grads);

    //save/load model weights
    void        saveWeights(const string &filePath);
    void        loadWeights(const string &filePath);

    //load config from json file
    static TransformerConfig    loadConfig(const string &filePath);
};

//cross entropy loss ;for next-token prediction
//loss = -sum(y_true * log(y_pred)) averaged over sequence
class CrossEntropyLoss {
public:
    Matrix*     lastProbs;  //softmax output ;cached for backward

    CrossEntropyLoss();
    ~CrossEntropyLoss();

    //compute loss: logits [seqLen x vocabSize], targets [seqLen] -> scalar loss
    double      forward(Matrix *logits, const vector<int> &targets);

    //backward: returns dL/dLogits [seqLen x vocabSize]
    Matrix*     backward(const vector<int> &targets);
};

//Adam optimizer with weight decay (AdamW)
//m_t = beta1*m_{t-1} + (1-beta1)*g_t
//v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
//param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
class AdamOptimizer {
public:
    double      lr;
    double      beta1;
    double      beta2;
    double      eps;
    double      weightDecay;
    int         timestep;

    //first and second moment estimates ;one per parameter matrix
    vector<Matrix*>     m;
    vector<Matrix*>     v;
    bool                initialized;

    AdamOptimizer(double lr, double beta1, double beta2, double eps, double weightDecay);
    ~AdamOptimizer();

    //update all params using their gradients
    void        step(vector<Matrix*> &params, vector<Matrix*> &grads);

    //initialize moment buffers ;called on first step
    void        initMoments(vector<Matrix*> &params);
};

#endif
