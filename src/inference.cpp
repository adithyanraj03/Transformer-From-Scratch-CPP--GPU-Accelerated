#include "Transformer.hpp"
#include "Tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

using namespace std;

int main(int argc, char *argv[]) {
    cout << "========================================" << endl;
    cout << "  Transformer from Scratch (C++)" << endl;
    cout << "  Text Generation / Inference" << endl;
    cout << "========================================" << endl;

    //init GPU if available
#ifdef USE_CUDA
    if(cuda_init() == 0) {
        cout << "  GPU acceleration: ENABLED" << endl;
    } else {
        cout << "  GPU acceleration: DISABLED (no CUDA device)" << endl;
    }
#else
    cout << "  GPU acceleration: NOT COMPILED (build with CUDA)" << endl;
#endif

    srand(time(nullptr));

    //parse args
    string configPath   = "config/model.json";
    string prompt       = "the ";
    int numTokens       = 200;
    double temperature  = 0.9;

    for(int i = 1; i < argc; i++) {
        string arg = argv[i];
        if(arg == "--config" && i + 1 < argc)       configPath  = argv[++i];
        else if(arg == "--prompt" && i + 1 < argc)   prompt      = argv[++i];
        else if(arg == "--tokens" && i + 1 < argc)   numTokens   = atoi(argv[++i]);
        else if(arg == "--temp" && i + 1 < argc)     temperature = atof(argv[++i]);
        else if(arg == "--help") {
            cout << "usage: inference [options]" << endl;
            cout << "  --config <path>   config file (default: config/model.json)" << endl;
            cout << "  --prompt <text>   starting prompt text" << endl;
            cout << "  --tokens <n>      number of tokens to generate (default: 200)" << endl;
            cout << "  --temp <float>    sampling temperature (default: 0.9)" << endl;
            return 0;
        }
    }

    //load config
    TransformerConfig config = Transformer::loadConfig(configPath);

    cout << "\nmodel config:" << endl;
    cout << "  d_model:     " << config.dModel << endl;
    cout << "  n_heads:     " << config.nHeads << endl;
    cout << "  n_layers:    " << config.nLayers << endl;
    cout << "  vocab_size:  " << config.vocabSize << endl;

    //load tokenizer
    cout << "\nloading tokenizer..." << endl;
    Tokenizer tokenizer;
    tokenizer.loadVocab(config.vocabFile);
    config.vocabSize = tokenizer.getVocabSize();

    //build model and load weights
    cout << "loading model weights..." << endl;
    Transformer model(config);
    model.loadWeights(config.weightsFile);

    //encode prompt
    vector<int> tokens = tokenizer.encode(prompt);
    cout << "\nprompt: \"" << prompt << "\" (" << tokens.size() << " tokens)" << endl;
    cout << "generating " << numTokens << " tokens (temperature=" << temperature << ")..." << endl;
    cout << "\n--- generated text ---" << endl;
    cout << prompt;

    //autoregressive generation with top-p nucleus sampling + repetition penalty
    double topP = 0.92;         //nucleus: keep smallest token set summing to p
    double repPenalty = 1.2;    //penalize previously generated tokens

    for(int i = 0; i < numTokens; i++) {
        //context window ;last maxSeqLen tokens
        int ctxStart = max(0, (int)tokens.size() - config.maxSeqLen);
        vector<int> context(tokens.begin() + ctxStart, tokens.end());

        //forward pass ;get logits
        Matrix *logits = model.forward(context);
        int lastRow = logits->numRows - 1;

        //apply repetition penalty ;reduce logits of already-generated tokens
        for(int t : tokens) {
            if(t >= 0 && t < config.vocabSize) {
                double val = logits->at(lastRow, t);
                logits->at(lastRow, t) = (val > 0) ? val / repPenalty : val * repPenalty;
            }
        }

        //temperature-scaled softmax
        vector<double> probs(config.vocabSize);
        double maxLogit = logits->at(lastRow, 0);
        for(int j = 1; j < config.vocabSize; j++) {
            if(logits->at(lastRow, j) > maxLogit)
                maxLogit = logits->at(lastRow, j);
        }

        double sumExp = 0.0;
        for(int j = 0; j < config.vocabSize; j++) {
            probs[j] = exp((logits->at(lastRow, j) - maxLogit) / temperature);
            sumExp += probs[j];
        }
        for(int j = 0; j < config.vocabSize; j++) {
            probs[j] /= sumExp;
        }

        //top-p (nucleus) sampling: sort by prob, keep until cumulative >= topP
        vector<pair<double, int>> sorted;
        for(int j = 0; j < config.vocabSize; j++) {
            sorted.push_back({probs[j], j});
        }
        sort(sorted.begin(), sorted.end(), greater<pair<double, int>>());

        double cumP = 0.0;
        double filteredSum = 0.0;
        int cutoff = 0;
        for(int j = 0; j < config.vocabSize; j++) {
            cumP += sorted[j].first;
            cutoff = j + 1;
            filteredSum += sorted[j].first;
            if(cumP >= topP) break;
        }

        //sample from nucleus
        double r = ((double)rand() / RAND_MAX) * filteredSum;
        double cumSum = 0.0;
        int nextToken = sorted[0].second;
        for(int j = 0; j < cutoff; j++) {
            cumSum += sorted[j].first;
            if(r < cumSum) {
                nextToken = sorted[j].second;
                break;
            }
        }

        tokens.push_back(nextToken);

        //decode and print the new token
        string decoded = tokenizer.decode({nextToken});
        cout << decoded;
        cout.flush();

        delete logits;
    }

    cout << endl;
    cout << "\n--- end ---" << endl;
    cout << "total tokens generated: " << numTokens << endl;

#ifdef USE_CUDA
    cuda_shutdown();
#endif

    return 0;
}
