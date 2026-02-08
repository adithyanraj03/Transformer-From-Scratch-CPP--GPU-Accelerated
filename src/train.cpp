#include "Transformer.hpp"
#include "Tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

using namespace std;

//read entire text file into string
string readFile(const string &path) {
    ifstream f(path);
    if(!f.is_open()) {
        cerr << "error: could not open data file: " << path << endl;
        return "";
    }
    stringstream ss;
    ss << f.rdbuf();
    f.close();
    return ss.str();
}

//create training sequences from token ids
//sliding window of seqLen+1 ;input = first seqLen tokens, target = last seqLen tokens (shifted by 1)
vector<pair<vector<int>, vector<int>>> createSequences(const vector<int> &tokens, int seqLen) {
    vector<pair<vector<int>, vector<int>>> sequences;

    for(int i = 0; i + seqLen < (int)tokens.size(); i += seqLen / 2) {
        //input: tokens[i : i+seqLen]
        //target: tokens[i+1 : i+seqLen+1] ;shifted by 1 for next-token prediction
        vector<int> input(tokens.begin() + i, tokens.begin() + i + seqLen);
        vector<int> target(tokens.begin() + i + 1, tokens.begin() + i + seqLen + 1);
        sequences.push_back({input, target});
    }

    return sequences;
}

int main(int argc, char *argv[]) {
    cout << "========================================" << endl;
    cout << "  Transformer from Scratch (C++)" << endl;
    cout << "  Decoder-Only GPT Training" << endl;
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

    //parse CLI args
    string configPath = "config/model.json";
    bool resume = false;
    int epochsOverride = -1;

    for(int i = 1; i < argc; i++) {
        string arg = argv[i];
        if(arg == "--resume")                            resume = true;
        else if(arg == "--config" && i + 1 < argc)       configPath = argv[++i];
        else if(arg == "--epochs" && i + 1 < argc)        epochsOverride = atoi(argv[++i]);
        else if(arg == "--help") {
            cout << "usage: train [options]" << endl;
            cout << "  --config <path>   config file (default: config/model.json)" << endl;
            cout << "  --resume          continue training from existing weights" << endl;
            cout << "  --epochs <n>      override number of epochs" << endl;
            return 0;
        }
    }

    //load config
    TransformerConfig config = Transformer::loadConfig(configPath);
    if(epochsOverride > 0) config.epochs = epochsOverride;

    cout << "\nmodel config:" << endl;
    cout << "  d_model:       " << config.dModel << endl;
    cout << "  n_heads:       " << config.nHeads << endl;
    cout << "  n_layers:      " << config.nLayers << endl;
    cout << "  d_ff:          " << config.dFF << endl;
    cout << "  max_seq_len:   " << config.maxSeqLen << endl;
    cout << "  learning_rate: " << config.learningRate << endl;
    cout << "  epochs:        " << config.epochs << endl;
    cout << "  batch_size:    " << config.batchSize << endl;

    // ---- step 1: load and tokenize data ----
    cout << "\n[1/4] loading data..." << endl;
    string text = readFile(config.dataFile);
    if(text.empty()) {
        cerr << "error: no training data found" << endl;
        return 1;
    }
    cout << "  loaded " << text.size() << " characters" << endl;

    // ---- step 2: build or load tokenizer ----
    cout << "\n[2/4] building tokenizer..." << endl;
    Tokenizer tokenizer;

    //check if vocab file exists
    ifstream vocabCheck(config.vocabFile);
    if(vocabCheck.good()) {
        vocabCheck.close();
        tokenizer.loadVocab(config.vocabFile);
    } else {
        tokenizer.buildVocab(text, config.vocabSize);
        tokenizer.saveVocab(config.vocabFile);
    }

    //update config vocab size to match actual tokenizer
    config.vocabSize = tokenizer.getVocabSize();
    cout << "  vocab size: " << config.vocabSize << endl;

    //tokenize entire corpus
    vector<int> allTokens = tokenizer.encode(text);
    cout << "  total tokens: " << allTokens.size() << endl;

    //create training sequences
    auto sequences = createSequences(allTokens, config.maxSeqLen);
    cout << "  training sequences: " << sequences.size() << endl;

    // ---- step 3: build model ----
    cout << "\n[3/4] building model..." << endl;
    Transformer model(config);

    //resume from existing weights if requested
    if(resume) {
        ifstream wCheck(config.weightsFile);
        if(wCheck.good()) {
            wCheck.close();
            cout << "  resuming from " << config.weightsFile << "..." << endl;
            model.loadWeights(config.weightsFile);
        } else {
            cout << "  no existing weights found, starting fresh" << endl;
        }
    }

    //count parameters
    vector<Matrix*> params, grads;
    model.getParameters(params, grads);
    long long totalParams = 0;
    for(auto p : params) {
        totalParams += (long long)p->numRows * p->numCols;
    }
    cout << "  total parameters: " << totalParams << endl;
    cout << "  parameter matrices: " << params.size() << endl;

    // ---- step 4: training loop ----
    cout << "\n[4/4] training..." << endl;

    CrossEntropyLoss lossFunc;
    AdamOptimizer optimizer(config.learningRate, config.beta1, config.beta2,
                            config.epsilon, config.weightDecay);

    auto trainStart = chrono::high_resolution_clock::now();

    for(int epoch = 0; epoch < config.epochs; epoch++) {
        double epochLoss = 0.0;
        int numBatches = 0;

        auto epochStart = chrono::high_resolution_clock::now();

        //shuffle sequences (simple Fisher-Yates)
        for(int i = (int)sequences.size() - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            swap(sequences[i], sequences[j]);
        }

        for(int b = 0; b < (int)sequences.size(); b += config.batchSize) {
            double batchLoss = 0.0;

            //zero gradients before accumulating over batch
            model.zeroGrad();

            int batchEnd = min(b + config.batchSize, (int)sequences.size());
            int actualBatchSize = batchEnd - b;

            for(int s = b; s < batchEnd; s++) {
                auto &seq = sequences[s];
                vector<int> &inputIds = seq.first;
                vector<int> &targetIds = seq.second;

                //forward pass ;get logits [seqLen x vocabSize]
                Matrix *logits = model.forward(inputIds);

                //compute loss
                double loss = lossFunc.forward(logits, targetIds);
                batchLoss += loss;

                //backward pass through loss ;get dL/dLogits
                Matrix *gradLogits = lossFunc.backward(targetIds);

                //backward pass through model
                model.backward(gradLogits);

                delete logits;
                delete gradLogits;
            }

            //average gradients over batch
            for(auto g : grads) {
                g->scaleInPlace(1.0 / actualBatchSize);
            }

            //gradient clipping ;prevent exploding gradients
            for(auto g : grads) {
                g->clip(-config.gradClip, config.gradClip);
            }

            //optimizer step
            optimizer.step(params, grads);

            batchLoss /= actualBatchSize;
            epochLoss += batchLoss;
            numBatches++;

            //progress every 10 batches
            if(numBatches % 10 == 0) {
                cout << "  epoch " << epoch + 1 << "/" << config.epochs
                     << " | batch " << numBatches
                     << " | loss: " << fixed << setprecision(4) << batchLoss << endl;
            }
        }

        epochLoss /= numBatches;

        auto epochEnd = chrono::high_resolution_clock::now();
        double epochTime = chrono::duration<double>(epochEnd - epochStart).count();

        cout << "  epoch " << epoch + 1 << "/" << config.epochs
             << " completed | avg loss: " << fixed << setprecision(4) << epochLoss
             << " | time: " << fixed << setprecision(1) << epochTime << "s" << endl;

        //save checkpoint every epoch
        string checkpoint = config.weightsFile;
        model.saveWeights(checkpoint);
    }

    auto trainEnd = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(trainEnd - trainStart).count();

    cout << "\n========================================" << endl;
    cout << "  training complete!" << endl;
    cout << "  total time: " << fixed << setprecision(1) << totalTime << "s" << endl;
    cout << "  weights saved to: " << config.weightsFile << endl;
    cout << "========================================" << endl;

    // ---- quick test: generate a few tokens ----
    cout << "\ntest generation:" << endl;
    string prompt = "First Citizen:\nBefore we proceed any further, hear me speak.";
    vector<int> promptTokens = tokenizer.encode(prompt);
    cout << "  prompt: \"" << prompt << "\" (" << promptTokens.size() << " tokens)" << endl;

    //autoregressive generation ;predict 100 tokens
    //uses top-p (nucleus) sampling + repetition penalty
    double temperature = 0.9;
    double topP = 0.92;             //nucleus: keep smallest set summing to p
    double repPenalty = 1.2;        //penalize repeated tokens

    vector<int> generated = promptTokens;
    for(int i = 0; i < 100; i++) {
        int ctxStart = max(0, (int)generated.size() - config.maxSeqLen);
        vector<int> context(generated.begin() + ctxStart, generated.end());

        Matrix *logits = model.forward(context);
        int lastRow = logits->numRows - 1;

        //apply repetition penalty ;divide logits of previously seen tokens
        for(int t : generated) {
            if(t >= 0 && t < config.vocabSize) {
                double val = logits->at(lastRow, t);
                logits->at(lastRow, t) = (val > 0) ? val / repPenalty : val * repPenalty;
            }
        }

        //temperature-scaled softmax
        vector<double> probs(config.vocabSize);
        double maxLogit = logits->at(lastRow, 0);
        for(int j = 1; j < config.vocabSize; j++) {
            if(logits->at(lastRow, j) > maxLogit) maxLogit = logits->at(lastRow, j);
        }
        double sumExp = 0.0;
        for(int j = 0; j < config.vocabSize; j++) {
            probs[j] = exp((logits->at(lastRow, j) - maxLogit) / temperature);
            sumExp += probs[j];
        }
        for(int j = 0; j < config.vocabSize; j++) {
            probs[j] /= sumExp;
        }

        //top-p nucleus sampling: sort by prob, keep until cumulative >= topP
        vector<pair<double, int>> sorted;
        for(int j = 0; j < config.vocabSize; j++) sorted.push_back({probs[j], j});
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
            if(r < cumSum) { nextToken = sorted[j].second; break; }
        }

        generated.push_back(nextToken);
        delete logits;
    }

    string generatedText = tokenizer.decode(generated);
    cout << "  generated: \"" << generatedText << "\"" << endl;

#ifdef USE_CUDA
    cuda_shutdown();
#endif

    return 0;
}
