#ifndef _TOKENIZER_HPP_
#define _TOKENIZER_HPP_

#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

//byte pair encoding tokenizer ;builds vocab from corpus then encodes/decodes
class Tokenizer {
public:
    int                                     vocabSize;
    map<string, int>                        tokenToId;
    map<int, string>                        idToToken;
    vector<pair<string, string>>            merges;

    //special tokens
    int                                     padId;
    int                                     unkId;
    int                                     bosId;
    int                                     eosId;

    Tokenizer();
    ~Tokenizer();

    //build BPE vocab from text corpus ;learns merge rules
    void            buildVocab(const string &text, int targetVocabSize);

    //encode text -> token ids
    vector<int>     encode(const string &text);

    //decode token ids -> text
    string          decode(const vector<int> &ids);

    //save/load vocab to/from json
    void            saveVocab(const string &filePath);
    void            loadVocab(const string &filePath);

    //get vocab size
    int             getVocabSize() { return this->vocabSize; }

private:
    //BPE helper ;find most frequent pair in token list
    pair<string, string>    getMostFrequentPair(const vector<vector<string>> &tokenizedWords);

    //apply one merge rule to all tokenized words
    void                    applyMerge(vector<vector<string>> &tokenizedWords,
                                       const string &first, const string &second);

    //split text into chars for initial BPE state
    vector<vector<string>>  initCharTokens(const string &text);

    //GPT-2-style pre-tokenization ;whitespace attaches as prefix to following word
    vector<string>          preTokenize(const string &text);
};

#endif
