#include "Tokenizer.hpp"

//constructor ;init special tokens
Tokenizer::Tokenizer() {
    this->vocabSize = 0;
    this->padId     = -1;
    this->unkId     = -1;
    this->bosId     = -1;
    this->eosId     = -1;
}

Tokenizer::~Tokenizer() {}

//build BPE vocabulary from raw text corpus
//1. start with character-level vocabulary
//2. count all adjacent pairs
//3. merge most frequent pair, add to vocab
//4. repeat until targetVocabSize reached
void Tokenizer::buildVocab(const string &text, int targetVocabSize) {
    cout << "building BPE vocabulary... target size: " << targetVocabSize << endl;

    //step 1: init special tokens
    this->tokenToId["<pad>"]    = 0;
    this->tokenToId["<unk>"]    = 1;
    this->tokenToId["<bos>"]    = 2;
    this->tokenToId["<eos>"]    = 3;
    this->idToToken[0]          = "<pad>";
    this->idToToken[1]          = "<unk>";
    this->idToToken[2]          = "<bos>";
    this->idToToken[3]          = "<eos>";
    this->padId = 0;
    this->unkId = 1;
    this->bosId = 2;
    this->eosId = 3;

    int nextId = 4;

    //step 2: add all unique characters to vocab
    set<char> chars;
    for(char c : text) {
        chars.insert(c);
    }
    for(char c : chars) {
        string s(1, c);
        if(!this->tokenToId.count(s)) {
            this->tokenToId[s]      = nextId;
            this->idToToken[nextId] = s;
            nextId++;
        }
    }

    cout << "  base char vocab: " << nextId << " tokens" << endl;

    //step 3: split text into words, then chars per word
    vector<vector<string>> tokenizedWords = this->initCharTokens(text);

    //step 4: iterative BPE merges
    int mergeCount = 0;
    while(nextId < targetVocabSize) {
        //find most frequent adjacent pair across all words
        pair<string, string> bestPair = this->getMostFrequentPair(tokenizedWords);

        //no more pairs to merge
        if(bestPair.first.empty() && bestPair.second.empty()) {
            break;
        }

        //create merged token
        string merged = bestPair.first + bestPair.second;

        //add to vocab
        this->tokenToId[merged]     = nextId;
        this->idToToken[nextId]     = merged;
        nextId++;

        //record merge rule
        this->merges.push_back(bestPair);

        //apply merge to all tokenized words
        this->applyMerge(tokenizedWords, bestPair.first, bestPair.second);

        mergeCount++;
        if(mergeCount % 100 == 0) {
            cout << "  merges: " << mergeCount << " | vocab: " << nextId << endl;
        }
    }

    this->vocabSize = nextId;
    cout << "vocab built: " << this->vocabSize << " tokens, " << mergeCount << " merges" << endl;
}

//find the most frequent adjacent pair in tokenized words
pair<string, string> Tokenizer::getMostFrequentPair(const vector<vector<string>> &tokenizedWords) {
    map<pair<string, string>, int> pairCounts;

    for(auto &word : tokenizedWords) {
        for(int i = 0; i + 1 < (int)word.size(); i++) {
            pairCounts[{word[i], word[i + 1]}]++;
        }
    }

    //find max
    pair<string, string> bestPair = {"", ""};
    int bestCount = 0;
    for(auto &pc : pairCounts) {
        if(pc.second > bestCount) {
            bestCount = pc.second;
            bestPair = pc.first;
        }
    }

    return bestPair;
}

//apply a merge rule to all tokenized words ;replace adjacent (first,second) with merged
void Tokenizer::applyMerge(vector<vector<string>> &tokenizedWords,
                            const string &first, const string &second) {
    for(auto &word : tokenizedWords) {
        vector<string> newWord;
        int i = 0;
        while(i < (int)word.size()) {
            if(i + 1 < (int)word.size() && word[i] == first && word[i + 1] == second) {
                newWord.push_back(first + second);
                i += 2;
            } else {
                newWord.push_back(word[i]);
                i++;
            }
        }
        word = newWord;
    }
}

//split text into words then chars ;initial BPE tokenization
//uses GPT-2-style pre-tokenization: whitespace is attached as prefix to following word
vector<vector<string>> Tokenizer::initCharTokens(const string &text) {
    vector<vector<string>> result;
    vector<string> words = this->preTokenize(text);

    for(auto &word : words) {
        vector<string> chars;
        for(int i = 0; i < (int)word.size(); i++) {
            chars.push_back(string(1, word[i]));
        }
        if(!chars.empty()) {
            result.push_back(chars);
        }
    }

    return result;
}

//GPT-2-style pre-tokenization
//whitespace characters are attached as prefix to the following word
//e.g. "hello world\nfoo" -> ["hello", " world", "\nfoo"]
//preserves newlines, tabs, spaces — model learns them as part of subword tokens
vector<string> Tokenizer::preTokenize(const string &text) {
    vector<string> words;
    string current = "";

    for(int i = 0; i < (int)text.size(); i++) {
        char c = text[i];
        bool isWs = (c == ' ' || c == '\n' || c == '\t' || c == '\r');

        if(isWs) {
            //flush current non-whitespace word (if any)
            if(!current.empty()) {
                words.push_back(current);
                current = "";
            }
            //start new word with this whitespace char as prefix
            current += c;
        } else {
            current += c;
        }
    }

    if(!current.empty()) {
        words.push_back(current);
    }

    return words;
}
