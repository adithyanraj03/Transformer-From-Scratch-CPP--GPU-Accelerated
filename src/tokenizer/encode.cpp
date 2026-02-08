#include "Tokenizer.hpp"

//encode text -> sequence of token ids using learned BPE merges
//1. pre-tokenize (GPT-2 style: whitespace as prefix to following word)
//2. split each word into characters
//3. apply merge rules in order
//4. map final tokens to ids
vector<int> Tokenizer::encode(const string &text) {
    vector<int> result;
    if(text.empty()) return result;

    //GPT-2-style split: whitespace attaches as prefix to following word
    vector<string> words = this->preTokenize(text);

    for(auto &word : words) {
        //init as char-level tokens
        vector<string> tokens;
        for(int i = 0; i < (int)word.size(); i++) {
            tokens.push_back(string(1, word[i]));
        }

        //apply all merges in order ;greedy left-to-right
        for(auto &mergePair : this->merges) {
            vector<string> newTokens;
            int i = 0;
            while(i < (int)tokens.size()) {
                if(i + 1 < (int)tokens.size() &&
                   tokens[i] == mergePair.first &&
                   tokens[i + 1] == mergePair.second) {
                    //merge this pair
                    newTokens.push_back(tokens[i] + tokens[i + 1]);
                    i += 2;
                } else {
                    newTokens.push_back(tokens[i]);
                    i++;
                }
            }
            tokens = newTokens;
        }

        //map tokens to ids ;no artificial space injection needed
        for(auto &tok : tokens) {
            if(this->tokenToId.count(tok)) {
                result.push_back(this->tokenToId[tok]);
            } else {
                result.push_back(this->unkId);  //unknown token
            }
        }
    }

    return result;
}
