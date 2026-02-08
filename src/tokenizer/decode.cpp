#include "Tokenizer.hpp"

//decode token ids -> string ;reverse lookup from id to token
string Tokenizer::decode(const vector<int> &ids) {
    string result = "";
    for(int id : ids) {
        if(this->idToToken.count(id)) {
            result += this->idToToken[id];
        } else {
            result += "<unk>";
        }
    }
    return result;
}
