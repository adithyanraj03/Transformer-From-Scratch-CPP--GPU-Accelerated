#include "Tokenizer.hpp"
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

//save vocab + merges to json file
void Tokenizer::saveVocab(const string &filePath) {
    json j;

    //save token->id mapping
    j["tokenToId"] = this->tokenToId;

    //save merge rules as array of pairs
    json mergesJson = json::array();
    for(auto &m : this->merges) {
        mergesJson.push_back({m.first, m.second});
    }
    j["merges"] = mergesJson;

    //save special token ids
    j["padId"]      = this->padId;
    j["unkId"]      = this->unkId;
    j["bosId"]      = this->bosId;
    j["eosId"]      = this->eosId;
    j["vocabSize"]  = this->vocabSize;

    //write to file
    ofstream f(filePath);
    if(!f.is_open()) {
        cerr << "error: could not open vocab file for writing: " << filePath << endl;
        return;
    }
    f << j.dump(2);
    f.close();
    cout << "vocab saved to " << filePath << endl;
}

//load vocab + merges from json file
void Tokenizer::loadVocab(const string &filePath) {
    ifstream f(filePath);
    if(!f.is_open()) {
        cerr << "error: could not open vocab file: " << filePath << endl;
        return;
    }

    json j;
    f >> j;
    f.close();

    //restore token->id and id->token mappings
    this->tokenToId.clear();
    this->idToToken.clear();
    for(auto &[key, val] : j["tokenToId"].items()) {
        int id = val.get<int>();
        this->tokenToId[key]    = id;
        this->idToToken[id]     = key;
    }

    //restore merge rules
    this->merges.clear();
    for(auto &m : j["merges"]) {
        this->merges.push_back({m[0].get<string>(), m[1].get<string>()});
    }

    //restore special tokens
    this->padId     = j["padId"].get<int>();
    this->unkId     = j["unkId"].get<int>();
    this->bosId     = j["bosId"].get<int>();
    this->eosId     = j["eosId"].get<int>();
    this->vocabSize = j["vocabSize"].get<int>();

    cout << "vocab loaded from " << filePath << " (" << this->vocabSize << " tokens)" << endl;
}
