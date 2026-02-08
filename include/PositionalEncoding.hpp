#ifndef _POSITIONAL_ENCODING_HPP_
#define _POSITIONAL_ENCODING_HPP_

#include "Matrix.hpp"

//sinusoidal positional encoding ;precomputed table
//PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
//PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
class PositionalEncoding {
public:
    int         maxSeqLen;
    int         dModel;
    Matrix*     encodingTable;  //[maxSeqLen x dModel] precomputed

    PositionalEncoding(int maxSeqLen, int dModel);
    ~PositionalEncoding();

    //add PE to input embeddings ;input [seqLen x dModel] -> output [seqLen x dModel]
    Matrix*     forward(Matrix *input);

    //backward: PE is constant so gradient just passes through
    Matrix*     backward(Matrix *gradOutput);

private:
    void        buildTable();
};

#endif
