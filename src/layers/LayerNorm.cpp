#include "LayerNorm.hpp"

//constructor ;init gamma=1, beta=0
LayerNorm::LayerNorm(int dModel) {
    this->dModel        = dModel;
    this->eps           = 1e-5;
    this->gamma         = new Matrix(1, dModel, 1.0);   //scale, init to 1
    this->beta          = new Matrix(1, dModel, 0.0);   //shift, init to 0
    this->gradGamma     = new Matrix(1, dModel, 0.0);
    this->gradBeta      = new Matrix(1, dModel, 0.0);
    this->normalized    = nullptr;
    this->stdInv        = nullptr;
    this->inputCentered = nullptr;
}

LayerNorm::~LayerNorm() {
    delete this->gamma;
    delete this->beta;
    delete this->gradGamma;
    delete this->gradBeta;
    if(this->normalized)    delete this->normalized;
    if(this->stdInv)        delete this->stdInv;
    if(this->inputCentered) delete this->inputCentered;
}

//forward: layer normalization
//for each row (position in sequence):
//  mean = (1/d) * sum(x_j)
//  var  = (1/d) * sum((x_j - mean)^2)
//  x_hat = (x - mean) / sqrt(var + eps)
//  y = gamma * x_hat + beta
Matrix* LayerNorm::forward(Matrix *input) {
    int seqLen = input->numRows;
    int d = this->dModel;

    //cleanup cached values from previous forward
    if(this->normalized)    delete this->normalized;
    if(this->stdInv)        delete this->stdInv;
    if(this->inputCentered) delete this->inputCentered;

    this->normalized    = new Matrix(seqLen, d, 0.0);
    this->stdInv        = new Matrix(seqLen, 1, 0.0);   //1/sqrt(var+eps) per row
    this->inputCentered = new Matrix(seqLen, d, 0.0);

    Matrix *output = new Matrix(seqLen, d, 0.0);

    for(int i = 0; i < seqLen; i++) {
        //compute mean of row i
        double mean = 0.0;
        for(int j = 0; j < d; j++) {
            mean += input->at(i, j);
        }
        mean /= d;

        //compute variance of row i
        double var = 0.0;
        for(int j = 0; j < d; j++) {
            double diff = input->at(i, j) - mean;
            var += diff * diff;
            this->inputCentered->at(i, j) = diff;
        }
        var /= d;

        //1/sqrt(var + eps) ;stored for backward
        double invStd = 1.0 / sqrt(var + this->eps);
        this->stdInv->at(i, 0) = invStd;

        //normalize and apply gamma/beta
        for(int j = 0; j < d; j++) {
            double xHat = this->inputCentered->at(i, j) * invStd;
            this->normalized->at(i, j) = xHat;
            output->at(i, j) = this->gamma->at(0, j) * xHat + this->beta->at(0, j);
        }
    }

    return output;
}

//backward: compute dL/dx, dL/dgamma, dL/dbeta
//dL/dbeta  = sum over seq of dL/dy
//dL/dgamma = sum over seq of dL/dy * x_hat
//dL/dx_hat = dL/dy * gamma
//dL/dx = (1/d) * invStd * (d*dL/dx_hat - sum(dL/dx_hat) - x_hat*sum(dL/dx_hat*x_hat))
Matrix* LayerNorm::backward(Matrix *gradOutput) {
    int seqLen  = gradOutput->numRows;
    int d       = this->dModel;

    Matrix *gradInput = new Matrix(seqLen, d, 0.0);

    for(int i = 0; i < seqLen; i++) {
        double invStd = this->stdInv->at(i, 0);

        //dL/dx_hat = dL/dy * gamma
        vector<double> dxHat(d);
        for(int j = 0; j < d; j++) {
            dxHat[j] = gradOutput->at(i, j) * this->gamma->at(0, j);
        }

        //accumulate dL/dgamma += dL/dy * x_hat
        //accumulate dL/dbeta  += dL/dy
        for(int j = 0; j < d; j++) {
            this->gradGamma->at(0, j) += gradOutput->at(i, j) * this->normalized->at(i, j);
            this->gradBeta->at(0, j)  += gradOutput->at(i, j);
        }

        //sum terms for dL/dx
        double sum_dxhat = 0.0;
        double sum_dxhat_xhat = 0.0;
        for(int j = 0; j < d; j++) {
            sum_dxhat       += dxHat[j];
            sum_dxhat_xhat  += dxHat[j] * this->normalized->at(i, j);
        }

        //dL/dx = invStd/d * (d*dxHat - sum_dxhat - xhat*sum_dxhat_xhat)
        for(int j = 0; j < d; j++) {
            gradInput->at(i, j) = (invStd / d) *
                (d * dxHat[j] - sum_dxhat - this->normalized->at(i, j) * sum_dxhat_xhat);
        }
    }

    return gradInput;
}

void LayerNorm::zeroGrad() {
    this->gradGamma->zeroOut();
    this->gradBeta->zeroOut();
}
