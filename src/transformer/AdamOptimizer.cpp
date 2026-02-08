#include "Transformer.hpp"

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

//constructor
AdamOptimizer::AdamOptimizer(double lr, double beta1, double beta2, double eps, double weightDecay) {
    this->lr            = lr;
    this->beta1         = beta1;
    this->beta2         = beta2;
    this->eps           = eps;
    this->weightDecay   = weightDecay;
    this->timestep      = 0;
    this->initialized   = false;
}

AdamOptimizer::~AdamOptimizer() {
    for(auto p : this->m) delete p;
    for(auto p : this->v) delete p;
}

//initialize first (m) and second (v) moment buffers ;zeros matching param shapes
void AdamOptimizer::initMoments(vector<Matrix*> &params) {
    for(auto p : this->m) delete p;
    for(auto p : this->v) delete p;
    this->m.clear();
    this->v.clear();

    for(auto p : params) {
        this->m.push_back(new Matrix(p->numRows, p->numCols, 0.0));
        this->v.push_back(new Matrix(p->numRows, p->numCols, 0.0));
    }
    this->initialized = true;
}

//AdamW update step
//GPU: fused kernel does m/v update + bias correction + weight update in ONE kernel launch
//CPU fallback: element-wise loops
void AdamOptimizer::step(vector<Matrix*> &params, vector<Matrix*> &grads) {
    if(!this->initialized) {
        this->initMoments(params);
    }

    this->timestep++;

    //bias correction terms
    double bc1 = 1.0 - pow(this->beta1, this->timestep);
    double bc2 = 1.0 - pow(this->beta2, this->timestep);

#ifdef USE_CUDA
    if(cuda_is_available()) {
        //fused GPU AdamW ;one kernel per parameter matrix
        for(int p = 0; p < (int)params.size(); p++) {
            int sz = params[p]->size();

            //get GPU pointers ;uploads if needed
            double *dParam = params[p]->gpu();
            double *dGrad  = grads[p]->gpu();
            double *dM     = this->m[p]->gpu();
            double *dV     = this->v[p]->gpu();

            cuda_adamw_gpu(dParam, dGrad, dM, dV,
                          this->lr, this->beta1, this->beta2,
                          bc1, bc2, this->eps, this->weightDecay, sz);

            //mark GPU data as authoritative
            params[p]->gpuValid = true;
            params[p]->gpuDirty = true;
            this->m[p]->gpuValid = true;
            this->m[p]->gpuDirty = true;
            this->v[p]->gpuValid = true;
            this->v[p]->gpuDirty = true;
        }
        return;
    }
#endif

    //CPU fallback
    for(int p = 0; p < (int)params.size(); p++) {
        int rows = params[p]->numRows;
        int cols = params[p]->numCols;

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                double g = grads[p]->at(i, j);

                //update first moment ;m = beta1*m + (1-beta1)*g
                this->m[p]->at(i, j) =
                    this->beta1 * this->m[p]->at(i, j) + (1.0 - this->beta1) * g;

                //update second moment ;v = beta2*v + (1-beta2)*g^2
                this->v[p]->at(i, j) =
                    this->beta2 * this->v[p]->at(i, j) + (1.0 - this->beta2) * g * g;

                //bias-corrected moments
                double mHat = this->m[p]->at(i, j) / bc1;
                double vHat = this->v[p]->at(i, j) / bc2;

                //AdamW update ;decoupled weight decay
                double update = mHat / (sqrt(vHat) + this->eps);
                params[p]->at(i, j) -= this->lr * (update + this->weightDecay * params[p]->at(i, j));
            }
        }
    }
}
