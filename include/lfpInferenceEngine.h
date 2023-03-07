#ifndef LFPINFERENCEENGINE
#define LFPINFERENCEENGINE

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <complex.h>

#include <vector>
#include <deque>
#include <iostream>

#include <cmath>
#include <algorithm>
#include <armadillo>

#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")

#include <string>
#include <vector>

class lfpInferenceEngine {
    public:
        // Member functions
        lfpInferenceEngine(); // constructor
        ~lfpInferenceEngine(); // destructor

        int callPythonFunction(std::vector<std::string> args, std::vector<PyObject*> pyArgs);
        //int callPredictFunction(std::vector<PyObject*> pyArgs);

        void printInPython();

        void load();
        void load_ll();
        void load_data();
        void loadModule(std::string name, std::string funcname);
        void init(std::string recording, std::string modelname);

        void setFeats(PyObject *newFeats);
        void setModel(PyObject *newModel);
        void setScaler(PyObject *newScaler);
        
        // Data manipulation
        void setData(std::vector<std::vector<double>> newData);
        void pushFFTSample(std::vector<double> fft);
        void reportFFTdata();
        void reportPs();

        PyObject* getResult() {return pResult;};
        PyObject* getModel() {return pModel;};
        PyObject* getFeats() {return pFeats;};
        PyObject* getScaler() {return pScaler;};
        PyObject* getInference() {return pInference;};
        PyObject* getData() {return pData;};

        arma::mat getPi0() {return pi0;};
        std::vector<arma::mat> getPs(){return Ps;};
        arma::mat getLl() {return ll;};

        std::vector<int> predict();

        std::vector<int> PyList_toVecInt(PyObject* py_list);
        std::vector<double> PyList_toVecDouble(PyObject* py_list);
        std::vector<arma::mat> buildPs(std::vector<double> Ps_flat, long T, long K);

        arma::vec viterbi(arma::mat pi0, std::vector<arma::mat> Ps, arma::mat ll);
        arma::vec mapStates(arma::vec states);
        
    protected:

    private:
    
        PyObject *pModel;
        PyObject *pFeats;
        PyObject *pScaler;
        PyObject *pInference;
        PyObject *pResult;
        PyObject *pData;
        PyObject *pStateMap;

        //PyObject *pyModule;
        //PyObject *pyPredictFunc;

        int N; //how many FFT samples to hold in memory
        int M; //how many frequency bins in each FFT: SHOULD MAKE THIS LOADED FROM MODEL
        std::vector<std::vector<double>> fftdata;
        

        arma::mat pi0;
        std::vector<arma::mat> Ps;
        arma::mat ll;

        std::vector<int> state_map;

};


#endif