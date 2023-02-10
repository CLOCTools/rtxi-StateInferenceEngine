#ifndef LFPRATIOMETER_H
#define LFPRATIOMETER_H

// just here so clangd can resolve these symbols
#ifndef LFPRATIOMETER
#include "lfpRatiometer.h"
#endif

#include <vector>
#include <malloc.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

class lfpRatiometer {
    public:
        // constructor
        lfpRatiometer(int N_input, double sampling_input);

        // destructor
        ~lfpRatiometer(void);

        // changing FFT plan
        void changeFFTPlan(int N_input, double sampling_input);

        // calculate LF/HF ratio
        void calcRatio(); 

        // setting windowing function
        void window_rect();
        void window_hamming();

        // setting parameters
        // note that to change FFT parameters, a new object needs to be created
        void setRatioParams(double lf_low_input, double lf_high_input, double hf_low_input, double hf_high_input); // ADDED

        // getting parameters
        double getRatio() const { return lf_hf_ratio; }; 
        double getLFpower() const { return lf_total; }; 
        double getHFpower() const { return hf_total; };

        // getting parameters
        std::vector<double> getFreqs() { return allfreqs; };
        std::vector<double> getPSD() { return psd; }
        std::vector<double> getFFTabs() { return fft_abs; };
        std::vector<double> getBuffer() { return in_raw; };
        std::vector<double> getWindow() { return window; };

        std::vector<double> getFreqBounds() { 
            std::vector<double> freqbounds;
            freqbounds.push_back(lf_low);
            freqbounds.push_back(lf_high);
            freqbounds.push_back(hf_low);
            freqbounds.push_back(hf_high);
            return freqbounds;
        };  

        // modifying raw time series
        void pushTimeSample(double input);
        void setTimeSeries(std::vector<double> inputSeries);
        void clrTimeSeries();

        // FFT calculations
        void makePSD();
        void makeFFTabs();
        
    protected:

    private:
    
        int N; // time window in samples
        int f_size; // nonredundant size of DFT
        double sampling; // sampling rate (Hz)
        double *in; // pointer to (windowed) time series
        std::vector<double> in_raw; // vector to hold raw time series
        fftw_complex *out; // pointer to DFT
        fftw_plan p; // stores FFTW3 plan
        std::vector<double> allfreqs;

        std::vector<double> psd;
        std::vector<double> fft_abs;
        double lf_hf_ratio;
        double lf_total;
        double hf_total;
        double lf_low;
        double lf_high;
        double hf_low;
        double hf_high;

        std::vector<double> window; // time domain of window
        double s2; // window scaling factor


};


#endif