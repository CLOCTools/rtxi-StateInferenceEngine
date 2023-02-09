#ifndef LFPRATIOMETER_H
#define LFPRATIOMETER_H

// just here so clangd can resolve these symbols
#ifndef LFPRATIOMETER
#include <lfpRatiometer>
#endif

class lfpRatiometer {
    public:
        // constructor
        lfpRatiometer(int N_input, double sampling_input);

        // destructor
        ~lfpRatiometer(void);

        // changing FFT plan
        void changeFFTPlan(int N_input, double sampling_input);

        // setting windowing function
        void window_rect();
        void window_hamming();

        // getting parameters
        std::vector<double> getFreqs() { return allfreqs; };
        std::vector<double> getPSD() { return psd; }
        std::vector<double> getFFTabs() { return fft_abs; };
        std::vector<double> getBuffer() { return in_raw; };
        std::vector<double> getWindow() { return window; };

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

        std::vector<double> window; // time domain of window
        double s2; // window scaling factor


};


#endif