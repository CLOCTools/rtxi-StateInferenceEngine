#include <lfpRatiometer>

using namespace std;


// defining what's in the object's constructor
// user defines time window length (in samples) and sampling rate
lfpRatiometer::lfpRatiometer(int N_input, double sampling_input) {
    // setting user defined variables
    N=N_input;
    f_size=N/2 + 1;
    sampling=sampling_input;
    
    // setting default window
    window_rect();

    // establishing frequencies that will be associated with DFT output
    for (int n=0; n<f_size;n++){
        allfreqs.push_back(sampling*n/N);
    }

    // allocating space for input time array
    in = (double*) fftw_malloc(sizeof(double)*N);
    // allocating space for N DFT terms
    // note: this has to be N, not N/2 + 1, for fftw_free()
    out = fftw_alloc_complex(N);
    // creating the FFTW3 plan
    // FFTW_MEASURE makes initialization long for faster...
    // calculations
    p = fftw_plan_dft_r2c_1d(N,in,out,FFTW_MEASURE);


}

// defining what's in the object's destructor
lfpRatiometer::~lfpRatiometer(void) {
    // destroy plan
    fftw_destroy_plan(p);
    // free output array
    fftw_free(out);
    // free input array
    fftw_free(in);

    return;
}

// this function changes the FFT plan (previously defined in the constructor)
// the existence of this function is very imperfect 
// this function probably shouldn't be used except for the RTXI plugin
void lfpRatiometer::changeFFTPlan(int N_input, double sampling_input) {
    // destroying previous FFT plan (similar to destructor)
    fftw_destroy_plan(p);
    fftw_free(out);
    fftw_free(in);

    // making new plan (similar to constructor)
    N=N_input;
    f_size=N/2 + 1;
    sampling=sampling_input;

    allfreqs.clear();
    for (int n=0; n<f_size;n++){
        allfreqs.push_back(sampling*n/N);
    }

    in = (double*) fftw_malloc(sizeof(double)*N);
    out = fftw_alloc_complex(N);
    p = fftw_plan_dft_r2c_1d(N,in,out,FFTW_MEASURE);

}


// allows users to push a new time series data point
void lfpRatiometer::pushTimeSample(double input) {
    in_raw.push_back(input);
    if (in_raw.size() > N) {
        // cut it to size
        in_raw.erase(in_raw.begin());
    }
}

// allows users to clear the raw time series
void lfpRatiometer::clrTimeSeries() {
    in_raw.clear();
}

// allows users to input a specific sequence of time series data
void lfpRatiometer::setTimeSeries(std::vector<double> inputSeries) {
    clrTimeSeries();

    // we require that the input time series be of the appropriate size
    if (inputSeries.size() == N) {
        in_raw = inputSeries;
    }
}

// method sets window as rectangle
void lfpRatiometer::window_rect() {
    window.clear();
    s2 = 0;
    for (int j=0; j<N; j++){
        double val = 1;
        window.push_back(val);
        s2 = s2 + val*val;
    }
}

// method sets window as Hamming
// supposed to be consistent with MATLAB hamming() function
void lfpRatiometer::window_hamming(){
    window.clear();
    s2 = 0;
    double pi = atan(1) * 4;
    for (int j=0; j<N; j++){
        double z = 2*pi*j/(N-1);
        double val = 0.54 - 0.46*cos(z);
        window.push_back(val);
        s2 = s2 + val*val;
    }
}
    

// function that calculates the power spectral density
void lfpRatiometer::makePSD() {
    psd.clear(); // clear vector which stores PSD

    if (in_raw.size() == N) {
        // applying window
        for (int n=0; n<N; n++){
            in[n] = in_raw[n]*window[n];
        }

        fftw_execute(p); // FFTW3 calculate the DFT

        // loop through DFT entries
        for (int n=0; n<f_size; n++){
            // calculate |y_n|^2
            double fftsqr = out[n][0]*out[n][0] + out[n][1]*out[n][1];

            // calculate PSD (described in Heinzel et al., 2002)
            // DC & Nyquist freq is scaled to be consistent with MATLAB's spectrogram
            // Usually, the edges should not matter because all frequencies of interest
            // should be contained in the interior
            // WILL CHANGE WITH WINDOWING
            if (n==0 || ((n==f_size-1) && (N%2 == 0))) {
                psd.push_back((1/(sampling*s2)) * fftsqr);
            }
            else {
                psd.push_back(2*(1/(sampling*s2)) * fftsqr);
            }
        }
    }

    
}

// function that calculates the power spectral density
void lfpRatiometer::makeFFTabs() {
    fft_abs.clear(); // clear vector which stores PSD

    // applying window
    for (int n=0; n<N; n++){
        in[n] = in_raw[n]*window[n];
    }

    fftw_execute(p); // FFTW3 calculate the DFT

    // loop through DFT entries
    for (int n=0; n<f_size; n++){
        // calculate |y_n|^2
        double fftsqr = out[n][0]*out[n][0] + out[n][1]*out[n][1];

        fft_abs.push_back(sqrt(fftsqr));
    }
}