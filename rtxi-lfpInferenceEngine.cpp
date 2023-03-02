#include <rtxi-lfpInferenceEngine.h>

#include <chrono>

using namespace std;

std::vector<int> PyList_toVecInt(PyObject* py_list);

extern "C" Plugin::Object *createRTXIPlugin(void){
    return new rtxilfpInferenceEngine();
}

static DefaultGUIModel::variable_t vars[] = {

  // Parameters
  { "Time Window (s)", "Time Window (s)", DefaultGUIModel::PARAMETER | DefaultGUIModel::DOUBLE},
  { "Sampling Rate (Hz)", "Sampling Rate (Hz)", DefaultGUIModel::STATE | DefaultGUIModel::DOUBLE,},
  { "LF Lower Bound", "LF Lower Bound", DefaultGUIModel::PARAMETER | DefaultGUIModel::DOUBLE,},
  { "LF Upper Bound", "LF Upper Bound", DefaultGUIModel::PARAMETER | DefaultGUIModel::DOUBLE,},
  { "HF Lower Bound", "HF Lower Bound", DefaultGUIModel::PARAMETER | DefaultGUIModel::DOUBLE,},
  { "HF Upper Bound", "HF Upper Bound", DefaultGUIModel::PARAMETER | DefaultGUIModel::DOUBLE,},
  { "Animal", "Animal name", DefaultGUIModel::COMMENT, },  
  { "Model", "Model name", DefaultGUIModel::COMMENT, },  

  // Inputs
  { "input_LFP", "Input LFP", DefaultGUIModel::INPUT | DefaultGUIModel::DOUBLE,},

  //Outputs
  { "ratio", "Output LFP Power Ratio", DefaultGUIModel::OUTPUT | DefaultGUIModel::DOUBLE,},
  { "LF Power", "Power in LF Band", DefaultGUIModel::OUTPUT | DefaultGUIModel::DOUBLE,},
  { "HF Power", "Power in HF Band", DefaultGUIModel::OUTPUT | DefaultGUIModel::DOUBLE,},
  { "State", "Estimated State", DefaultGUIModel::OUTPUT | DefaultGUIModel::UINTEGER,}
};

static size_t num_vars = sizeof(vars) / sizeof(DefaultGUIModel::variable_t);
std::vector<int> my_vector;
std::chrono::time_point<std::chrono::system_clock> start, stop;
std::chrono::microseconds duration;

// defining what's in the object's constructor
// sampling set by RT period
rtxilfpInferenceEngine::rtxilfpInferenceEngine(void) : DefaultGUIModel("lfpInferenceEngine with Custom GUI", ::vars, ::num_vars),
period(((double)RT::System::getInstance()->getPeriod())*1e-9), // grabbing RT period
sampling(1.0/period), // calculating RT sampling rate
lfpratiometer(N, sampling), // constructing lfpRatiometer object
lfpinferenceengine(),
DEFAULT_ANIMAL("AP103_1"),
DEFAULT_MODEL("a")
{
    setWhatsThis("<p><b>lfpInferenceEngine:</b><br>Given an lfp input, this module estimates the cortical state.</p>");
    
    animal = DEFAULT_ANIMAL;
    model = DEFAULT_MODEL;
    count = 1;
    fftstep = 40;
    lfpinferenceengine.init(animal.toStdString(),model.toStdString());

    DefaultGUIModel::createGUI(vars, num_vars);
    customizeGUI();
    update(INIT);
    refresh();
    QTimer::singleShot(0, this, SLOT(resizeMe()));
    
}

// defining what's in the object's destructor
rtxilfpInferenceEngine::~rtxilfpInferenceEngine(void) { }

// real-time RTXI function
void rtxilfpInferenceEngine::execute(void) {

  // push new time series reading to lfpRatiometer
  lfpratiometer.pushTimeSample(input(0)/0.01);

  // calculate LF/HF ratio
  lfpratiometer.calcRatio();

  // put the LF/HF ratio into the output
  output(0) = lfpratiometer.getRatio();
  output(1) = lfpratiometer.getLFpower();
  output(2) = lfpratiometer.getHFpower();

  
  // estimate cortical state from FFT data
  if (count == fftstep)
  {
    lfpratiometer.makeFFTabs();
    fft = lfpratiometer.truncateFFT(lfpratiometer.getFreqs(),lfpratiometer.getFFTabs());
    lfpinferenceengine.pushFFTSample(fft);
    count = 1;
  }
  count++;
  //state_vec = lfpinferenceengine.predict();

  arguments_predict = {"pyfuncs","log_likes"};
  pyArgs = {lfpinferenceengine.getModel(),
                                    lfpinferenceengine.getFeats(),
                                    lfpinferenceengine.getScaler(),
                                    lfpinferenceengine.getData()};

  lfpinferenceengine.callPythonFunction(arguments_predict, pyArgs);
  lfpinferenceengine.load_ll();
  //std::cout << "pi0: " << pi0 << endl;
    //std::cout << "Ps: " << Ps << endl;
    //std::cout << "ll: " << ll << endl;
  state_vec = lfpinferenceengine.viterbi(lfpinferenceengine.getPi0(),
                                          lfpinferenceengine.getPs(),
                                          lfpinferenceengine.getLl());

  state_vec = lfpinferenceengine.mapStates(state_vec);


  //lfpinferenceengine.callPredictFunction(pyArgs);
  //state_vec = PyList_toVecInt(lfpinferenceengine.getResult());
  
  /*
  printf("Result of call: \n");
  my_vector = PyList_toVecInt(lfpinferenceengine.getResult());
  for (int j = 0; j < my_vector.size(); j++) {
  std::cout << my_vector[j] << " ";
  }
  std::cout << std::endl;
  */
  if (!state_vec.empty()) {
    state = state_vec.back();
  }else{
    state = -1;
  }
  output(3) = state;
  
  
    
}

// update function (not running in real time)
void rtxilfpInferenceEngine::update(DefaultGUIModel::update_flags_t flag)
{
  //std::vector<int> my_vec;
  switch (flag) {
    case INIT:
      setParameter("Time Window (s)", sampling/N);
      setState("Sampling Rate (Hz)", sampling);
      // get bounds from lfpratiometer object
      setParameter("LF Lower Bound", lfpratiometer.getFreqBounds()[0]);
      setParameter("LF Upper Bound", lfpratiometer.getFreqBounds()[1]);
      setParameter("HF Lower Bound", lfpratiometer.getFreqBounds()[2]);
      setParameter("HF Upper Bound", lfpratiometer.getFreqBounds()[3]);

      setParameter("Animal",animal);
      setParameter("Model",model);

      break;

    case MODIFY:
      // defining parameters needed for constructor
      period = ((double)RT::System::getInstance()->getPeriod())*1e-9;
      sampling = 1.0/period;
      setState("Sampling Rate (Hz)", sampling); // updating GUI
      N = (int) (getParameter("Time Window (s)").toDouble() * sampling);

      // making new FFT plan
      lfpratiometer.changeFFTPlan(N, sampling);

      // setting frequency bounds based on user input
      lfpratiometer.setRatioParams(getParameter("LF Lower Bound").toDouble(),
          getParameter("LF Upper Bound").toDouble(),
          getParameter("HF Lower Bound").toDouble(),
          getParameter("HF Upper Bound").toDouble());

      start = std::chrono::high_resolution_clock::now();
     
      //lfpinferenceengine.init(getComment("Animal").toStdString(),getComment("Model").toStdString());
      //pModel = Py_NewRef(lfpinferenceengine.getModel());
      //pFeats = Py_NewRef(lfpinferenceengine.getFeats());
      //pScaler = Py_NewRef(lfpinferenceengine.getScaler());
      //pData = Py_NewRef(lfpinferenceengine.getData());

      //arguments_predict = {"pyfuncs","predict"};
      //start = std::chrono::high_resolution_clock::now();
      //state_vec = lfpinferenceengine.predict();
      
      arguments_predict = {"pyfuncs","log_likes"};
      pyArgs = {lfpinferenceengine.getModel(),
                                        lfpinferenceengine.getFeats(),
                                        lfpinferenceengine.getScaler(),
                                        lfpinferenceengine.getData()};

      //PyRun_SimpleString("print(lfpinferenceengine.getData().size())");

      lfpinferenceengine.callPythonFunction(arguments_predict, pyArgs);
      
      lfpinferenceengine.load_ll();
      //std::cout << "pi0: " << lfpinferenceengine.getPi0() << endl;
      //std::cout << "Ps[0]: " << lfpinferenceengine.getPs()[0] << endl;
      //lfpinferenceengine.reportPs();
      //std::cout << "ll: " << lfpinferenceengine.getLl() << endl;
      state_vec = lfpinferenceengine.viterbi(lfpinferenceengine.getPi0(),
                                              lfpinferenceengine.getPs(),
                                              lfpinferenceengine.getLl());

      state_vec = lfpinferenceengine.mapStates(state_vec);
      
      //state_vec = PyList_toVecInt(lfpinferenceengine.getResult());
      //my_vec = PyList_toVecInt(lfpinferenceengine.getResult());
      
      stop = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

      std::cout << "Predict time: " << duration.count() << std::endl;
      printf("Result of call: \n");
      //my_vector = PyList_toVecInt(lfpinferenceengine.getResult());
      for (int j = 0; j < state_vec.size(); j++) {
      std::cout << state_vec[j] << " ";
      }
      std::cout << std::endl;
      
      // setting DFT windowing function choice
      if (windowShape->currentIndex() == 0) {
        lfpratiometer.window_rect();
      }
      else if (windowShape->currentIndex() == 1) {
        lfpratiometer.window_hamming();
      }

      // clearing time series
      lfpratiometer.clrTimeSeries();

      break;

    case UNPAUSE:
      break;

    case PAUSE:
      //lfpinferenceengine.reportFFTdata();
      lfpratiometer.clrTimeSeries();
      break;

    case PERIOD:
      break;

    default:
      break;
  }
}

// RTXI's customizeGUI function
void rtxilfpInferenceEngine::customizeGUI(void)
{
  QGridLayout* customlayout = DefaultGUIModel::getLayout();

  // adding dropdown menu for choosing FFT window shape
  windowShape = new QComboBox;
  windowShape->insertItem(1, "Rectangular");
  windowShape->insertItem(2, "Hamming");

  customlayout->addWidget(windowShape, 2, 0);
  setLayout(customlayout);
}

std::vector<int> PyList_toVecInt(PyObject* py_list) {
  if (PySequence_Check(py_list)) {
    PyObject* seq = PySequence_Fast(py_list, "expected a sequence");
    if (seq != NULL){
      std::vector<int> my_vector;
      my_vector.reserve(PySequence_Fast_GET_SIZE(seq));
      for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq,i);
        if(PyNumber_Check(item)){
          Py_ssize_t value = PyNumber_AsSsize_t(item, PyExc_OverflowError);
          if (value == -1 && PyErr_Occurred()) {
            //handle error
          }
          my_vector.push_back(value);
        } else {
          //handle error
        }
      }
      Py_DECREF(seq);
      return my_vector;
    } else {
      //handle error
    }
  }else{
    //handle error
  }
}

