#ifndef RTXILFPINFERENCEENGINE_H
#define RTXILFPINFERENCEENGINE_H

#include <cstring>
#include <string>

#include <default_gui_model.h>
#include <main_window.h>
#include <plotdialog.h>
#include "lfpRatiometer.h"
#include "lfpInferenceEngine.h"

class rtxilfpInferenceEngine : public DefaultGUIModel {

    Q_OBJECT

    public:
        // constructor
        rtxilfpInferenceEngine(void);

        // destructor
        virtual ~rtxilfpInferenceEngine(void);

        // execute
        void execute(void);

        // functions to make GUI
        void createGUI(DefaultGUIModel::variable_t*, int);
        void customizeGUI(void);

        //void resetVars(void);
        
    protected:

        // update function
        virtual void update(DefaultGUIModel::update_flags_t);

    private:

        // needed to initialize lfpratiometer object
        int N = 1000; // initialized to 1000 samples (1s for 1kHz sampling)
        double period; // set from RT period
        double sampling; // set based on RT period

        // Model loading
        const QString DEFAULT_ANIMAL;
        const QString DEFAULT_MODEL;
        QString animal;
        QString model;
        
        // declarations for state inference
        int state;
        std::vector<int> state_vec;

        // parameters for inputs into python functions
        std::vector<std::string> arguments_predict;
        std::vector<PyObject*> pyArgs;

        // lfpRatiometer object
        lfpRatiometer lfpratiometer;
        lfpInferenceEngine lfpinferenceengine;

        // variables for GUI
        QComboBox* windowShape;

        PyObject *pModel;
        PyObject *pFeats;
        PyObject *pScaler;
        PyObject *pData;

        std::vector<double> fft;
        int count;
        int fftstep;

};


#endif