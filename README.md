 # rtxi-StateInferenceEngine  
This code is for a Real-Time eXperiment Inferface (http://rtxi.org/) module for inferring 
brain states from local field potential observations using Hidden Semi-Markov
Models (HSMMs).  For an in depth description of the methods, see the accompanying paper.

## Installation

Installation has been tested on Ubuntu 16.04 & 22.04.  This software may work 
with other Linux distributions that support RTXI but it is not guaranteed.  If using
other distributions, the user may need to find suitable replacement dependencies. 
These installation instructions assume that the user has already installed RTXI
on their system.  If not, follow the instructions [here](http://rtxi.org/install/) to install RTXI.

1. Clone this project
~~~bash
git clone https://github.com/CLOCTools/rtxi-StateInferenceEngine.git
~~~

2. Install system binaries
~~~bash
sudo apt-get install cmake pkg-config fftw3-dev
~~~
3. Install [Anaconda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-22-04).
After installed create and activate a conda environment.
~~~bash
conda create -n hmm python=3.10
conda activate hmm
~~~
4. Download and install python dependencies. Follow instructions to install 
[ssm](https://github.com/lindermanlab/ssm). While your conda environment is activated
use pip to install mat73 and scikit-learn.
~~~bash
pip install mat73
pip install scikit-learn==1.1.2
~~~
5. Point the dynamic linker to the lib directory of your Anaconda environment.
To do this you must create a file /etc/ld.so.conf.d/99local.conf using sudo privileges
and write the path to your lib directory on the first line.  Using standard Anaconda installation,
your lib directory should located at **/home/${USER}/anaconda3/envs/${ENV_NAME}/lib**.
Once the configuration file is created, you must run ldconfig to tell the dynamic linker to
update its path.
~~~bash
sudo ldconfig
~~~
6. Set up Makefiles to find relevant libraries. You must set up an installation specific
Makefile.plugin_compile following the instructions below.  For reference, an example of a 
working Makefile.plugin_compile can be found in the root directory of this project. 
First, you must copy the file Makefile.plugin_compile
from your rtxi directory to the rtxi-StateInferenceEngine project directory.
This file should have been created in the rtxi root directory during installation
of RTXI and has system/installation specific include flags that are automatically
generated.  Now open the project's version of Makefile.plugin_compile and edit the 
build instructions to include ${PYTHON_CXX_FLAGS} and ${PYTHON_LD_FLAGS} 
After completion, the build instructions should look like this:
~~~bash
%.lo: %.cpp
	$(CXXCOMPILE) $(CXXFLAGS) $(PYTHON_CXX_FLAGS) $(PYTHON_LD_FLAGS) -c $< -o $@

$(PLUGIN_NAME).la: $(OBJECTS) $(SOURCES) $(HEADERS)
	$(CXXLINK) $(CXXFLAGS) $(PYTHON_CXX_FLAGS) $(LIBS) $(LDFLAGS) $(PYTHON_LD_FLAGS) $(OBJECTS) -rpath `readlink -f $(exec_modeldir)` -o $(PLUGIN_NAME).la

install: $(PLUGIN_NAME).la
	$(LIBTOOL) --mode=install cp $(PLUGIN_NAME).la `readlink -f $(exec_modeldir)`
~~~

7. Build and install the project by running
~~~bash
make
make install
~~~

## Run Locally  

Clone the project  

~~~bash  
  git clone https://link-to-project
~~~

Go to the project directory  

~~~bash  
  cd my-project
~~~

Install dependencies  

~~~bash  
npm install
~~~

Start the server  

~~~bash  
npm run start
~~~

## Contributing  

Contributions are always welcome!  

See `contributing.md` for ways to get started.  

Please adhere to this project's `code of conduct`.  

## License  

[MIT](https://choosealicense.com/licenses/mit/)
