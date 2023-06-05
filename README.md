 # rtxi-StateInferenceEngine  
This code is for a Real-Time eXperiment Inferface (http://rtxi.org/) module for inferring 
brain states from local field potential observations using Hidden Semi-Markov
Models (HSMMs).  For an in depth description of the methods, see the accompanying paper.

## Installation

Installation has been tested on Ubuntu 16.04 & 22.04.  This software may work 
with other Linux distributions that support RTXI but it is not guaranteed.  If using
other distributions, the user may need to find suitable replacement dependencies.

1. Install system binaries
~~~bash
sudo apt-get install cmake pkg-config fftw3-dev
~~~
2. Install [Anaconda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-22-04).
After installed create and activate a conda environment.
~~~bash
conda create -n hmm python=3.10
conda activate hmm
~~~
3. Download and install python dependencies. Follow instructions to install 
[ssm](https://github.com/lindermanlab/ssm). While your conda environment is activated
use pip to install mat73 and scikit-learn.
~~~bash
pip install mat73
pip install scikit-learn==1.1.2
~~~


## Contributing  

Contributions are always welcome!  

See `contributing.md` for ways to get started.  

Please adhere to this project's `code of conduct`.  

## License  

[MIT](https://choosealicense.com/licenses/mit/)
