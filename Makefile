PLUGIN_NAME = rtxilfpInferenceEngine

HEADERS = rtxi-lfpInferenceEngine.h

SOURCES = rtxi-lfpInferenceEngine.cpp \
			src/lfpInferenceEngine.cpp \
			include/lfpInferenceEngine.h \
			src/lfpRatiometer.cpp \
			include/lfpRatiometer.h \
			moc_rtxi-lfpRatiometer.cpp

LIBS =

OS := $(shell uname)
CXX = g++

CXX_FLAGS=-std=c++11 -Iinclude
PYTHON_CXX_FLAGS=-I/home/dweiss/anaconda3/envs/hmm/include/python3.10 -I/home/dweiss/anaconda3/envs/hmm/include/python3.10 -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -isystem /home/dweiss/anaconda3/envs/hmm/include -fdebug-prefix-map=/croot/python-split_1669298683653/work=/usr/local/src/conda/python-3.10.8 -fdebug-prefix-map=/home/dweiss/anaconda3/envs/hmm=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall

LD_FLAGS=-lfftw3 -lm
PYTHON_LD_FLAGS=-L/home/dweiss/anaconda3/envs/hmm/lib/python3.10/config-3.10-x86_64-linux-gnu -L/home/dweiss/anaconda3/envs/hmm/lib -lpython3.10 -lcrypt -lpthread -ldl -lutil -lm -lm


# FFTW3 (not sure if necessary)
#CXXFLAGS := $(CXXFLAGS) $(shell pkg-config --cflags fftw3)
#LDFLAGS := $(LDFLAGS) $(shell pkg-config --libs fftw3)

# lfpRatiometer
#CXXFLAGS := $(CXXFLAGS) $(shell pkg-config --cflags lfpRatiometer)
#LDFLAGS := $(LDFLAGS) $(shell pkg-config --libs lfpRatiometer)
#CXXFLAGS := $(CXXFLAGS) -I/home/amborsa10/.local/include
#LDFLAGS := $(LDFLAGS)  -L/usr/lib/x86_64-linux-gnu -lfftw3 -L/home/amborsa10/.local/lib -llfpRatiometer

#LDFLAGS := $(LDFLAGS) -Wl,-rpath -Wl,/home/amborsa10/.local/lib -L/home/amborsa10/.local/lib -llfpRatiometer

# CXXFLAGS := $(CXXFLAGS) -rpath  /home/amborsa10/.local/lib

# RTXI plug-in stuff
include Makefile.plugin_compile

print-% : ; @echo $* = $($*)

cleanrtxi :
		sudo rm -rf /usr/local/lib/rtxi/$(PLUGIN_NAME).*
