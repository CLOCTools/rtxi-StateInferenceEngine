PLUGIN_NAME = rtxi-lfpInferenceEngine

HEADERS = rtxi-lfpInferenceEngine.h

SOURCES = rtxi-lfpInferenceEngine.cpp \
			moc_rtxi-lfpInferenceEngine.cpp \
			include/lfpInferenceEngine.h \
			src/lfpInferenceEngine.cpp \
			include/lfpRatiometer.h \
			src/lfpRatiometer.cpp 
			

LIBS =

OS := $(shell uname)
CXX = g++

CXX_FLAGS=-std=c++11 -Iinclude
PYTHON_CXX_FLAGS:=$(shell python3.10-config --cflags --embed)
LD_FLAGS=-lfftw3 -lm
PYTHON_LD_FLAGS=$(shell python3.10-config --embed --ldflags)

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
