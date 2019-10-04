PROJECT_ROOT := $(shell pwd)

GCC ?= g++

CCFLAGS := -std=c++11 -g 

CUDA_PATH ?= /usr/local/cuda-9.0

NVCC := $(CUDA_PATH)/bin/nvcc
 
NVCCFLAGS := $(CCFLAGS)

LDFLAGS := -Wl,-rpath=$(PROJECT_ROOT)/lib/ffmpeg -L$(PROJECT_ROOT)/lib/ffmpeg -lavformat -lavcodec -lavutil -lswscale -lswresample # ffmpeg library
LDFLAGS += -Wl,-rpath=$(PROJECT_ROOT)/lib/opencv -L$(PROJECT_ROOT)/lib/opencv -lavformat  -lopencv_core -lopencv_imgproc -lopencv_highgui -lz -lpthread # opencv library
LDFLAGS += -L$(CUDA_PATH)/lib64/stubs -lcuda
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
LDFLAGS += -shared

ifndef CPU_ONLY
LDFLAGS += -Wl,-rpath=/usr/lib/nvidia-415 -L/usr/lib/nvidia-415 -lnvcuvid # nvcuvid
else
LDFLAGS += -Wl,-rpath=$(PROJECT_ROOT)/lib/nvcuvid -L$(PROJECT_ROOT)/lib/nvcuvid -lnvcuvid
endif

# Project includes
INCLUDES := -I./include 
INCLUDES += -I./include/opencv 
INCLUDES += -I./include/ffmpeg 
INCLUDES += -I./include/NvDecoder 
INCLUDES += -I./include/cuda 
INCLUDES += -I./include/Utils
# CUDA includes
INCLUDES += -I$(CUDA_PATH)/include
# pybind11 includes
INCLUDES += $(shell python -m pybind11 --includes)

SUFFIX ?= $(shell python3-config --extension-suffix)

BUILD_DIR := build

$(BUILD_DIR)/PyDecoder$(SUFFIX): $(BUILD_DIR)/PyDecoder.o $(BUILD_DIR)/CPUDecoder.o $(BUILD_DIR)/GPUDecoder.o $(BUILD_DIR)/NvDecoder.o $(BUILD_DIR)/DeviceChecker.o $(BUILD_DIR)/ColorSpace.o $(BUILD_DIR)/AudioDec.o
	$(GCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/PyDecoder.o: python/PyDecoder.cpp
	$(GCC) $(CCFLAGS) -fPIC $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/CPUDecoder.o: src/CpuDec.cpp
	$(GCC) $(CCFLAGS) -fPIC $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/GPUDecoder.o: src/GpuDec.cpp
	$(GCC) $(CCFLAGS) -fPIC $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/AudioDec.o: src/AudioDec.cpp
	$(GCC) $(CCFLAGS) -fPIC $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/NvDecoder.o: src/NvDecoder/NvDecoder.cpp
	$(GCC) $(CCFLAGS) -fPIC $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/DeviceChecker.o: src/CheckDevice.cpp
	$(GCC) $(CCFLAGS) -fPIC $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/ColorSpace.o: src/Utils/ColorSpace.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC $(INCLUDES) -o $@ -c $<


clean:
	rm -rf $(BUILD_DIR)/PyDecoder$(SUFFIX) $(BUILD_DIR)/PyDecoder.o $(BUILD_DIR)/CPUDecoder.o $(BUILD_DIR)/GPUDecoder.o $(BUILD_DIR)/NvDecoder.o $(BUILD_DIR)/DeviceChecker.o $(BUILD_DIR)/ColorSpace.o



