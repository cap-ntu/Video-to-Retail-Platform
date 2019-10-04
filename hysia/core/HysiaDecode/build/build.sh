#! /bin/bash

# compile ColorSpace.o
nvcc -Xcompiler -fPIC -std=c++11 -g -c ../utils/ColorSpace.cu -o ColorSpace.o -I/opt/cuda/include

# compile NvDecoder.o

g++ -std=c++11 -g -fPIC -c ../src/NvDecoder/NvDecoder.cpp -o NvDecoder.o -I../include -I/opt/cuda/include -I..

# compile GpuDec.o

g++ -std=c++11 -g -fPIC -c ../src/GpuDec.cpp -o GpuDec.o -I../include -I/opt/cuda/include -I/usr/include/opencv4 -I..
