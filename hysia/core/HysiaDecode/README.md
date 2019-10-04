# HysiaDecode

This repository aims to provide a highly efficient and user-friendly video preprocessing library. 

The whole pipeline of the HysiaDecode can be illustrated in the following diagram. The Decode module firstly detect the GPU exist. If GPU exists, the module will select the GPU decoding function priorly. Otherwise, the module will select the CPU decoding function. It depends on user‘s hardware configuration.So    it support various types of users. After decoding, we save the data into a queue, and the applications can fetch the frames from the queue. To facilitate Python users,  we  warp the C++ function to Python using Pybind11.  You can follow the example you provided in test directory.  Enjoy your experience and don't forget to report the issue you encountered! 



![](https://github.com/iversonicter/HysiaDecode/blob/develop/images/pipeline.png)



| pATH    | DESCRIPTION                                     |
| ------- | ----------------------------------------------- |
| include | header files(include the dependencies)          |
| lib     | required libraries(E.g., OpenCV, FFmpeg, Cuvid) |
| src     | implementation of decoding function             |
| python  | warp the C++ function to Python                 |
| test    | test files                                      |
| build   | save the generated files(.o and .so) after make |
| utils   | some pre-implemented functions in NvDecode      |

# Required Packages and Platform

Requirement | Minimal Version
---|---
Platform | Ubutun 16/14.04, Manjaro 17.10 etc.
Nvidia GPU | Titan X, M6000 etc. |
Driver | nvidia-396 or later|
CUDA   | CUDA 8.0 or later  |


# Installation

Before using our repository, you should install Nvidia-driver first. For ubuntu user, you can install it under this instruction belowing:
```
sudo apt-get purge nvidia*
sudo add-apt-repository ppa:graphics-drivers 
sudo apt-get update
sudo apt-get install nvidia-396 
lsmod | grep nvidia 
```

Pybind11 is also necessary, because we use Pybind wrap the C++ function to Python

```
pip install pybind11
```
Then: 

```
git clone https://github.com/iversonicter/HysiaDecode.git

cd HysiaDecode

```
If the required hardware or driver is not available:
```
make CPU_ONLY=TRUE
```
Otherwise:
```
make
```

After compliation and link, this reporistory will generate a shared file like this 'PyDecoder.cpython-3xm-x86_64-linux-gnu.so' in build directory. You can import it in Python like this, don't forget add it into the system path unless you will meet this error " no moudle named PyDecoder ":

```
import PyDecoder
```

# Prefermance 

we test the decoding speed on our server(Titan X, Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz) using several video clips from The Big Bang Series.  We also compare the decoding speed with released algorithms in OpenCV.  We achieved the considerable performance. Need to notice, our library provide Python interface which can be easily used for users who don't familiar with C/C++.  This is only a alpha version of decoding. In the next step, we will continue optimizing the library to achieve a large throughput.  You are also welcomed to join our team to contribute your code. We also found that GPU initialization takes up much time. If you want to decode a short video clip, we recommend using CPU. Otherwise, GPU is preferred.

Hardware | Speed
---|---
GPU decoding(Titan X) | ~800 frames/s
CPU (Intel E5-2630) | ~200 frames/s 
Python  cv2.VideoCapture | 375 frames /s 
OpenCV VideoCaptur* | 399 frames /s 
OpenCV  cv::cudacodec::VideoReader* | 865 frames/s 


#  Issues

Some issues we met in develop and test. If you meet the same error, you can refer the  solution we provide here.

- issue 1

```
GPU in use: Quadro M6000
Traceback (most recent call last):
  File "test_gpu_decoder.py", line 19, in <module>
    dec.ingestVideo(video_path)
RuntimeError: NvDecoder : cuvidCtxLockCreate(&m_ctxLock, cuContext) returned error -1282253296 at src/NvDecoder/NvDecoder.cpp:524
```
 Solution : This error comes from cuvid link conflicts. You should link 

- Issue 2

```
the decoding processing is blocked and does not start the decoded process
```

Solution: When it happens, maybe your GPU does support NVDecode. You should try to consult the Nvidia documents to check whether you graphical card are suitable. In this repository, we detached the decoded function from Video_Codec_SDK_8.2.16. 



# Reference

We referred the released repositories listed below:

[Nvidia video_codec_SDK_8.2.16](https://developer.nvidia.com/nvidia-video-codec-sdk) 

[scanner video decoding](https://github.com/scanner-research/scanner/tree/master/scanner/video/nvidia)

[Nvidia NVVL](https://github.com/NVIDIA/nvvl)

[FFmpeg official tutorials](https://ffmpeg.org/doxygen/trunk/encoding-example_8c-source.html#l00325)
