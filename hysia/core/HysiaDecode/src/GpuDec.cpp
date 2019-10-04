//
// Created by WENBO JIANG on 14/12/18.
//
#include <queue>
#include <iostream>
#include "GpuDec.h"
#include "cuda.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "NvDecoder/NvDecoder.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegDemuxer.h"


GPUDecoder::GPUDecoder(int device_id) {
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (device_id < 0 || device_id > nGpu - 1) {
        throw std::invalid_argument("Invalid devide id");
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, device_id));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
}


GPUDecoder::~GPUDecoder() {
    if(pTmpImage) {
        cuMemFree(pTmpImage);
    }
}


int GPUDecoder::GetHeight() {
    return (demuxer)? demuxer->GetHeight() : 0;
}

int GPUDecoder::GetWidth() {
    return (demuxer)? demuxer->GetWidth() : 0;
}

int GPUDecoder::IngestVideo(const char* videoFile) {
    demuxer.reset(new FFmpegDemuxer(videoFile));
    dec.reset(new NvDecoder(cuContext, demuxer->GetWidth(), demuxer->GetHeight(), true, FFmpeg2NvCodecId(demuxer->GetVideoCodec())));
    frameSize = 3 * demuxer->GetWidth() * demuxer->GetHeight();
    // Delete last image
    pImage.reset(new uint8_t[frameSize]);
    // Free last cuda buffer
    if(pTmpImage) {
        cuMemFree(pTmpImage);
    }
    // Create new frame container
    cuMemAlloc(&pTmpImage, frameSize);
	return 0;
}


void GPUDecoder::GetImage(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight)
{
    CUDA_MEMCPY2D m = { 0 };
    m.WidthInBytes = nWidth;
    m.Height = nHeight;
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)dpSrc;
    m.srcPitch = m.WidthInBytes;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
    m.dstPitch = m.WidthInBytes;
    cuMemcpy2D(&m);
}


cv::Mat* GPUDecoder::ToMat(uint8_t* rawData) {
    cv::Mat channelB(dec->GetHeight(), dec->GetWidth(), CV_8UC1, rawData);
    cv::Mat channelG(dec->GetHeight(), dec->GetWidth(), CV_8UC1, rawData + dec->GetHeight() * dec->GetWidth());
    cv::Mat channelR(dec->GetHeight(), dec->GetWidth(), CV_8UC1, rawData + 2 * dec->GetHeight() * dec->GetWidth());

    std::vector<cv::Mat> channels{channelB, channelG, channelR};
    auto rgbImage = new cv::Mat();
    cv::merge(channels, *rgbImage);
    return rgbImage;
}


int GPUDecoder::DecodeFrames(DecodeQueue<cv::Mat*> &queue) {
    while(nVideoBytes) {
        uint8_t** ppFrame;
        demuxer->Demux(&pVideo, &nVideoBytes);
        int nFrameReturned = 0;
        dec->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

        for(int i = 0; i < nFrameReturned; i++) {
            Nv12ToBgrPlanar((uint8_t*)ppFrame[i], dec->GetWidth(), (uint8_t*)pTmpImage, dec->GetWidth(), dec->GetWidth(), dec->GetHeight());
            GetImage(pTmpImage, reinterpret_cast<uint8_t*>(pImage.get()), dec->GetWidth(), 3 * dec->GetHeight());
            cv::Mat* rgbImage = ToMat(pImage.get());
            queue.push(rgbImage);
        }
    }
    // End of video
    nVideoBytes = 1;
    queue.push(nullptr);
    return 0;
}


