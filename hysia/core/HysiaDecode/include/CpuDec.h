/*
 * Author:		Wang Yongjie
 * Email:		yongjie.wang@ntu.edu.sg
 * Description:	video decoded with CPU
 */

#ifndef _CPUDEC_H
#define _CPUDEC_H

#include "BaseDec.h"
#include <opencv2/opencv.hpp>
#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"

#ifdef __cplusplus
}
#endif



class CPUDecoder: public BaseDecoder
{
private:
	AVCodec *pVideoCodec = NULL;
	AVCodec *pAudioCodec = NULL;
	AVCodecContext *pVideoCodecCtx = NULL;
	AVCodecContext *pAudioCodecCtx = NULL;
	AVIOContext *pb = NULL;
	AVInputFormat *piFmt = NULL;
	AVFormatContext *pFmt = NULL;
	AVStream *pVst,*pAst;
	//AVFrame *pframe = av_frame_alloc();
	//AVPacket pkt;
	int videoindex = -1;
	int audioindex = -1;

public:
	CPUDecoder();
	~CPUDecoder();
	int IngestVideo(const char*) override;
	int DecodeFrames(DecodeQueue<cv::Mat*> &queue) override;
	int GetWidth() override;
	int GetHeight() override;
};

#endif
