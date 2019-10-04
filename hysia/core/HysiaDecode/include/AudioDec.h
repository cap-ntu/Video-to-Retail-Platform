/*
 * Author:		Wang Yongjie
 * Email:		yongjie.wang@ntu.edu.sg
 * Description:	audio decode with CPU
 */

#ifndef _AUDIODEC_H 
#define _AUDIODEC_H
#include <cstdint>
#include <fstream>
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"

#ifdef __cplusplus
}
#endif



class AudioDecoder
{
private:
	AVCodec *pVideoCodec = NULL;
	AVCodec *pAudioCodec = NULL;
	AVCodecContext *pAudioCodecCtx = NULL;
	AVIOContext *pb = NULL;
	AVInputFormat *piFmt = NULL;
	AVFormatContext *pFmt = NULL;
	AVStream *pVst,*pAst;
	int videoindex = -1;
	int audioindex = -1;
	uint8_t* audio_buffer = NULL; // audio buffer
	int length = 0; // bytes of audio data

public:
	AudioDecoder();
	~AudioDecoder();
	int ingestVideo(const char*);
	int decodeClips();
	int saveWav(const char * filename);
	int savePcm(const char *filename);
	void write_little_endian(std::ofstream &file, unsigned int word, int num_bytes);
	int getSize() const;
	uint8_t* getData() const;
};

#endif
