/*
 * Filename : CpuDec
 * Author	: Wang Yongjie
 * Description: video decode with CPU
 */ 
#include "CpuDec.h"
#include <iostream>
#include <cstdint>

using namespace std;

CPUDecoder::CPUDecoder(){
}

CPUDecoder::~CPUDecoder(){
}


int CPUDecoder::IngestVideo(const char* filename){

	av_register_all(); //initialize decoding environments
	if (avformat_open_input(&pFmt, filename, piFmt, NULL) < 0)
	{
		cerr<<"avformat open failed";
		return -1;
	}
	else
	{
		cout<<"Open stream success"<<endl;
	}

	if (avformat_find_stream_info(pFmt,NULL) < 0)
	{
		cerr<<"could not find stream";
		return -1;
	}
	av_dump_format(pFmt, 0, "", 0);

	for (int i = 0; i < pFmt->nb_streams; i++)
	{
		if ( (pFmt->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) &&
				(videoindex < 0) )
		{
			videoindex = i;
		}
		if ( (pFmt->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) &&
				(audioindex < 0) )
		{
			audioindex = i;
		}
	}

	if (videoindex < 0 || audioindex < 0)
	{
		cerr<<"video index = "<<videoindex<<"audioindex = "<<audioindex;
		return -1;
	}

	pVst = pFmt->streams[videoindex];
	pAst = pFmt->streams[audioindex];

	pVideoCodecCtx = pVst->codec;
	pAudioCodecCtx = pAst->codec;

	pVideoCodec = avcodec_find_decoder(pVideoCodecCtx->codec_id);
	if (!pVideoCodec)
	{
		cerr<<"could not find video decoder";
		return -1;
	}
	if (avcodec_open2(pVideoCodecCtx, pVideoCodec,NULL) < 0)
	{
		cerr<<"could not open video codec";
		return -1;
	}

	pAudioCodec = avcodec_find_decoder(pAudioCodecCtx->codec_id);
	if (!pAudioCodec)
	{
		cerr<<"could not find audio decoder";
		return -1;
	}
	if (avcodec_open2(pAudioCodecCtx, pAudioCodec, NULL) < 0)
	{
		cerr<<"could not find audio codec";
		return -1;
	}
	return 0;

}

int CPUDecoder::DecodeFrames(DecodeQueue<cv::Mat*> &queue){

	int got_picture;
	AVFrame *pframe = av_frame_alloc();
	AVFrame *rgbframe = av_frame_alloc();
	AVPacket pkt;
	av_init_packet(&pkt);
	unsigned char *out_buffer = (unsigned char*)av_malloc(av_image_get_buffer_size(AV_PIX_FMT_BGR24, pVideoCodecCtx->width, pVideoCodecCtx->height, 1));
	av_image_fill_arrays(rgbframe->data, rgbframe->linesize, out_buffer, AV_PIX_FMT_BGR24, pVideoCodecCtx->width, pVideoCodecCtx->height, 1);

	struct SwsContext *img_convert_ctx = NULL;
	img_convert_ctx = sws_getContext(pVideoCodecCtx->width, pVideoCodecCtx->height, pVideoCodecCtx->pix_fmt, pVideoCodecCtx->width, pVideoCodecCtx->height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
	while(1)
	{
		if (av_read_frame(pFmt, &pkt) >= 0)
		{
			if (pkt.stream_index == videoindex)
			{
				avcodec_decode_video2(pVideoCodecCtx, pframe, &got_picture, &pkt);
				if (got_picture)
				{
					sws_scale(img_convert_ctx, (const unsigned char* const*)pframe->data, pframe->linesize, 0, pVideoCodecCtx->height, rgbframe->data, rgbframe->linesize);
					int buffer_size = pVideoCodecCtx->height * pVideoCodecCtx->width * 3;
					uint8_t *tmp = (uint8_t *)av_malloc(buffer_size);
					memcpy(tmp, rgbframe->data[0], sizeof(uint8_t) * buffer_size);
					cv::Mat *img = new cv::Mat(pVideoCodecCtx->height, pVideoCodecCtx->width, CV_8UC3, tmp); //rgbframe->data[0]);
					queue.push(img);
				}
			}
		}else{
            queue.push(nullptr);
			break;
		}
		av_free_packet(&pkt);
	}
	av_free(pframe);
	av_free(rgbframe);
	sws_freeContext(img_convert_ctx);
	return 0;
}

int CPUDecoder::GetWidth(){
	return pVideoCodecCtx->width;
}


int CPUDecoder::GetHeight(){
	return pVideoCodecCtx->height;
}

