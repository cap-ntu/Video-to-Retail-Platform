/*
 * Author:		Wang Yongjie
 * Email:		yongjie.wang@ntu.edu.sg
 * Description:	audio decode with CPU
 */

#include "AudioDec.h"

AudioDecoder::AudioDecoder(){//constuctor function
	//empty
}

AudioDecoder::~AudioDecoder(){
	// empty deconsturctor
	// user must release memory mannually to avoid memory leak
}

int AudioDecoder::ingestVideo(const char* filename){
	av_register_all();
	if (avformat_open_input(&pFmt, filename, piFmt, NULL) < 0)
	{
		std::cerr<<"avformat open failed.";
		return -1;
	}
	else
	{
		std::cout<<"open stream successfully!"<<std::endl;
	}

	if (avformat_find_stream_info(pFmt,NULL) < 0)
	{
		std::cerr<<"could not find stream.";
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
		std::cerr<<"video index = "<<videoindex <<", audio index = "<<audioindex;
		return -1;
	}

	AVStream *pVst,*pAst;
	pVst = pFmt->streams[videoindex];
	pAst = pFmt->streams[audioindex];

	pAudioCodecCtx = pAst->codec;

	pAudioCodec = avcodec_find_decoder(pAudioCodecCtx->codec_id);
	if (!pAudioCodec)
	{
		std::cerr<<"could not find audio codec";
		return -1;
	}
	if (avcodec_open2(pAudioCodecCtx, pAudioCodec,NULL) < 0)
	{
		std::cerr<<"could not open audio codec";
		return -1;
	}
	return 0;

}

int AudioDecoder::decodeClips(){

	AVFrame *pframe = av_frame_alloc();
	AVPacket pkt;
	av_init_packet(&pkt);
	SwrContext *swrCtr = swr_alloc();
	enum AVSampleFormat in_sample_fmt = pAudioCodecCtx->sample_fmt;
	enum AVSampleFormat out_sample_fmt = AV_SAMPLE_FMT_S16;//AV_SAMPLE_FMT_S16 signed 16 bits
	int in_sample_rate = pAudioCodecCtx->sample_rate;
	int out_sample_rate = 44100;
	uint64_t in_ch_layout = pAudioCodecCtx->channel_layout; // input layer mono or stereo
	uint64_t out_ch_layout = AV_CH_LAYOUT_MONO; // stereo or mono
	swr_alloc_set_opts(swrCtr,out_ch_layout,out_sample_fmt,out_sample_rate,in_ch_layout,in_sample_fmt,in_sample_rate,0,NULL);
	swr_init(swrCtr);

	int out_channel_nb = av_get_channel_layout_nb_channels(out_ch_layout);
	int got_frame = 0, ret;
	//*audio_buffer = NULL;
	//*size = 0;
	while(av_read_frame(pFmt, &pkt) >= 0){
		if(pkt.stream_index == audioindex)
		{
			ret = avcodec_decode_audio4(pAudioCodecCtx, pframe, &got_frame, &pkt);
			if(got_frame > 0){
				//std::cout<<"audio decoding"<<std::endl;
				//resample frames
				int buffer_size = 44100; // sample rate * 16 bits if stereo double
				uint8_t *buffer = (uint8_t* )av_malloc(buffer_size);
				swr_convert(swrCtr, &buffer, pframe->nb_samples, (const uint8_t**)pframe->data, pframe->nb_samples);
				// append resampled frames to data
				int out_buffer_size = av_samples_get_buffer_size(NULL, out_channel_nb, pframe->nb_samples, out_sample_fmt, 1);
				
				this->audio_buffer = (uint8_t *)realloc(this->audio_buffer, (this->length + out_buffer_size) * sizeof(uint8_t));
				memcpy(audio_buffer + this->length, buffer, out_buffer_size * sizeof(uint8_t));
				this->length += out_buffer_size;
			}

		}else{
			//std::cout<<"no audio stream"<<std::endl;
		}
		av_frame_unref(pframe);
		av_free_packet(&pkt);
	}

	av_free(pframe);
	//av_free(out_buffer);  free the memory after dequeue
	avcodec_close(pAudioCodecCtx);
	swr_free(&swrCtr);
	avformat_close_input(&pFmt);
	avformat_free_context(pFmt);

	return 0;

}
void AudioDecoder::write_little_endian(std::ofstream &file, unsigned int word, int num_bytes)
{
	uint8_t buf;
	while(num_bytes > 0)
	{
		buf = word & 0xff;
		file<<buf;
		word >>= 8;
		num_bytes--;
	}
}

int AudioDecoder::saveWav(const char *filename)
{
	std::ofstream file;
	file.open(filename, std::ios::binary);
	if(!file.is_open()){
		std::cerr<<"open file error";
		return -1;
	}
	// ChunkID RIFF
	file<<"RIFF"; // 4 bytes
	// ChunkSize 36 + subChunk2Size
	this->write_little_endian(file, this->length + 36, 4); // 8 bytes
	// format + subchunk1ID
	file<<"WAVEfmt "; // 16 bytes
	// subchunk1size 16 for pcm
	this->write_little_endian(file, 16, 4); // 20 bytes
	// audioformat pcm = 1
	this->write_little_endian(file, 1, 2); // 22 bytes
	// NumChannels Mono = 1
	this->write_little_endian(file, 1, 2); // 24 bytes
	// SampleRate default 44100
	this->write_little_endian(file, 44100, 4); // 28 bytes
	// ByteRate sampleRate * NumChannels * BitsPerSample / 8
	this->write_little_endian(file, 44100 * 2, 4); // 32 bytes
	// BlockAlign == NumChannels * BitsPerSample / 8
	this->write_little_endian(file, 2, 2); // 34 bytes
	// bitsPerSample 
	this->write_little_endian(file, 16, 2); // 36 bytes
	// Subchunk2ID
	file<<"data"; // 40 bytes
	// Subchunk2Size
	this->write_little_endian(file, this->length, 4); // 44 bytes

	for(int i = 0; i < this->length; i++){
		file<<*(this->audio_buffer + i);
	}
	file.close();
	return 0;

}

int AudioDecoder::savePcm(const char *filename)
{
	std::ofstream file;
	file.open(filename, std::ios::binary);
	if(!file.is_open())
	{
		std::cerr<<"open file error";
		return -1;
	}
	for(int i = 0; i < this->length; i++)
	{
		file<<*(this->audio_buffer + i);
	}
	file.close();
	return 0;
}

uint8_t* AudioDecoder::getData() const
{
	return this->audio_buffer;
}

int AudioDecoder::getSize() const
{
	return this->length;
}

