/*
 * Author : Wang Yongjie
 * Email  : yongjie.wang@ntu.edu.sg
 * Description: Video decoder
 */

#ifndef _MULTIDECODER_HPP
#define _MULTIDECODER_HPP

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "BaseDec.h"
#include "CpuDec.h"
#include "GpuDec.h"
#include "AudioDec.h"
#include "DecodeQueue.hpp"
#include "CheckDevice.h"
#include <sys/stat.h>
#include <cstring>
#include <exception>

class MultiDecoder{
public:
	//default constructor using GPU 
	MultiDecoder(char *filename){
		if(check_device(1) == 0){
			std::cout<<"Using GPU"<<std::endl;
			vdec.reset(new GPUDecoder(0));
		}else{
			vdec.reset(new CPUDecoder());
		}
		adec.reset(new AudioDecoder());
		this->filename = filename;
	}

	MultiDecoder(char *filename, char *device, int device_id = 0){
		if(strcmp(device, "CPU") == 0){
			vdec.reset(new CPUDecoder());
		}else if(strcmp(device, "GPU") == 0){
			try{
				vdec.reset(new GPUDecoder(device_id));
			}catch(std::exception &e){
				std::cout<<"exception in create GPU decoder:\t"<<e.what()<<std::endl;
			}
		}
		this->filename = filename;
		adec.reset(new AudioDecoder());
	}

	~MultiDecoder(){ // release allocated resources
		vdec.release();
		adec.release();
	}

	int GetFrames(){ // video decode
		vdec->IngestVideo(this->filename);
		vdec->DecodeFrames(this->queue);
	}

	int GetAudios(){ // audio decode
		adec->IngestVideo(this->filename);
		adec->DecodeClips(&this->audio_buffer, &this->size);
	}

	int SaveFile(const char *path){
		// path the path saved the decoded videos
		//save the decoded file into disk
		if(path == NULL) return -1;
		char directory[128] = {0};
		sprintf(directory, "%s/%s", path, this->filename);

		if(! mkdir(directory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)){
			std::cout<<"create "<<directory<<" failed"<<std::endl;
		}
		char audio_dir[128], video_dir[128];
		sprintf(audio_dir, "%s/%s", directory, "pcm");
		sprintf(video_dir, "%s/%s", directory, "frame");

		if(! mkdir(audio_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)){// create video directory
			std::cout<<"create "<<audio_dir<<" failed"<<std::endl;
		}

		if(! mkdir(video_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)){ // create audio directory
			std::cout<<"create "<<video_dir<<" failed"<<std::endl;
		}

		char audio_filename[128] = {0};
		char video_filename[128] = {0};
		
		sprintf(audio_filename, "%s/%s.pcm", audio_dir, this->filename);

		// save audio pcm file
		FILE *fp_pcm = fopen(audio_filename, "wb");
		fwrite(this->audio_buffer, 1, this->size, fp_pcm);
		fclose(fp_pcm);

		// save video frames
		int cnt = 0;

		while(queue.get_size() > 0){
			sprintf(video_filename, "%s/%s-%08d.jpg", video_dir, this->filename, cnt);
			printf("%s\n", video_filename);
			cnt = cnt + 1;
			cv::Mat *tmp = queue.pop();
			if(!tmp){continue;}
			cv::imwrite(video_filename, *tmp);
		}
		return 0;
	}

	int SaveMemory(const char *ip = NULL, const int port = 0){
		// save data into redis
		// left empty temperoraily
		if(ip == NULL && port == 0)
		{
			std::cerr<<"errors in:";
			return -1;
		}
	}

	int SetAlignedParam(const float duration = 1.0){// default audio length is 1. second for a frame
		int sample_rate = 44100;
		this->length = (int)(duration * sample_rate / 2 * 2 + 1);
		this->start = -length / 2;
		this->end = length / 2;
		this->step = sample_rate * (1.0 / 30.); // 30 frames per second
		this->ptr = length / 2;
	};

	int GetAlignedData(uint8_t **buffer){
		// Get aligned audio data
		
		*buffer = new uint8_t[this->length];
		memset(*buffer, 0, sizeof(uint8_t) * this->length);
		this->ptr = std::max(this->ptr, 0);
		this->start = std::max(this->start, 0);
		this->end = std::min(this->end, this->size);
		int copy_len = this->end - this->start;
		memcpy(*buffer + this->ptr, this->audio_buffer + this->start, sizeof(uint8_t) * copy_len); // memory copy
		//update 
		this->start = this->start + this->step;
		this->end = this->end + this->step;
		this->ptr = this->ptr - this->step;
		return 0;
	}


private:
	DecodeQueue<cv::Mat* > queue = DecodeQueue<cv::Mat* >(1000000); // video decode queue
	std::unique_ptr<BaseDecoder> vdec = nullptr;  // video codec
	std::unique_ptr<AudioDecoder> adec = nullptr; // audio codec
	uint8_t *audio_buffer = nullptr;  // audio decode buffer

public:
	int size;  // audio buffer length
	char *filename = nullptr; // video files to be decoded
	int start, end; // memory copy start/end point
	int length; // audio length for a single frame
	int ptr;  //  
	int step; // sample numbers of per audio frame
};
#endif
