#include "AudioDec.h"
#include <iostream>


int main(int argc, char **argv){
	char* filename = argv[1];
	char* savename = argv[2];
	uint8_t* audio_buffer = NULL;
	int size;
	FILE *fp_pcm = fopen(savename, "wb");
	AudioDecoder test = AudioDecoder();
	test.IngestVideo(filename);
	test.DecodeClips(&audio_buffer, &size);
	/*
	float sum = 0.0;
	std::cout<<size<<std::endl;
	for(int i = 0; i < size / 2; i++){
		short tmp = audio_buffer[i];
		tmp = tmp<<8;
		tmp += tmp + audio_buffer[i * 2 + 1];
		sum += tmp;
	}
	*/
	std::cout<<size<<std::endl;
	int sum = 0.0;
	short *ptr = (short *)audio_buffer;
	for(int i = 0; i < size / 2; i++)
	{
		sum += *(ptr + i);
	}

	fwrite(audio_buffer, 1, size, fp_pcm);
	fclose(fp_pcm);

	std::cout<<sum<<std::endl;

	return 0;

}

