/*
 * Author: Wang Yongjie
 * Email : yongjie.wang@ntu.edu.sg
 * Description: test file of decoder
 * 
 */

#include "../include/Decoder.hpp"
#include "../include/opencv/opencv2/opencv.hpp"
#include "../include/opencv/opencv2/core.hpp"
#include "sys/time.h"


int main(int argc, char **argv){
	char* filename = argv[1];
	char* config = argv[2];
	struct timeval start, end;
	Decoder<cv::Mat*> test(config);
	test.ingestVideo(filename);
	test.decode();
	int frame_num = 0;
	gettimeofday(&start, NULL);
	while(1)
	{
		
		cv::Mat* tmp = test.fetchFrame();
		if(!tmp){
			std::cout<<"empty image"<<std::endl;
			break;
		}
		frame_num++;
		std::cout<<tmp->rows<<"\t"<<tmp->cols<<std::endl;
		delete(tmp);
	}
	gettimeofday(&end, NULL);
	float duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.;
	cout<<"time elapsed\t"<<duration<<"\tframe_Num\t"<<frame_num<<endl;
	cout<<1 / (duration / frame_num)<<" frames per second"<<endl;

};
