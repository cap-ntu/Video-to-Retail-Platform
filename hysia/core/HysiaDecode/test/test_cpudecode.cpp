#include "CpuDec.h"
#include <pthread.h>
#include <iostream>

using namespace std;


void *get_frame(DecodeQueue<cv::Mat> &myqueue)
{
	cout<<"DecodeFrames "<<myqueue.size()<<endl;
	while(1)
	{
		cout<<myqueue.size()<<"\t"<<myqueue._head<<"\t"<<myqueue._end<<endl;
		Mat tmp = myqueue.pop();
		cout<<tmp.rows<<"\t"<<tmp.cols<<endl;
	}
	return NULL;
}

int main(int argc, char **argv){

	pthread_t my_thread[10];
	void *res;
	DecodeQueue<cv::Mat> myqueue = DecodeQueue<Mat>(100000);
	for(int i = 0; i < 1; i++){
		int ret = pthread_create(&my_thread[i], NULL, get_frame, &myqueue);
		if(ret != 0){
			cout<<"create thread "<<i<<" error"<<endl;
		}else
		{
			cout<<"create thread "<<i<<" sucessfully"<<endl;
		}
	}

	char* filename = argv[1];
	CPUDecoder test = CPUDecoder();
	test.IngestVideo(filename);
	test.DecodeFrames(myqueue);

	for(int i = 0; i < 1; i++){
		pthread_join(my_thread[i], &res);
	}
	return 0;
}


