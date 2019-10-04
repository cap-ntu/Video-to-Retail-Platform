/*
 * Author : Wang Yongjie
 * Email  : yongjie.wang@ntu.edu.sg
 * Description: Video decoder
 */
 

#ifndef _DECODER_H
#define _DECODER_H

#include "BaseDec.h"
#include "CpuDec.h"
#include "GpuDec.h"
#include "DecodeQueue.hpp"
#include "CheckDevice.h"
#include <pthread.h>
#include <thread>
#include "string.h"
#include "string"

using namespace std;

template<class T>
class Decoder{
public:
    // Default constructor that performs GPU detection
	Decoder(){
        if(check_device(1) == 0){ // GPUs exist
            cout << "Using GPU" << endl;
            // Default to use GPU:0
            dec.reset(new GPUDecoder(0));
        }else{ // no GPUs
            cout << "Using CPU" << endl;
            dec.reset(new CPUDecoder());
        }
	};
    
	Decoder(const char* device){
        if(strcmp(device, "CPU") == 0) {
            dec.reset(new CPUDecoder());
        } else {
            if(strncmp(device, "GPU", 3) != 0) {
                throw invalid_argument("Constructor argument should be GPU:<device_id>\n");
            }
            int device_id = 0;
            try {
                device_id = stoi(device + 4);
            } catch (invalid_argument const &e) {
                throw invalid_argument("The divice id provided is not an integer\n");
            }
            dec.reset(new GPUDecoder(device_id));
        }
	}
    
	~Decoder(){
        if(decodeThread.joinable()) {
            decodeThread.join();
        }
	}
    
    int ingestVideo(const char* filename) {
        dec->IngestVideo(filename);
        return 0;
    }
    
    int getWidth() {
        return dec->GetWidth();
    }
    
    int getHeight() {
        return dec->GetHeight();
    }
    
	int decode(){
        if (decodeThread.joinable()) {
            decodeThread.join();
        }
        thread newThread(&Decoder::threadWrapper, this);
        decodeThread.swap(newThread);
        return 0;
	}

	T fetchFrame(){
		return this->queue.pop();
	}
	int get_size(){
		return  queue.get_size();
	}
	
private:
	DecodeQueue<T> queue = DecodeQueue<T>(100000); // video decode queue
    unique_ptr<BaseDecoder> dec = nullptr;
    thread decodeThread;
private:
    int threadWrapper() {
        dec->DecodeFrames(queue);
    }
};

#endif
