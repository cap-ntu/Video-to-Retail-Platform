/*
 * GpuDevice.cpp
 *
 *  Created on: 21 Feb 2019
 *      Author: wangyongjie
 */


#include "GpuDevice.h"


GpuDevice::GpuDevice(){
	//empty constructor
}

int GpuDevice::GetGpuNum(){
	cudaError_t error_id = cudaGetDeviceCount(&this->num);
	if(error_id != cudaSuccess){
		std::cerr<<"cudaGetDeviceCount returns "<<(int)error_id<<" "<<cudaGetErrorString(error_id);
		return -1;
	}
	return this->num;
}


int GpuDevice::GetGpuInfo(){
	this->gpus.resize(this->num);
	if(this->num == 0){
		std::cout<<"There are no available gpu devices that support cuda"<<std::endl;
		return -1;
	}
	for(int dev = 0; dev < this->num; dev++){
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		size_t total_bytes = 0, free_bytes = 0;
		cudaMemGetInfo(&free_bytes, &total_bytes);
		cudaGetDeviceProperties(&deviceProp, dev);
		gpus[dev].name = (std::string)deviceProp.name;
		gpus[dev].totalMemoryMB = (double)deviceProp.totalGlobalMem / 1048675.0f;
		gpus[dev].availMemoryMB = (double)free_bytes / (1024. * 1024);
		gpus[dev].compute = deviceProp.major + 0.1 * deviceProp.minor;
		gpus[dev].frequency = deviceProp.clockRate * 1e-6f;
	}

}

const std::vector<GpuInfo> GpuDevice::ReturnGpuInfo() const{
	return this->gpus;
}
