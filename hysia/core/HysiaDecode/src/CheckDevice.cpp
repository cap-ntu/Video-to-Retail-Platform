/*
 * Author: Wang Yongjie
 * Email : yongjie.wang@ntu.edu.sg
 * Description:	check the gpu existing
 */

#include "CheckDevice.h"
#include <memory.h>
#include <iostream>

using namespace std;


int check_device(int flag)
{
	int device_count = 0;
	cudaError_t error_id = cudaGetDeviceCount(&device_count);
	if(error_id != cudaSuccess)
	{
		cout<<"cudaGetDeviceCount returned "<<(int)error_id<<" "<<cudaGetErrorString(error_id)<<endl;
		return -1; // return error
	}
	if(device_count == 0)
	{
		cout<<"there are no avaliable gpu devices that support cuda"<<endl;
		return -1;
	}else{
		cout<<"detected "<<device_count<<" cuda capable device(s)"<<endl;
	}
	
	// if flag set 1, print gpu information
	if(flag){
		int dev, driverVersion = 0, runtimeVersion = 0;
		for(dev = 0; dev < device_count; dev++)
		{
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);
			cout<<"\nDevice"<<dev<<":"<<deviceProp.name<<endl;
			//console log
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			cout<<"\tCuda Driver Version/Runtime Version: "<<driverVersion/1000<<
				"."<<(driverVersion % 100)/10<<"/"<<runtimeVersion/1000<<"."<<(runtimeVersion % 100)/10<<endl;
			cout<<"\tCuda Capability Major/Minor version number: "<<deviceProp.major<<"."<<deviceProp.minor<<endl;
			cout<<"\tTotal amount of global memory: "<<(float)deviceProp.totalGlobalMem / 1048675.0f<<" MBytes"<<endl;
			cout<<"\t"<<deviceProp.multiProcessorCount<<" MultiProcessors, "<<_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)<<" Cuda cores/MP: "<<_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount<<endl;
			cout<<"\tGPU max clock rate: "<<deviceProp.clockRate * 1e-3f<<" MHz "<<deviceProp.clockRate * 1e-6f<<"GHz"<<endl;
			//end print


		}
		
	}
	return 0;

}

