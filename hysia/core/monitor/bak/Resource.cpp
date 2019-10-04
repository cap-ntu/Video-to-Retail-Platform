/*
 * Resource.cpp
 *
 *  Created on: 21 Feb 2019
 *      Author: wangyongjie
 */


#include "Resource.h"


Resource::Resource(){
	// empty constructor
}

int Resource::GetCpuCores(){
	// get cpu cores
	config.cpuCores = sysconf(_SC_NPROCESSORS_ONLN);
	return 0;
}

int Resource::GetTotalMemoryMB(){
	// get system memory MB
	const auto phys_pages = sysconf(_SC_PHYS_PAGES);
	const auto page_size = sysconf(_SC_PAGESIZE);
	config.totalMemoryMB = (phys_pages / 1024.) * (page_size / 1024);
	return 0;
}

int Resource::GetAvailMemoryMB(){
	// Get available memory
	std::ifstream f("/proc/meminfo");
	std::stringstream ss;
	char str[1000] = {0};
	if(f.is_open()){
		f.getline(str, 1000); // read total memory 
		f.getline(str, 1000); // read memory free
		f.getline(str, 1000); // read available memory
		ss.str(str);
		std::string MemoryAvailable;
		long bytes;
		ss>>MemoryAvailable>>bytes;
		std::cout<<MemoryAvailable<<std::endl;
		std::cout<<bytes<<std::endl;
		config.availMemoryMB = bytes / (1024.);  // convert to MB
	}
	f.close();
	return 0;
	
}

int Resource::GetGpus(){
	// get GPU specifications
	GpuDevice tmp;
	tmp.GetGpuNum();
	tmp.GetGpuInfo();
	config.gpus = tmp.ReturnGpuInfo();
}

Config Resource::GetConfig() const{
	// return system configuration
	return this->config;
}
