/*
 * common.h
 *
 *  Created on: 21 Feb 2019
 *      Author: wangyongjie
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <vector>
#include <cstdint>

typedef struct{
	std::string name;
	double frequency; //GHz
	double totalMemoryMB; // total physical memory
	double availMemoryMB; // available GPU memory
	double compute; // GPU compute capacity
}GpuInfo;

typedef struct{
	int cpuCores;
	double totalMemoryMB; // total system memory
	double availMemoryMB; // available memory 
	std::vector<GpuInfo> gpus;
}Config;

typedef struct {
	int pid; // the process ID
	std::string comm; // filename of the executable
	char state; // running state
	int ppid; // parent of this process
	int pgrp; // the process group ID of the process
	// jump the field 6-13
	unsigned long utime; // time in user mode
	unsigned long stime; // time in kernel mode
	unsigned long cutime; // waiting time in user mode
	unsigned long cstime; // waiting time in kernel mode
	// jump the field 18-23
	double rss; // resident memory


}PidResourceOccupy;

#endif /* COMMON_H_ */
