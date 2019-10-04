//============================================================================
// Name        : Minitor.cpp
// Author      : Wang Yongjie
// Version     : Version 1.0
// Copyright   : Your copyright notice
// Description : System Monitor in C++, Ansi-style
//============================================================================ 
#include <iostream>
#include "Resource.h"
#include "TaskResource.h"
#include "common.h"

int main()
{
	Resource res;
	res.GetTotalMemoryMB();
	res.GetAvailMemoryMB();
	res.GetCpuCores();
	res.GetGpus();

	Config config;
	config = res.GetConfig();
	std::cout<<"CPU cores\t"<<config.cpuCores<<std::endl;
	std::cout<<"Memory capacity\t"<<config.totalMemoryMB<<std::endl;
	std::cout<<"Memory available\t"<<config.availMemoryMB<<std::endl;
	std::cout<<"GPU numbers\t"<<config.gpus.size()<<std::endl;

	std::cout<<"------------------------------------------------"<<std::endl;
	for(int i = 0; i < config.gpus.size(); i++)
	{
		std::cout<<"GPU ID\t"<<i<<std::endl;
		std::cout<<"GPU name\t"<<config.gpus[i].name<<std::endl;
		std::cout<<"GPU frequency\t"<<config.gpus[i].frequency<<std::endl;
		std::cout<<"GPU total memory\t"<<config.gpus[i].totalMemoryMB<<std::endl;
		std::cout<<"GPU avail memory\t"<<config.gpus[i].availMemoryMB<<std::endl;
		std::cout<<"Gpu compute\t"<<config.gpus[i].compute<<std::endl;
	}
	std::cout<<"------------------------------------------------"<<std::endl;

	pid_t pid = getpid();
	TaskResource task;
	task.GetPidResourceOccupy(pid);

	std::cout<<"------------------------------------------------"<<std::endl;


	PidResourceOccupy pid_t = task.ReturnPidCpuOccupy();
	std::cout<<"return pid\t"<<pid<<std::endl;

	std::cout<<"pid\t"<<pid_t.pid<<std::endl;
	std::cout<<"name\t"<<pid_t.comm<<std::endl;
	std::cout<<"state\t"<<pid_t.state<<std::endl;
	std::cout<<"parent pid\t"<<pid_t.ppid<<std::endl;
	std::cout<<"group pid\t"<<pid_t.pgrp<<std::endl;
	std::cout<<"utime\t"<<pid_t.utime<<std::endl;
	std::cout<<"stime\t"<<pid_t.stime<<std::endl;
	std::cout<<"cutime\t"<<pid_t.cutime<<std::endl;
	std::cout<<"cstime\t"<<pid_t.cstime<<std::endl;
	std::cout<<"rss memory\t"<<pid_t.rss<<" M"<<std::endl;

	std::cout<<"------------------------------------------------"<<std::endl;
	return 0;
}
