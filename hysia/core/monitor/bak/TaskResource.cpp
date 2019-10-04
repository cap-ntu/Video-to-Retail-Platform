/*
 * TaskResource.cpp
 *
 *  Created on: 22 Feb 2019
 *      Author: wangyongjie
 */

#include "TaskResource.h"

TaskResource::TaskResource(){
	// empty constructor
}

int TaskResource::GetPidResourceOccupy(int pid){
	char name[64] = {0};
	char buff[1024] = {0};
	sprintf(name, "/proc/%d/stat", pid);

	std::stringstream ss;

	std::ifstream fi(name);
	if(fi.is_open()){
		fi.getline(buff, 1024);
		ss.str(buff);
		ss>>pid_t.pid;
		ss>>pid_t.comm;
		ss>>pid_t.state;
		ss>>pid_t.ppid;
		ss>>pid_t.pgrp;
		long trash;
		for(int i = 0; i < 8; i++){
			ss>>trash;
		}
		ss>>pid_t.utime; // time in user mode
		ss>>pid_t.stime; // time in kernel mode
		ss>>pid_t.cutime; // wait time in user mode
		ss>>pid_t.cstime; // wait time in kernel mode

		for(int i = 0; i < 6; i++){
			ss>>trash;
		}
		ss>>pid_t.rss;
		pid_t.rss = pid_t.rss / (1024. * 1024.); // convert bytes to Metabytes

		return 0;
	}
	return -1;
}

PidResourceOccupy TaskResource::ReturnPidCpuOccupy() const{
	return this->pid_t;
}
