/*
 * TaskResource.h
 *
 *  Created on: 22 Feb 2019
 *      Author: wangyongjie
 */

#ifndef TASKRESOURCE_H_
#define TASKRESOURCE_H_

#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include "common.h"
#include <sstream>
#include <string.h>

#define VMRSS_LINE 20
#define VMSIZE_LINE 16
#define PROCESS_ITEM 14

class TaskResource{

private:
	PidResourceOccupy pid_t;

public:

	TaskResource();

	int GetPidResourceOccupy(int pid);

	PidResourceOccupy ReturnPidCpuOccupy() const;

};

#endif /* TASKRESOURCE_H_ */
