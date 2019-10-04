/*
 * Resource.h
 *
 *  Created on: 21 Feb 2019
 *      Author: wangyongjie
 */

#ifndef RESOURCE_H_
#define RESOURCE_H_

#include "common.h"
#include "GpuDevice.h"
#include "unistd.h"
#include <string>
#include <pcre.h>
#include <iostream>
#include <fstream>
#include <sstream>

class Resource{
private:
	Config config;

public:
	Resource();
	int GetTotalMemoryMB();
	int GetAvailMemoryMB();
	int GetCpuCores();
	int GetGpus();
	Config GetConfig() const;
};

#endif /* RESOURCE_H_ */
