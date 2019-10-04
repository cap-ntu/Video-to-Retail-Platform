/*
 * GpuDevice.h
 *
 *  Created on: 21 Feb 2019
 *      Author: wangyongjie
 */

#ifndef GPUDEVICE_H_
#define GPUDEVICE_H_

#include "common.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string>
#include <vector>
#include <iostream>

class GpuDevice{
private:
	int num;
	std::vector<GpuInfo> gpus;
public:
	GpuDevice();
	int GetGpuNum();
	int GetGpuInfo();
	const std::vector<GpuInfo> ReturnGpuInfo() const;
};

#endif /* GPUDEVICE_H_ */
