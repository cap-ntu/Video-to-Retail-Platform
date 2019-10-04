/*
 * Author: Wang Yongjie
 * Email : yongjie.wang@ntu.edu.sg
 * Description:	check the gpu existing
 */


#ifndef _CHECK_DEVICE_
#define _CHECK_DEVICE_

#include <cuda_runtime.h>
#include <helper_cuda.h>

// if gpu exists, return 0, else return -1
// if flag sets 1, print the gpu information
int check_device(int flag = 0);


#endif
