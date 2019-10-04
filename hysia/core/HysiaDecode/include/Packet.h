/*
 * Author:		Wang Yongjie
 * Email:		yongjie.wang@ntu.edu.sg
 * Description:	the packet for video frame and audio clip
 */

#ifndef _PACKET_H
#define _PACKET_H

#include <cstdint>
#include "opencv2/opencv.hpp"

struct packet{
	cv::Mat* frame = nullptr; // save the frame
	uint8_t* clip = nullptr; // save the audio clip
};

#endif
