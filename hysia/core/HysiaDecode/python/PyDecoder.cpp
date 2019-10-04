#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "opencv2/opencv.hpp"
#include "../include/Decoder.hpp"
#include <iostream>
#include "../include/AudioDec.h"

namespace py = pybind11;


// Wrap the class method 
// Convert opencv mat to numpy array
py::array_t<uint8_t> fetchFrame(Decoder<cv::Mat*>* dec) {
	cv::Mat* frame = dec->fetchFrame();
	if(frame) {
		// Transfer memory management to Python
		py::capsule cleanUp(frame, [](void* d){
			delete reinterpret_cast<cv::Mat*>(d);
		});
		// Construct Numpy array from data pointer without copying
		// With the help of capsule
		return py::array_t<uint8_t>(
			{dec->getHeight(), dec->getWidth(), 3},
			{dec->getWidth() * 3, 3, 1},
			frame->data,
			cleanUp
		);
	}
	else {
		return py::array_t<uint8_t>();
	}
}

py::array_t<int16_t> getData(AudioDecoder adec){
	int size = adec.getSize();
	int16_t *tmp = (int16_t *)adec.getData();
	if(size > 0){
		py::capsule cleanUp(tmp, [](void *d){
				free(d);
				});
		return py::array_t<int16_t>(
				{size / 2},
				{sizeof(int16_t)},
				tmp,
				cleanUp
				);
	}
	return py::array_t<int16_t>();
}

PYBIND11_MODULE(PyDecoder, m) {
	py::class_<Decoder<cv::Mat*>>(m, "Decoder")
		.def(py::init<>())
		.def(py::init<const char*>()) // Constructor overloading
		.def("ingestVideo", &Decoder<cv::Mat*>::ingestVideo)
		.def("decode", &Decoder<cv::Mat*>::decode)
		.def("getWidth", &Decoder<cv::Mat*>::getWidth)
		.def("getHeight", &Decoder<cv::Mat*>::getHeight)
		.def("fetchFrame", &fetchFrame);

	py::class_<AudioDecoder>(m, "AudioDecoder")
		.def(py::init<>())
		.def("ingestVideo", &AudioDecoder::ingestVideo)
		.def("decodeClips", &AudioDecoder::decodeClips)
		.def("saveWav", &AudioDecoder::saveWav)
		.def("savePcm", &AudioDecoder::savePcm)
		.def("getSize", &AudioDecoder::getSize)
		.def("getData", &getData);
}

