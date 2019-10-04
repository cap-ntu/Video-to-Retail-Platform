cc=g++
flag=-std=c++11 -fpermissive
flag+=-Wdeprecated-declarations
LDFLAGS=-Wl,-rpath=/home/wyj/code/HysiaDecode/lib/ffmpeg -L/home/wyj/code/HysiaDecode/lib/ffmpeg -lavformat -lavcodec -lavutil -lswscale  -Wl,-rpath=/home/wyj/code/HysiaDecode/lib/opencv -L/home/wyj/code/HysiaDecode/lib/opencv  -lopencv_core -lopencv_imgproc -lopencv_highgui -lz -lpthread

CPUDecoder:CPUDecoder.o test_cpudecode.o
	g++ $(flag) -Wdeprecated-declarations -g -o CPUDecoder CPUDecoder.o test_cpudecode.o $(LDFLAGS)

test_cpudecode.o:test_cpudecode.cpp
	g++ $(flag) -Wdeprecated-declarations -g -c test_cpudecode.cpp -o test_cpudecode.o -I ../include/ -I ../include/ffmpeg

CPUDecoder.o:../src/CpuDec.cpp
	g++ $(flag) -Wdeprecated-declarations -g -c ../src/CpuDec.cpp -o CPUDecoder.o -I ../include/ffmpeg -I ../include/ 

clean:
	-rm CPUDecoder.o CPUDecoder test_cpudecode.o
