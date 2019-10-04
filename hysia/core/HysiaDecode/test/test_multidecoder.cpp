#include "MultiDecoder.hpp"

int main(int argc, char **argv){
	char *filename = argv[1];
	char *config = argv[2];
	char *save_dir = argv[3];
	MultiDecoder test(filename, config);
	test.GetFrames();
	test.GetAudios();
	test.SaveFile(save_dir);

	return 0;
}

