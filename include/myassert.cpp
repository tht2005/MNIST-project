#include "myassert.h"

void myassert(bool a, const char* msg) {
	if(!a) {
		std::cout << std::endl << msg << std::endl << std::endl;
		exit(1);
	}
}

