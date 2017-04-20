#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.h"
#else



typedef struct THMklTensor
{
	THTensor *tensor;   
	char freeFlag;
    int mklStorage;  //0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
    long mklLayout;
} THMklTensor;



#endif
