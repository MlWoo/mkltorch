#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.h"
#else



typedef struct THMKLTensor
{
	THTensor *tensor;   
	char freeFlag;
    int mklStorage;  //0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
    long mklLayout;
} THMKLTensor;


static int TH_MKL_(copyFromTH)(THMKLTensor * pTensor, THTensor * src);
static int TH_MKL_(copyBacktoTH)(THTensor * pTensor, THMKLTensor * src);



#endif
