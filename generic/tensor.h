#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.h"
#else



typedef struct THMKLTensor
{
	THTensor *tensor;   
	char freeFlag;
    int mklStorage;  //0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
    long long mkldnnLayout;
    char flagBackup;
    long * size;
} THMKLTensor;

#define MKL_TENSOR_FLAG  0x00010000

static int TH_MKL_(copyFromTH)(THMKLTensor * pTensor, THTensor * src);
static int TH_MKL_(copyBacktoTH)(THTensor * pTensor, THMKLTensor * src);
void TH_MKL_(resize4d)(THMKLTensor *self, long size0, long size1, long size2, long size3);
real* TH_MKL_(data)(THMKLTensor *self);

#endif
