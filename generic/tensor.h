#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.h"
#else

#define MKL_INFO_MAINTAIN 3

typedef struct THMKLTensor
{
  THTensor *tensor; 
#ifdef TH_REAL_IS_FLOAT  
  dnnWorkspace* workspace;
#endif
  long *size;  
  int  refcount;      // 0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn   
  char flag;
  char dnnMem;
} THMKLTensor;

static void TH_MKL_(rawInit)(THMKLTensor *self);
void TH_MKL_(inplace2MKL)(THMKLTensor * pTensor, THTensor * src);
THTensor* TH_MKL_(inplace2TH)(THMKLTensor * src);
void TH_MKL_(TH2MKL)(THMKLTensor* pTensor, THTensor * src);
THTensor* TH_MKL_(MKL2TH)(THMKLTensor * src);
void TH_MKL_(resize4d)(THMKLTensor *self, long size0, long size1, long size2, long size3);
void TH_MKL_(resizeAs)(THMKLTensor *self, THMKLTensor *src);
real* TH_MKL_(data)(THMKLTensor *self);
int TH_MKL_(nElement)(const THMKLTensor *self);
void TH_MKL_(free)(THMKLTensor *self);

#ifdef TH_REAL_IS_FLOAT
void TH_MKL_(changeWorkspace)(THMKLTensor *self, dnnWorkspace* workspace);
#endif

#endif
