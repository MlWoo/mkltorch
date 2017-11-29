#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/dnnWorkspace.h"
#else

typedef struct tensorDNNWorkspace
{
  dnnPrimitive_t cvtPrmt;     // -1: not initialized; 0: no need; other: need convert
  dnnLayout_t layout;
  int  refcount;    // 0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
  char flag;
  char sync;  // memory management machenism 0: unknown 1:extern regular tensor; 2:mkldnn 3:self
} dnnWorkspace;

void WORKSPACE_(RawInit)(dnnWorkspace* self);
dnnWorkspace* WORKSPACE_(New)(dnnLayout_t layout);
void WORKSPACE_(Retain)(dnnWorkspace* self);
void WORKSPACE_(Free)(dnnWorkspace* self);
void WORKSPACE_(ShallowCopy)(dnnWorkspace* src, dnnWorkspace* dst);
void WORKSPACE_(Change)(dnnWorkspace* self, dnnWorkspace* workspace);
//void dnnWorkspaceChangeLayout(dnnWorkspace* self, dnnLayout_t layout);


#endif
