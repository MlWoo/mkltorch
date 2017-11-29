#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/dnnWorkspace.c"
#else

void WORKSPACE_(RawInit)(dnnWorkspace* self)
{
  self->cvtPrmt  = NULL;
  self->layout   = NULL;
  self->refcount = 1;
  self->flag     = TH_TENSOR_REFCOUNTED;
  self->sync     = 1;
}

dnnWorkspace* WORKSPACE_(New)(dnnLayout_t layout)
{
  dnnWorkspace* pWorkspace = THAlloc(sizeof(dnnWorkspace));
  if(NULL == pWorkspace) {
    printf("Cannot allocate memory for mklTensor\n");
  }
  WORKSPACE_(RawInit)(pWorkspace);
  pWorkspace->layout = layout;
  pWorkspace->sync = 0;
  return pWorkspace;
}

void WORKSPACE_(Retain)(dnnWorkspace* self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED) {
    THAtomicIncrementRef(&self->refcount);
  }
}

void WORKSPACE_(Free)(dnnWorkspace* self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED) {
    if(THAtomicDecrementRef(&self->refcount)) {
      dnnError_t err;
      if(self->layout) {
        CHECK_ERR( MKLDNN_(dnnLayoutDelete)(self->layout), err );
        if(self->cvtPrmt) {
          CHECK_ERR( MKLDNN_(dnnDelete)(self->cvtPrmt) , err );
        }
      }
      THFree(self);
      self = NULL;
    }
  }
}

void WORKSPACE_(ShallowCopy)(dnnWorkspace* src, dnnWorkspace* dst)
{
  if(src != dst) {
    WORKSPACE_(Free)(dst);
    dst = src;
    WORKSPACE_(Retain)(src);
  }
}

void WORKSPACE_(Change)(dnnWorkspace* self, dnnWorkspace* workspace)
{
  WORKSPACE_(ShallowCopy)(workspace, self);
}

#endif
