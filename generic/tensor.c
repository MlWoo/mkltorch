#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.c"
#else

////////////////////////////////////////////////////////////////////////
//----------------------- access function------------------------------
#ifdef TH_REAL_IS_FLOAT
real* TH_MKL_(data)(THMKLTensor *self)
{
  //printf("data --permission-----------refcount = %4d\n", self->freeFlag);
  if( self->tensor && self->tensor->storage)
    return (self->tensor->storage->data+self->tensor->storageOffset);
  else
    return NULL;
}

int TH_MKL_(nElement)(const THMKLTensor *self)
{
  if( self->tensor && self->tensor->storage)
    return THTensor_(nElement)(self->tensor);
  else
    return 0;
}

//------------------  math function-------------------------------------------------- 

static int torch_mkl_(add)(lua_State *L)
{
  THMKLTensor *dst = luaT_checkudata(L, 1, torch_mkl_tensor);
  THMKLTensor *src = luaT_checkudata(L, 2, torch_mkl_tensor);
  THTensor_(cadd)(dst->tensor, dst->tensor, 1, src->tensor);
  return 1;
}

//------------------ memory relative function --------------------------------------

void TH_MKL_(changeWorkspace)(THMKLTensor *self, dnnWorkspace* workspace)
{
  if (!self) 
    return;

  self->workspace = workspace;

}

void TH_MKL_(resize4d)(THMKLTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};
  THTensor_(resizeNd)(self->tensor, 4, size, NULL);
  self->size = self->tensor->size;    
}

void TH_MKL_(resizeAs)(THMKLTensor *self, THMKLTensor *src)
{
  if ((NULL != src->tensor) && (NULL != self->tensor)) {
    if (!THTensor_(isSameSizeAs)(self->tensor, src->tensor)){
      THTensor_(resizeNd)(self->tensor, src->tensor->nDimension, src->tensor->size, NULL);
      self->size = self->tensor->size;
    }
  } else {
    printf("free self = %x, src = %x \n", self, src);
    printf("The input MKLTensor should be initialized!\n");
  }
}


static int torch_mkl_(resizeAs)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  THMKLTensor *src = luaT_checkudata(L, 2, torch_mkl_tensor);
  TH_MKL_(resizeAs)(pTensor, src);
  return 1;
}

/* helpful functions */
static void torch_mkl_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_)
{
  THLongStorage *size = NULL;
  THLongStorage *stride = NULL;

  if( (size = luaT_toudata(L, index, "torch.LongStorage")) )
  {
    if(!lua_isnoneornil(L, index+1))
    {
      if( (stride = luaT_toudata(L, index+1, "torch.LongStorage")) )
        THArgCheck(stride->size == size->size, index+1, "provided stride and size are inconsistent");
      else
        THArgCheck(0, index+1, "torch.LongStorage expected");
    }
    THLongStorage_retain(size);
    if(stride)
      THLongStorage_retain(stride);
  }
  else
  {
    int i;

    size = THLongStorage_newWithSize(8);
    stride = THLongStorage_newWithSize(8);
    THLongStorage_fill(size, -1);
    THLongStorage_fill(stride, -1);

    if(allowStride)
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+2*i))
          break;
        size->data[i] = luaL_checklong(L, index+2*i);

        if(lua_isnone(L, index+2*i+1))
          break;
        stride->data[i] = luaL_checklong(L, index+2*i+1);
      }
    }
    else
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+i))
          break;
        size->data[i] = luaL_checklong(L, index+i);
      }
    }
  }

  *size_ = size;
  *stride_ = stride;
}

static int torch_mkl_(resize)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  THLongStorage *size, *stride;
  torch_mkl_(c_readSizeStride)(L, 2, 0, &size, &stride);
  THTensor_(resize)(pTensor->tensor, size, stride);

  pTensor->size = pTensor->tensor->size;
  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_mkl_(set)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  int index = 2;
  int arg1Type = lua_type(L, index);
  if( arg1Type == LUA_TNONE) {
    if(NULL != pTensor->tensor) {      //detach the previous regular tensor, should free the previous tensor
      if(pTensor->workspace) {
        WORKSPACE_(Free)(pTensor->workspace);
      }
      if(0 == pTensor->dnnMem) {
        THTensor_(free)(pTensor->tensor);
        pTensor->tensor = NULL;
      }

    }
  } else {
    THMKLTensor *src = luaT_checkudata(L, 2, torch_mkl_tensor);
    if (src->dnnMem) {
      printf("Illegal operation, %s dnnMemory could not be modified by external operation\n", __func__);
    } else {
      THTensor_(set)(pTensor->tensor, src->tensor);
      WORKSPACE_(ShallowCopy)(pTensor->workspace, src->workspace);
      pTensor->size = src->size;
    }

  }
  lua_settop(L, 1);
  return 1;
}

static int torch_mkl_(copy)(lua_State *L)
{
  THMKLTensor *dst = luaT_checkudata(L, 1, torch_mkl_tensor);
  THMKLTensor *src = luaT_checkudata(L, 2, torch_mkl_tensor);
  THTensor_(copy)(dst->tensor, src->tensor);
  WORKSPACE_(ShallowCopy)(src->workspace, dst->workspace);
  dst->size = dst->tensor->size;
  dst->dnnMem = 0;
  return 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------- MKL tensor <---> regular tensor  Implementation -------------------------
void TH_MKL_(createWorkspace)(THMKLTensor* pTensor)
{
  dnnLayout_t layout = (dnnLayout_t)(pTensor->workspace->layout);
  assert(layout != NULL);

  dnnLayout_t usrLayout = NULL;
  dnnPrimitive_t cvtPrmt = NULL;

  int dimension = 4;

  size_t N = pTensor->tensor->size[0];
  size_t C = pTensor->tensor->size[1];
  size_t H = pTensor->tensor->size[2];
  size_t W = pTensor->tensor->size[3];
  size_t size[] = {W, H, C, N};
  size_t strides[] = {1, W, H*W, C*H*W};

  dnnError_t err;
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&usrLayout, dimension, size, strides) , err );

  if (!MKLDNN_(dnnLayoutCompare)(layout, usrLayout)) {
    CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvtPrmt, layout, usrLayout), err );
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(usrLayout), err );
 
  pTensor->workspace->cvtPrmt = cvtPrmt;

}

void TH_MKL_(convertToTH)(THTensor** ppTensor, THMKLTensor * src)
{
  if(NULL == src) {
    printf("fatal error, %s src is NULL\n", __func__);
  }
  // need initialized
  if (0 == src->workspace->sync){
    TH_MKL_(createWorkspace)(src);
  }
  if (src->workspace->cvtPrmt > 0) {
    //converting the data from a MKL tensor to a regualr tensor will reuse the data of MKL tensor. There is no need to retain MKL tensor.
    *ppTensor = THTensor_(new)();
    THTensor_(resizeAs)(*ppTensor, src->tensor); 
    dnnError_t err;
    dnnPrimitive_t cvtPrmt = (dnnPrimitive_t)src->workspace->cvtPrmt;
    
    if(cvtPrmt) {
      CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvtPrmt, src->tensor->storage->data, (*ppTensor)->storage->data), err);
    } else  {
       printf("fatal error, %s cvt prmt is NULL\n", __func__);
    }
  } else {
    (*ppTensor) = TH_MKL_(inplace2TH)(src);
  }
}
#else
static int torch_mkl_(zero)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  THTensor_(zero)(pTensor->tensor);
}

static int torch_mkl_(set)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  int index = 2;
  int arg1Type = lua_type(L, index);
  if( arg1Type == LUA_TNONE) {
    if(NULL != pTensor->tensor) {      //detach the previous regular tensor, should free the previous tensor
      if(0 == pTensor->dnnMem) {
        THTensor_(free)(pTensor->tensor);
      }
      pTensor->tensor = NULL;
    }
  } else {
    printf("fatal error, %s MKLLongTensor should only be set to NULL\n", __func__);
  }
  lua_settop(L, 1);
  return 1;
}
#endif

void TH_MKL_(inplace2MKL)(THMKLTensor * pTensor, THTensor * src)
{
  if(NULL != src) {
    if(NULL != pTensor->tensor) {      //detach the previous regular tensor, should free the previous tensor
      THTensor_(free)(pTensor->tensor);
    }
    pTensor->tensor = src;              
    pTensor->size = src->size;
    pTensor->dnnMem = 0;
    THTensor_(retain)(src);
  } else {
    printf("fatal error, %s src is NULL\n", __func__);
  }
}

void TH_MKL_(TH2MKL)(THMKLTensor * pTensor, THTensor * src)
{
  TH_MKL_(inplace2MKL)(pTensor, src);
}

THTensor* TH_MKL_(inplace2TH)(THMKLTensor * src)
{
  
  if(NULL != src) {
    //free the previous regular tensor
    THTensor* pTensor = src->tensor;
    THTensor_(retain)(src->tensor);              // should be care of that situation tensorA -> mkl tensor --> tensorB, the refcount of tensorA buffer will be 3
    return pTensor;
  } else {
    printf("fatal error, %s src is NULL\n", __func__);
    return NULL;
  }
}


THTensor* TH_MKL_(MKL2TH)(THMKLTensor * src)
{
  //printf("%s start\n", __func__);
#ifdef TH_REAL_IS_FLOAT
  THTensor* pTensor = NULL;
  if(0 == src->dnnMem) {
    if(0 != src->workspace) {
      THTensor** ppTensor = (THTensor**)&pTensor;
      TH_MKL_(convertToTH)(ppTensor, src);
    } else {
      pTensor = TH_MKL_(inplace2TH)(src);
    }
  } else if (1 == src->dnnMem){
    THTensor** ppTensor = (THTensor**)&pTensor;
    TH_MKL_(convertToTH)(ppTensor, src);
  } else {
    printf("fatal error, %s src is NULL\n", __func__);
  }
#else
  THTensor* pTensor = NULL;
  if(0 == src->dnnMem) {
    pTensor = TH_MKL_(inplace2TH)(src);
  } else {
    printf("fatal error, %s THMKLLongTensor must get the tensor from regular tensor\n", __func__);
  } 
#endif
  return pTensor;
}


//------------------------- MKL tensor <---> regular tensor -------------------------
static int torch_mkl_(TH2MKL)(lua_State *L)
{
  void *src;
  void *dst;
  if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLFloatTensor"))){
    THMKLTensor* pTensor = (THMKLTensor*)dst;
    if(NULL != pTensor->tensor) {
      printf("fatal error, %s dst should be clean AND %s should not be used by external users.\n", __func__, __func__);
    } else {
      THMKLFloatTensorTH2MKL(dst, src);   
    }
  } else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLDoubleTensor"))){
    THMKLTensor* pTensor = (THMKLTensor*)dst;
    if(NULL != pTensor->tensor) {
      printf("fatal error, %s dst should be clean AND %s should not be used by external users.\n", __func__, __func__);
    } else {
      THMKLDoubleTensorTH2MKL(dst, src);
    }
  } else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLLongTensor"))){
    THMKLTensor* pTensor = (THMKLTensor*)dst;
    if(NULL != pTensor->tensor) {
      printf("fatal error, %s dst should be clean AND %s should not be used by external users.\n", __func__, __func__);
    } else {
      THMKLLongTensorTH2MKL(dst, src);
    }
  } else {
    luaL_typerror(L, 2, "torch.*Tensor");
  }
  //lua_remove(L, 2);   // remove the stack bottom, aka, the TH Tensor is reserved
  lua_settop(L, 1);   //reserve stack bottom, aka, the MKL tensor
  return 1;           // return the MKL tensor right!
}

static int torch_mkl_(MKL2TH)(lua_State *L)
{
  void *src;
  THTensor* pTensor = NULL;
  if( src = luaT_toudata(L, 1, torch_mkl_tensor) ){
    THMKLTensor* pSrc = (THMKLTensor*)src;
    pTensor = TH_MKL_(MKL2TH)(pSrc);

  } else {
    luaL_typerror(L, 1, "torch.MKL*Tensor");
  }
  luaT_pushudata(L, pTensor, torch_Tensor);
  return 1;
}

/*---------------------- dangerous function------------------------------------------------------*/
static int torch_mkl_(directGetTH)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  THTensor *tensor = pTensor->tensor;
  THTensor_(retain)(pTensor->tensor);              // should be care of that situation tensorA -> mkl tensor --> tensorB, the refcount of tensorA buffer will be 3
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

//////////////////////////////////////////////////////////////////////
//----------------------------- fundamental functions Implementation-------------------------------
static void TH_MKL_(rawInit)(THMKLTensor *self)
{
  self->tensor = NULL;
#ifdef TH_REAL_IS_FLOAT
  self->workspace = NULL;
#endif
  self->size = NULL;
  self->refcount = 1;
  self->dnnMem = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void TH_MKL_(retain)(THMKLTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED) {
    THAtomicIncrementRef(&self->refcount);
  }
}

void TH_MKL_(free)(THMKLTensor *self)
{
  if(!self )
    return;

#ifdef TH_REAL_IS_FLOAT  
  if(1 == self->dnnMem) {
    if(0 && (self->flag & TH_TENSOR_REFCOUNTED)) {
      if(THAtomicDecrementRef(&self->refcount)) {
        THFree(self);
      }
    }
  } else if (0 == self->dnnMem) {
    if(self->flag & TH_TENSOR_REFCOUNTED) {
      if(THAtomicDecrementRef(&self->refcount))
      {
        dnnError_t err;
//        printf("free self->tensor = %x, self = %x\n", self->tensor, self);
        if(self->workspace) {
          WORKSPACE_(Free)(self->workspace);
        }
        THTensor_(free)(self->tensor);
        THFree(self);
      }
    }
  } else {
    printf("Cannot free memory for mklTensor\n");
  }
#else
  if(self->flag & TH_TENSOR_REFCOUNTED) {
    if(THAtomicDecrementRef(&self->refcount))
    {
      dnnError_t err;
      //printf("free self->tensor = %x, self = %x\n", self->tensor, self);
      // release mkldnn workspace first
      int float_type = self->tensor->storage->data[0];
      int primitive_count = self->tensor->storage->data[1];
      int buffer_count = self->tensor->storage->data[2];
      int i = 0;
      //primitives release
      for (i = 0; i < primitive_count; ++i) {
        dnnPrimitive_t primitive = (dnnPrimitive_t)(self->tensor->storage->data[i+MKL_INFO_MAINTAIN]);
        if( primitive) {
          if(float_type) {
            CHECK_ERR( dnnDelete_F64((dnnPrimitive_t)primitive) , err );
          } else {
            CHECK_ERR( dnnDelete_F32((dnnPrimitive_t)primitive) , err );
          }
          primitive = NULL;
        }
      }
      //buffer release
      for (i = 0; i < buffer_count; ++i) {
        void* buffer = (void*)(self->tensor->storage->data[i+MKL_INFO_MAINTAIN+primitive_count]);
        if( buffer ) {
          if(float_type) {
            CHECK_ERR( dnnReleaseBuffer_F64(buffer), err );
          } else {
            CHECK_ERR( dnnReleaseBuffer_F32(buffer), err );
          }
          buffer = NULL;
        }
      }
      THTensor_(free)(self->tensor);
      THFree(self);
    }
  }
#endif
}


//----------------------------- fundamental functions -------------------------------
static int torch_mkl_(factory)(lua_State *L)
{
  THMKLTensor* pTensor = THAlloc(sizeof(THMKLTensor));
  if(pTensor == NULL){
    printf("Cannot allocate memory for mklTensor\n");
  }
  //set metetable for THMKLTensor
  TH_MKL_(rawInit)(pTensor); 
  luaT_pushudata(L, pTensor, torch_mkl_tensor);
  return 1;
}

static int torch_mkl_(new)(lua_State *L)
{
  int argc = lua_gettop(L);
  THMKLTensor* pTensor = THAlloc(sizeof(THMKLTensor));
  if(pTensor == NULL){
    printf("Cannot allocate memory for mklTensor\n");
  }
  //printf("mklTensor address %x\n", pTensor);
  //set metetable for THMKLTensor
  TH_MKL_(rawInit)(pTensor); 
  //buffer built-in mmm is of itself 
  if( 1 == argc ) {
    int arg1Type = lua_type(L, 1);
    THStorage *storage;
    THTensor *tensor;
    ptrdiff_t storageOffset;
    THLongStorage *size, *stride;
    THMKLTensor *src = NULL;
    if((arg1Type == LUA_TUSERDATA) && (src = luaT_toudata(L, 1, torch_mkl_tensor)) )
    {
      storage = src->tensor->storage;
      storageOffset = src->tensor->storageOffset;
      size = THTensor_(newSizeOf)(src->tensor);
      stride = THTensor_(newStrideOf)(src->tensor);
      tensor = THTensor_(newWithStorage)(storage, storageOffset, size, stride);
      pTensor->tensor = tensor;
      pTensor->size = tensor->size;
      pTensor->dnnMem = 0; 

      THLongStorage_free(size);
      THLongStorage_free(stride);

    } else {
      printf("Cannot support with the type! NULL or MKLTensor expected!\n");
    }    
  }else if (argc >1){
    printf("Cannot support with mutiple argument! NULL or MKLTensor expected!\n");
  }
  
  luaT_pushudata(L, pTensor, torch_mkl_tensor);  
  return 1;
}


static int torch_mkl_(retain)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  TH_MKL_(retain)(pTensor);
  return 0;
}

static int torch_mkl_(free)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  //printf("free tensor = %p,refount = %ld  --> self = %p, refcount = %ld\n", pTensor->tensor, pTensor->tensor->refcount, pTensor, pTensor->refcount);
  TH_MKL_(free)(pTensor);
  return 0;
}

static int torch_mkl_(write)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  THFile *file = luaT_checkudata(L, 2, "torch.File");
  
  lua_getfield(L, 2, "writeObject"); /* the method */
  lua_pushvalue(L, 2); /* the file */
  /* the storage */
  if(pTensor->tensor)
  {
#ifdef TH_REAL_IS_FLOAT
    if(pTensor->workspace && pTensor->workspace->layout) {
      THTensor* tensor = TH_MKL_(MKL2TH)(pTensor);
      THFree(pTensor->tensor);
      pTensor->tensor = tensor;
    }
#endif
    THFile_writeIntScalar(file, pTensor->tensor->nDimension);
    THFile_writeLongRaw(file, pTensor->tensor->size, pTensor->tensor->nDimension);
    THFile_writeLongRaw(file, pTensor->tensor->stride, pTensor->tensor->nDimension);
    THFile_writeLongScalar(file, pTensor->tensor->storageOffset+1); /* to respect Lua convention */
    THTensor_(retain)(pTensor->tensor);
    if(pTensor->tensor->storage)
    {
      THStorage_(retain)(pTensor->tensor->storage);
      luaT_pushudata(L, pTensor->tensor->storage, torch_Storage);
    }
    else
      lua_pushnil(L); 
  } else {
    lua_pushnil(L);
  }
  lua_call(L, 2, 0); /* call the method */

  return 0;
}

static int torch_mkl_(read)(lua_State *L)
{
  THMKLTensor *pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  THFile *file = luaT_checkudata(L, 2, "torch.File");
  pTensor->tensor = THTensor_(new)();
  pTensor->size = pTensor->tensor->size;
  pTensor->dnnMem = 0;
#ifdef TH_REAL_IS_FLOAT
  pTensor->workspace = NULL;
#endif
  int nDimension = THFile_readIntScalar(file);
  if (nDimension > 0) {
    pTensor->tensor->nDimension = nDimension;
    pTensor->tensor->size = THAlloc(sizeof(long)*pTensor->tensor->nDimension);
    pTensor->tensor->stride = THAlloc(sizeof(long)*pTensor->tensor->nDimension);
    THFile_readLongRaw(file,pTensor-> tensor->size, pTensor->tensor->nDimension);
    THFile_readLongRaw(file,pTensor-> tensor->stride, pTensor->tensor->nDimension);
    pTensor->tensor->storageOffset = THFile_readLongScalar(file);
    pTensor->tensor->storageOffset--;  /* to respect Lua convention */

    lua_getfield(L, 2, "readObject"); /* the method */
    lua_pushvalue(L, 2); /* the file */
    lua_call(L, 1, 1); /* call the method */

    pTensor->tensor->storage = luaT_toudata(L, -1, torch_Storage);
    if(pTensor->tensor->storage) {
      THStorage_(retain)(pTensor->tensor->storage);
    } 
  }

  return 0;
}

static const struct luaL_Reg torch_mkl_(_) [] = {
  {"retain",        torch_mkl_(retain)},
  {"new",           torch_mkl_(new)},
  {"free",          torch_mkl_(free)},
  {"MKL2TH",        torch_mkl_(MKL2TH)},
  {"TH2MKL",        torch_mkl_(TH2MKL)},
  {"directTH",      torch_mkl_(directGetTH)},
  {"read",          torch_mkl_(read)},
  {"write",         torch_mkl_(write)},
  {"set",           torch_mkl_(set)},
#ifdef TH_REAL_IS_FLOAT
  {"resizeAs",      torch_mkl_(resizeAs)},
  {"resize",        torch_mkl_(resize)},
  {"copy",          torch_mkl_(copy)},
  {"add",           torch_mkl_(add)},
#else
  {"zero",          torch_mkl_(zero)},
#endif
  {NULL, NULL}
};

void torch_mkl_(init)(lua_State *L)
{
  luaT_newmetatable(L, torch_mkl_tensor, NULL,
                    torch_mkl_(new), torch_mkl_(free), torch_mkl_(factory));
  luaT_setfuncs(L, torch_mkl_(_), 0);
  lua_pop(L, 1);
}
#endif
