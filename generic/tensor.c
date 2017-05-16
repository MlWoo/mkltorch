#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.c"
#else

////////////////////////////////////////////////////////////////////////
void TH_MKL_(createWorkspace)(THMKLTensor* pTensor)
{

  dnnLayout_t mkldnnLayout = (dnnLayout_t)(pTensor->mkldnnLayout);
  assert(mkldnnLayout != NULL);
  dnnLayout_t usrLayout = NULL;
  dnnPrimitive_t cvtPrmt = NULL;
  real* cvtBuffer = NULL;
  int dimension = 4;

  size_t N = pTensor->tensor->size[0];
  size_t C = pTensor->tensor->size[1];
  size_t H = pTensor->tensor->size[2];
  size_t W = pTensor->tensor->size[3];
  size_t Size[] = {W,H,C,N};
  size_t strides[] = { 1, W, H * W, C * H * W };

  dnnError_t err;
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)((dnnLayout_t*)&usrLayout, dimension, Size, strides) , err );
  //printf("createWorkspace  mkldnnLayout =  %p  usrLayout = %p\n", mkldnnLayout, usrLayout);

  if (! MKLDNN_(dnnLayoutCompare)((dnnLayout_t)mkldnnLayout, (dnnLayout_t)usrLayout)) {
    CHECK_ERR( MKLDNN_(dnnConversionCreate)((dnnPrimitive_t*)&cvtPrmt, (dnnLayout_t)mkldnnLayout, (dnnLayout_t)usrLayout), err );
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)&cvtBuffer, (dnnLayout_t)usrLayout), err );
  }
 
  //printf("workspace primitives = %p    buffer = %p\n", cvtPrmt, cvtBuffer);
  pTensor->workspace[0] = (long)cvtPrmt;
  pTensor->workspace[1] = (long)cvtBuffer;

}

void TH_MKL_(convertToTH)(THTensor * pTensor, THMKLTensor * src)
{
  //pTensor->size = src->size;
  if ((0 == src->workspace[0]) && (0 == src->workspace[1])){
	TH_MKL_(createWorkspace)(src);
  }
  if (0 != src->workspace[0]*src->workspace[1])
  {
    THTensor_(resizeAs)(pTensor, src->tensor); 
    dnnError_t err;
    dnnPrimitive_t cvtPrmt = (dnnPrimitive_t)src->workspace[0];
    //  real * cvtbuffer = (real *)src->workspace[1];
	
    if( cvtPrmt != 0)
    {
      CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvtPrmt, src->tensor->storage->data, pTensor->storage->data), err );
      //src->tensor->storage->data =  cvtbuffer;
    }
    src->tensor->flag = src->flagBackup;
  }
  else
  {
    TH_MKL_(copyBacktoTH)(pTensor, src);
  }
}

//////////////////////////////////////////////////////////////////////
int TH_MKL_(nElement)(const THMKLTensor *self)
{
	//printf("data --permission-----------refcount = %4d\n", self->freeFlag);
  if( self->tensor && self->tensor->storage)
    return THTensor_(nElement)(self->tensor);
  else
    return 0;
}


//////////////////////////////////////////////////////////////////////
real* TH_MKL_(data)(THMKLTensor *self)
{
	//printf("data --permission-----------refcount = %4d\n", self->freeFlag);
  if( self->tensor && self->tensor->storage)
    return (self->tensor->storage->data+self->tensor->storageOffset);
  else
    return NULL;
}

void TH_MKL_(setMKLdata)(THMKLTensor *self, real * mkldata)
{
  //mkldata is allocated by MKLDNN, should not be freed by torch
  //free original torch buffer
  if(self->mkldnnLayout == 0){
    int memSize = self->tensor->storage->size;
    THStorage_(free)(self->tensor->storage);
    self->tensor->storage = THStorage_(newWithData)(mkldata, memSize);
  }
  self->tensor->storage->data = mkldata;
  self->mklStorage = 1;

}

void TH_MKL_(resize4d)(THMKLTensor *self, long size0, long size1, long size2, long size3)
{
	//printf("retain --permission-----------refcount = %4d\n", self->freeFlag);
  long size[4] = {size0, size1, size2, size3};
  THTensor_(resizeNd)(self->tensor, 4, size, NULL);
  self->tensor->flag = MKL_TENSOR_FLAG;
  self->tensor->refcount = 2;
  self->size = self->tensor->size;	
}

void TH_MKL_(resizeAs)(THMKLTensor *self, THMKLTensor *src)
{
	//printf("retain --permission-----------refcount = %4d\n", self->freeFlag);
	
  if(!THTensor_(isSameSizeAs)(self->tensor, src->tensor))
    THTensor_(resizeNd)(self->tensor, src->tensor->nDimension, src->tensor->size, NULL);
    self->size = self->tensor->size;	
}

void TH_MKL_(type)(THMKLTensor *self)
{
  
}

//----------------------------------------------------------------------------
void TH_MKL_(retain)(THMKLTensor *self)
{
	printf("retain --permission-----------refcount = %4d\n", self->freeFlag);
}

void TH_MKL_(free)(THMKLTensor *self)
{
  if(!self )
    return;
  
  if(self->freeFlag & TH_MKL_FREE_PERMISSION) 
  {
	printf("free   --permission-----------refcount = %4d\n", self->freeFlag);
	if(1 == self->mklStorage){
		printf("you should free the tensor memory using mkldnn method");
	}
	if(NULL != self->workspace){
		printf("you should free the workspace memory using mkldnn method");
	}
	THFree(self);
  } 
}


void TH_MKL_(copyFromTH)(THMKLTensor * pTensor, THTensor * src)
{

  //printf("TH_MKL_(copyFromTH) called, pTensor = %p, src = %p\n", pTensor, src);
  pTensor->tensor = src;
  pTensor->size = src->size;
  pTensor->flagBackup = src->flag;
  pTensor->tensor->flag = MKL_TENSOR_FLAG;
  pTensor->workspace[0] = 0;
  pTensor->workspace[1] = 0;
}

void TH_MKL_(TH2MKL)(THMKLTensor * pTensor, THTensor * src)
{
  //printf("TH_MKL_(TH2MKL) called, pTensor = %p, src = %p\n", pTensor, src);
  TH_MKL_(copyFromTH)(pTensor, src);
}

void TH_MKL_(copyBacktoTH)(THTensor * pTensor, THMKLTensor * src)
{
 
  THTensor_(resizeAs)(pTensor, src->tensor);  
  src->tensor->flag = src->flagBackup;
  THTensor_(copy)(pTensor, src->tensor);
}

void TH_MKL_(MKL2TH)(THTensor * pTensor, THMKLTensor * src)
{
  //printf("TH_MKL_(MKL2TH) called, pTensor = %p, src = %p layout = %ld \n", pTensor, src, src->mkldnnLayout);
  
  if(0 == src->mkldnnLayout)
    TH_MKL_(copyBacktoTH)(pTensor, src);
  else
    TH_MKL_(convertToTH)(pTensor, src);
}


//////////////////////////////////////////////////////////////////////

static int torch_mkl_(nElement)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  //printf("nElement -- pTensor = %p\n", pTensor);
  //printf("nElement -- tensor  = %p\n", pTensor->tensor);
  luaT_pushinteger(L, TH_MKL_(nElement)(pTensor));
  return 1;
}

static int torch_mkl_(new)(lua_State *L)
{
  //printf("enter new tensor\n");
  THMKLTensor* pTensor = THAlloc(sizeof(THMKLTensor));
  if(pTensor == NULL){
    printf("Cannot allocate memory for mklTensor\n");
  }
  //set metetable for THMKLTensor
  pTensor->freeFlag = 0;
  pTensor->mklStorage = 0;
  pTensor->mkldnnLayout = 0;
  pTensor->workspace[0] = 0;
  pTensor->workspace[1] = 0;

  luaT_pushudata(L, pTensor, torch_mkl_tensor);    
  //printf("construct THMKLTensor = %p\n", pTensor);
  //printf("construct tensor      = %p\n", pTensor->tensor);
	
  return 1;
}

static int torch_mkl_(retain)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  //printf("retain -- recycle heap memory pTensor = %p\n", pTensor);
  //printf("retain -- recycle heap memory tensor  = %p\n", pTensor->tensor);
  TH_MKL_(retain)(pTensor);
  return 0;
}

static int torch_mkl_(free)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  //printf("free -- recycle heap memory pTensor = %p\n", pTensor);
  //printf("free -- recycle heap memory tensor  = %p\n", pTensor->tensor);
  TH_MKL_(free)(pTensor);
  return 0;
}

static int torch_mkl_(TH2MKL)(lua_State *L)
{
  //printf("copyFromTH 0 \n");
  //THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  //printf("copyFromTH 1 \n");
  void *src;
  void *dst;
  if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLFloatTensor"))){
    THMKLFloatTensorTH2MKL(dst, src);
    
    }
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLDoubleTensor"))){
    THMKLDoubleTensorTH2MKL(dst, src);
    }
  else{
    luaL_typerror(L, 2, "torch.*Tensor");
    }
  lua_settop(L, 1);   //reserve stack bottom, aka, the MKL tensor
  return 1;           // return the MKL tensor right!
}

static int torch_mkl_(MKL2TH)(lua_State *L)
{
  //printf("torch_mkl_(copyBacktoTH) 1 \n");
  //THTensor* pTensor = luaT_checkudata(L, 2, torch_tensor);
  //printf("torch_mkl_(copyBacktoTH) 2 \n");
  void *src;
  void *dst;
  if( (src = luaT_toudata(L, 1, "torch.MKLFloatTensor")) && (dst = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THMKLFloatTensorMKL2TH(dst, src);
  else if( (src = luaT_toudata(L, 1, "torch.MKLDoubleTensor")) && (dst = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THMKLDoubleTensorTH2MKL(dst, src);
  else{
    luaL_typerror(L, 1, "torch.MKL*Tensor");
    }
  //lua_settop(L, 2);
  lua_remove(L, 1);   // remove the stack bottom, aka, the TH Tensor is reserved
  return 1;
}

static int torch_mkl_(copyFromTH)(lua_State *L)
{
  //printf("copyFromTH 0 \n");
  //THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  //printf("copyFromTH 1 \n");
  void *src;
  void *dst;
  if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLFloatTensor"))){
    THMKLFloatTensorcopyFromTH(dst, src);
   
    }
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) && (dst = luaT_toudata(L, 1, "torch.MKLDoubleTensor"))){
    THMKLDoubleTensorcopyFromTH(dst, src);
    }
  else{
    luaL_typerror(L, 2, "torch.*Tensor");
    }
  lua_settop(L, 1);
  return 1;
}

static int torch_mkl_(copyBacktoTH)(lua_State *L)
{
  //printf("torch_mkl_(copyBacktoTH) 1 \n");
  //THTensor* pTensor = luaT_checkudata(L, 2, torch_tensor);
  //printf("torch_mkl_(copyBacktoTH) 2 \n");
  void *src;
  void *dst;
  if( (src = luaT_toudata(L, 1, "torch.MKLFloatTensor")) && (dst = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THMKLFloatTensorcopyBacktoTH(dst, src);
  else if( (src = luaT_toudata(L, 1, "torch.MKLDoubleTensor")) && (dst = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THMKLDoubleTensorcopyBacktoTH(dst, src);
  else{
    luaL_typerror(L, 1, "torch.*Tensor");
    }
  //lua_settop(L, 2);
  lua_remove(L, 1);
  return 1;
}


static int torch_mkl_(factory)(lua_State *L)
{
  THMKLTensor* pTensor = THAlloc(sizeof(THMKLTensor));
  luaT_pushudata(L, pTensor, torch_mkl_tensor);
  return 1;
}


static const struct luaL_Reg torch_mkl_(_) [] = {
  {"retain", torch_mkl_(retain)},
  {"new", torch_mkl_(new)},
  {"free", torch_mkl_(free)},
  {"nElement", torch_mkl_(nElement)},
  //{"type", torch_mkl_(type)},
 // {"__len__", torch_mpi_(size)},
  {"copyFromTH", torch_mkl_(copyFromTH)},
  {"copyFromTH", torch_mkl_(copyFromTH)},
  {"MKL2TH", torch_mkl_(MKL2TH)},
  {"TH2MKL", torch_mkl_(TH2MKL)},
 // {"clone", torch_mpi_(clone)},               //deep copy
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
