#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.c"
#else



//////////////////////////////////////////////////////////////////////
real* TH_MKL_(data)(THMKLTensor *self)
{
	//printf("data --permission-----------refcount = %4d\n", self->freeFlag);
  if( self->tensor && self->tensor->storage)
    return (self->tensor->storage->data+self->tensor->storageOffset);
  else
    return NULL;
}

real* TH_MKL_(setMKLdata)(THMKLTensor *self, real * mkldata)
{
  //mkldata is allocated by MKLDNN, should not be freed by torch
  //free original torch buffer
  if(self->mkldnnLayout == 0){
    int memSize = self->tensor->storage->size;
    THStorage_(free)(self->tensor->storage);
    self->tensor->storage = THStorage_(newWithData)(mkldata,memSize);
  }
  self->tensor->storage->data = mkldata;
  self->mklStorage = 1;

}

void TH_MKL_(resize4d)(THMKLTensor *self, long size0, long size1, long size2, long size3)
{
	//printf("retain --permission-----------refcount = %4d\n", self->freeFlag);
  long size[4] = {size0, size1, size2, size3};
  THTensor_(resizeNd)(self->tensor, 4, size, NULL);

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
		printf("you should free the memory using mkldnn method");
	}
	THFree(self);
  } 
}

//////////////////////////////////////////////////////////////////////



static int torch_mkl_(new)(lua_State *L)
{
  printf("enter new tensor\n");
  THMKLTensor* pTensor = THAlloc(sizeof(THMKLTensor));
  if(pTensor == NULL){
    printf("Cannot allocate memory for Strategy\n");
  }
  //set metetable for THMKLTensor
  pTensor->freeFlag = 0;
  pTensor->mklStorage = 0;
  pTensor->mkldnnLayout = 0;

  printf("enter new tensor3\n");
  luaT_pushudata(L, pTensor, torch_mkl_tensor);    
  printf("construct THMKLTensor = %p\n", pTensor);
  printf("construct tensor      = %p\n", pTensor->tensor);
	
	return 1;
}

static int torch_mkl_(retain)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  printf("retain -- recycle heap memory pTensor = %p\n", pTensor);
  printf("retain -- recycle heap memory tensor  = %p\n", pTensor->tensor);
  TH_MKL_(retain)(pTensor);
  return 0;
}


static int torch_mkl_(free)(lua_State *L)
{
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  printf("free -- recycle heap memory pTensor = %p\n", pTensor);
  printf("free -- recycle heap memory tensor  = %p\n", pTensor->tensor);
  TH_MKL_(free)(pTensor);
  return 0;
}


static int TH_MKL_(copyFromTH)(THMKLTensor * pTensor, THTensor * src)
{
  printf("TH_MKL_(copyFromTH) called, pTensor = 0x%x, src = 0x%x\n", pTensor, src);
  pTensor->tensor = src;
  pTensor->size = src->size;
  pTensor->flagBackup = src->flag;
  pTensor->tensor->flag = MKL_TENSOR_FLAG;
}

static int TH_MKL_(copyBacktoTH)(THTensor * pTensor, THMKLTensor * src)
{
  printf("TH_MKL_(copyBacktoTH) called, pTensor = 0x%x, src = 0x%x\n", pTensor, src);
  //pTensor->size = src->size;
  THTensor_(resizeAs)(pTensor, src->tensor);
  src->tensor->flag = src->flagBackup;
  THTensor_(copy)(pTensor, src->tensor);
}

static int torch_mkl_(copyFromTH)(lua_State *L)
{
  printf("copyFromTH 0 \n");
  THMKLTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  printf("copyFromTH 1 \n");
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.LongTensor")) ){
    THMKLLongTensorcopyFromTH(pTensor, src);
    }
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) ){
    printf("copyFromTH 2 \n");
    THMKLFloatTensorcopyFromTH(pTensor, src);
    printf("copyFromTH 3 \n");
    }
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) ){
    THMKLDoubleTensorcopyFromTH(pTensor, src);
    }
    else{
    luaL_typerror(L, 2, "torch.*Tensor");
    }
  lua_settop(L, 1);
  return 1;
}

static int torch_mkl_(copyBacktoTH)(lua_State *L)
{
  printf("torch_mkl_(copyBacktoTH) 1 \n");
  THTensor* pTensor = luaT_checkudata(L, 2, torch_tensor);
  printf("torch_mkl_(copyBacktoTH) 2 \n");
  void *src;
  if( (src = luaT_toudata(L, 1, "torch.MKLLongTensor")) )
    THMKLLongTensorcopyBacktoTH(pTensor, src);
  else if( (src = luaT_toudata(L, 1, "torch.MKLFloatTensor")) )
    THMKLFloatTensorcopyBacktoTH(pTensor, src);
  else if( (src = luaT_toudata(L, 1, "torch.MKLDoubleTensor")) )
    THMKLDoubleTensorcopyBacktoTH(pTensor, src);
  else{
    luaL_typerror(L, 1, "torch.*Tensor");
    }
  lua_settop(L, 2);
  return 2;
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
  //{"type", torch_mkl_(type)},
 // {"__len__", torch_mpi_(size)},
  {"copyFromTH", torch_mkl_(copyFromTH)},
  {"copyBacktoTH", torch_mkl_(copyBacktoTH)},
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
