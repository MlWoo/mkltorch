#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.c"
#else



//////////////////////////////////////////////////////////////////////
void TH_MKL_(retain)(THMklTensor *self)
{
	printf("retain --permission-----------refcount = %4d\n", self->freeFlag);
}

void TH_MKL_(free)(THMklTensor *self)
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


static int torch_mkl_(new)(lua_State *L)
{
  printf("enter new tensor\n");
  THMklTensor* pTensor = THAlloc(sizeof(THMklTensor));
  if(pTensor == NULL){
    printf("Cannot allocate memory for Strategy\n");
  }
  //set metetable for THMklTensor
  pTensor->freeFlag = 0;
  pTensor->mklStorage = 0;
  pTensor->mklLayout = 0;

  printf("enter new tensor3\n");
  luaT_pushudata(L, pTensor, torch_mkl_tensor);    
  printf("construct THMklTensor = %p\n", pTensor);
  printf("construct tensor      = %p\n", pTensor->tensor);
	
	return 1;
}

static int torch_mkl_(retain)(lua_State *L)
{
  THMklTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  printf("retain -- recycle heap memory pTensor = %p\n", pTensor);
  printf("retain -- recycle heap memory tensor  = %p\n", pTensor->tensor);
  TH_MKL_(retain)(pTensor);
  return 0;
}


static int torch_mkl_(free)(lua_State *L)
{
  THMklTensor* pTensor = luaT_checkudata(L, 1, torch_mkl_tensor);
  printf("free -- recycle heap memory pTensor = %p\n", pTensor);
  printf("free -- recycle heap memory tensor  = %p\n", pTensor->tensor);
  TH_MKL_(free)(pTensor);
  return 0;
}

static int torch_mkl_(factory)(lua_State *L)
{
  THMklTensor* pTensor = THAlloc(sizeof(THMklTensor));
  luaT_pushudata(L, pTensor, torch_mkl_tensor);
  return 1;
}


static const struct luaL_Reg torch_mkl_(_) [] = {
  {"retain", torch_mkl_(retain)},
  {"new", torch_mkl_(new)},
  {"free", torch_mkl_(free)},
 // {"__len__", torch_mpi_(size)},
 // {"copy", torch_mpi_(copy)},
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
