
mkltorch={}
mkltorch.C=require 'libmkltorch'

local ok, ffi = pcall(require, 'ffi')

if ok then

   local Real2real = {
      Float='float',
      Double='double'
   }
   local  cdefs_prev = [[
typedef struct _uniPrimitive_s* dnnPrimitive_t;
typedef struct _dnnLayout_s* dnnLayout_t;

typedef struct THMKLLongTensor
{
  THLongTensor* tensor;
  long *size;
  int  refcount;      // 0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
  char flag;
  char dnnMem;
} THMKLLongTensor;
]]
   ffi.cdef(cdefs_prev)
   local Tensor_type = 'torch.MKLLongTensor'
   local Tensor = torch.getmetatable(Tensor_type)
   local Tensor_tt = ffi.typeof('THMKLLongTensor**')
   
   rawset(Tensor, "cdata", function(self)
                             if not self then return nil; end
                             return Tensor_tt(self)[0]
                           end)

   -- Tensor
   for Real, real in pairs(Real2real) do

      local cdefs = [[
typedef struct RealTensorWorkspace
{
  dnnPrimitive_t cvtPrmt;     // -1: not initialized; 0: no need; other: need convert
  dnnLayout_t layout;
  int  refcount;              // 0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
  char flag;
  char sync;                  // memory management machenism 0: unknown 1:extern regular tensor; 2:mkldnn 3:self
} DNNRealWorkspace;

typedef struct THMKLRealTensor
{
  THRealTensor* tensor;
  DNNRealWorkspace* workspace;
  long *size;
  int  refcount;      // 0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
  char flag;
  char dnnMem;
} THMKLRealTensor;
]]

      cdefs = cdefs:gsub('Real', Real):gsub('real', real)
      ffi.cdef(cdefs)
      local Tensor_type = string.format('torch.MKL%sTensor', Real)
      local Tensor = torch.getmetatable(Tensor_type)
      local Tensor_tt = ffi.typeof('THMKL' .. Real .. 'Tensor**')
	  
      rawset(Tensor, "cdata", function(self)
                              if not self then return nil; end
                              return Tensor_tt(self)[0]
                              end)
  end
end
--mklLongTensor = torch.MKLLongTensor.new()
--print('method cdata')
--print(mklLongTensor:cdata())



