
require 'libmkltorch'

local ok, ffi = pcall(require, 'ffi')

if ok then

   local Real2real = {
      Long='long',
      Float='float',
      Double='double'
   }
   
   -- Tensor
   for Real, real in pairs(Real2real) do

      local cdefs = [[
typedef struct THMKLRealTensor
{
	THRealTensor *tensor;   
	char freeFlag;
    int mklStorage;  //0:storage buffer allocated by THTensor, 1:storage buffer allocated by mklnn
    long mklLayout;
} THMKLRealTensor;
]]

      cdefs = cdefs:gsub('Real', Real):gsub('real', real)
      ffi.cdef(cdefs)
      local Tensor_type = string.format('torch.MKL%sTensor', Real)
      local Tensor = torch.getmetatable('torch.MKLLongTensor')
	  local Tensor_tt = ffi.typeof('THMKL' .. Real .. 'Tensor**')
	  
	  
	  rawset(Tensor,
	 "cdata",
	 function(self)
		if not self then return nil; end
		return Tensor_tt(self)[0]
	 end)
	 end
end	 


--mklLongTensor = torch.MKLLongTensor.new()
--print('method cdata')
--print(mklLongTensor:cdata())



