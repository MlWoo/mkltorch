
-- additional methods for Tensor
local MKLTensor = {}


local MklTensorTypes = {
   float  = 'torch.MKLFloatTensor',
   double = 'torch.MKLDoubleTensor',
   long   = 'torch.MKLLongTensor'
}



for _, MKLTensorType in pairs(MklTensorTypes) do
   local metatable = torch.getmetatable(MKLTensorType)
   for funcname, func in pairs(MKLTensor) do
      rawset(metatable, funcname, func)
   end
end

