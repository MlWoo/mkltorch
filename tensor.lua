
-- additional methods for Tensor
local MKLTensor = {}


local MKLTensorTypes = {
   float  = 'torch.MKLFloatTensor',
   double = 'torch.MKLDoubleTensor',
   long   = 'torch.MKLLongTensor'
}



for _, MKLTensorType in pairs(MKLTensorTypes) do
   local metatable = torch.getmetatable(MKLTensorType)
   for funcname, func in pairs(MKLTensor) do
      rawset(metatable, funcname, func)
   end
end

