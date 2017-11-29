
-- additional methods for Tensor


--[[
  If you want convertions to more types of a MKLTensor, just add the 
  corresponding tensor to the table.
]]--
local TensorTypes = {
   float  = 'torch.FloatTensor',
   double = 'torch.DoubleTensor',
   long   = 'torch.LongTensor',
}

local TH2MKL = {
   float = 'mklFloat',
   double = 'mklDouble',
   long = 'mklLong',
}

local MKLTensorTypes = {
   mklFloat  = 'torch.MKLFloatTensor',
   mklDouble = 'torch.MKLDoubleTensor',
   mklLong   = 'torch.MKLLongTensor',
}

local MKL2TH = {
   mklFloat  = 'float' ,
   mklDouble = 'double' ,
   mklLong = 'long' ,
}

local function Tensor__TH2MKL__converter(type)
   return function(self)
            local current = torch.typename(self)
            if not type then return current end
            if type ~= current then
              local new = torch.getmetatable(type).new()
              new:TH2MKL(self)
              return new
            else
              return self
            end
          end
end


local function Tensor__MKL2TH__converter(type)
   return function(self)
             local current = torch.typename(self)
             if not type then return current end
             if type ~= current then
               return self:MKL2TH()
             else
               return self
             end
          end
end

local function getType(type)
   return function(self)
             return type
          end
end

local function doNothing()
   return function(self)
             return self
          end
end

local function Tensor__Patch_Op(op, ...)
  return function(self, ...)
    local tensor = nil
    for _,v in pairs({...}) do 
      local typeString = torch.type(v)
      if (string.find(typeString, 'MKL')) then
        tensor = self:mkl()
      end
    end
	local metatable = getmetatable(tensor)
	local func = metatable[op]	
	return func(tensor, ...)
  end
end

local function MKLTensor__tostring__(...)
  return function(self, ...) 
        local tensor = self:directTH() 
        if (nil == tensor) then
           print("this mkltensor is a built-in tensor, no print method")
           return self['__tostring']
        end
        local metatable = getmetatable(tensor)
        local func = metatable['__tostring']
        local tensorType = torch.type(self)
        local stringTable = func(tensor, ...)
        local outputString = string.gsub(stringTable, "torch.*Tensor", tensorType)
        return outputString
  end
end

for type, SrcType in pairs(TensorTypes) do
   local metatable = torch.getmetatable(SrcType)
   local MKLType = TH2MKL[type]
   rawset(metatable, 'mkl', Tensor__TH2MKL__converter(MKLTensorTypes[MKLType]))
   rawset(metatable, MKLType, Tensor__TH2MKL__converter(MKLTensorTypes[MKLType]))
   --metatable.resizeAs = Tensor__Patch_Op('resizeAs', ...)
end

for type, SrcType in pairs(MKLTensorTypes) do
   local metatable = torch.getmetatable(SrcType)
   local THType = MKL2TH[type]
   rawset(metatable, 'th', Tensor__MKL2TH__converter(TensorTypes[THType]))
   rawset(metatable, THType, Tensor__MKL2TH__converter(TensorTypes[THType]))
   rawset(metatable, 'type', getType(SrcType))
   rawset(metatable, 'mkl', doNothing())
   metatable.__tostring = MKLTensor__tostring__(...)
end
-------------------------------------------------------------------------------


--[[
  If you want operations to query the infos of a MKLTensor, just add the ops to
  the table.
]]--

local query_operations = {
	'dim' ,
	'size' ,
	'nElement' ,
}

local function Tensor__MKL__query_Op(op, ...)
  return function(self, ...)
	local tensor = self:directTH()
	local metatable = getmetatable(tensor)
	local func = metatable[op]	
	return func(tensor, ...)
  end
end

for type, SrcType in pairs(MKLTensorTypes) do
  local metatable = torch.getmetatable(SrcType)
  for _, query_op in pairs(query_operations) do
    rawset(metatable, query_op, Tensor__MKL__query_Op(query_op, ...))
  end
end
-------------------------------------------------------------------------------
