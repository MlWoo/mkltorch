
-- additional methods for Tensor


--[[
  If you want convertions to more types of a MKLTensor, just add the 
  corresponding tensor to the table.
]]--
local TensorTypes = {
   float  = 'torch.FloatTensor',
   double = 'torch.DoubleTensor',
}

local TH2MKL = {
   float = 'mklFloat',
   double = 'mklDouble',
}

local MKLTensorTypes = {
   mklFloat  = 'torch.MKLFloatTensor',
   mklDouble = 'torch.MKLDoubleTensor',
}

local MKL2TH = {
   mklFloat  = 'float' ,
   mklDouble = 'double' ,
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
               local new = torch.getmetatable(type).new()
               self:MKL2TH(new)
               return new
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

--[[
-- tostring() functions for Tensor and Storage
local function Storage__printformat(self)
   if self:size() == 0 then
     return "", nil, 0
   end
   local intMode = true
   local type = torch.typename(self)
--   if type == 'torch.FloatStorage' or type == 'torch.DoubleStorage' then
      for i=1,self:size() do
         if self[i] ~= math.ceil(self[i]) then
            intMode = false
            break
         end
      end
--   end
   local tensor = torch.DoubleTensor(torch.DoubleStorage(self:size()):copy(self), 1, self:size()):abs()
   local expMin = tensor:min()
   if expMin ~= 0 then
      expMin = math.floor(log10(expMin)) + 1
   else
      expMin = 1
   end
   local expMax = tensor:max()
   if expMax ~= 0 then
      expMax = math.floor(log10(expMax)) + 1
   else
      expMax = 1
   end

   local format
   local scale
   local sz
   if intMode then
      if expMax > 9 then
         format = "%11.4e"
         sz = 11
      else
         format = "%SZd"
         sz = expMax + 1
      end
   else
      if expMax-expMin > 4 then
         format = "%SZ.4e"
         sz = 11
         if math.abs(expMax) > 99 or math.abs(expMin) > 99 then
            sz = sz + 1
         end
      else
         if expMax > 5 or expMax < 0 then
            format = "%SZ.4f"
            sz = 7
            scale = math.pow(10, expMax-1)
         else
            format = "%SZ.4f"
            if expMax == 0 then
               sz = 7
            else
               sz = expMax+6
            end
         end
      end
   end
   format = string.gsub(format, 'SZ', sz)
   if scale == 1 then
      scale = nil
   end
   return format, scale, sz
end




local function Tensor__printMatrix(self, indent)
   local format,scale,sz = Storage__printformat(self:storage())
   if format:sub(2,4) == 'nan' then format = '%f' end
--   print('format = ' .. format)
   scale = scale or 1
   indent = indent or ''
   local strt = {indent}
   local nColumnPerLine = math.floor((80-#indent)/(sz+1))
--   print('sz = ' .. sz .. ' and nColumnPerLine = ' .. nColumnPerLine)
   local firstColumn = 1
   local lastColumn = -1
   while firstColumn <= self:size(2) do
      if firstColumn + nColumnPerLine - 1 <= self:size(2) then
         lastColumn = firstColumn + nColumnPerLine - 1
      else
         lastColumn = self:size(2)
      end
      if nColumnPerLine < self:size(2) then
         if firstColumn ~= 1 then
            table.insert(strt, '\n')
         end
         table.insert(strt, 'Columns ' .. firstColumn .. ' to ' .. lastColumn .. '\n' .. indent)
      end
      if scale ~= 1 then
         table.insert(strt, string.format('%g', scale) .. ' *\n ' .. indent)
      end
      for l=1,self:size(1) do
         local row = self:select(1, l)
         for c=firstColumn,lastColumn do
            table.insert(strt, string.format(format, row[c]/scale))
            if c == lastColumn then
               table.insert(strt, '\n')
               if l~=self:size(1) then
                  if scale ~= 1 then
                     table.insert(strt, indent .. ' ')
                  else
                     table.insert(strt, indent)
                  end
               end
            else
               table.insert(strt, ' ')
            end
         end
      end
      firstColumn = lastColumn + 1
   end
   local str = table.concat(strt)
   return str
end

local function Tensor__printTensor(self)
   local counter = torch.LongStorage(self:nDimension()-2)
   local strt = {''}
   local finished
   counter:fill(1)
   counter[1] = 0
   while true do
      for i=1,self:nDimension()-2 do
         counter[i] = counter[i] + 1
         if counter[i] > self:size(i) then
            if i == self:nDimension()-2 then
               finished = true
               break
            end
            counter[i] = 1
         else
            break
         end
      end
      if finished then
         break
      end
--      print(counter)
      if #strt > 1 then
         table.insert(strt, '\n')
      end
      table.insert(strt, '(')
      local tensor = self
      for i=1,self:nDimension()-2 do
         tensor = tensor:select(1, counter[i])
         table.insert(strt, counter[i] .. ',')
      end
      table.insert(strt, '.,.) = \n')
      table.insert(strt, Tensor__printMatrix(tensor, ' '))
   end
   return table.concat(strt)
end
]]--


