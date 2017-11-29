require 'mkltorch'
local a = nil
print("===========")
for i = 1, 10 do
  a = torch.FloatTensor(10, 20)
  print(a:cdata())
  local b = a:mkl()
  local c = b:th()
  print(c:cdata())
end
collectgarbage()
os.execute("sleep " .. 1)
print("end==============>")
print(collectgarbage("collect"))
--d = a
--print(d)

