#include "luaT.h"


extern void torch_MKLFloatTensor_init(lua_State* L);
extern void torch_MKLDoubleTensor_init(lua_State* L);


LUA_EXTERNC DLL_EXPORT int luaopen_libmkltorch(lua_State *L);

int luaopen_libmkltorch(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "mkltorch");
  torch_MKLFloatTensor_init(L);
  torch_MKLDoubleTensor_init(L);

  return 1;
}
