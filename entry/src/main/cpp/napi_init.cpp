#include "napi/native_api.h"
#include <aki/jsbind.h>
#include <string> 

// 1、用户自定义业务 
std::string SayHello(std::string msg){  return msg + " too.";}  
 
// 2、导出业务接口 
// Step 1 注册 AKI 插件 
JSBIND_ADDON(entry) // 注册 AKI 插件名: 即为编译*.so名称，规则与Node-API一致 
 
// Step 2 注册 FFI 特性 
JSBIND_GLOBAL() 
{ 
  JSBIND_FUNCTION(SayHello); 
}