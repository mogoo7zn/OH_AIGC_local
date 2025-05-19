#include "napi/native_api.h"
#include <aki/jsbind.h>
#include <string> 

// 用户自定义业务函数 
std::string SayHello(std::string msg) {  
  return msg + " too.";
}  

// 添加一个新的Add函数示例
int Add(int a, int b) {
  return a + b;
}
 
// 导出业务接口 
// 注册 AKI 插件 
JSBIND_ADDON(entry) // 注册 AKI 插件名: 即为编译*.so名称
 
// 注册 FFI 特性 
JSBIND_GLOBAL() 
{ 
  JSBIND_FUNCTION(SayHello, "SayHello"); 
  JSBIND_FUNCTION(Add, "Add");  // 导出Add函数
}