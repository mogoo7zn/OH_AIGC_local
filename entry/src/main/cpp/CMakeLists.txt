cmake_minimum_required(VERSION 3.5.0)
project(OH_AIGC_local)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# 设置 AKI
set(AKI_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../oh_modules/@ohos/aki)
set(CMAKE_MODULE_PATH ${AKI_ROOT_PATH})
find_package(Aki REQUIRED)

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

# 构建entry库（使用AKI绑定）
add_library(entry SHARED napi_init.cpp)
target_link_libraries(entry PUBLIC libace_napi.z.so)
target_link_libraries(entry PUBLIC Aki::libjsbind)

# MNN库配置
set(MNN_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/MNN")
if(EXISTS ${MNN_SOURCE_ROOT})
  include_directories(${MNN_SOURCE_ROOT}/include)
  include_directories(${MNN_SOURCE_ROOT}/source)
  include_directories(${MNN_SOURCE_ROOT}/express)

  # 添加MNN库到构建系统
  add_subdirectory(${MNN_SOURCE_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/MNN)

  # 构建mnn_wrapper库（也使用AKI绑定）
  add_library(mnn_wrapper SHARED mnn_wrapper.cpp)
  target_link_libraries(mnn_wrapper 
                       MNN 
                       MNN_Express
                       libace_napi.z.so
                       Aki::libjsbind)
endif()