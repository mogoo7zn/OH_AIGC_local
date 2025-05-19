#include <aki/jsbind.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>

// 创建MNN解释器函数
int64_t CreateInterpreter(std::string modelPath) {
    // 创建解释器
    std::shared_ptr<MNN::Interpreter> interpreter(MNN::Interpreter::createFromFile(modelPath.c_str()));
    return (int64_t)(interpreter.get());
}

// 创建会话函数
int64_t CreateSession(int64_t interpreterPtr, int numThreads = 4) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    if (!interpreter) return 0;
    
    MNN::ScheduleConfig config;
    config.numThread = numThreads;
    config.type = MNN_FORWARD_CPU;
    
    // 创建会话
    MNN::Session* session = interpreter->createSession(config);
    return (int64_t)session;
}

// 运行会话函数
void RunSession(int64_t interpreterPtr, int64_t sessionPtr) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    
    if (interpreter && session) {
        interpreter->runSession(session);
    }
}

// 注册AKI插件
JSBIND_ADDON(mnn_wrapper)

// 注册FFI特性 
JSBIND_GLOBAL() 
{ 
  JSBIND_FUNCTION(CreateInterpreter);
  JSBIND_FUNCTION(CreateSession);
  JSBIND_FUNCTION(RunSession);
}