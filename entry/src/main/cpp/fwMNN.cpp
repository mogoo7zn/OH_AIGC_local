#include "fwMNN.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "hilog/log.h"


/**
 * 编码器进行性能调优
 * @param llm
 */
static void tuning_prepare(MNN::Transformer::Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

MNN_cpp::~MNN_cpp() {
    MNN_llm->reset();
    MNN_chatMessages.clear();
    OH_LOG_INFO(LOG_APP, "MNN model resources released");
}

/**
 * 加载模型准备
 * @param path
 * @param prompt
 */
MNN_cpp::MNN_cpp(std::string path, std::string prompt) {
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, MNN_backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    std::unique_ptr<MNN::Transformer::Llm> MNN_llm(MNN::Transformer::Llm::createLLM(path));
    {
        AUTOTIME;
        MNN_llm->load();
    }
    {
        AUTOTIME;
        tuning_prepare(MNN_llm.get());
    }
    MNN_chatMessages.emplace_back("system", "我是一个大语言模型助手");
    
    OH_LOG_INFO(LOG_APP,"load model success");
}

/**
 * 
 * @param userInput
 * @param ref
 */
void MNN_cpp::MNN_cpp_inference_start(std::string userInput, std::function<void(std::string)> ref) {
    OH_LOG_INFO(LOG_APP,"prompt=%{public}s",userInput.c_str());
    std::string response;
    
//    MNN_chatMessages.clear();
    MNN_chatMessages.emplace_back("user", userInput);
    
    MNN_llm->set_config(R"({"async":false})");
    
    std::ostringstream nullStream;
    MNN_llm->response(MNN_chatMessages, &nullStream, nullptr, 0);
    int last_lens = 0;
    
    auto context = MNN_llm->getContext();
    
    while(!MNN_llm -> stoped()) {
        MNN_llm->generate(1);
        response = context->generate_str;
        if (response.length() > last_lens) {
            ref(response);
            last_lens = response.length();
        }
    }
    
    MNN_chatMessages.emplace_back("system", response);
    OH_LOG_INFO(LOG_APP, "response=%{public}s", response.c_str());
}


