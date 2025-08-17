#ifndef BACKEND_FWMNN_H
#define BACKEND_FWMNN_H

#include <string>
#include <functional>
#include <llm.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#define n_ctx_num 1024

class MNN_cpp {
    private:
        MNN::Transformer::ChatMessages MNN_chatMessages;
        MNN::BackendConfig MNN_backendConfig;
        std::string MNN_modelName;
        MNN::Transformer::Llm* MNN_llm;
        
    public:
        MNN_cpp(std::string path, std::string prompt);      // load MNN model
        bool check_model_load(std::string path);
        std:: string test();
        void MNN_cpp_inference_start(std::string prompt, std::function<void(std::string)> ref);    
        ~MNN_cpp();
};

#endif
