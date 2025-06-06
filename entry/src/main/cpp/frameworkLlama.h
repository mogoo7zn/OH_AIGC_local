//
// Created on 2025/6/2.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef BACKEND_FRAMEWORKLLAMA_H
#define BACKEND_FRAMEWORKLLAMA_H

#include <string>
#include <functional>
#include "llama.h"

#define n_ctx_num 256


class llama_cpp{
    private:
        llama_model *model;
        std::string model_name;
        llama_context *ctx;
        const llama_vocab *vocab; 
        llama_sampler *sampler;
        int prev_len = 0;
    
        std::vector<llama_chat_message> messages={{"system","你是一个有用的AI助手，负责回答问题"}};
    
    public:
        llama_cpp(std::string path);
        ~llama_cpp();
        bool check_model_load(std::string path);
        std::string test();
        void llama_cpp_inference_start(std::string prompt, std::function<void(std::string)> ref);
};
#endif