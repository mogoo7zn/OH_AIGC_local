//
// Created on 2025/6/2.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef BACKEND_FRAMEWORKLLAMA_H
#define BACKEND_FRAMEWORKLLAMA_H

#include <string>
#include <functional>
#include "sampling.h"
#include "chat.h"
#include "llama.h"
#include "mtmd.h"
#include "common.h"

#define n_ctx_num 512


class llama_cpp{
    private:
        llama_model *model;
        std::string model_name;
        llama_context *ctx;
        const llama_vocab *vocab; 
        llama_sampler *sampler;
        int prev_len = 0;
        std::vector<llama_chat_message> messages={};
    
    public:
        llama_cpp(std::string path,std::string prompt);
        ~llama_cpp();
        bool check_model_load(std::string path);
        std::string test();
        void llama_cpp_inference_start(std::string prompt, std::function<void(std::string)> ref);
};

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    llama_batch         batch;
    int                 n_batch;

    common_chat_templates_ptr tmpls;

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;
    
    mtmd::bitmaps bitmaps;

    int n_threads    = 1;
    llama_pos n_past = 0;
    
    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }
};


class llama_cpp_mtmd{
    private:
        std::string model_name;
        llama_sampler *sampler;
        mtmd_cli_context ctx;
        mtmd::context_ptr ctx_vision;
        struct common_sampler * smpl;
        
    public:
        int                 image_num = 0;
        llama_cpp_mtmd(std::string module_path , std::string mmproj_path);
        ~llama_cpp_mtmd();
        bool check_model_load(std::string path);
        std::string test();
        void load_image(const std::string & fname);
        void llama_cpp_inference_start(std::string prompt,std::function<void(std::string)> ref);
};
#endif