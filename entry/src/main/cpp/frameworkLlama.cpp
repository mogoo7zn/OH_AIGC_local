#include "frameworkLlama.h"
#include <thread>
#include "hilog/log.h"
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0721  // 全局domain宏，标识业务领域
#define LOG_TAG "TEST"   // 全局tag宏，标识模块日志tag


std::string llama_cpp::test(){
    return model_name;
}

llama_cpp::llama_cpp(std::string path){
    model_name = path;
    OH_LOG_INFO(LOG_APP,"load model%{public}s",path.c_str());
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    //model
    model = llama_model_load_from_file(path.c_str(), model_params);
    if (model == nullptr){
        OH_LOG_ERROR(LOG_APP,"load model error!");
        exit(0);
    }
    //ctx
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_num;
    ctx_params.n_batch = 128;
    ctx_params.no_perf = false;
    ctx_params.n_threads = 4;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        OH_LOG_ERROR(LOG_APP,"initial context error!");
        exit(0);
    }
    //vocab
    vocab = llama_model_get_vocab(model);
    //sample
    auto sampler_params = llama_sampler_chain_default_params(); 
    sampler_params.no_perf = false;
    sampler = llama_sampler_chain_init(sampler_params);
    if (sampler == nullptr) {
        OH_LOG_ERROR(LOG_APP,"initial sampler error!");
        exit(0);
    }
    //llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    OH_LOG_INFO(LOG_APP,"load model success");
}

llama_cpp::~llama_cpp(){
    llama_sampler_free(sampler);
    llama_model_free(model);
    llama_free(ctx);
    for(auto &msg:messages){
        free(const_cast<char *>(msg.content));
    }
    delete vocab;
}

bool llama_cpp::check_model_load(std::string path){
    if (path != model_name){
        return false;
    }
    return true;
}

void llama_cpp::llama_cpp_inference_start(std::string prompt, std::function<void(std::string)> ref){
    OH_LOG_INFO(LOG_APP,"prompt=%{public}s",prompt.c_str());
    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);
    std::vector<char> formatted(llama_n_ctx(ctx));
    messages.push_back({"user",strdup(prompt.c_str())});
    int new_len = llama_chat_apply_template(tmpl, messages.data(),messages.size(),true,formatted.data(),formatted.size());
    if (new_len > (int)formatted.size()){
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(),messages.size(),true,formatted.data(),formatted.size());
    }
    if (new_len<0){
        OH_LOG_ERROR(LOG_APP,"new_len error!");
        exit(0);
    }
    OH_LOG_INFO(LOG_APP,"formatted=%{public}s",formatted.data());
    std::string new_prompt(formatted.begin() + prev_len,formatted.begin()+new_len);
    //start forward
    std::string response;
    const bool is_first = llama_kv_self_used_cells(ctx) == 0;
    const int n_prompt_tokens = -llama_tokenize(vocab, new_prompt.c_str(), new_prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, new_prompt.c_str(), new_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        OH_LOG_ERROR(LOG_APP,"failed to tokenize the prompt");
        exit(0);
    }
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    
    while(true){
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_kv_self_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            OH_LOG_INFO(LOG_APP,"context size exceeded");
            break;
        }
        if (llama_decode(ctx, batch)) {
            OH_LOG_ERROR(LOG_APP,"failed to decode");
            exit(0);
        }
        new_token_id = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            OH_LOG_ERROR(LOG_APP,"failed to convert token to piece");
            exit(0);
        }
        std::string piece(buf, n);
        OH_LOG_INFO(LOG_APP,"token:%{public}s,n_ctx_used=%{public}d,batch.n_tokens=%{public}d",piece.c_str(),n_ctx_used,batch.n_tokens);
        response += piece;
        ref(response.c_str());
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    
    //change message
    messages.push_back({"assistant",strdup(response.c_str())});
}
