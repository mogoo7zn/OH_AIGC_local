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

llama_cpp::llama_cpp(std::string path,std::string prompt){
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
    ctx_params.n_batch = 512;
    ctx_params.no_perf = false;
    ctx_params.n_threads = 4;   //设置推理启用线程数
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
    
    //message
    messages.push_back({"system",strdup(prompt.c_str())});
    OH_LOG_INFO(LOG_APP,"prompt:%{public}s",prompt.c_str());
    
    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);
    std::vector<char> formatted(llama_n_ctx(ctx));
    int new_len = llama_chat_apply_template(tmpl, messages.data(),messages.size(),false,formatted.data(),formatted.size());
    if (new_len > (int)formatted.size()){
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(),messages.size(),false,formatted.data(),formatted.size());
    }
    if (new_len<0){
        OH_LOG_ERROR(LOG_APP,"new_len error!");
        exit(0);
    }
    
    std::string new_prompt(formatted.begin() , formatted.begin() + new_len);
    OH_LOG_INFO(LOG_APP,"new_prompt=%{public}s ,peev_len=%{public}d",new_prompt.c_str(),prev_len);

    const bool is_first = true;
    const int n_prompt_tokens = -llama_tokenize(vocab, new_prompt.c_str(), new_prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, new_prompt.c_str(), new_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        OH_LOG_ERROR(LOG_APP,"failed to tokenize the prompt");
        exit(0);
    }
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_decode(ctx, batch);
    prev_len = new_len;
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
    stop = false;
    OH_LOG_INFO(LOG_APP,"userinput=%{public}s",prompt.c_str());
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
    OH_LOG_INFO(LOG_APP,"formatted=%{public}s ,peev_len=%{public}d",formatted.data(),prev_len);
    std::string new_prompt(formatted.begin() + prev_len,formatted.begin() + new_len);
    OH_LOG_INFO(LOG_APP,"new_prompt=%{public}s ,peev_len=%{public}d",new_prompt.c_str(),prev_len);
    //start forward
    std::string response;
    const bool is_first = false;
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
            
            OH_LOG_INFO(LOG_APP,"used %{public}d, batch %{public}d, context size exceeded",n_ctx_used,batch.n_tokens);
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
        if (stop)   return;
        ref(response.c_str());
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    //change message
    messages.push_back({"assistant",strdup(response.c_str())});
    prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
    ref("炸薯丸出品");
}

void llama_cpp::add_message(std::string role,std::string content){
    messages.push_back({role.c_str(),strdup(content.c_str())});
}

llama_cpp_mtmd::llama_cpp_mtmd(std::string module_path , std::string mmproj_path){
    ggml_time_init();
    
    model_name = module_path;
    OH_LOG_INFO(LOG_APP,"load model%{public}s",model_name.c_str());
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    //model
    ctx.model = llama_model_load_from_file(model_name.c_str(), model_params);
    if (ctx.model == nullptr){
        OH_LOG_ERROR(LOG_APP,"load model error!");
        exit(0);
    }
    //ctx
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_num;
    ctx_params.n_batch = 128;
    ctx_params.no_perf = false;
    ctx_params.n_threads = 4;   //设置推理启用线程数
    ctx.lctx = llama_init_from_model(ctx.model, ctx_params);
    if (ctx.lctx == nullptr) {
        OH_LOG_ERROR(LOG_APP,"initial context error!");
        exit(0);
    }
    ctx.vocab = llama_model_get_vocab(ctx.model);
    
    ctx.n_threads = 8;
    
    ctx.n_batch = ctx_params.n_batch;
    ctx.batch = llama_batch_init(ctx.n_batch, 0, 1);
    
    ctx.tmpls = common_chat_templates_init(ctx.model, "deepseek");;
    ctx.antiprompt_tokens = common_tokenize(ctx.lctx, "###", false, true);
    
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = 0;
    mparams.print_timings = true;
    mparams.n_threads = 8;
    ctx_vision.reset(mtmd_init_from_file(mmproj_path.c_str(), ctx.model, mparams));
    if (!ctx_vision.get()) {
        OH_LOG_ERROR(LOG_APP,"Failed to load vision model");
        exit(1);
    }
    common_params_sampling sampling_params;
    smpl = common_sampler_init(ctx.model, sampling_params);
}

void llama_cpp_mtmd::load_image(const std::string & fname){
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(fname.c_str()));
    if (!bmp.ptr) {
        OH_LOG_ERROR(LOG_APP,"load image error!");
    }
    ctx.bitmaps.entries.push_back(std::move(bmp));
    image_num += 1;
    OH_LOG_INFO(LOG_APP,"load_image_success");
}

std::string llama_cpp_mtmd::test(){
    return model_name;
}

llama_cpp_mtmd::~llama_cpp_mtmd(){
    llama_sampler_free(sampler);
    common_sampler_free(smpl);
    llama_free(ctx.lctx);
    llama_model_free(ctx.model);
    delete ctx.vocab;
}

bool llama_cpp_mtmd::check_model_load(std::string path){
    if (path != model_name){
        return false;
    }
    return true;
}

void llama_cpp_mtmd::llama_cpp_inference_start(std::string prompt ,std::function<void(std::string)> ref){
    if (prompt.find("<__image__>") == std::string::npos) {
            prompt += " <__image__>";
    }
    common_chat_msg msg;
    msg.role = "user";
    msg.content = prompt;
    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false; // jinja is buggy here
    auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);
    OH_LOG_INFO(LOG_APP,"formatted_chat.prompt:%{public}s",formatted_chat.prompt.c_str());

    mtmd_input_text text;
    text.text          = formatted_chat.prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    if (!ctx_vision.get()){
        OH_LOG_ERROR(LOG_APP,"ctx_vision error!");
    }
    int32_t res = mtmd_tokenize(ctx_vision.get(),
                        chunks.ptr.get(), // output
                        &text, // text
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        OH_LOG_ERROR(LOG_APP,"Unable to tokenize prompt：%{public}d",res);
        exit(1);
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past;
    OH_LOG_INFO(LOG_APP,"1");
    if (mtmd_helper_eval_chunks(ctx_vision.get(),
                ctx.lctx, // lctx
                chunks.ptr.get(), // chunks
                ctx.n_past, // n_past
                0, // seq_id
                ctx.n_batch, // n_batch
                true, // logits_last
                &new_n_past)) {
        OH_LOG_ERROR(LOG_APP,"Unable to eval prompt");
        exit(1);
    }
    OH_LOG_INFO(LOG_APP,"2");
    ctx.n_past = new_n_past;
    OH_LOG_INFO(LOG_APP,"new_n_past: %{public}d",new_n_past);
    
    llama_tokens generated_tokens;
    std::string response;
    for (int i = 0; i < 256; i++) {

        llama_token token_id = common_sampler_sample(smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            OH_LOG_INFO(LOG_APP,"finish");
            break; // end of generation
        }
        
        response += common_token_to_piece(ctx.lctx, token_id).c_str();
        OH_LOG_INFO(LOG_APP,"token:%{public}s",response.c_str());
        ref(response.c_str());
        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            OH_LOG_ERROR(LOG_APP,"failed to decode token");
            exit(1);
        }
    }
}
