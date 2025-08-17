#include "napi/native_api.h"
#include "hilog/log.h"

#include <thread>
#include "frameworkLlama.h"

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0721  // 全局domain宏，标识业务领域
#define LOG_TAG "TEST"   // 全局tag宏，标识模块日志tag

static llama_cpp_mtmd *multimodal_model = nullptr;
static llama_cpp *model = nullptr;
static bool waiting = false;

std::string GetStringArgument(napi_env env,napi_value arg){
    size_t length;
    char error_message[]= "get str error";
    napi_status status = napi_get_value_string_utf8(env, arg, nullptr, 0, &length);
    if (status != napi_ok){
        napi_throw_error(env, nullptr, error_message);
        return "";
    }
    std::string result(length + 1, '\0');
    status = napi_get_value_string_utf8(env, arg, &result[0], length + 1, nullptr);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, error_message);
        return "";
    }
    result.resize(length);
    return result;
}

void load_module_thread(std::string path,std::string prompt){
    model = new llama_cpp(path,prompt);
    OH_LOG_INFO(LOG_APP,"end load module");
    waiting = false;
}

static napi_value load_module(napi_env env, napi_callback_info info){
    size_t argc = 2;            //the number of arguments
    napi_value args[2];      //the list
    napi_get_cb_info(env, info , &argc, args, nullptr, nullptr);        //get the info of args
    
    std::string path = GetStringArgument(env, args[0]);
    std::string prompt = GetStringArgument(env, args[1]);
    if(model!=nullptr && model->check_model_load(path)){
        return nullptr;
    }
    OH_LOG_INFO(LOG_APP,"start load module");
    waiting = true;
    std::thread thread(load_module_thread,path,prompt);
    thread.detach(); 
    return nullptr;
}

static napi_value unload_module(napi_env env, napi_callback_info info) {
    if (model!=nullptr){
        model->stop = true;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        delete model;
        model = nullptr;
    }
    OH_LOG_INFO(LOG_APP,"unload model");
    return nullptr;
}

struct callTs_context{
    napi_env env;
    std::string output;
};

static void callTS(napi_env env, napi_value jsCb, void *context, void *data) {
    callTs_context *arg = (callTs_context*) data;
    napi_value result;
    napi_create_string_utf8(arg->env, arg->output.c_str(),arg->output.length(), &result);
    napi_call_function(arg->env, nullptr, jsCb, 1, &result, nullptr);
}

void inference_start_thread(napi_env env,napi_ref callback,std::string prompt){
    while(waiting)    std::this_thread::sleep_for(std::chrono::seconds(1));
    waiting = true;
    OH_LOG_INFO(LOG_APP,"start inference");
    napi_value jsCb;
    napi_get_reference_value(env, callback, &jsCb);
    napi_value workName;
    napi_create_string_utf8(env, "Inference", NAPI_AUTO_LENGTH, &workName);
    napi_threadsafe_function tsFn;
    napi_create_threadsafe_function(env, jsCb, nullptr, workName, 0, 1, nullptr, nullptr, nullptr, callTS, &tsFn);    
    callTs_context *ctx = new callTs_context;
    
    OH_LOG_INFO(LOG_APP,"load model%{public}s",model->test().c_str());
    model->llama_cpp_inference_start(prompt, [=](std::string prompt) -> void{
            ctx->env = env;
            ctx->output = prompt;                   
            napi_call_threadsafe_function(tsFn, (void*)ctx, napi_tsfn_blocking);
       });
    waiting = false;
}

static napi_value inference_start(napi_env env, napi_callback_info info) {
    size_t argc = 2;            //the number of arguments
    napi_value args[2];      //the list
    napi_get_cb_info(env, info , &argc, args, nullptr, nullptr);        //get the info of args
    
    std::string prompt = GetStringArgument(env, args[0]);
    if(!waiting && model == nullptr){
        OH_LOG_ERROR(LOG_APP,"didn't load module'");
        return nullptr;
    }
    napi_ref callback;
    napi_create_reference(env, args[1], 1, &callback);
    std::thread thread(inference_start_thread,env,callback,prompt);
    thread.detach();
    return nullptr;
}

static napi_value NAPI_Global_inference_stop(napi_env env, napi_callback_info info) {
    if (model != nullptr){
        model->stop = true;
    }else if(multimodal_model !=nullptr){
        multimodal_model->stop =true;
    }
    return nullptr;
}

static napi_value Add(napi_env env, napi_callback_info info){
    size_t argc = 2;
    napi_value args[2] = {nullptr};

    napi_get_cb_info(env, info, &argc, args , nullptr, nullptr);

    napi_valuetype valuetype0;
    napi_typeof(env, args[0], &valuetype0);

    napi_valuetype valuetype1;
    napi_typeof(env, args[1], &valuetype1);

    double value0;
    napi_get_value_double(env, args[0], &value0);

    double value1;
    napi_get_value_double(env, args[1], &value1);

    napi_value sum;
    napi_create_double(env, value0 + value1, &sum);

    return sum;

}


void load_image_thread(std::string path){
    while(waiting)    std::this_thread::sleep_for(std::chrono::seconds(1));
    waiting = true;
    multimodal_model->load_image(path);
    OH_LOG_INFO(LOG_APP,"end load module");
    waiting = false;
}

static napi_value NAPI_Global_load_image(napi_env env, napi_callback_info info) {
     size_t argc = 1;            //the number of arguments
    napi_value args[1];      //the list
    napi_get_cb_info(env, info , &argc, args, nullptr, nullptr);        //get the info of args
    
    std::string path = GetStringArgument(env, args[0]);
    std::thread thread(load_image_thread,path);
    thread.detach();
    return nullptr;
}

void load_multimodal_module_thread(std::string module_path , std::string mmproj_path){
    multimodal_model = new llama_cpp_mtmd(module_path,mmproj_path);
    OH_LOG_INFO(LOG_APP,"end load module");
    waiting = false;
}

static napi_value NAPI_Global_load_multimodal_module(napi_env env, napi_callback_info info)
{
    size_t argc = 2;            //the number of arguments
    napi_value args[2];      //the list
    napi_get_cb_info(env, info , &argc, args, nullptr, nullptr);        //get the info of args
    
    std::string module_path = GetStringArgument(env, args[0]);
    std::string mmproj_path = GetStringArgument(env, args[1]);
    if(model!=nullptr && model->check_model_load(module_path)){
        return nullptr;
    }
    OH_LOG_INFO(LOG_APP,"start load module");
    waiting = true;
    std::thread thread(load_multimodal_module_thread,module_path,mmproj_path);
    thread.detach();
    return nullptr;
}

static napi_value NAPI_Global_unload_multimodal_module(napi_env env, napi_callback_info info) {
    delete multimodal_model;
    multimodal_model = nullptr;
    return nullptr;
}

void multimodal_inference_start_thread(napi_env env,napi_ref callback,std::string prompt){
    while(waiting || multimodal_model->image_num == 0)    std::this_thread::sleep_for(std::chrono::seconds(1));
    waiting = true;
    OH_LOG_INFO(LOG_APP,"start inference");
    napi_value jsCb;
    napi_get_reference_value(env, callback, &jsCb);
    napi_value workName;
    napi_create_string_utf8(env, "Inference", NAPI_AUTO_LENGTH, &workName);
    napi_threadsafe_function tsFn;
    napi_create_threadsafe_function(env, jsCb, nullptr, workName, 0, 1, nullptr, nullptr, nullptr, callTS, &tsFn);    
    callTs_context *ctx = new callTs_context;
    
    OH_LOG_INFO(LOG_APP,"load model%{public}s",model->test().c_str());
    model->llama_cpp_inference_start(prompt, [=](std::string prompt) -> void{
            ctx->env = env;
            ctx->output = prompt;                   
            napi_call_threadsafe_function(tsFn, (void*)ctx, napi_tsfn_blocking);
       });
    waiting = false;
}

static napi_value NAPI_Global_inference_multimodal_start(napi_env env, napi_callback_info info) {
    size_t argc = 2;            //the number of arguments
    napi_value args[2];      //the list
    napi_get_cb_info(env, info , &argc, args, nullptr, nullptr);        //get the info of args
    
    std::string prompt = GetStringArgument(env, args[0]);
    if(!waiting && model == nullptr){
        OH_LOG_ERROR(LOG_APP,"didn't load module'");
        return nullptr;
    }
    napi_ref callback;
    napi_create_reference(env, args[1], 1, &callback);
    std::thread thread(multimodal_inference_start_thread,env,callback,prompt);
    thread.detach();
    return nullptr;
}

static napi_value NAPI_Global_add_message(napi_env env, napi_callback_info info) {
    size_t argc = 2;            //the number of arguments
    napi_value args[2];      //the list
    napi_get_cb_info(env, info , &argc, args, nullptr, nullptr);        //get the info of args
    
    std::string role = GetStringArgument(env, args[0]);
    std::string content = GetStringArgument(env, args[1]);
    if(model!=nullptr){
        model->add_message(role, content);
    }
    return nullptr;
}
EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"add", nullptr, Add, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"load_module", nullptr, load_module, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"unload_module", nullptr, unload_module, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"inference_start", nullptr, inference_start, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"inference_stop", nullptr, NAPI_Global_inference_stop, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"load_multimodal_image", nullptr, NAPI_Global_load_image, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"load_multimodal_module", nullptr, NAPI_Global_load_multimodal_module, nullptr, nullptr, nullptr, napi_default,
         nullptr},
        {"unload_multimodal_module", nullptr, NAPI_Global_unload_multimodal_module, nullptr, nullptr, nullptr,
         napi_default, nullptr},
        {"inference_multimodal_start", nullptr, NAPI_Global_inference_multimodal_start, nullptr, nullptr, nullptr,
         napi_default, nullptr},
        {"add_message", nullptr, NAPI_Global_add_message, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
}
