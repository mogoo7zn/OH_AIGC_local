#include <aki/jsbind.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>
#include <vector>

// Create MNN interpreter from model file
int64_t CreateInterpreter(std::string modelPath) {
    std::shared_ptr<MNN::Interpreter> interpreter(MNN::Interpreter::createFromFile(modelPath.c_str()));
    return (int64_t)(interpreter.get());
}

// Create inference session
int64_t CreateSession(int64_t interpreterPtr, int numThreads = 4) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    if (!interpreter) return 0;
    
    MNN::ScheduleConfig config;
    config.numThread = numThreads;
    config.type = MNN_FORWARD_CPU;
    
    MNN::Session* session = interpreter->createSession(config);
    return (int64_t)session;
}

// Run model inference
void RunSession(int64_t interpreterPtr, int64_t sessionPtr) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    
    if (interpreter && session) {
        interpreter->runSession(session);
    }
}

// Set input tensor data
void SetInputTensor(int64_t interpreterPtr, int64_t sessionPtr, const std::string& name, 
                   const std::vector<int>& shape, const std::vector<float>& data) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    
    if (!interpreter || !session) return;
    
    MNN::Tensor* tensor = interpreter->getSessionInput(session, name.c_str());
    if (!tensor) return;
    
    // Resize if shapes don't match
    if (tensor->elementSize() != data.size()) {
        interpreter->resizeTensor(tensor, shape);
        interpreter->resizeSession(session);
    }
    
    auto host = tensor->host<float>();
    auto size = tensor->elementSize();
    
    memcpy(host, data.data(), size * sizeof(float));
}

// Get output tensor data
std::vector<float> GetOutputTensor(int64_t interpreterPtr, int64_t sessionPtr, const std::string& name) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    std::vector<float> result;
    
    if (!interpreter || !session) return result;
    
    MNN::Tensor* tensor = interpreter->getSessionOutput(session, name.c_str());
    if (!tensor) return result;
    
    // Create host tensor for CPU access
    MNN::Tensor hostTensor(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&hostTensor);
    
    // Copy data to result vector
    auto size = hostTensor.elementSize();
    auto host = hostTensor.host<float>();
    result.resize(size);
    memcpy(result.data(), host, size * sizeof(float));
    
    return result;
}

// Get input tensor names
std::vector<std::string> GetInputNames(int64_t interpreterPtr, int64_t sessionPtr) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    std::vector<std::string> names;
    
    if (!interpreter || !session) return names;
    
    auto inputs = interpreter->getSessionInputAll(session);
    for (auto& iter : inputs) {
        names.push_back(iter.first);
    }
    
    return names;
}

// Get output tensor names
std::vector<std::string> GetOutputNames(int64_t interpreterPtr, int64_t sessionPtr) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    std::vector<std::string> names;
    
    if (!interpreter || !session) return names;
    
    auto outputs = interpreter->getSessionOutputAll(session);
    for (auto& iter : outputs) {
        names.push_back(iter.first);
    }
    
    return names;
}

// Process image for model input
std::vector<float> ProcessImage(int64_t interpreterPtr, int64_t sessionPtr, 
                              const std::string& inputName, 
                              const std::vector<uint8_t>& imageData,
                              int width, int height, int channels) {
    MNN::Interpreter* interpreter = (MNN::Interpreter*)(interpreterPtr);
    MNN::Session* session = (MNN::Session*)(sessionPtr);
    std::vector<float> result;
    
    if (!interpreter || !session) return result;
    
    MNN::Tensor* inputTensor = interpreter->getSessionInput(session, inputName.c_str());
    if (!inputTensor) return result;
    
    // Create image process config
    MNN::CV::ImageProcess::Config config;
    config.sourceFormat = MNN::CV::ImageFormat::RGB;
    config.destFormat = MNN::CV::ImageFormat::RGB;
    
    // Set mean and norm values for common models
    config.mean[0] = 127.5f;
    config.mean[1] = 127.5f;
    config.mean[2] = 127.5f;
    config.normal[0] = 1.0f / 127.5f;
    config.normal[1] = 1.0f / 127.5f;
    config.normal[2] = 1.0f / 127.5f;
    
    // Create image process
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
    
    // Process image and convert to tensor
    pretreat->convert(imageData.data(), width, height, width * channels, inputTensor);
    
    return result;
}

// Register AKI plugin
JSBIND_ADDON(mnn_wrapper)

// Register FFI features
JSBIND_GLOBAL() 
{ 
    JSBIND_FUNCTION(CreateInterpreter);
    JSBIND_FUNCTION(CreateSession);
    JSBIND_FUNCTION(RunSession);
    JSBIND_FUNCTION(SetInputTensor);
    JSBIND_FUNCTION(GetOutputTensor);
    JSBIND_FUNCTION(GetInputNames);
    JSBIND_FUNCTION(GetOutputNames);
    JSBIND_FUNCTION(ProcessImage);
}