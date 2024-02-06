//#include <onnxruntime/core/providers/providers.h>
//#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
//#include <onnxruntime/core/providers/cpu/cpu_execution_provider.h>
//#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

//#include <onnxruntime/core/providers/providers.h>
//#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
//#include <onnxruntime/core/providers/providers.h>

#include "cpu_provider_factory.h"
#include "cuda_provider_factory.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
//#include "onnxruntime_cxx_inline.h"
#include "onnxruntime_session_options_config_keys.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

#define BATCH (1)
#define WSIZE (32)
#define HSIZE (32)
#define NCH (3)
#define SHAPE_SIZE (4)

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    
    const char* model_path = "model.onnx";  // Replace with the path to your ONNX model file

    printf("BBB\n");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    printf("AAA\n");
    // Prepare input data (adjust dimensions and values based on your model's input requirements)
    const int input_tensor_size = WSIZE * HSIZE * NCH * BATCH;  // Example size for an image classification model
    float input_data[input_tensor_size];

    for(int i=0; i<BATCH*WSIZE*HSIZE*NCH; i++)
        input_data[i] = 0.0f;
    
    // Define OrtMemoryInfo for CPU memory
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    // Create Ort::Value with the input tensor
    int64_t* shape = (int64_t*) malloc(sizeof(int64_t)*SHAPE_SIZE);
    shape[0] = BATCH; shape[1] = NCH; shape[2] = WSIZE; shape[3] = HSIZE;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_tensor_size, shape, SHAPE_SIZE);

    // Run inference
    const char* input_names[] = {"input.1"};  // Replace with the actual input name from your model
    const char* output_names[] = {"119"};  // Replace with the actual output name from your model

    clock_t start = clock();
    std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, 1, output_names, 1);
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("time: %.20f\n", time);

    // Process the output data (adjust based on your model's output requirements)
    float* output_data = output_tensor.at(0).GetTensorMutableData<float>();

    for(int i=0; i<10; i++) {
        printf("%f ", output_data[i]);
    }
    printf("\n");
    return 0;
}