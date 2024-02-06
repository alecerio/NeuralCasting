#include "cpu_provider_factory.h"
#include "cuda_provider_factory.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

int main() {
    const char* model_path = "gru_reimplemented_4.onnx"; 
    const int SIZE_INPUT = 3;
    const int SIZE_HIDDEN = 4;
    const int NUM_EXPERIMENTS = 10000;

    const int INPUT_SHAPE_SIZE = 2;
    const int HIDDEN_SHAPE_SIZE = 3;

    // create onnxruntime env
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // create ort session
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    // prepare input data
    const int input_tensor_size = SIZE_INPUT;
    float input_data[input_tensor_size];
    for(int i=0; i<input_tensor_size; i++)
        input_data[i] = 0.0f;
    
    // prepare hidden state
    const int hidden_tensor_size = SIZE_INPUT * SIZE_HIDDEN;
    float hidden_data[hidden_tensor_size];
    for(int i=0; i<hidden_tensor_size; i++)
        hidden_data[i] = 1.0f;
    
    // Define OrtMemoryInfo for CPU memory
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    // Create Ort::Value with the input tensor
    int64_t* input_shape = (int64_t*) malloc(sizeof(int64_t)*INPUT_SHAPE_SIZE);
    input_shape[0] = 1; input_shape[1] = SIZE_INPUT;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_tensor_size, input_shape, INPUT_SHAPE_SIZE);

    // Create Ort::Value with the hidden tensor
    int64_t* hidden_shape = (int64_t*) malloc(sizeof(int64_t)*HIDDEN_SHAPE_SIZE);
    hidden_shape[0] = 1; hidden_shape[1] = SIZE_INPUT; hidden_shape[2] = SIZE_HIDDEN;
    Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(memory_info, hidden_data, hidden_tensor_size, hidden_shape, HIDDEN_SHAPE_SIZE);
    
    // run inference
    const char* input_names[] = {"onnx::Gemm_0", "onnx::MatMul_1"};
    const char* output_names[] = {"37"};

    double total_time = 0.0;
    double min_time = 10000.0;
    double max_time = -100.0;
    for(int i=0; i<NUM_EXPERIMENTS; i++) {
        // run inference
        clock_t start = clock();
        std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, 2, output_names, 1);
        clock_t end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC;
        
        // update total time
        total_time += time;

        // update min time
        if(time < min_time)
            min_time = time;
        
        // update max time
        if(time > max_time)
            max_time = time;
    }

    // compute average time
    double avg_time = total_time / (double)NUM_EXPERIMENTS;
    
    // print average time
    printf("%.20f,", avg_time);

    // print min time
    printf("%.20f,", min_time);

    // print max time
    printf("%.20f", max_time);

    return 0;
}