#include "nsnet.h"
#include <stdio.h>

#define INPUT_SIZE (257)
#define OUTPUT_SIZE (INPUT_SIZE)
#define HIDDEN_SIZE (400)

int main() {
    float32_t in_noisy[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++)
        in_noisy[i] = 1.0f;
    
    float32_t in_hidden1[HIDDEN_SIZE];
    for(int i=0; i<HIDDEN_SIZE; i++)
        in_hidden1[i] = 0.0f;
    
    float32_t in_hidden2[HIDDEN_SIZE];
    for(int i=0; i<HIDDEN_SIZE; i++)
        in_hidden2[i] = 0.0f;
    
    float32_t output[OUTPUT_SIZE];
    float32_t out_hidden1[HIDDEN_SIZE];
    float32_t out_hidden2[HIDDEN_SIZE];
    
    allocnn();
    run_inference(in_noisy, in_hidden1, in_hidden2, output, out_hidden1, out_hidden2);
    freenn();

    for(int i=0; i<OUTPUT_SIZE; i++) {
        printf("%f ", output[i]);
        if((i+1)%4 == 0)
            printf("\n");
    }
    printf("\n");

    return 0;
}