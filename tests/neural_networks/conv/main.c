#include <stdio.h>

#define INPUT_SIZE (1*3*8*8)
#define OUTPUT_SIZE (1*2*8*8)

int main() {
    float input[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++)
        input[i] = 1.0f;
    
    float output[OUTPUT_SIZE];
    
    run_inference(input, output);

    FILE* file;
    file = fopen("test_output.txt", "w");
    for(int i=0; i<OUTPUT_SIZE; i++) {
        fprintf(file, "%f", output[i]);
        if(i < OUTPUT_SIZE-1)
            fprintf(file, " ");
    }
    fclose(file);

    return 0;
}