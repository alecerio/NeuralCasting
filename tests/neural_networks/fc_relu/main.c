#include <stdio.h>

#define INPUT_SIZE (2)
#define OUTPUT_SIZE (3)

int main() {
    float input[INPUT_SIZE] = {
        1.0f, 1.0f
    };

    float output[OUTPUT_SIZE] = {
        0.0f, 0.0f
    };

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