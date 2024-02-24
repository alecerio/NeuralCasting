#include <stdio.h>

#define INPUT_SIZE (1*3*1*4)

int main() {
    float input[INPUT_SIZE] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };

    float output[INPUT_SIZE];

    run_inference(input, output);

    FILE* file;
    file = fopen("test_output.txt", "w");
    for(int i=0; i<INPUT_SIZE; i++) {
        fprintf(file, "%f", output[i]);
        if(i < INPUT_SIZE-1)
            fprintf(file, " ");
    }
    fclose(file);

    return 0;
}