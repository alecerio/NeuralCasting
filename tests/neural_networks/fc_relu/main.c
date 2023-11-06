#include <stdio.h>

int main() {
    float input[2] = {
        1.0f, 1.0f
    };

    float output[3] = {
        0.0f, 0.0f, 0.0f
    };

    run_inference(input, output);

    FILE* file;
    file = fopen("test_output.txt", "w");
    for(int i=0; i<3; i++) {
        fprintf(file, "%f", output[i]);
        if(i < 2)
            fprintf(file, " ");
    }
    fclose(file);

    return 0;
}