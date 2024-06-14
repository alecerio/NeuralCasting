#include <stdio.h>

#define INPUT_SIZE (16)
#define OUTPUT_SIZE (4)

int main() {
    float input[INPUT_SIZE] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };

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
