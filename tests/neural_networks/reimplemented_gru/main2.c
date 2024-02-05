#include <stdio.h>

#define INPUT_SIZE (3)
#define HIDDEN_SIZE (4)

int main() {
    float input[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++)
        input[i] = 1.0f;

    float hidden[INPUT_SIZE * HIDDEN_SIZE];
    for(int i=0; i<INPUT_SIZE * HIDDEN_SIZE; i++)
        hidden[i] = 0.0f;

    float output[INPUT_SIZE*HIDDEN_SIZE];

    run_inference(input, hidden, output);

    FILE* file;
    file = fopen("test_output.txt", "w");
    for(int i=0; i<INPUT_SIZE * HIDDEN_SIZE; i++) {
        fprintf(file, "%f", output[i]);
        if(i < INPUT_SIZE * HIDDEN_SIZE-1)
            fprintf(file, " ");
    }
    fclose(file);

    return 0;
}