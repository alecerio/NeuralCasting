#include <stdio.h>

#define INPUT_SIZE (10)
#define HIDDEN_SIZE (20)
#define SEQ_LEN (5)

int main() {
    float input[INPUT_SIZE * SEQ_LEN];
    for(int i=0; i<INPUT_SIZE * SEQ_LEN; i++)
        input[i] = 1.0f;

    float hidden[HIDDEN_SIZE];
    for(int i=0; i<HIDDEN_SIZE; i++)
        hidden[i] = 0.0f;

    float output[HIDDEN_SIZE*SEQ_LEN];
    float next_hidden[HIDDEN_SIZE];

    run_inference(input, hidden, output, next_hidden);

    FILE* file;
    file = fopen("test_output.txt", "w");
    for(int i=0; i<HIDDEN_SIZE; i++) {
        fprintf(file, "%f", output[i]);
        if(i < HIDDEN_SIZE-1)
            fprintf(file, " ");
    }
    fclose(file);

    return 0;
}