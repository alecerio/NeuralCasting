#include <stdio.h>

#define ROWS (2)
#define COLUMNS (3)

int main() {
    float input[ROWS*COLUMNS] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float output[COLUMNS*ROWS];

    run_inference(input, output);

    FILE* file;
    file = fopen("test_output.txt", "w");
    for(int i=0; i<COLUMNS*ROWS; i++) {
        fprintf(file, "%f", output[i]);
        if(i < COLUMNS*ROWS-1)
            fprintf(file, " ");
    }
    fclose(file);

    return 0;
}