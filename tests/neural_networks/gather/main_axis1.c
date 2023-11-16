#include <stdio.h>

// if you want to change the axis and the other parameters here reported, you also need to change the input onnx file
#define AXIS (1)

#define VAL_SIZE_ROWS (4)
#define VAL_SIZE_COLS (3)
#define VAL_SIZE (VAL_SIZE_ROWS * VAL_SIZE_COLS)

#define IND_SIZE_ROWS (3)
#define IND_SIZE_COLS (2)
#define IND_SIZE (IND_SIZE_ROWS * IND_SIZE_COLS)

#if AXIS == 0
#define OUTPUT_SIZE (VAL_SIZE_COLS * IND_SIZE_ROWS * IND_SIZE_COLS)
#elif AXIS == 1
#define OUTPUT_SIZE (VAL_SIZE_ROWS * IND_SIZE_ROWS * IND_SIZE_COLS)
#else
printf("Error: invalid axis value\n");
return -1;
#endif

int main() {
    float32_t values[VAL_SIZE] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };

    int64_t indices[IND_SIZE] = {
        0, 2,
        0, 2,
        0, 2,
    };

    float output[OUTPUT_SIZE];

    run_inference(values, indices, output);

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