// MAXPOOL OPERATOR $(NAME)

for (int i = 0; i < $(OUTPUT_HEIGHT); ++i) {
    for (int j = 0; j < $(OUTPUT_WIDTH); ++j) {
        float max_val = -FLT_MAX;
        for (int m = 0; m < $(POOL_HEIGHT); ++m) {
            for (int n = 0; n < $(POOL_WIDTH); ++n) {
                int input_row = i * $(STRIDE) + m;
                int input_col = j * $(STRIDE) + n;
                float val = tensor_$(INPUT_NAME)[input_row * $(INPUT_WIDTH) + input_col];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        tensor_$(OUTPUT_NAME)[i * $(OUTPUT_WIDTH) + j] = max_val;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<$(OUTPUT_SIZE); i++) {
    printf("%f ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif