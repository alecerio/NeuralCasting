
// OPERATOR GLOBAL AVERAGE POOL $(NAME)

for (int n = 0; n < $(N); ++n) {
    for (int c = 0; c < $(C); ++c) {
        float sum = 0.0;
        for (int h = 0; h < $(H); ++h) {
            for (int w = 0; w < $(W); ++w) {
                sum += tensor_$(INPUT_NAME)[((n * $(C) + c) * $(H) + h) * $(W) + w];
            }
        }
        tensor_$(OUTPUT_NAME)[n * $(C) + c] = sum / ($(H) * $(W));
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");

for(int i=0; i<$(OUTPUT_SIZE); i++) {
    printf("%f, ", tensor_$(OUTPUT_NAME)[i]);
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
