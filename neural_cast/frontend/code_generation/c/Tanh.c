// TANH OPERATOR $(NAME)

for(int i=0; i<TANH_$(NAME)_OUTPUT_SIZE; i++) {
    float y;
    NCAST_ACCESS_LUT(NCAST_TANH_LUT, tensor_$(INPUT_NAME)[i], y, NCAST_TANH_LUT_MINRANGE, NCAST_TANH_LUT_MAXRANGE, NCAST_TANH_LUT_UPPER, NCAST_TANH_LUT_LOWER, NCAST_TANH_LUT_SIZE)
    tensor_$(OUTPUT_NAME)[i] = y;
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<TANH_$(NAME)_OUTPUT_SIZE; i++) {
    printf("%f ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif