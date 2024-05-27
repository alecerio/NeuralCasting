
// QLINEARSIGMOID $(NAME)

NCAST_DQUANT8(tensor_$(INPUT_NAME), tensor_$(NAME)_qsigmoid_temp, QSIGMOID_$(NAME)_OUTPUT_SIZE, tensor_$(QSIGMOID_SX)[0], tensor_$(QSIGMOID_ZX)[0])

for(int i=0; i<QSIGMOID_$(NAME)_OUTPUT_SIZE; i++) {
    float y;
    NCAST_ACCESS_LUT(NCAST_SIGMOID_LUT, tensor_$(NAME)_qsigmoid_temp[i], y, NCAST_SIGMOID_LUT_MINRANGE, NCAST_SIGMOID_LUT_MAXRANGE, NCAST_SIGMOID_LUT_UPPER, NCAST_SIGMOID_LUT_LOWER, NCAST_SIGMOID_LUT_SIZE)
    tensor_$(NAME)_qsigmoid_temp[i] = y;
}

NCAST_QUANT8(tensor_$(NAME)_qsigmoid_temp, tensor_$(OUTPUT_NAME), QSIGMOID_$(NAME)_OUTPUT_SIZE, tensor_$(QSIGMOID_SY)[0], tensor_$(QSIGMOID_ZY)[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<QLINEARMUL_$(NAME)_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
