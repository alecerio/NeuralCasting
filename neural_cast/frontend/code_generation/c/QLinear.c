
// -------------------------------------------------------
//                         QLIN $(NAME)
// -------------------------------------------------------

NCAST_QUANT8(tensor_$(INPUT_NAME), tensor_$(OUTPUT_NAME), QLIN_$(NAME)_OUTPUT_SIZE, tensor_$(SCALING_FACTOR_NAME)[0], tensor_$(ZERO_NAME)[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<QLIN_$(NAME)_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif