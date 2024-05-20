
// -------------------------------------------------------
//                         QLIN $(NAME)
// -------------------------------------------------------

NCAST_QUANT8(tensor_$(INPUT_NAME), tensor_$(OUTPUT_NAME), QLIN_$(NAME)_OUTPUT_SIZE, tensor_$(SCALING_FACTOR_NAME), tensor_$(ZERO_NAME))

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$(OUTPUT_SIZE); i++) {
    printf("%f ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif