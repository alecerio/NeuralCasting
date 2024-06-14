// SOFTMAX OPERATOR $(NAME)

{
    float max_x; NCAST_REL_OP(tensor_$(INPUT_NAME), SOFTMAX_$(NAME)_OUTPUT_SIZE, max_x, >)

    float sum_exps = 0.0f;
    float softmax_exps_temp[SOFTMAX_$(NAME)_OUTPUT_SIZE];
    for(int i=0; i<SOFTMAX_$(NAME)_OUTPUT_SIZE; i++) {
        float input_val = tensor_$(INPUT_NAME)[i] - max_x;
        NCAST_ACCESS_LUT(NCAST_EXP_LUT, input_val, softmax_exps_temp[i], NCAST_EXP_MINRANGE, NCAST_EXP_MAXRANGE, NCAST_EXP_UPPER, NCAST_EXP_LOWER, SOFTMAX_OUTPUT_SIZE)
        sum_exps += softmax_exps_temp[i];
    }
    
    for(int i=0; i<SOFTMAX_$(NAME)_OUTPUT_SIZE; i++) {
        tensor_$(OUTPUT_NAME)[i] = softmax_exps_temp[i] / sum_exps;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<SOFTMAX_$(NAME)_OUTPUT_SIZE; i++) {
    printf("%f ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
