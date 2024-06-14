// QGEMM OPERATOR $(NAME)

{
    int32_t temp;
    for(int i=0; i<QGEMM_$(NAME)_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_$(NAME)_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_$(INPUT_NAME_QW)[i*QGEMM_$(NAME)_INPUT_SIZE+j] * ((int32_t)tensor_$(INPUT_NAME)[j]-(int32_t)tensor_$(INPUT_NAME_ZX)[0]);        
        }
        temp += tensor_$(INPUT_NAME_QB)[i];

        float sw = tensor_$(INPUT_NAME_SW)[0];
        float sx = tensor_$(INPUT_NAME_SX)[0];
        float sy = tensor_$(INPUT_NAME_SY)[0];
        float swxy = (sw*sx)/sy;

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_$(INPUT_NAME_ZY)[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_$(OUTPUT_NAME)[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<QGEMM_$(NAME)_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif