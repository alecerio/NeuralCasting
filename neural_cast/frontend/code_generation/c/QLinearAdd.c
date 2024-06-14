
// QLINEARADD OPERATOR $(NAME)

{
float32_t say = tensor_$(INPUT_SA)[0] / tensor_$(INPUT_SY)[0];
float32_t sby = tensor_$(INPUT_SB)[0] / tensor_$(INPUT_SY)[0];

$(FOR_LOOPS_BEGIN)
int32_t qaz_i = (int32_t)tensor_$(INPUT1_NAME)[$(INDEX1)] - (int32_t)tensor_$(INPUT_ZA)[0];
int32_t qbz_i = (int32_t)tensor_$(INPUT2_NAME)[$(INDEX2)] - (int32_t)tensor_$(INPUT_ZB)[0];
int32_t res_i = ((int32_t)NCAST_ROUND(say*qaz_i)) + ((int32_t)NCAST_ROUND(sby*qbz_i)) + (int32_t)tensor_$(INPUT_ZY)[0];
NCAST_CLIP_INT8(res_i)
tensor_$(OUTPUT_NAME)[$(INDEX_TOT)] = (int8_t)res_i;
$(FOR_LOOPS_END)
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<QLINEARADD_$(NAME)_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
