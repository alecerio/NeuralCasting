
// FLATTEN OPERATOR $(NAME)

#ifdef CONNECTED_OUTPUT
tensor_$(OUTPUT_NAME) = tensor_$(INPUT_NAME);
#else
memcpy(tensor_$(OUTPUT_NAME), tensor_$(INPUT_NAME), $(OUTPUT_SIZE) * sizeof(float));
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");

for(int i=0; i<$(OUTPUT_SIZE); i++) {
    printf("%f, ", tensor_$(OUTPUT_NAME)[i]);
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
