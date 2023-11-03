// SIGMOID OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

for(int i=0; i<$INPUT_SIZE; i++) {
    float ex = exp(tensor_$INPUT_NAME[i]);
    tensor_$OUTPUT_NAME[i] = ex / (1.0f + ex);
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif