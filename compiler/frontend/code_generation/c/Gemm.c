// GEMM OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_$OUTPUT_NAME[$OUTPUT_SIZE * $BATCH_SIZE] = {
    $OUTPUT_INIT
};
#undef CONNECTED_OUTPUT
#endif

for(int b=0; b<$BATCH_SIZE; b++) {
    for(int i=0; i<$OUTPUT_SIZE; i++) {
        float temp = 0.0f;
        for(int j=0; j<$INPUT_SIZE; j++) {
            temp += weight_$NAME[i * $INPUT_SIZE + j] * tensor_$INPUT_NAME[j + b * $BATCH_SIZE];
        }
        tensor_$OUTPUT_NAME[i + b * $BATCH_SIZE] = temp + bias_$NAME[i];
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$OUTPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
