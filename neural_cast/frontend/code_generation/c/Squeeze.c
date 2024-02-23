// SQUEEZE OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float* tensor_$OUTPUT_NAME = tensor_$INPUT_NAME;
#undef CONNECTED_OUTPUT
#else
for(int i=0; i<$INPUT_SIZE; i++) {
    tensor_$OUTPUT_NAME[i] = tensor_$INPUT_NAME[i];
}
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif