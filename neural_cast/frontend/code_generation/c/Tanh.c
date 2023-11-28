// TANH OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

$FOR_LOOPS_BEGIN
$OUTPUT_TYPE ex = exp(tensor_$INPUT_NAME[$INDEX]);
$OUTPUT_TYPE emx = exp(-tensor_$INPUT_NAME[$INDEX]);
tensor_$OUTPUT_NAME[$INDEX] = (ex - emx) / (ex + emx);
$FOR_LOOPS_END

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif