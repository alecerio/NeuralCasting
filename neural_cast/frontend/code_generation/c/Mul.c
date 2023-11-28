// ELEMENT WISE MULTIPLICATION $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

$FOR_LOOPS_BEGIN
tensor_$OUTPUT_NAME[$INDEX] = tensor_$INPUT1_NAME[$INDEX] * tensor_$INPUT2_NAME[$INDEX];
$FOR_LOOPS_END

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif