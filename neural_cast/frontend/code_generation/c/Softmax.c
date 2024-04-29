// SOFTMAX OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

{
    float maximum = tensor_$INPUT_NAME[0];
    $FOR_LOOPS_BEGIN
        if(tensor_$INPUT_NAME[$INDEX] > maximum)
            maximum = tensor_$INPUT_NAME[$INDEX];
    $FOR_LOOPS_END
    
    float sum = 0.0f;
    float exps[$INPUT_SIZE];
    $FOR_LOOPS_BEGIN
        exps[$INDEX] = exp(tensor_$INPUT_NAME[$INDEX]-maximum);
        sum += exps[$INDEX];
    $FOR_LOOPS_END

    $FOR_LOOPS_BEGIN
        tensor_$OUTPUT_NAME[$INDEX] = exps[$INDEX] / sum;
    $FOR_LOOPS_END
    
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif