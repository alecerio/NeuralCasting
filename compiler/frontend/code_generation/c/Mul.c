// ELEMENT WISE MULTIPLICATION $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

for(int i=0; i<$INPUT_SIZE; i++) {
    tensor_$OUTPUT_NAME[i] = tensor_$INPUT1_NAME[i] * tensor_$INPUT2_NAME[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif