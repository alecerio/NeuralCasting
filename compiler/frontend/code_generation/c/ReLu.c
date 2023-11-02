// RELU OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_$OUTPUT_NAME[$INPUT_SIZE * $BATCH_SIZE];
#undef CONNECTED_OUTPUT
#endif

for(int i=0; i<$INPUT_SIZE; i++) {
    for(int j=0; j<$BATCH_SIZE; j++) {
        tensor_$OUTPUT_NAME[i * $BATCH_SIZE + j] = tensor_$INPUT_NAME[i * $BATCH_SIZE + j] > 0.0f ? tensor_$INPUT_NAME[i * $BATCH_SIZE + j] : 0.0f;
    }
}
