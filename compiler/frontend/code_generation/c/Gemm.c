// *********************************************************************************
//                CODE AUTOMATICALLY GENERATED FOR GEMM OPERATOR
// *********************************************************************************

float weight_$NAME[$OUTPUT_SIZE][$INPUT_SIZE] = {
    $WEIGHTS
};

float bias_$NAME[$OUTPUT_SIZE] = {
    $BIAS
};

float tensor_$OUTPUT_NAME[$OUTPUT_SIZE][$BATCH_SIZE] = {
    $OUTPUT_INIT
};

for(int b=0; b<$BATCH_SIZE; b++) {
    for(int i=0; i<$OUTPUT_SIZE; i++) {
        float temp = 0.0f;
        for(int j=0; j<$INPUT_SIZE; j++) {
            temp += weight_$NAME[i][j] * tensor_$INPUT_NAME[j][b];
        }
        tensor_$OUTPUT_NAME[i][b] = temp + bias_$NAME[i];
    }
}