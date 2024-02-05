// MATMUL OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$N_ROWS_LEFT * $N_COLS_RIGHT];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<$N_ROWS_LEFT; i++) {
    for(int32_t j=0; j<$N_COLS_RIGHT; j++) {
        $OUTPUT_TYPE temp = 0.0f;
        for(int32_t k=0; k<$N_COLS_LEFT; k++) {
            int32_t index1 = i*$N_COLS_LEFT+k;
            int32_t index2 = k*$N_COLS_RIGHT+j;
            temp += tensor_$INPUT_NAME_1[index1] * tensor_$INPUT_NAME_2[index2];
        }
        tensor_$OUTPUT_NAME[i*$N_COLS_RIGHT + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");

for(int i=0; i<$N_ROWS_LEFT; i++) {
    for(int j=0; j<$N_COLS_RIGHT; j++) {
        printf("%f, ", tensor_$OUTPUT_NAME[i*$N_COLS_RIGHT+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif