// GATHER OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#define AXIS ($AXIS)

#if AXIS == 0

int32_t out_index_$NAME = 0;
for(int32_t i=0; i<$IND_SIZE; i++) {
    $INDEX_TYPE row = tensor_$INDICES[i];
    for(int32_t col=0; col<$VAL_SIZE_COLS; col++) {
        tensor_$OUTPUT_NAME[out_index_$NAME] = tensor_$VALUES[row*$VAL_SIZE_COLS+col];
        out_index_$NAME++;
    }
}

#elif AXIS == 1

$VALUE_TYPE temp_$NAME[$VAL_SIZE_ROWS][$IND_SIZE];

for(int32_t i=0; i<$IND_SIZE; i++) {
    $INDEX_TYPE r = tensor_$INDICES[i];
    for(int32_t j=0; j<$VAL_SIZE_ROWS; j++) {
        temp_$NAME[j][i] = tensor_$VALUES[j*$VAL_SIZE_COLS+r];
    }
}

int32_t out_index_$NAME = 0;
for(int32_t i=0; i<$VAL_SIZE_ROWS; i++) {
    for(int32_t j=0; j<$IND_SIZE; j++) {
        tensor_$OUTPUT_NAME[out_index_$NAME] = temp_$NAME[i][j];
        out_index_$NAME++;
    }
}

#else

printf("Error: only axis 0 and 1 are supported\n");
return -1;

#endif

#undef AXIS

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_$NAME", $NFLOPS)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$OUTPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif