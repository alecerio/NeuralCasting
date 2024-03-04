// GEMM OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

$OMP_PARALLEL_FOR
for(int32_t i=0; i<$OUTPUT_SIZE; i++) {
    $OUTPUT_TYPE temp = 0.0f;
$OMP_REDUCTION
    for(int32_t j=0; j<$INPUT_SIZE; j++) {
        temp += tensor_$INPUT_NAME_W[i * $INPUT_SIZE + j] * tensor_$INPUT_NAME_X[j];
    }
    tensor_$OUTPUT_NAME[i] = temp + tensor_$INPUT_NAME_B[i];
}

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
