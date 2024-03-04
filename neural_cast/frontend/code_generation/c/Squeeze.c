// SQUEEZE OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#ifdef CONNECTED_OUTPUT
float* tensor_$OUTPUT_NAME = tensor_$INPUT_NAME;
#undef CONNECTED_OUTPUT
#else

$OMP_PARALLEL_FOR
for(int i=0; i<$INPUT_SIZE; i++) {
    tensor_$OUTPUT_NAME[i] = tensor_$INPUT_NAME[i];
}
#endif

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_$NAME", $NFLOPS)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f ", tensor_$OUTPUT_NAME[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif