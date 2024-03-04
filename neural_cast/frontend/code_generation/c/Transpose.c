// TRANSPOSE $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

$FOR_LOOPS_BEGIN
tensor_$OUTPUT_NAME[$INDEX_OUT] = tensor_$INPUT_NAME[$INDEX_IN];
$FOR_LOOPS_END

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