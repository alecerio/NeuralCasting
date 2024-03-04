// ELEMENT WISE SUBTRACTION $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
$OUTPUT_TYPE tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

$FOR_LOOPS_BEGIN
tensor_$OUTPUT_NAME[$INDEX_TOT] = tensor_$INPUT1_NAME[$INDEX_1] - tensor_$INPUT2_NAME[$INDEX_2];
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