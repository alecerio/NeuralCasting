#define BENCHMARK(NAME, NFLOPS) \
    printf("BENCHMARK\n"); \
    printf("node name: %s\n", NAME); \
    neuralcasting_end_benchmark = omp_get_wtime(); \
    neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark; \
    printf("time: %f\n", neuralcasting_time_benchmark); \
    printf("nflops: %d\n", NFLOPS); \
    printf("nflops/time: %f\n", NFLOPS/neuralcasting_time_benchmark);