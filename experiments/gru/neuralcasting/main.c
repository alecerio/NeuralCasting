#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "dummy.h"

int main(int argc, char* argv[]) {
    const int INPUT_SIZE = atoi(argv[1]);
    const int HIDDEN_SIZE = atoi(argv[2]);
    const int NUM_EXPERIMENTS = atoi(argv[3]);

    float input[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++)
        input[i] = 1.0f;

    float hidden[INPUT_SIZE * HIDDEN_SIZE];
    for(int i=0; i<INPUT_SIZE * HIDDEN_SIZE; i++)
        hidden[i] = 0.0f;

    float output[INPUT_SIZE*HIDDEN_SIZE];

    double total_time = 0.0;
    double min_time = 10000.0;
    double max_time = -100.0;
    float* times = malloc(sizeof(float) * NUM_EXPERIMENTS);

    for(int i=0; i<NUM_EXPERIMENTS; i++) {
        clock_t start = clock();
        run_inference(input, hidden, output);
        clock_t end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC;

        // update total time
        total_time += time;

        // update min time
        if(time < min_time)
            min_time = time;
        
        // update max time
        if(time > max_time)
            max_time = time;
        
        // update times
        times[i] = time;
    }

    // compute average time
    double avg_time = total_time / (double) NUM_EXPERIMENTS;

    // compute standard deviation
    double stddev_time = 0.0;
    for(int i=0; i<NUM_EXPERIMENTS; i++) {
        stddev_time += (times[i] - avg_time) * (times[i] - avg_time);
    }
    stddev_time /= (NUM_EXPERIMENTS - 1);
    stddev_time = sqrtf(stddev_time);

    printf("%d,%d,%.20f,%.20f,%.20f,%.20f\n", INPUT_SIZE, HIDDEN_SIZE, avg_time, min_time, max_time, stddev_time);

    free(times);

    return 0;
}