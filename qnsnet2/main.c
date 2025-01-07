#include "nsnet2.h"

int main() {
    float32_t h1[400];
    float32_t h2[400];
    for(int i=0; i<400; i++) { h1[i] = 0.0f; h2[i] = 0.0f; }
    float32_t in_noisy[257];
    for(int i=0; i<257; i++) in_noisy[i] = 1.0f;
    run_inference(in_noisy, h1, h2);
    return 0;
}