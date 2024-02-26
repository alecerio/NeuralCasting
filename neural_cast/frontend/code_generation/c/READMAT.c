#define READMAT(OUT_MATRIX, SIZE, MATRIX_NAME, TYPE) \
    OUT_MATRIX = (TYPE*) malloc(SIZE * sizeof(TYPE)); \
    fp = fopen(MATRIX_NAME, "rb"); \
    if (fp != NULL) { \
        for (int i = 0; i < SIZE; i++) { \
            int res; \
            TYPE x; \
            res = fread(&x, sizeof(TYPE), 1, fp); \
            OUT_MATRIX[i] = x; \
        } \
        fclose(fp); \
    }