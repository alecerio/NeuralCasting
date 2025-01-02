#ifndef __NCAST_SUB__
#define __NCAST_SUB__

#define NCAST_SUB(A, B, Y, SIZE) \
for(int i=0; i<SIZE; i++) { \
    Y[i] = 1 - B[i]; \
}

#endif // __NCAST_SUB__
