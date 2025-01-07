#ifndef __NCAST_SQUEEZE__
#define __NCAST_SQUEEZE__

#include <stdint.h>
#include <stdio.h>
#include "utils.h"

#define SQUEEZE(INPUT,OUTPUT,SIZE) \
{ \
    for(int i=0; i<SIZE; i++) { \
        OUTPUT[i] = INPUT[i]; \
    } \
}

#endif // __NCAST_SQUEEZE__

