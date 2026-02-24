
#include "ncast_lib.h"
#include <stdio.h>

int main() { 
    int8_t xq[5] = { -128,-64,-1,62,126 };

    int8_t yq[5];
    NC_UNSQUEEZE(xq,yq,5);

    NC_OUTTNS("/media/alessandro/SecondDisk1/ncast/NeuralCasting/other/tmp/out.txt",yq,5,"%d");

    return 0;
}
