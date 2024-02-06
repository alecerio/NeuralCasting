// *****************************************************************************
// 	THIS CODE WAS AUTOMATICALLY GENERATED ON 2024-02-06 18:04:14
// *****************************************************************************

#include "dummy.h"

static float32_t tensor_W_irweight[12] = {0.46674994, -0.012937614, 0.09054168, -0.19359227, -0.06476433, 0.20658116, -0.1462408, -0.35687667, -0.30548102, 0.3484161, -0.47218513, 0.48714018, };

static float32_t tensor_W_irbias[4] = {-0.28864518, 0.43087587, -0.008655221, -0.10712354, };

static float32_t tensor_W_hrbias[4] = {-0.4636972, 0.0007826686, 0.3267094, 0.27961892, };

static float32_t tensor_W_izweight[12] = {-0.30469105, 0.05652557, 0.49149525, -0.30121252, 0.1995443, 0.11192054, 0.018884068, -0.5069513, 0.29635936, 0.2714757, 0.15862244, -0.06227064, };

static float32_t tensor_W_izbias[4] = {-0.06286619, -0.18161856, -0.30184647, 0.41956475, };

static float32_t tensor_W_hzbias[4] = {0.3989945, 0.013421118, -0.41111135, 0.4008327, };

static float32_t tensor_W_inweight[12] = {-0.21213889, -0.024935966, 0.25322402, 0.2583493, -0.30564213, 0.30236357, -0.5374175, 0.030127611, -0.56708723, -0.3198601, 0.556541, 0.43072623, };

static float32_t tensor_W_inbias[4] = {-0.295387, -0.27431998, -0.036316883, 0.22921449, };

static float32_t tensor_W_hnbias[4] = {0.06980789, -0.47165936, 0.26637852, 0.2540927, };

static float32_t tensor_onnxMatMul_38[16] = {-0.46672726, -0.03609824, 0.44504708, 0.16678083, -0.3910575, -0.063376725, -0.37558955, 0.21092665, 0.45582968, 0.23471081, -0.058329523, 0.047926724, -0.35623407, -0.4798109, -0.30336297, -0.28352922, };

static float32_t tensor_onnxMatMul_39[16] = {0.4543351, -0.040812492, 0.05735624, 0.0588724, 0.334311, -0.24387366, -0.0031642914, 0.42479157, 0.24996674, 0.052757025, 0.06375873, 0.3743441, 0.1258493, 0.48212075, -0.14819872, 0.1751042, };

static float32_t tensor_onnxMatMul_40[16] = {-0.32797217, 0.08479667, -0.47155553, 0.06700957, 0.075098336, -0.13785875, -0.02381897, 0.32947582, 0.35899687, -0.48313314, 0.099408865, 0.44990313, -0.062055767, 0.04492426, -0.29394072, -0.024787009, };

// CONSTANT DECLARATION Constant

static float32_t tensor_Constant_output_0[1] = {
    1.0, 
};
void run_inference(float32_t* tensor_onnxGemm_0, float32_t* tensor_onnxMatMul_1, float32_t* tensor_37) {






// MATMUL OPERATOR W_hrMatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hrMatMul_output_0[3 * 4];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<3; i++) {
    for(int32_t j=0; j<4; j++) {
        float32_t temp = 0.0f;
        for(int32_t k=0; k<4; k++) {
            int32_t index1 = i*4+k;
            int32_t index2 = k*4+j;
            temp += tensor_onnxMatMul_1[index1] * tensor_onnxMatMul_38[index2];
        }
        tensor_W_hrMatMul_output_0[i*4 + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hrMatMul -----------------\n");

for(int i=0; i<3; i++) {
    for(int j=0; j<4; j++) {
        printf("%f, ", tensor_W_hrMatMul_output_0[i*4+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// GEMM OPERATOR W_irGemm

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_irGemm_output_0[4];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<4; i++) {
    float32_t temp = 0.0f;
    for(int32_t j=0; j<3; j++) {
        temp += tensor_W_irweight[i * 3 + j] * tensor_onnxGemm_0[j];
    }
    tensor_W_irGemm_output_0[i] = temp + tensor_W_irbias[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_irGemm -----------------\n");
for(int i=0; i<4; i++) {
    printf("%f ", tensor_W_irGemm_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// ELEMENT WISE ADDITION W_hrAdd

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hrAdd_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_W_hrAdd_output_0[i1*4 + i2*1] = tensor_W_hrbias[i2*1] + tensor_W_hrMatMul_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hrAdd -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_W_hrAdd_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Add_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Add_output_0[i1*4 + i2*1] = tensor_W_irGemm_output_0[i2*1] + tensor_W_hrAdd_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// MATMUL OPERATOR W_hzMatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hzMatMul_output_0[3 * 4];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<3; i++) {
    for(int32_t j=0; j<4; j++) {
        float32_t temp = 0.0f;
        for(int32_t k=0; k<4; k++) {
            int32_t index1 = i*4+k;
            int32_t index2 = k*4+j;
            temp += tensor_onnxMatMul_1[index1] * tensor_onnxMatMul_39[index2];
        }
        tensor_W_hzMatMul_output_0[i*4 + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hzMatMul -----------------\n");

for(int i=0; i<3; i++) {
    for(int j=0; j<4; j++) {
        printf("%f, ", tensor_W_hzMatMul_output_0[i*4+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// GEMM OPERATOR W_izGemm

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_izGemm_output_0[4];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<4; i++) {
    float32_t temp = 0.0f;
    for(int32_t j=0; j<3; j++) {
        temp += tensor_W_izweight[i * 3 + j] * tensor_onnxGemm_0[j];
    }
    tensor_W_izGemm_output_0[i] = temp + tensor_W_izbias[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_izGemm -----------------\n");
for(int i=0; i<4; i++) {
    printf("%f ", tensor_W_izGemm_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// ELEMENT WISE ADDITION W_hzAdd

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hzAdd_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_W_hzAdd_output_0[i1*4 + i2*1] = tensor_W_hzbias[i2*1] + tensor_W_hzMatMul_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hzAdd -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_W_hzAdd_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add_1

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Add_1_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Add_1_output_0[i1*4 + i2*1] = tensor_W_izGemm_output_0[i2*1] + tensor_W_hzAdd_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_1 -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Add_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// MATMUL OPERATOR W_hnMatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hnMatMul_output_0[3 * 4];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<3; i++) {
    for(int32_t j=0; j<4; j++) {
        float32_t temp = 0.0f;
        for(int32_t k=0; k<4; k++) {
            int32_t index1 = i*4+k;
            int32_t index2 = k*4+j;
            temp += tensor_onnxMatMul_1[index1] * tensor_onnxMatMul_40[index2];
        }
        tensor_W_hnMatMul_output_0[i*4 + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hnMatMul -----------------\n");

for(int i=0; i<3; i++) {
    for(int j=0; j<4; j++) {
        printf("%f, ", tensor_W_hnMatMul_output_0[i*4+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// SIGMOID OPERATOR Sigmoid

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sigmoid_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

float32_t ex = exp(tensor_Add_output_0[i1*4 + i2*1]);
tensor_Sigmoid_output_0[i1*4 + i2*1] = ex / (1.0f + ex);
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Sigmoid_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION W_hnAdd

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hnAdd_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_W_hnAdd_output_0[i1*4 + i2*1] = tensor_W_hnbias[i2*1] + tensor_W_hnMatMul_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hnAdd -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_W_hnAdd_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// GEMM OPERATOR W_inGemm

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_inGemm_output_0[4];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<4; i++) {
    float32_t temp = 0.0f;
    for(int32_t j=0; j<3; j++) {
        temp += tensor_W_inweight[i * 3 + j] * tensor_onnxGemm_0[j];
    }
    tensor_W_inGemm_output_0[i] = temp + tensor_W_inbias[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_inGemm -----------------\n");
for(int i=0; i<4; i++) {
    printf("%f ", tensor_W_inGemm_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// ELEMENT WISE MULTIPLICATION Mul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Mul_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Mul_output_0[i1*4 + i2*1] = tensor_Sigmoid_output_0[i1*4 + i2*1] * tensor_W_hnAdd_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Mul_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add_2

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Add_2_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Add_2_output_0[i1*4 + i2*1] = tensor_W_inGemm_output_0[i2*1] + tensor_Mul_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_2 -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Add_2_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// SIGMOID OPERATOR Sigmoid_1

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sigmoid_1_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

float32_t ex = exp(tensor_Add_1_output_0[i1*4 + i2*1]);
tensor_Sigmoid_1_output_0[i1*4 + i2*1] = ex / (1.0f + ex);
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid_1 -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Sigmoid_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE SUBTRACTION Sub

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sub_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Sub_output_0[i1*4 + i2*1] = tensor_Constant_output_0[0] - tensor_Sigmoid_1_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sub -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Sub_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TANH OPERATOR /Tanh

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Tanh_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

float32_t ex = exp(tensor_Add_2_output_0[i1*4 + i2*1]);
float32_t emx = exp(-tensor_Add_2_output_0[i1*4 + i2*1]);
tensor_Tanh_output_0[i1*4 + i2*1] = (ex - emx) / (ex + emx);
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /Tanh -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Tanh_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE MULTIPLICATION Mul_1

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Mul_1_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Mul_1_output_0[i1*4 + i2*1] = tensor_Sub_output_0[i1*4 + i2*1] * tensor_Tanh_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_1 -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Mul_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE MULTIPLICATION Mul_2

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Mul_2_output_0[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_Mul_2_output_0[i1*4 + i2*1] = tensor_Sigmoid_1_output_0[i1*4 + i2*1] * tensor_onnxMatMul_1[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_2 -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_Mul_2_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add_3



#ifdef CONNECTED_OUTPUT
float32_t tensor_37[12];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<3; i1++) {
for(int i2=0; i2<4; i2++) {

tensor_37[i1*4 + i2*1] = tensor_Mul_1_output_0[i1*4 + i2*1] + tensor_Mul_2_output_0[i1*4 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_3 -----------------\n");
for(int i=0; i<12; i++) {
    printf("%f ", tensor_37[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
}