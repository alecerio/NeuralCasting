// *****************************************************************************
// 	THIS CODE WAS AUTOMATICALLY GENERATED ON 2024-02-06 21:36:26
// *****************************************************************************

#include "dummy.h"

static float32_t tensor_W_irweight[100] = {-0.29285067, -0.28309342, -0.283437, 0.2184147, -0.17327106, 0.23833545, 0.15737678, -0.16843475, 0.039452285, -0.23379892, 0.09347235, -0.23728813, 0.03231042, -0.036525354, -0.28832975, -0.15785117, -0.05432258, -0.24212693, 0.089454226, -0.1700626, 0.09071208, 0.1797867, -0.10905286, 0.17766, 0.181695, 0.066193834, 0.11380683, -0.28692386, -0.31284297, 0.14399561, -0.23105296, 0.110620804, -0.04468919, -0.00018128625, -0.027087234, 0.14575554, 0.03823794, 0.25445077, -0.038562667, -0.059811532, -0.026537534, -0.27720642, -0.2047216, -0.2655133, 0.22523971, -0.23853712, -0.25604782, -0.19374898, -0.07910497, -0.01046096, 0.30686975, -0.060346007, -0.22044379, -0.044419732, -0.2940631, -0.28519928, -0.25815162, 0.09058956, 0.24008407, 0.13154073, -0.22694547, -0.13582031, -0.26084116, 0.093618535, -0.24713048, 0.30844703, -0.074258715, -0.16834636, 0.27650303, -0.03256989, 0.23728105, -0.26433405, -0.29433987, -0.02638565, -0.017528748, -0.014068062, -0.242656, -0.06046607, 0.23342161, 0.117494754, -0.3154811, 0.031570535, -0.21582533, -0.08713958, -0.19467576, 0.041631225, -0.09043496, -0.0025259822, 0.14830448, -0.20965596, 0.13579275, -0.28177622, -0.13711521, -0.062112954, 0.18313308, 0.182657, 0.20685266, -0.024268156, 0.14658794, 0.288582, };

static float32_t tensor_W_irbias[10] = {-0.005961675, 0.1872212, -0.31041563, -0.19367856, -0.1904922, -0.07967823, -0.29727846, 0.10728987, -0.2178294, 0.012341677, };

static float32_t tensor_W_hrbias[10] = {0.10894388, -0.2890948, -0.13467194, -0.176969, 0.21871996, -0.158391, 0.19545583, -0.31565973, 0.064912274, 0.11449646, };

static float32_t tensor_W_izweight[100] = {0.26773462, 0.2830944, -0.11292332, 0.30836493, -0.1216781, -0.041200608, -0.037691444, 0.15395953, 0.24741627, -0.17943898, -0.20465246, 0.17487553, -0.09595844, 0.07408886, 0.15236108, 0.06792037, -0.039417, -0.20041902, -0.271347, -0.06925395, 0.25470874, 0.15329541, -0.13501818, -0.036371246, -0.08432736, 0.18592845, -0.27326676, -0.2126715, 0.20647007, -0.193241, 0.057922747, 0.3110094, -0.11045, -0.10110718, 0.29912868, 0.21797019, -0.20803343, 0.014839575, 0.12063196, 0.30043328, -0.2695717, 0.048754204, 0.19033873, 0.1803296, 0.29886672, 0.20406431, 0.1616365, -0.10082505, 0.05424225, 0.13411002, 0.27230635, 0.07411253, -0.13930237, -0.29168528, 0.009723412, -0.28794932, 0.21034676, 0.029326377, 0.13636813, 0.093909405, 0.070644036, 0.008007809, -0.2534081, -0.2839206, -0.17447323, 0.047563534, -0.20263898, 0.016667629, -0.26207927, -0.18147682, 0.30195436, -0.25901625, -0.13359006, -0.123582415, 0.3029754, 0.2309717, 0.01654564, 0.0045695896, -0.09807398, -0.21713962, -0.26659802, -0.031439237, 0.092777245, 0.27653587, 0.04725777, -0.18731669, -0.21758369, 0.05769502, 0.09597654, 0.12990376, 0.22347483, -0.11740703, -0.21285072, -0.25682378, -0.30467686, -0.0593946, -0.0757003, -0.18802102, -0.007203386, 0.27951062, };

static float32_t tensor_W_izbias[10] = {-0.2753345, -0.19656628, 0.09027686, 0.003612531, -0.23013425, -0.26534796, -0.30111316, 0.20572752, 0.19971156, 0.19964121, };

static float32_t tensor_W_hzbias[10] = {-0.22879, -0.22368157, -0.104192436, -0.13579634, -0.048725553, 0.313393, -0.27706268, 0.24894837, -0.050921157, 0.2555579, };

static float32_t tensor_W_inweight[100] = {-0.28352275, -0.021244004, -0.09232714, 0.10711598, -0.06954821, -0.26585492, -0.22796318, -0.2249828, -0.2735831, 0.23901147, 0.0041153, -0.18492046, 0.3002962, -0.22171569, 0.11005365, -0.19892614, 0.13131492, -0.1620854, 0.17437848, -0.2219226, -0.29021242, 0.2011744, -0.006843226, 0.1393265, -0.10256708, 0.043846995, 0.22996876, -0.09204969, -0.14533368, -0.012180107, -0.13281463, 0.26092872, 0.25776923, -0.2455567, 0.06075461, 0.060983993, 0.11284287, -0.1302234, -0.2329589, 0.14526246, -0.09590175, -0.059672885, 0.16603412, 0.27695695, 0.2821291, -0.17253442, 0.15884344, -0.24226022, 0.25013885, -0.08988816, -0.036438197, -0.15686369, 0.21942998, 0.069455706, -0.17549519, -0.022120655, -0.11709328, 0.09448896, 0.24521607, -0.004044127, 0.15296115, -0.07632766, -0.21337901, -0.007563772, 0.27700034, 0.06128199, -0.095248036, -0.07520059, 0.07866553, -0.16313471, -0.16535161, 0.16448861, 0.05058467, -0.16272987, -0.2692948, 0.26830268, -0.31157744, 0.15137307, -0.15806517, 0.13822581, 0.22235107, -0.18206531, -0.18484747, 0.19598699, 0.2432855, -0.1580262, 0.104711115, 0.26227456, -0.026982285, 0.20518196, 0.08975336, 0.09024327, -0.10604318, 0.23068056, 0.30041134, 0.114750996, -0.015794786, 0.26461133, 0.16453241, -0.12103657, };

static float32_t tensor_W_inbias[10] = {-0.05123574, 0.20529875, -0.13628587, -0.29515278, -0.02967391, -0.20448181, 0.21894161, -0.085814066, 0.31063744, 0.23564017, };

static float32_t tensor_W_hnbias[10] = {0.24807929, 0.19395804, 0.14630449, 0.30052078, -0.16525133, -0.12784341, 0.013139653, 0.21787226, 0.01675226, 0.15555458, };

static float32_t tensor_onnxMatMul_38[100] = {-0.02493359, -0.17421903, 0.2543743, -0.21387517, -0.13044682, 0.06182325, 0.11693936, 0.2607059, 0.019040033, -0.04613918, -0.10596805, -0.0012558474, 0.09990101, -0.053809635, -0.0121771665, 0.14401823, -0.2599928, -0.19604711, 0.29882422, 0.05843528, -0.22086875, 0.26653472, -0.2903983, 0.044357266, 0.027859539, 0.19338015, -0.090145595, -0.06425582, 0.13796031, -0.15023763, 0.06445207, -0.14877461, 0.25752738, -0.25451282, 0.023290968, 0.19804949, -0.16278197, 0.2542234, 0.06361519, -0.25543204, -0.1983108, 0.22821452, -0.05101623, -0.06550039, -0.09847014, 0.025835836, 0.05119612, -0.20843086, -0.17135789, -0.046303727, 0.18507652, 0.0993984, 0.25345594, 0.18811217, -0.13176593, 0.26117215, 0.19015692, 0.100493506, -0.14231351, 0.1403133, -0.18070862, 0.31449878, 0.05354519, -0.08026981, -0.048567075, 0.24611348, -0.175161, 0.27276227, -0.023552964, 0.048772674, -0.23585813, -0.23749623, -0.20778081, 0.0033556616, 0.30493474, 0.015052489, -0.051958963, -0.06958267, 0.30641517, 0.1354243, -0.025192946, 0.05721321, 0.16136867, 0.29781207, -0.2813533, 0.11243955, -0.28443077, 0.13132226, 0.11930185, 0.18755923, -0.29999837, -0.07960408, 0.0689742, -0.098182134, -0.27435404, -0.045142237, 0.28293344, 0.22295777, 0.1392385, 0.098996356, };

static float32_t tensor_onnxMatMul_39[100] = {0.14940803, 0.2721928, 0.08615933, -0.28772035, 0.025435189, -0.08684897, -0.19699128, 0.030097853, 0.01756524, 0.25139597, 0.1878573, 0.009658648, -0.105114326, 0.15969563, -0.28356707, 0.10000065, 0.03880713, 0.28584853, -0.09789944, -0.3133719, -0.0775961, -0.12976345, 0.1580743, -0.07396279, -0.19389004, 0.092863046, 0.18929395, -0.3065708, 0.21667014, -0.13059457, -0.28028506, -0.31330746, -0.04392371, -0.24034844, 0.14832374, 0.15668982, -0.22752771, -0.29063362, -0.24391815, -0.30340195, -0.019754699, -0.24349594, 0.24082376, 0.21900657, 0.25618646, -0.0057666292, 0.089861736, 0.2717516, -0.0071457094, -0.15512076, -0.06619233, -0.106069274, 0.061511755, 0.063947946, 0.00028725332, 0.22844736, 0.16983464, 0.047102083, -0.25641924, 0.21500006, 0.07912005, 0.27489653, 0.01593913, -0.13158193, -0.25809166, -0.014605814, 0.051936496, 0.08058493, -0.04518559, -0.06298463, 0.01935537, 0.18757372, 0.24595554, 0.25565973, -0.28733554, -0.24971083, -0.11242993, -0.01561056, 0.17144206, -0.0014576787, -0.12680045, -0.050557263, -0.016244365, -0.19877087, 0.12910604, -0.021636358, 0.26481968, 0.3038045, 0.25200474, -0.26568323, 0.3075637, -0.11187579, -0.16105326, -0.3018375, 0.11250989, 0.19904397, -0.20930733, 0.08184695, -0.08142968, -0.21961565, };

static float32_t tensor_onnxMatMul_40[100] = {0.002375042, -0.11592326, -0.31588733, 0.20919102, 0.061421245, -0.23268582, 0.23168859, 0.26453552, -0.076105885, 0.3038859, -0.11993022, -0.14253521, 0.29150617, -0.28080344, -0.11085815, 0.09460251, 0.08741658, -0.08521664, -0.26094538, 0.2899317, -0.16092433, -0.14789939, -0.16302294, 0.31611648, 0.1743167, -0.1108543, 0.14763252, -0.042684373, -0.061557822, 0.10009463, 0.011205594, -0.121232174, 0.2607243, -0.07388363, -0.1827382, 0.1395592, -0.027361898, -0.3079718, -0.2695195, 0.034620736, 0.14344044, -0.14526062, -0.12738734, -0.31375343, 0.24523239, -0.31346184, -0.0386914, -0.07930416, -0.28435147, -0.11445978, -0.027980585, -0.14397962, -0.13170682, 0.020312127, -0.13493058, -0.14511472, -0.021908721, 0.10546792, -0.26398584, 0.08819133, -0.21852326, -0.21913938, -0.29926452, -0.15077648, 0.08319671, -0.1608731, 0.17833139, -0.013285165, -0.2703106, -0.148053, 0.13292089, 0.21486583, -0.019770041, 0.15554002, -0.14554077, -0.08194357, 0.26325417, -0.09148283, -0.29043427, 0.1513105, 0.007202255, -0.2872882, -0.046972554, -0.24874106, -0.05720371, 0.21345289, 0.11403384, 0.209595, -0.13872145, 0.18990181, -0.10438108, -0.18107516, -0.10515055, -0.003967112, 0.20165266, -0.26539588, 0.09659319, 0.27118045, 0.2936415, -0.20628235, };

// CONSTANT DECLARATION Constant

static float32_t tensor_Constant_output_0[1] = {
    1.0, 
};
void run_inference(float32_t* tensor_onnxGemm_0, float32_t* tensor_onnxMatMul_1, float32_t* tensor_37) {






// MATMUL OPERATOR W_hrMatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hrMatMul_output_0[10 * 10];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<10; i++) {
    for(int32_t j=0; j<10; j++) {
        float32_t temp = 0.0f;
        for(int32_t k=0; k<10; k++) {
            int32_t index1 = i*10+k;
            int32_t index2 = k*10+j;
            temp += tensor_onnxMatMul_1[index1] * tensor_onnxMatMul_38[index2];
        }
        tensor_W_hrMatMul_output_0[i*10 + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hrMatMul -----------------\n");

for(int i=0; i<10; i++) {
    for(int j=0; j<10; j++) {
        printf("%f, ", tensor_W_hrMatMul_output_0[i*10+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// GEMM OPERATOR W_irGemm

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_irGemm_output_0[10];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<10; i++) {
    float32_t temp = 0.0f;
    for(int32_t j=0; j<10; j++) {
        temp += tensor_W_irweight[i * 10 + j] * tensor_onnxGemm_0[j];
    }
    tensor_W_irGemm_output_0[i] = temp + tensor_W_irbias[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_irGemm -----------------\n");
for(int i=0; i<10; i++) {
    printf("%f ", tensor_W_irGemm_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// ELEMENT WISE ADDITION W_hrAdd

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hrAdd_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_W_hrAdd_output_0[i1*10 + i2*1] = tensor_W_hrbias[i2*1] + tensor_W_hrMatMul_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hrAdd -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_W_hrAdd_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Add_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Add_output_0[i1*10 + i2*1] = tensor_W_irGemm_output_0[i2*1] + tensor_W_hrAdd_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// MATMUL OPERATOR W_hzMatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hzMatMul_output_0[10 * 10];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<10; i++) {
    for(int32_t j=0; j<10; j++) {
        float32_t temp = 0.0f;
        for(int32_t k=0; k<10; k++) {
            int32_t index1 = i*10+k;
            int32_t index2 = k*10+j;
            temp += tensor_onnxMatMul_1[index1] * tensor_onnxMatMul_39[index2];
        }
        tensor_W_hzMatMul_output_0[i*10 + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hzMatMul -----------------\n");

for(int i=0; i<10; i++) {
    for(int j=0; j<10; j++) {
        printf("%f, ", tensor_W_hzMatMul_output_0[i*10+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// GEMM OPERATOR W_izGemm

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_izGemm_output_0[10];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<10; i++) {
    float32_t temp = 0.0f;
    for(int32_t j=0; j<10; j++) {
        temp += tensor_W_izweight[i * 10 + j] * tensor_onnxGemm_0[j];
    }
    tensor_W_izGemm_output_0[i] = temp + tensor_W_izbias[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_izGemm -----------------\n");
for(int i=0; i<10; i++) {
    printf("%f ", tensor_W_izGemm_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// ELEMENT WISE ADDITION W_hzAdd

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hzAdd_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_W_hzAdd_output_0[i1*10 + i2*1] = tensor_W_hzbias[i2*1] + tensor_W_hzMatMul_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hzAdd -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_W_hzAdd_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add_1

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Add_1_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Add_1_output_0[i1*10 + i2*1] = tensor_W_izGemm_output_0[i2*1] + tensor_W_hzAdd_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_1 -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Add_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// MATMUL OPERATOR W_hnMatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hnMatMul_output_0[10 * 10];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<10; i++) {
    for(int32_t j=0; j<10; j++) {
        float32_t temp = 0.0f;
        for(int32_t k=0; k<10; k++) {
            int32_t index1 = i*10+k;
            int32_t index2 = k*10+j;
            temp += tensor_onnxMatMul_1[index1] * tensor_onnxMatMul_40[index2];
        }
        tensor_W_hnMatMul_output_0[i*10 + j] = temp;
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hnMatMul -----------------\n");

for(int i=0; i<10; i++) {
    for(int j=0; j<10; j++) {
        printf("%f, ", tensor_W_hnMatMul_output_0[i*10+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// SIGMOID OPERATOR Sigmoid

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sigmoid_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

float32_t ex = exp(tensor_Add_output_0[i1*10 + i2*1]);
tensor_Sigmoid_output_0[i1*10 + i2*1] = ex / (1.0f + ex);
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Sigmoid_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION W_hnAdd

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_hnAdd_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_W_hnAdd_output_0[i1*10 + i2*1] = tensor_W_hnbias[i2*1] + tensor_W_hnMatMul_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_hnAdd -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_W_hnAdd_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// GEMM OPERATOR W_inGemm

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_W_inGemm_output_0[10];
#undef CONNECTED_OUTPUT
#endif

for(int32_t i=0; i<10; i++) {
    float32_t temp = 0.0f;
    for(int32_t j=0; j<10; j++) {
        temp += tensor_W_inweight[i * 10 + j] * tensor_onnxGemm_0[j];
    }
    tensor_W_inGemm_output_0[i] = temp + tensor_W_inbias[i];
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT W_inGemm -----------------\n");
for(int i=0; i<10; i++) {
    printf("%f ", tensor_W_inGemm_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// ELEMENT WISE MULTIPLICATION Mul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Mul_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Mul_output_0[i1*10 + i2*1] = tensor_Sigmoid_output_0[i1*10 + i2*1] * tensor_W_hnAdd_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Mul_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add_2

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Add_2_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Add_2_output_0[i1*10 + i2*1] = tensor_W_inGemm_output_0[i2*1] + tensor_Mul_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_2 -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Add_2_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// SIGMOID OPERATOR Sigmoid_1

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sigmoid_1_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

float32_t ex = exp(tensor_Add_1_output_0[i1*10 + i2*1]);
tensor_Sigmoid_1_output_0[i1*10 + i2*1] = ex / (1.0f + ex);
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sigmoid_1 -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Sigmoid_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE SUBTRACTION Sub

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Sub_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Sub_output_0[i1*10 + i2*1] = tensor_Constant_output_0[0] - tensor_Sigmoid_1_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Sub -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Sub_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TANH OPERATOR /Tanh

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Tanh_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

float32_t ex = exp(tensor_Add_2_output_0[i1*10 + i2*1]);
float32_t emx = exp(-tensor_Add_2_output_0[i1*10 + i2*1]);
tensor_Tanh_output_0[i1*10 + i2*1] = (ex - emx) / (ex + emx);
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /Tanh -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Tanh_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE MULTIPLICATION Mul_1

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Mul_1_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Mul_1_output_0[i1*10 + i2*1] = tensor_Sub_output_0[i1*10 + i2*1] * tensor_Tanh_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_1 -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Mul_1_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE MULTIPLICATION Mul_2

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_Mul_2_output_0[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_Mul_2_output_0[i1*10 + i2*1] = tensor_Sigmoid_1_output_0[i1*10 + i2*1] * tensor_onnxMatMul_1[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Mul_2 -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_Mul_2_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION Add_3



#ifdef CONNECTED_OUTPUT
float32_t tensor_37[100];
#undef CONNECTED_OUTPUT
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<10; i1++) {
for(int i2=0; i2<10; i2++) {

tensor_37[i1*10 + i2*1] = tensor_Mul_1_output_0[i1*10 + i2*1] + tensor_Mul_2_output_0[i1*10 + i2*1];
}
}
}


#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT Add_3 -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_37[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
}