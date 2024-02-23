// GRU OPERATOR $NAME

$DEFINE_CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_$OUTPUT_NAME[$INPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

$DEFINE_CONNECTED_HIDDEN

#ifdef CONNECTED_HIDDEN
float tensor_$OUTPUT_HIDDEN_NAME[$HIDDEN_SIZE];
#undef CONNECTED_HIDDEN
#endif

{
    // a = W_ir @ x + b_ir
    float a[$HIDDEN_SIZE];
    for(int i=0; i< $HIDDEN_SIZE; i++) {
        a[i] = 0.0f;
        for(int j=0; j<$INPUT_SIZE; j++) {
            a[i] += tensor_$W_NAME[($HIDDEN_SIZE*$INPUT_SIZE) + i*$INPUT_SIZE + j] * tensor_$INPUT_NAME[j];
        }
        a[i] += tensor_$B_NAME[$HIDDEN_SIZE + i];
    }

    // b = W_hr @ h1 + b_hr
    float b[$HIDDEN_SIZE];
    for(int i=0; i< $HIDDEN_SIZE; i++) {
        b[i] = 0.0f;
        for(int j=0; j<$HIDDEN_SIZE; j++) {
            b[i] += tensor_$R_NAME[($HIDDEN_SIZE*$HIDDEN_SIZE) + i*$HIDDEN_SIZE + j] * tensor_$INPUT_HIDDEN_NAME[j];
        }
        b[i] += tensor_$B_NAME[4*$HIDDEN_SIZE + i];
    }

    // c = W_iz @ x + b_iz
    float c[$HIDDEN_SIZE];
    for(int i=0; i< $HIDDEN_SIZE; i++) {
        c[i] = 0.0f;
        for(int j=0; j<$INPUT_SIZE; j++) {
            c[i] += tensor_$W_NAME[i*$INPUT_SIZE + j] * tensor_$INPUT_NAME[j];
        }
        c[i] += tensor_$B_NAME[i];
    }

    // d = W_hz @ h1 + b_hz
    float d[$HIDDEN_SIZE];
    for(int i=0; i< $HIDDEN_SIZE; i++) {
        d[i] = 0.0f;
        for(int j=0; j<$HIDDEN_SIZE; j++) {
            d[i] += tensor_$R_NAME[i*$HIDDEN_SIZE + j] * tensor_$INPUT_HIDDEN_NAME[j];
        }
        d[i] += tensor_$B_NAME[3*$HIDDEN_SIZE + i];
    }

    // e = W_in @ x + b_in
    float e[$HIDDEN_SIZE];
    for(int i=0; i< $HIDDEN_SIZE; i++) {
        e[i] = 0.0f;
        for(int j=0; j<$INPUT_SIZE; j++) {
            e[i] += tensor_$W_NAME[(2*$HIDDEN_SIZE*$INPUT_SIZE) + i*$HIDDEN_SIZE + j] * tensor_$INPUT_NAME[j];
        }
        e[i] += tensor_$B_NAME[2*$HIDDEN_SIZE + i];
    }

    // f = W_hn @ h1 + b_hn
    float f[$HIDDEN_SIZE];
    for(int i=0; i< $HIDDEN_SIZE; i++) {
        f[i] = 0.0f;
        for(int j=0; j<$HIDDEN_SIZE; j++) {
            f[i] += tensor_$R_NAME[(2*$HIDDEN_SIZE*$HIDDEN_SIZE) + i*$HIDDEN_SIZE + j] * tensor_$INPUT_HIDDEN_NAME[j];
        }
        f[i] += tensor_$B_NAME[5*$HIDDEN_SIZE + i];
    }

    // r = sigmoid(a + b)
    float r[$HIDDEN_SIZE];
    for(int i=0; i<$HIDDEN_SIZE; i++) {
        float s = a[i] + b[i];
        r[i] = 1.0f / (1.0f + expf(-s));
    }

    // z = sigmoid(c + d)
    float z[$HIDDEN_SIZE];
    for(int i=0; i<$HIDDEN_SIZE; i++) {
        float s = c[i] + d[i];
        z[i] = 1.0f / (1.0f + expf(-s));
    }

    // n = tanh(e + r*f)
    float n[$HIDDEN_SIZE];
    for(int i=0; i<$HIDDEN_SIZE; i++) {
        n[i] = tanh(e[i] + r[i] * f[i]);
    }

    // hn = (1-z) * n + z * h1
    float hn[$HIDDEN_SIZE];
    for(int i=0; i<$HIDDEN_SIZE; i++) {
        tensor_$OUTPUT_HIDDEN_NAME[i] = (1.0f - z[i]) * n[i] + z[i] * tensor_$INPUT_HIDDEN_NAME[i];
        tensor_$OUTPUT_NAME[i] = tensor_$OUTPUT_HIDDEN_NAME[i];
    }
}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");

for(int i=0; i<$INPUT_SIZE; i++) {
    printf("%f, ", tensor_$OUTPUT_NAME[i]);
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif