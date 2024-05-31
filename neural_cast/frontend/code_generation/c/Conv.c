
// CONV OPERATOR $(NAME)

{

$(OMP_PRAGMA1)
for (int oc = 0; oc < $(OUTPUT_CHANNELS); oc++) {
    for (int oh = 0; oh < $(OUTPUT_HEIGHT); oh++) {
        for (int ow = 0; ow < $(OUTPUT_WIDTH); ow++) {
            tensor_$(OUTPUT_NAME)[oc * $(OUTPUT_HEIGHT) * $(OUTPUT_WIDTH) + oh * $(OUTPUT_WIDTH) + ow] = tensor_$(BIASES_NAME)[oc];
        }
    }
}

$(OMP_PRAGMA2)
for (int oc = 0; oc < $(OUTPUT_CHANNELS); oc++) {
    for (int ic = 0; ic < $(INPUT_CHANNELS); ic++) {
        for (int kh = 0; kh < $(KERNEL_SIZE); kh++) {
            for (int kw = 0; kw < $(KERNEL_SIZE); kw++) {
                for (int oh = 0; oh < $(OUTPUT_HEIGHT); oh++) {
                    for (int ow = 0; ow < $(OUTPUT_WIDTH); ow++) {
                        int ih = oh * $(STRIDE) - $(PADDING) + kh;
                        int iw = ow * $(STRIDE) - $(PADDING) + kw;
                        if (ih >= 0 && ih < $(INPUT_HEIGHT) && iw >= 0 && iw < $(INPUT_WIDTH)) {
                            $(OMP_PRAGMA3)
                            tensor_$(OUTPUT_NAME)[oc * $(OUTPUT_HEIGHT) * $(OUTPUT_WIDTH) + oh * $(OUTPUT_WIDTH) + ow] +=
                                tensor_$(INPUT_NAME)[ic * $(INPUT_HEIGHT) * $(INPUT_WIDTH) + ih * $(INPUT_WIDTH) + iw] *
                                tensor_$(WEIGHTS_NAME)[oc * $(INPUT_CHANNELS) * $(KERNEL_SIZE) * $(KERNEL_SIZE) + ic * $(KERNEL_SIZE) * $(KERNEL_SIZE) + kh * $(KERNEL_SIZE) + kw];
                        }
                    }
                }
            }
        }
    }
}

}

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $(NAME) -----------------\n");
for(int i=0; i<$(OUTPUT_SIZE); i++) {
    printf("%f ", tensor_$(OUTPUT_NAME)[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif