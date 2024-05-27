
// -------------------------------------------------------
//             DEQUANTIZE LINEAR $(NAME)
// -------------------------------------------------------

#define DEQUANT_$(NAME)_OUTPUT_SIZE ($(OUTPUT_SIZE))

$(DEFINE_CONNECTED_OUTPUT)
#ifdef CONNECTED_OUTPUT
static float tensor_$(OUTPUT_NAME)[DEQUANT_$(NAME)_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
