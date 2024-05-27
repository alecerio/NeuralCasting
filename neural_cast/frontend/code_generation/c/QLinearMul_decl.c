
// -------------------------------------------------------
//                   QLINEARMUL $(NAME)
// -------------------------------------------------------

#define QLINEARMUL_$(NAME)_OUTPUT_SIZE ($(OUTPUT_SIZE))

$(DEFINE_CONNECTED_OUTPUT)
#ifdef CONNECTED_OUTPUT
static int8_t tensor_$(OUTPUT_NAME)[QLINEARMUL_$(NAME)_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
