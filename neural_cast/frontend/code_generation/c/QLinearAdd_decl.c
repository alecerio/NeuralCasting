
// -------------------------------------------------------
//                   QLINEARADD $(NAME)
// -------------------------------------------------------

#define QLINEARADD_$(NAME)_OUTPUT_SIZE ($(OUTPUT_SIZE))

$(DEFINE_CONNECTED_OUTPUT)
#ifdef CONNECTED_OUTPUT
static int8_t tensor_$(OUTPUT_NAME)[QLINEARADD_$(NAME)_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
