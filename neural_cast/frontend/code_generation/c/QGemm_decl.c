
// -------------------------------------------------------
//                         QGEMM $(NAME)
// -------------------------------------------------------

#define QGEMM_$(NAME)_OUTPUT_SIZE ($(OUTPUT_SIZE))
#define QGEMM_$(NAME)_INPUT_SIZE ($(INPUT_SIZE))

$(DEFINE_CONNECTED_OUTPUT)
#ifdef CONNECTED_OUTPUT
static int8_t tensor_$(OUTPUT_NAME)[QGEMM_$(NAME)_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
