
// -------------------------------------------------------
//             GLOBAL AVERAGEPOOL $(NAME)
// -------------------------------------------------------

$(DEFINE_CONNECTED_OUTPUT)
#ifdef CONNECTED_OUTPUT
static float tensor_$(OUTPUT_NAME)[$(OUTPUT_SIZE)];
#undef CONNECTED_OUTPUT
#endif
