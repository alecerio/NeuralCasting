#include "$MODEL_NAME.h"

// weights initialization
$WEIGHTS_INITIALIZATION

// activations initialization
$ACTIVATIONS_INITIALIZATION

// attributes initialization
$ATTRIBUTES_INITIALIZATION

void run_inference($INPUTS, $OUTPUTS) {
$RUN_OPS
}
