from neural_cast.ncrun import ncgencode

ncgencode(
    name='nsnet2', 
    onnx_path='/media/alessandro/SecondDisk1/NeuralCasting/examples/qnsnet2/nsnet2_reimplemented_int8_static.onnx',
    output_path='/home/alessandro/Desktop/qnsnet2_out',
    debug=True
)