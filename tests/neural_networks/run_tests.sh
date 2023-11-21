tests_dir=$1

# run test fc_relu
echo ""
echo "#####################################################################"
echo "                  RUN TESTS FOR FC-RELU                              "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_relu/main_test.py

# run test fc_relu_fc_relu
echo ""
echo "#####################################################################"
echo "               RUN TESTS FOR FC-RELU-FC-RELU                         "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_relu_fc_relu/main_test.py

# run test fc_sigmoid
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR FC-SIGMOID                         "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_sigmoid/main_test.py

# run test fc_tanh
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR FC-TANH                            "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_tanh/main_test.py

# run test fc_add
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR FC-ADD                             "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_add/main_test.py

# run test fc_mul
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR FC-MUL                             "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_mul/main_test.py

# run test fc_sub
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR FC-SUB                             "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/fc_sub/main_test.py

# run test matmul
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR MATMUL                             "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/matmul/main_test.py

# run test constant
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR CONSTANT                           "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/constant/main_test.py

# run test gather
echo ""
echo "#####################################################################"
echo "                    RUN TESTS FOR GATHER                             "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/gather/main_test.py

# run test reimplemented gru
echo ""
echo "#####################################################################"
echo "                RUN TESTS FOR REIMPLEMENETD GRU                      "
echo "#####################################################################"
echo ""
python $tests_dir/neural_networks/reimplemented_gru/main_test.py