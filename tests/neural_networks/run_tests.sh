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