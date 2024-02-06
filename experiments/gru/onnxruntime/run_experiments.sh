#!/bin/bash

# experiment parameters
NUM_EXPERIMENTS=10000
SIZES=(10 20 30 40 50 60 70 80 90 100)
result="results.txt"

# if results file exists, delete it
if [ -f "$result" ]; then
    rm "$result"
    echo "Delete existing '$result' file."
fi

# create results file
echo "Create '$result' file."
touch "$result"

# append header
printf "%s\n" "INPUT_SIZE,HIDDEN_SIZE,AVG_TIME,MIN_TIME,MAX_TIME,STDDEV_TIME" >> "$result"

for SIZE in "${SIZES[@]}"; do
    echo "Experiment for size: $SIZE"

    echo "Compiling ..."
    g++ -o prog main.cpp -I ./include/ -L ./ ./libonnxruntime.so.1.6.0

    echo "Execute ..."
    DATA=$(./prog M$SIZE.onnx $SIZE $SIZE $NUM_EXPERIMENTS)
    printf "%s\n" "$DATA" >> "$result"
done