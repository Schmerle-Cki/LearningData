#!/usr/bin/env bash

# If project not ready, generate cmake file.
if [[ ! -d build ]]; then
    mkdir -p build
    cd build
    cmake ..
    cd ..
fi

# Build project.
cd build
make -j
cd ..

# Run all testcases. 
# You can comment some lines to disable the run of specific examples.
mkdir -p output

bin/PA1 testcases/bigSecond.txt output/Vase.bmp
bin/PA1 testcases/scene20_square.txt output/rabbit.bmp
