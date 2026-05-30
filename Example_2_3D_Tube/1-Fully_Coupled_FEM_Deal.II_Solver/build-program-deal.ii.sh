#!/bin/bash
mkdir -p build
cd build
cmake -DDEAL_II_DIR=~/directory/to/dealii-build -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../ ..
make release
make
cd ..

