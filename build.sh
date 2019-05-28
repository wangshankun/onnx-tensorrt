#!/bin/sh
cd build
sudo rm -rf ./*
cmake .. -DTENSORRT_ROOT=/home/shankun.shankunwan/trt/TensorRT-5.1.5.0/ -DCUDNN_INCLUDE_DIR=/usr/local/cuda-10.0/include/ -DCUBLAS_LIBRARY=/usr/local/cuda-10.0/lib64/libcublas.so
make -j12
sudo make install
cd ..
python setup.py build
sudo /home/shankun.shankunwan/anaconda/bin/python  setup.py install

cd example_centernet_dla34
chmod a+x run_example.py
python run_example.py

