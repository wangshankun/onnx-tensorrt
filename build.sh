#!/bin/sh
#如果要使用pycuda，那么需要提前在python环境中安装
#下载TensorRT-5.1.5.0，cd python; pip install tensorrt-5.1.5.0-cp37-none-linux_x86_64.whl

sudo rm -rf ./build
mkdir build && cd build

cmake .. -DTENSORRT_ROOT=/home/shankun.shankunwan/trt/TensorRT-5.1.5.0/ -DCUDNN_INCLUDE_DIR=/usr/local/cuda-10.0/include/ -DCUBLAS_LIBRARY=/usr/local/cuda-10.0/lib64/libcublas.so
make -j12
sudo make install

cd ..
python setup.py build
sudo /srv2/shankun/anaconda3/bin/python setup.py install

cd example_centernet_dla34
chmod a+x run_example.py
python run_example.py

./build_demo_center_net.sh && ./center_net
