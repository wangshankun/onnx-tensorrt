#!/bin/sh
/usr/bin/c++   -DONNX_NAMESPACE=onnx2trt_onnx -I/usr/local/cuda-10.0/include -I/home/shankun.shankunwan/trt/TensorRT-5.1.5.0/include -I/home/shankun.shankunwan/work/onnx-tensorrt/third_party/onnx -I/home/shankun.shankunwan/work/onnx-tensorrt/build/third_party/onnx  -Wall -O3 -DNDEBUG -fPIE   -std=gnu++11 -o center_net.o -c center_net.cpp


/usr/bin/c++   -Wall -O3 -DNDEBUG  -rdynamic center_net.o -o center_net -L/usr/local/cuda-10.0/lib64/stubs  -L/usr/local/cuda-10.0/lib64  -Wl,-rpath,/usr/local/lib:/usr/local/cuda-10.0/lib64:/home/shankun.shankunwan/trt/TensorRT-5.1.5.0/lib: -lpthread -ldl /usr/local/lib/libprotobuf.so -lpthread /usr/local/lib/libprotobuf.so -lcudnn /usr/local/cuda-10.0/lib64/libcublas.so /home/shankun.shankunwan/trt/TensorRT-5.1.5.0/lib/libnvinfer.so /home/shankun.shankunwan/trt/TensorRT-5.1.5.0/lib/libnvinfer_plugin.so  ../build/libnvonnxparser.so -lcudadevrt -lcudart -lrt -lpthread -ldl

rm -rf center_net.o
