## origin from: https://github.com/onnx/onnx-tensorrt

# 依赖环境 TensorRT,cuda,cublas,onnx
##TensorRT
 如果要使用pycuda，那么需要提前在python环境中安装

 下载TensorRT-5.1.5.0，cd python; pip install tensorrt-5.1.5.0-cp37-none-linux\_x86\_64.whl

#绕过onnx的parse
  /srv2/shankun/anaconda3/lib/python3.7/site-packages/onnx/checker.py", line 86, in check\_model
     C.check_model(model.SerializeToString())

    将C.check_model(model.SerializeToString()) 替换为 pass  

# 性能
     
     因为内存有限制，1080ti最大batch不能过12，一般最大为8
     当batch为4时候，TensorRT FP16 为22ms
     当batch为1时候，TensorRT FP16 为11ms
     当batch为1时候，TensorRT FP32 为15ms 
     当batch为1时候，pytorch的CenterNet为24ms

# 增加的内容：
## 增加DCNv2的插件;
    https://github.com/CharlesShang/DCNv2
## 增加centernet dla34执行例子
    https://github.com/xingyizhou/CenterNet
## 参考首次commit信息
    https://github.com/wangshankun/onnx-tensorrt/commit/23497e10d05e3038c2afb96ef31a0c601c876525
    
### Download the code
Clone the code from GitHub.

    git clone --recursive https://github.com/wangshankun/onnx-tensorrt
### Executable and libraries
    ./build.sh

## 与原始pytorch工程实现的CenterNet结果对比，差异在小数点后4位
### debug代码，调试信息和存储中间过程文件没有删除
![image](https://github.com/wangshankun/onnx-tensorrt/blob/master/readme.jpg)


### CenterNet导出onnx模型参考
    https://github.com/xingyizhou/CenterNet/issues/77
   
# TensorRT backend for ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

## ONNX Python backend usage

The TensorRT backend for ONNX can be used in Python as follows:

## trt的python的接口不支持fp16，可以用trt的c++接口测试fp16的CenterNet
```python
import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("/path/to/model.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)
```
