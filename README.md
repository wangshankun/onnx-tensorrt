#origin from: https://github.com/onnx/onnx-tensorrt

# ADD

##增加DCNv2的插件;

    https://github.com/CharlesShang/DCNv2

##增加centernet dla34执行例子

    https://github.com/xingyizhou/CenterNet


### Download the code
Clone the code from GitHub.

    git clone --recursive https://github.com/wangshankun/onnx-tensorrt

### Executable and libraries

    ./build.sh

##与原始pytorch工程实现的CenterNet结果对比，差异在小数点后4位
![image](https://github.com/wangshankun/onnx-tensorrt/blob/master/readme.jpg)


# TensorRT backend for ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

## ONNX Python backend usage

The TensorRT backend for ONNX can be used in Python as follows:

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
