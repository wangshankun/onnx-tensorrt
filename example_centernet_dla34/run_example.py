import sys
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import time

model = onnx.load("centernet_dla34.onnx")
graph = onnx.helper.printable_graph(model.graph)
#print(graph)

engine = backend.prepare(model, device='CUDA:1')
#input_data = np.random.random(size=(1, 3, 512, 512)).astype(np.float32)
images = np.load('images.npy')
#input_data = np.random.random(size=(1, 3, 512, 512)).astype(np.float32)
output_datas = engine.run(images)
#print(output_datas[0].shape)
#print(output_datas[0])
#print("===============(output_datas[0]===============================")
#print(output_datas[1].shape)
#print(output_datas[1])
#print("================(output_datas[1]==================================")
#print(output_datas[2].shape)
print(output_datas[2][0])
#print("=================output_datas[2]=================================")
#output_hm =  np.load('output_hm.npy')
#output_wh =  np.load('output_wh.npy')
output_reg =  np.load('output_reg.npy')
#print(output_hm.shape)
#print(output_hm)
#print("=============  output_hm =====================================")
#print(output_wh.shape)
#print(output_wh)
#print("================output_wh==================================")
#print(output_reg.shape)
print("==================output_reg================================")
print(output_reg[0])
