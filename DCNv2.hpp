/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//#pragma once

#include "plugin.hpp"
#include "serialize.hpp"

#include <thrust/device_vector.h>
#include <cassert>

#include <cuda_runtime.h> 
extern "C" {
#include "cublas_v2.h" 
}
#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUBLAS_CALL(f) { \
  cublasStatus_t  err = (f); \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

class DCNv2Plugin final : public onnx2trt::Plugin {
    int _deformable_groups;
    int _dilation;
    int _padding;
    int _stride;
    int _kernel_size;
    std::vector<int> _outdims;
    cublasHandle_t _cublas_handle;
    void *  weight_data;
    void *  bias_data;
    void *  ones;
    void *  columns; 
    void *  weight_data_cuda;
    void *  bias_data_cuda;
    void ** input_b;
    void ** output_b;
    void ** columns_b;
    void ** ones_b;
    void ** weight_b;
    void ** bias_b;
protected:
    void deserialize(void const* serialData, size_t serialLength)
    {
        deserializeBase(serialData, serialLength);
        deserialize_value(&serialData, &serialLength, &_deformable_groups);
        deserialize_value(&serialData, &serialLength, &_dilation);
        deserialize_value(&serialData, &serialLength, &_padding);
        deserialize_value(&serialData, &serialLength, &_stride);
        deserialize_value(&serialData, &serialLength, &_kernel_size);
        deserialize_value(&serialData, &serialLength, &_outdims);
        deserialize_value(&serialData, &serialLength, &weight_data);
        deserialize_value(&serialData, &serialLength, &bias_data);
        deserialize_value(&serialData, &serialLength, &_cublas_handle);
    }
    size_t getSerializationSize() override
    {
        return (serialized_size(_deformable_groups) +
                serialized_size(_dilation) +
                serialized_size(_padding) +
                serialized_size(_stride) +
                serialized_size(_kernel_size) +
                serialized_size(_outdims) +
                serialized_size(weight_data) +
                serialized_size(bias_data) +
                serialized_size(_cublas_handle) +
                getBaseSerializationSize());
    }
    void serialize(void *buffer) override 
    {
        serializeBase(buffer);
        serialize_value(&buffer, _deformable_groups);
        serialize_value(&buffer, _dilation);
        serialize_value(&buffer, _padding);
        serialize_value(&buffer, _stride);
        serialize_value(&buffer, _kernel_size);
        serialize_value(&buffer, _outdims);
        serialize_value(&buffer, weight_data);
        serialize_value(&buffer, bias_data);
        serialize_value(&buffer, _cublas_handle);
    }
public:
  DCNv2Plugin(int deformable_groups,
              int dilation,
              int padding,
              int stride,
              int kernel_size,
              std::vector<int> outdims,
              void* weight_data,
              void* bias_data)
      : 
            _deformable_groups(deformable_groups),
            _dilation(dilation),
            _padding(padding),
            _stride(stride),
            _kernel_size(kernel_size),
            _outdims(outdims),
            weight_data(weight_data),
            bias_data(bias_data)
      { }

  DCNv2Plugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }

  virtual const char* getPluginType() const override { return "DCNv2"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;

  template <typename Dtype>
  int do_initialize();

  int initialize() override;

  void terminate() override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;
                      
  template <typename Dtype>
  int do_enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream);

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};

