#include "ResizeBilinear.hpp"
#include <cuda_fp16.h>
#include <cassert>

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims ResizeBilinearPlugin::getOutputDimensions(int index,
                                                        const nvinfer1::Dims *inputDims,
                                                        int nbInputs) {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(is_CHW(input));
  assert(_ndims == 2);
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  int s = 0;
  for( int d=0; d<input.nbDims; ++d ) {
    output.type[d] = input.type[d];
    if( input.type[d] == nvinfer1::DimensionType::kSPATIAL ) {
      output.d[d] = int(input.d[d] * _scale[s++]);
    } else {
      output.d[d] = input.d[d];
    }
  }
  return output;
}

int ResizeBilinearPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  assert(is_CHW(this->getInputDims(0)));
  assert(is_CHW(_output_dims));
  assert(_ndims == 2);
  return 0;
}

//代码实现与pytorch保持一致
//参考aten/src/ATen/native/UpSampleBilinear2d.cpp中函数upsample_bilinear2d_out_frame
template <typename Data>
__global__
void resize_bilinear_kernel_2d(int nbatch,
                              float2 scale,
                              int2 osize,
                              Data const* idata, int istride, int ibatchstride,
                              Data*       odata, int ostride, int obatchstride);

template <>
__global__
void resize_bilinear_kernel_2d<__half>(int nbatch,
                              float2 scale,
                              int2 osize,
                              __half const* idata, int istride, int ibatchstride,
                              __half*       odata, int ostride, int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  int i_w = round(osize.x/scale.x);
  int i_h = round(osize.y/scale.y);
  for( int batch=z0; batch<nbatch; batch+=gridDim.z ) {
    for( int oy=y0; oy<osize.y; oy+=blockDim.y*gridDim.y ) {
      for( int ox=x0; ox<osize.x; ox+=blockDim.x*gridDim.x ) {
        float       h1r = min((oy + 0.5) / scale.y - 0.5, float(i_h - 1));
        if(h1r < 0) h1r = static_cast<float>(0);
        const int   h1   = h1r;
        const int   h1p  = (h1 < i_h - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1) - h1lambda;

        float       w1r = min((ox + 0.5) / scale.x - 0.5, float(i_w - 1));
        if(w1r < 0) w1r = static_cast<float>(0);
        const int w1    = w1r;
        const int w1p   = (w1 < i_w - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1) - w1lambda;

        float val =
            h0lambda *
                (w0lambda * __half2float(idata[batch * ibatchstride + h1 * istride + w1]) +
                 w1lambda * __half2float(idata[batch * ibatchstride + h1 * istride + (w1 + w1p)])) +
            h1lambda *
                (w0lambda * __half2float(idata[batch * ibatchstride + (h1 + h1p )* istride + w1]) +
                 w1lambda * __half2float(idata[batch * ibatchstride + (h1 + h1p )* istride + (w1 + w1p)]));

        odata[batch * obatchstride + oy * ostride + ox] = __float2half(val);
      }
    }
  }
}


template <>
__global__
void resize_bilinear_kernel_2d<float>(int nbatch,
                              float2 scale,
                              int2 osize,
                              float const* idata, int istride, int ibatchstride,
                              float*       odata, int ostride, int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  int i_w = round(osize.x/scale.x);
  int i_h = round(osize.y/scale.y);
  for( int batch=z0; batch<nbatch; batch+=gridDim.z ) {
    for( int oy=y0; oy<osize.y; oy+=blockDim.y*gridDim.y ) {
      for( int ox=x0; ox<osize.x; ox+=blockDim.x*gridDim.x ) {
        float       h1r = min((oy + 0.5) / scale.y - 0.5, float(i_h - 1));
        if(h1r < 0) h1r = static_cast<float>(0);
        const int   h1   = h1r;
        const int   h1p  = (h1 < i_h - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1) - h1lambda;

        float       w1r = min((ox + 0.5) / scale.x - 0.5, float(i_w - 1));
        if(w1r < 0) w1r = static_cast<float>(0);
        const int w1    = w1r;
        const int w1p   = (w1 < i_w - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1) - w1lambda;

        float val =
            h0lambda *
                (w0lambda * idata[batch * ibatchstride + h1 * istride + w1] +
                 w1lambda * idata[batch * ibatchstride + h1 * istride + (w1 + w1p)]) +
            h1lambda *
                (w0lambda * idata[batch * ibatchstride + (h1 + h1p )* istride + w1] +
                 w1lambda * idata[batch * ibatchstride + (h1 + h1p )* istride + (w1 + w1p)]);

        odata[batch * obatchstride + oy * ostride + ox] = val;
      }
    }
  }
}

int ResizeBilinearPlugin::enqueue(int batchSize,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int nchan = input_dims.d[0];
  switch( _ndims ) {
  case 2: {
    float2 scale = {_scale[1], _scale[0]};
    int2 osize = {_output_dims.d[2], _output_dims.d[1]};
    int istride =   input_dims.d[2];
    int ostride = _output_dims.d[2];
    int ibatchstride =   input_dims.d[1] * istride;
    int obatchstride = _output_dims.d[1] * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.x + 1,
              (osize.y - 1) / block.y + 1,
              std::min(batchSize * nchan, 65535));
    if (getDataType()==nvinfer1::DataType::kFLOAT) {
      resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, scale, osize,
         static_cast<float const*>( inputs[0]), istride, ibatchstride,
         static_cast<float*      >(outputs[0]), ostride, obatchstride);
    } else {
      resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, scale, osize,
         static_cast<__half const*>( inputs[0]), istride, ibatchstride,
         static_cast<__half*      >(outputs[0]), ostride, obatchstride);
    }
    return cudaGetLastError() != cudaSuccess;
  }
  default: return -1;
  }
}
