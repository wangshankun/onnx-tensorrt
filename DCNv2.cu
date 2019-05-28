#include "DCNv2.hpp"
#include <cuda_fp16.h>
#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)



#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    printf("Error trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb, int64_t *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  if(n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if(transa_)
  {
    if(m <= 1)
      *lda = std::max<int64_t>(k, 1);
  }
  else
  {
    if(k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if(transb_)
  {
    if(k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  }
  else
  {
    if(n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                           const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__global__ void modulated_deformable_im2col_gpu_kernel(int n,
                                                       float *data_im, float *data_offset,  float *data_mask,
                                                       int height,     int width,           int kernel_h, int kernel_w,
                                                       int pad_h,      int pad_w,
                                                       int stride_h,   int stride_w,
                                                       int dilation_h, int dilation_w,
                                                       int channel_per_deformable_group,
                                                       int batch_size, int num_channels,    int deformable_group,
                                                       int height_col, int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    // const int b_col = (index / width_col / height_col) % batch_size;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    // const int c_im = (index / width_col / height_col) / batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    // const int c_col = c_im * kernel_h * kernel_w;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        // data_col_ptr += batch_size * height_col * width_col;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void modulated_deformable_im2col_cuda(cudaStream_t stream,
  float* data_im,       float* data_offset, float* data_mask,
  int batch_size,       int channels,       int height_im, int width_im, 
  int height_col,       int width_col,      int kernel_h,  int kernel_w,
  int pad_h,            int pad_w,          int stride_h,  int stride_w, 
  int dilation_h,       int dilation_w,
  int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

void CudaBlas_SgemmBatched(cublasHandle_t handle,  cudaStream_t stream,
                                      char transa, char transb, int64_t m, int64_t n, int64_t k,
                                      float alpha, float *a[], int64_t lda, float *b[], int64_t ldb,
                                      float beta, float *c[], int64_t ldc, int64_t batchCount)
{
  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasSetStream(handle, stream);
  CUBLAS_CALL(cublasSgemmBatched(handle,
                               opa, opb, (int)m, (int)n, (int)k,
                               &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                               (int)batchCount));
}

__global__ void createBatchGemmBuffer(float **input_b, float **output_b,
                                      float **columns_b, float **ones_b,
                                      float **weight_b, float **bias_b,
                                      float *input, float *output,
                                      float *columns, float *ones,
                                      float *weight, float *bias,
                                      int input_stride,  int output_stride,
                                      int columns_stride, int ones_stride,
                                      int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}

nvinfer1::Dims DCNv2Plugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
    nvinfer1::Dims const& input = inputDims[0];
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    for(int d=0; d<input.nbDims; ++d ) 
    {
        output.d[d] = _outdims[d];
    }    
    printf("====================%s  %d outputdim:%d %d %d\r\n",__FUNCTION__,__LINE__,output.d[0],output.d[1],output.d[2]);
    return output;
}

int DCNv2Plugin::initialize() {

    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    auto const& input_dims = this->getInputDims(0);
    int batch      = 1;
    int c_in       = input_dims.d[0];
    //int height_in  = input_dims.d[1];
    //int width_in   = input_dims.d[2];
    int c_out      = _outdims[0];
    int height_out = _outdims[1];
    int width_out  = _outdims[1];

    int matrices_size = batch * sizeof(float *);

    CHECK_CUDA(cudaMalloc((void***)&input_b,   matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&output_b,  matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&columns_b, matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&ones_b,    matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&weight_b,  matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&bias_b,    matrices_size));
    CHECK_CUDA(cudaMalloc((void**)&weight_data_cuda, batch * c_in * _kernel_size * _kernel_size * c_out * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&bias_data_cuda,   batch * c_out * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&ones,          batch * height_out * width_out * sizeof(float)));
    //CHECK_CUDA(cudaMemset((void**)&ones,    1,    batch * height_out * width_out * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&columns,       batch * c_in * _kernel_size * _kernel_size * height_out * width_out * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(weight_data_cuda, weight_data, batch * c_in * _kernel_size * _kernel_size * c_out * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bias_data_cuda,   bias_data,   batch * c_out * sizeof(float), cudaMemcpyHostToDevice));

    CUBLAS_CALL(cublasCreate(&_cublas_handle));
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    
    thrust::device_ptr<float> dev_ptr(ones);
    thrust::fill(dev_ptr, dev_ptr + batch * height_out * width_out, (float)1);

    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    return 0;
}

void DCNv2Plugin::terminate() {
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    cudaFree(ones);
    cudaFree(columns);
    cudaFree(input_b);
    cudaFree(output_b);
    cudaFree(columns_b);
    cudaFree(ones_b);
    cudaFree(bias_b);
    cudaFree(weight_b);
    cudaFree(weight_data_cuda);
    cudaFree(bias_data_cuda);
    cublasDestroy(_cublas_handle);
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
}

int DCNv2Plugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    auto const& input_dims = this->getInputDims(0);
    int batch      = 1;
    int c_in       = input_dims.d[0];
    int height_in  = input_dims.d[1];
    int width_in   = input_dims.d[2];
    int c_out      = _outdims[0];
    int height_out = _outdims[1];
    int width_out  = _outdims[1];

    const int block = 128;
    const int grid = (batch + block - 1) / block;


    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    float* input_buf = (float*)malloc(c_in * height_in * height_in*sizeof(float));
    (cudaMemcpy(input_buf, (float*)inputs[0],  c_in*height_in * height_in*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("input_buf.bin",input_buf,c_in * height_in * height_in*sizeof(float));


    float* offset_buf = (float*)malloc(18 * height_in * width_in*sizeof(float));
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    (cudaMemcpy(offset_buf, (float*)inputs[1],  18*height_in * width_in*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("offset_buf.bin",offset_buf,18 * height_in * width_in*sizeof(float));


    float* mask_buf = (float*)malloc(9 * height_in * width_in*sizeof(float));
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    (cudaMemcpy(mask_buf, (float*)inputs[2],  9 * height_in * width_in*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("mask_buf.bin",mask_buf,9 * height_in * width_in*sizeof(float));
    
    
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    savefile("weight.bin", weight_data, c_in * _kernel_size * _kernel_size * c_out*sizeof(float));
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    savefile("bias.bin", bias_data, c_out*sizeof(float));


    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    float* columns_buf = (float*)malloc(c_in*_kernel_size*_kernel_size*height_out*width_out*sizeof(float));
    (cudaMemcpy(columns_buf, columns,  c_in*_kernel_size*_kernel_size*height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("columns_buf.bin",columns_buf,c_in*_kernel_size*_kernel_size*height_out*width_out*sizeof(float));

    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    float* ones_buf = (float*)malloc(height_out*width_out*sizeof(float));
    (cudaMemcpy(ones_buf, ones, height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("ones_buf.bin",ones_buf,height_out*width_out*sizeof(float));

    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    float* output_buf = (float*)malloc(height_out*width_out*c_out*sizeof(float));
    (cudaMemcpy(output_buf, (float*)outputs[0], height_out*width_out*c_out*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("output_buf.bin",output_buf,height_out*width_out*c_out*sizeof(float));
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);

    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    createBatchGemmBuffer<<<grid, block, 0, stream>>>(
        input_b, output_b,
        columns_b, ones_b,
        weight_b, bias_b,
        (float*)inputs[0],
        (float*)outputs[0],
        columns,
        ones,
        weight_data_cuda,
        bias_data_cuda,
        c_in * width_in * height_in,
        c_out * width_out * height_out,
        c_in * _kernel_size * _kernel_size * height_out * width_out,
        height_out * width_out,
        batch);
    
    long m_ = c_out;
    long n_ = height_out * width_out;
    long k_ = 1;
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    CudaBlas_SgemmBatched(_cublas_handle,
                          stream,
                          't',
                          'n',
                          n_,
                          m_,
                          k_,
                          1.0f,
                          ones_b, k_,
                          bias_b, k_,
                          0.0f,
                          output_b, n_,
                          batch);

    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    float* output_sgemm_buf = (float*)malloc(height_out*width_out*c_out*sizeof(float));
    (cudaMemcpy(output_sgemm_buf, (float*)outputs[0], height_out*width_out*c_out*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("output_sgemm_buf.bin",output_sgemm_buf,height_out*width_out*c_out*sizeof(float));
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);

    modulated_deformable_im2col_cuda(stream,
                                     (float*)inputs[0],
                                     (float*)inputs[1],
                                     (float*)inputs[2],
                                     batch, c_in, height_in, width_in,
                                     height_out, width_out, _kernel_size, _kernel_size,
                                     _padding, _padding, _stride, _stride, _dilation, _dilation,
                                     _deformable_groups,
                                     columns);

    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    float* columns_im2col_buf = (float*)malloc(c_in*_kernel_size*_kernel_size*height_out*width_out*sizeof(float));
    (cudaMemcpy(columns_im2col_buf, columns,  c_in*_kernel_size*_kernel_size*height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("columns_im2col_buf.bin",columns_im2col_buf,c_in*_kernel_size*_kernel_size*height_out*width_out*sizeof(float));
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    long m = c_out;
    long n = height_out * width_out;
    long k = c_in * _kernel_size * _kernel_size;
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    CudaBlas_SgemmBatched(_cublas_handle,
                          stream,
                           'n',
                           'n',
                           n,
                           m,
                           k,
                           1.0f,
                           columns_b, n,
                           weight_b, k,
                           1.0f,
                           output_b, n,
                           batch);
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    float* output_final_buf = (float*)malloc(height_out*width_out*c_out*sizeof(float));
    (cudaMemcpy(output_final_buf, (float*)outputs[0], height_out*width_out*c_out*sizeof(float), cudaMemcpyDeviceToHost));
    savefile("output_final_buf.bin",output_final_buf,height_out*width_out*c_out*sizeof(float));
    //exit(1);
    printf("======================%s  %d\r\n",__FUNCTION__,__LINE__);
    return cudaGetLastError() != cudaSuccess;
}

