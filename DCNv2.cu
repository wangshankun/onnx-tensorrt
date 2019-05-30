#include "DCNv2.hpp"
#include <cuda_fp16.h>
#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <time.h>

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

template <typename Dtype>
__device__ float dmcn_im2col_bilinear(const Dtype *bottom_data, const int data_width,
                           const int height, const int width, float h, float w)
{
  int h_low = floor((float)h);
  int w_low = floor((float)w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = (float)h - h_low;
  float lw = (float)w - w_low;
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
  return (float)val;
}

template <typename Dtype>
__global__ void modulated_deformable_im2col_gpu_kernel(int n,
                                                       Dtype *data_im, Dtype *data_offset,  Dtype *data_mask,
                                                       int height,     int width,           int kernel_h, int kernel_w,
                                                       int pad_h,      int pad_w,
                                                       int stride_h,   int stride_w,
                                                       int dilation_h, int dilation_w,
                                                       int channel_per_deformable_group,
                                                       int batch_size, int num_channels,    int deformable_group,
                                                       int height_col, int width_col,
                                                       Dtype *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
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

    int h_in = h_col * stride_h - pad_h;
    int w_in = w_col * stride_w - pad_w;

    //  Dtype *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    Dtype *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const Dtype* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const Dtype *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const Dtype *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const Dtype *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        float offset_h = data_offset_ptr[data_offset_h_ptr];
        float offset_w = data_offset_ptr[data_offset_w_ptr];
        float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        float h_im = h_in + i * dilation_h + offset_h;
        float w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          val = dmcn_im2col_bilinear<Dtype>(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = (Dtype)(val * mask);
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void modulated_deformable_im2col_cuda(cudaStream_t stream,
  Dtype* data_im,       Dtype* data_offset, Dtype* data_mask,
  int batch_size,       int channels,       int height_im, int width_im, 
  int height_col,       int width_col,      int kernel_h,  int kernel_w,
  int pad_h,            int pad_w,          int stride_h,  int stride_w, 
  int dilation_h,       int dilation_w,
  int deformable_group, Dtype* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel<Dtype>
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

template<typename Dtype>
void CudaBlas_gemmBatched(cublasHandle_t handle,  cudaStream_t stream,
                                      char transa, char transb, int64_t m, int64_t n, int64_t k,
                                      Dtype alpha, Dtype *a[],  int64_t lda, Dtype *b[], int64_t ldb,
                                      Dtype beta,  Dtype *c[],  int64_t ldc, int64_t batchCount);

template<>
void CudaBlas_gemmBatched<float>(cublasHandle_t handle,  cudaStream_t stream,
                                      char transa, char transb, int64_t m, int64_t n, int64_t k,
                                      float alpha, float *a[],  int64_t lda, float *b[], int64_t ldb,
                                      float beta,  float *c[],  int64_t ldc, int64_t batchCount)
{

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasSetStream(handle, stream);

struct timespec start, finish;
clock_gettime(CLOCK_MONOTONIC, &start);
  CUBLAS_CALL(cublasSgemmBatched(handle,
                     opa, opb, (int)m, (int)n, (int)k,
                     &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                     (int)batchCount));
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed += (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
printf("elapsed time:%f\r\n",elapsed);
}

template<>
void CudaBlas_gemmBatched<__half>(cublasHandle_t handle,  cudaStream_t stream,
                                      char transa, char transb, int64_t m, int64_t n, int64_t k,
                                      __half alpha, __half *a[],  int64_t lda, __half *b[], int64_t ldb,
                                      __half beta,  __half *c[],  int64_t ldc, int64_t batchCount)
{
  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasSetStream(handle, stream);

struct timespec start, finish;
double elapsed = 0;
clock_gettime(CLOCK_MONOTONIC, &start);
  CUBLAS_CALL(cublasHgemmBatched(handle,
                     opa, opb, (int)m, (int)n, (int)k,
                     &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                     (int)batchCount));
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed += (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
printf("elapsed time:%f\r\n",elapsed);
}

template <typename Dtype>
__global__ void createBatchGemmBuffer(Dtype **input_b,   Dtype **output_b,
                                      Dtype **columns_b, Dtype **ones_b,
                                      Dtype **weight_b,  Dtype **bias_b,
                                      Dtype *input,      Dtype *output,
                                      Dtype *columns,    Dtype *ones,
                                      Dtype *weight,     Dtype *bias,
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
    return output;
}


template <typename Dtype>
__global__ void cuda_memcpy_init(Dtype *cu_buf, int size, Dtype val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        cu_buf[i]=val;
    }
}

template <typename Dtype>
int DCNv2Plugin::do_initialize(){
    auto const& input_dims = this->getInputDims(0);
    int batch      = 1;
    int c_in       = input_dims.d[0];
    int c_out      = _outdims[0];
    int height_out = _outdims[1];
    int width_out  = _outdims[1];

    int matrices_size = batch * sizeof(Dtype *);

    CHECK_CUDA(cudaMalloc((void***)&input_b,   matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&output_b,  matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&columns_b, matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&ones_b,    matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&weight_b,  matrices_size));
    CHECK_CUDA(cudaMalloc((void***)&bias_b,    matrices_size));
    CHECK_CUDA(cudaMalloc((void**)&weight_data_cuda, batch * c_in * _kernel_size * _kernel_size * c_out * sizeof(Dtype)));
    CHECK_CUDA(cudaMalloc((void**)&bias_data_cuda,   batch * c_out * sizeof(Dtype)));
    CHECK_CUDA(cudaMalloc((void**)&ones,          batch * height_out * width_out * sizeof(Dtype)));
    CHECK_CUDA(cudaMalloc((void**)&columns,       batch * c_in * _kernel_size * _kernel_size * height_out * width_out * sizeof(Dtype)));

    CHECK_CUDA(cudaMemcpy((Dtype*)weight_data_cuda, (Dtype*)weight_data, batch * c_in * _kernel_size * _kernel_size * c_out * sizeof(Dtype), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((Dtype*)bias_data_cuda,   (Dtype*)bias_data,   batch * c_out * sizeof(Dtype), cudaMemcpyHostToDevice));

    CUBLAS_CALL(cublasCreate(&_cublas_handle));

    int buf_size = batch * height_out * width_out;
    int blockSize = 256;
    int numBlocks = (buf_size + blockSize - 1) / (buf_size);
    cuda_memcpy_init<Dtype><<<numBlocks, blockSize>>>((Dtype*)ones, buf_size , (Dtype)1.0f);
    
    return 0;
}

int DCNv2Plugin::initialize() {

    if (getDataType()==nvinfer1::DataType::kFLOAT)
    {
        return do_initialize<float>();
    }
    else if (getDataType()==nvinfer1::DataType::kHALF)
    {
        return do_initialize<__half>();
    }
    else
    {
        return -1;
    }
}

void DCNv2Plugin::terminate() {
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
}


template <typename Dtype>
int DCNv2Plugin::do_enqueue(int batchSize,
                            const void *const *inputs, void **outputs,
                            void *workspace, cudaStream_t stream){
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

    createBatchGemmBuffer<Dtype><<<grid, block, 0, stream>>>(
        (Dtype**)input_b,   (Dtype**)output_b,
        (Dtype**)columns_b, (Dtype**)ones_b,
        (Dtype**)weight_b,  (Dtype**)bias_b,
        (Dtype*)inputs[0],
        (Dtype*)outputs[0],
        (Dtype*)columns,
        (Dtype*)ones,
        (Dtype*)weight_data_cuda,
        (Dtype*)bias_data_cuda,
        c_in * width_in * height_in,
        c_out * width_out * height_out,
        c_in * _kernel_size * _kernel_size * height_out * width_out,
        height_out * width_out,
        batch);
    
    long m_ = c_out;
    long n_ = height_out * width_out;
    long k_ = 1;



    CudaBlas_gemmBatched<Dtype>(_cublas_handle,
                                 stream,
                                 't',
                                 'n',
                                 n_,
                                 m_,
                                 k_,
                                 (Dtype)1.0f,
                                 (Dtype**)ones_b, k_,
                                 (Dtype**)bias_b, k_,
                                 (Dtype)0.0f,
                                 (Dtype**)output_b, n_,
                                 batch);    

    modulated_deformable_im2col_cuda<Dtype>(stream,
                                           (Dtype*)inputs[0],
                                           (Dtype*)inputs[1],
                                           (Dtype*)inputs[2],
                                           batch, c_in, height_in, width_in,
                                           height_out, width_out, _kernel_size, _kernel_size,
                                           _padding, _padding, _stride, _stride, _dilation, _dilation,
                                           _deformable_groups,
                                           (Dtype*)columns);

    long m = c_out;
    long n = height_out * width_out;
    long k = c_in * _kernel_size * _kernel_size;

    CudaBlas_gemmBatched<Dtype>(_cublas_handle,
                             stream,
                              'n',
                              'n',
                              n,
                              m,
                              k,
                              (Dtype)1.0f,
                              (Dtype**)columns_b, n,
                              (Dtype**)weight_b,  k,
                              (Dtype)1.0f,
                              (Dtype**)output_b,  n,
                              batch);

    return cudaGetLastError() != cudaSuccess;
}

int DCNv2Plugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

    if (getDataType()==nvinfer1::DataType::kFLOAT)
    {
        return do_enqueue<float>(batchSize, inputs, outputs, workspace, stream);
    }
    else if (getDataType()==nvinfer1::DataType::kHALF)
    {
        return do_enqueue<__half>(batchSize, inputs, outputs, workspace, stream);
    }
    else
    {
        return -1;
    }
}

bool DCNv2Plugin::supportsFormat(nvinfer1::DataType type,
                                     nvinfer1::PluginFormat format) const {
  return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF);
}
