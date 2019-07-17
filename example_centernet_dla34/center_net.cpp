#include <assert.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "../onnx_utils.hpp"
#include "../common.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>


#define CHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }

#define readfile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


int main(int argc, char* argv[])
{
    nvinfer1::DataType model_dtype;
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    common::TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
    auto trt_builder = common::infer_object(nvinfer1::createInferBuilder(trt_logger));
    auto trt_network = common::infer_object(trt_builder->createNetwork());
    auto trt_parser  = common::infer_object(nvonnxparser::createParser(*trt_network, trt_logger));

    if ( !trt_parser->parseFromFile("centernet_dla34.onnx", verbosity) )
    {
        return false;
    }
    int maxBatchSize = 4;
    int batchSize    = 4;
    trt_builder->setMaxBatchSize(maxBatchSize);
    trt_builder->setMaxWorkspaceSize(1 << 20);
    trt_builder->setFp16Mode(true);
    trt_builder->setStrictTypeConstraints(true);
    nvinfer1::ICudaEngine* engine = trt_builder->buildCudaEngine(*trt_network);
    assert(engine);

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    

    float* input = (float*)malloc(batchSize * 3  * 512 * 512 * sizeof(float));
    for (int i = 0; i < batchSize; i++)
    {
        readfile("images.bin", input + i * 3  * 512 * 512 , 3  * 512 * 512 * sizeof(float));
    }

    void* buffers[4];
    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[0], batchSize * 3  * 512 * 512 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], batchSize * 80 * 128 * 128 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[2], batchSize * 2  * 128 * 128 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[3], batchSize * 2  * 128 * 128 * sizeof(float)));

    float* output_reg = (float*)malloc(batchSize * 2  * 128 * 128 * sizeof(float));
    float* output_hw  = (float*)malloc(batchSize * 2  * 128 * 128 * sizeof(float));
    float* output_cl  = (float*)malloc(batchSize * 80  * 128 * 128 * sizeof(float));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3  * 512 * 512 * sizeof(float), cudaMemcpyHostToDevice, stream));
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    context->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output_reg, buffers[3], batchSize * 2  * 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_hw, buffers[2], batchSize * 2  * 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_cl, buffers[1], batchSize * 80  * 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);
 
    printf("output[0]:%f\r\n",output_reg[0]);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaFree(buffers[2]));
    CHECK(cudaFree(buffers[3]));

    free(output_reg);
    free(input);
    context->destroy();
    engine->destroy();

    return 0;
}
