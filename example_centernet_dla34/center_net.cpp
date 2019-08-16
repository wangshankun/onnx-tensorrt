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


const char* gNetworkName = "centernet"; 

inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        if (dir.back() != '/')
            filepath = dir + "/" + filepathSuffix;
        else
            filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
                break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
                                                    [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << directoryList << std::endl;
        std::cout << "&&&& FAILED" << std::endl;
        exit(EXIT_FAILURE);
    }
    return filepath;
}

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
        , mDataDir(directories)
    {
        FILE* file = fopen(locateFile(mPrefix + std::string("0.batch"), mDataDir).c_str(), "rb");
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  //The number of dimensions.
        mDims.d[0] = d[0]; //Batch Size
        mDims.d[1] = d[1]; //Channels
        mDims.d[2] = d[2]; //Height
        mDims.d[3] = d[3]; //Width

        fclose(file);
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
                return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return &mBatch[0]; }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    int getImageSize() const { return mImageSize; }
    nvinfer1::Dims getDims() const { return mDims; }

private:
    float* getFileBatch() { return &mFileBatch[0]; }

    bool update()
    {
        std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + std::string(".batch"), mDataDir);
        FILE* file = fopen(inputFileName.c_str(), "rb");
        if (!file)
            return false;

        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
        size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
        assert(readInputCount == size_t(mDims.d[0] * mImageSize));

        fclose(file);
        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    nvinfer1::Dims mDims;
    std::vector<float> mBatch;
    std::vector<float> mFileBatch;
    std::string mPrefix;
    std::vector<std::string> mDataDir;
};

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(BatchStream& stream, int firstBatch, bool readCache = true)
        : mStream(stream)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator2()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], kINPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        return std::string("CalibrationTable") + gNetworkName;
    }
    BatchStream mStream;
    size_t mInputCount;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

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
    int maxBatchSize = 1;
    int batchSize    = 1;
    trt_builder->setMaxBatchSize(maxBatchSize);
    trt_builder->setMaxWorkspaceSize(1 << 20);
    //trt_builder->setFp16Mode(true);

    // INT8 calibration variables
    static const int kCAL_BATCH_SIZE = 1;   // Batch size
    static const int kFIRST_CAL_BATCH = 0;  // First batch
    static const int kNB_CAL_BATCHES = 100; // Number of batches

    static const std::vector<std::string> kDIRECTORIES{"/home/shankun.shankunwan/trt/TensorRT-5.1.5.0/data/ssd/","data/ssd/"};

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, kFIRST_CAL_BATCH));
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    trt_builder->setInt8Mode(true);
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    trt_builder->setInt8Calibrator(calibrator.get());
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    //trt_builder->setStrictTypeConstraints(true);
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    nvinfer1::ICudaEngine* engine = trt_builder->buildCudaEngine(*trt_network);
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    assert(engine);

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
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
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    context->enqueue(batchSize, buffers, stream, nullptr);
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
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
