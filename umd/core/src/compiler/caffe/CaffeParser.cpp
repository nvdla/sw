/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>

#include "ErrorMacros.h"
#include "priv/Check.h"
#include "priv/caffe/CaffeParser.h"

#include "ditcaffe/protobuf-2.6.1/ditcaffe.pb.h"

#include "half.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unistd.h>

using namespace nvdla;
namespace dc = ditcaffe;
typedef half_float::half float16;

namespace nvdla
{

namespace caffe
{


IBlobNameToTensor::~IBlobNameToTensor() { }
IBinaryProtoBlob::~IBinaryProtoBlob() { }
ICaffeParser::~ICaffeParser() { }

ICaffeParser *createCaffeParser()
{
    priv::CaffeParserFactory::CaffeParserPrivPair ppair;
    ppair = nvdla::caffe::priv::CaffeParserFactory::newCaffeParser();
    return ppair.i();
}

NvDlaError destroyCaffeParser(ICaffeParser *parser)
{
    NvDlaError e = NvDlaSuccess;
    PROPAGATE_ERROR_FAIL(priv::CaffeParserFactory::deleteCaffeParser(parser));

fail:
    return e;
}

namespace priv
{

CaffeParserFactory::CaffeParserPrivPair CaffeParserFactory::newCaffeParser()
{
    ICaffeParser *parser;
    CaffeParser *parser_priv;
    parser = parser_priv = new priv::CaffeParser();
    if (parser) {
        s_priv.insert(parser,parser_priv);
        s_self.insert(parser, parser);
    }
    return CaffeParserPrivPair(parser, parser_priv);
}

NvDlaError CaffeParserFactory::deleteCaffeParser(ICaffeParser *parser)
{
    if (parser != NULL) {
        CaffeParser *parser_priv = priv(parser);
        if (parser_priv != NULL) {
            delete(parser_priv);
        }

        s_priv.remove(parser);
        s_self.remove(parser);
    }

    return NvDlaSuccess;
}

CaffeParser *CaffeParserFactory::priv(ICaffeParser *parser)
{
    // gLogError << __func__ << " looking up priv for base_i=" << parser << endl;
    nvdla::priv::BiMap<ICaffeParser *, CaffeParser *>::left_iterator f = s_priv.find_left(parser);
    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ICaffeParser *CaffeParserFactory::i(CaffeParser *parser)
{
    nvdla::priv::BiMap<ICaffeParser *, CaffeParser *>::right_iterator f = s_priv.find_right(parser);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}


ICaffeParser *CaffeParserFactory::self(void *s)
{
    nvdla::priv::BiMap<void *, ICaffeParser *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}

nvdla::priv::BiMap<ICaffeParser *, CaffeParser*> CaffeParserFactory::s_priv;
nvdla::priv::BiMap<void *, ICaffeParser*> CaffeParserFactory::s_self;


void BlobNameToTensor::add(const std::string& name, ITensor* tensor)
{
    mMap[name] = tensor;
}

ITensor* BlobNameToTensor::find(const char* name) const
{
    std::map<std::string, ITensor*>::const_iterator p = mMap.find(name);
    if (p == mMap.end()) {
        return 0;
    }
    return p->second;
}

ITensor*& BlobNameToTensor::operator[](const std::string& name)
{
    return mMap[name];
}

void BlobNameToTensor::setTensorNames()
{
    std::map<std::string, ITensor*>::iterator p;
    for ( p = mMap.begin(); p != mMap.end(); p++) {
        p->second->setName(p->first.c_str());
    }
}


BlobNameToTensor::~BlobNameToTensor() { }

struct CaffeParserPoolingDimsCallback : public INetwork::OutputDimensionsFormula
{
    // FB pooling parameters
    // Use floor((height + 2 * padding - kernel) / stride) + 1
    // instead of ceil((height + 2 * padding - kernel) / stride) + 1
    std::set<std::string> mHasTorchPooling;

    // TODO: mostly duplicated with code in engine
    virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride,
                          Dims2 tlPadding, Dims2 brPadding, const char* layerName) const /* override */
    {
        // check for overflow before we delve into any further computations here...
        assert( input.h + tlPadding.h + brPadding.h >= kernel.h );
        assert( input.w + tlPadding.w + brPadding.w >= kernel.w );
        int pooledH, pooledW;
        if (mHasTorchPooling.find(std::string(layerName)) != mHasTorchPooling.end())
        {
            pooledH = static_cast<int>
                (floor(static_cast<float>(input.h + tlPadding.h + brPadding.h - kernel.h) / stride.h)) + 1;
            pooledW = static_cast<int>
                (floor(static_cast<float>(input.w + tlPadding.w + brPadding.w - kernel.w) / stride.w)) + 1;
        } else
        {
            pooledH = static_cast<int>
                (ceil(static_cast<float>(input.h + tlPadding.h + brPadding.h - kernel.h) / stride.h)) + 1;
            pooledW = static_cast<int>
                (ceil(static_cast<float>(input.w + tlPadding.w + brPadding.w - kernel.w) / stride.w)) + 1;
        }

        if (tlPadding.h || tlPadding.w)
        {
            // DS: caffe comment for this (which doesn't work if padding is very large) is:
            // "If we have padding, ensure that the last pooling starts strictly inside the image (instead of at the padding); otherwise clip the last."
            if ((pooledH - 1) * stride.h >= input.h + tlPadding.h)
                --pooledH;
            if ((pooledW - 1) * stride.w >= input.w + tlPadding.w)
                --pooledW;

            assert((pooledH - 1) * stride.h < input.h + tlPadding.h);
            assert((pooledW - 1) * stride.w < input.w + tlPadding.w);
        }

        return Dims2(pooledH, pooledW);
    }

    Dims2 compute(Dims2 /*input*/, Dims2 /*kernel*/, Dims2 /*stride*/,
                  Dims2 /*tlPadding*/, Dims2 /*brPadding*/, Dims2 /*dilation*/, const char*) const
    {
        return Dims2(-1, -1);
    }

};

void CaffeParser::shutdownProtobufLibrary()
{
    google::protobuf::ShutdownProtobufLibrary();
}

// There are some challenges associated with importing caffe models. One is that
// a .caffemodel file just consists of layers and doesn't have the specs for its
// input and output blobs.
//
// So we need to read the deploy file to get the input

static bool readBinaryProto(dc::NetParameter* net, const char* file, size_t bufSize)
{
    CHECK_NULL_RET_VAL(net, false);
    CHECK_NULL_RET_VAL(file, false);
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in | std::ios::binary);
    if (!stream)
    {
        std::cout << "could not open file " << file << std::endl;
        return false;
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(int(bufSize), -1);

    bool ok = net->ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
    {
        std::cout << "Caffe Parser: could not parse binary model file" << std::endl;
        return false;
    }

    return ok;
}

static bool readTextProto(dc::NetParameter* net, const char* file)
{
    CHECK_NULL_RET_VAL(net, false);
    CHECK_NULL_RET_VAL(file, false);
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in );
    if (!stream)
    {
        std::cout << "could not open file " << file;
        return false;
    }

    IstreamInputStream input(&stream);
    bool ok = google::protobuf::TextFormat::Parse(&input, net);
    stream.close();

    if (!ok)
    {
        std::cout << "Caffe Parser: could not parse text file" << std::endl;
        return false;
    }

    return ok;
}

enum /*class*/ WeightType
{
    // types for convolution, deconv, fully connected
    kGENERIC = 0,	// typical weights for the layer: e.g. filter (for conv) or matrix weights (for innerproduct)
    kBIAS = 1,		// bias weights

    kMEAN = 0,
    kVARIANCE = 1,
    kMOVING_AVERAGE = 2
};


class CaffeWeightFactory
{
public:
    CaffeWeightFactory(const dc::NetParameter& msg, bool convertTo16bit, std::vector<void*>& tmpAllocs)
        : mMsg(msg), mTmpAllocs(tmpAllocs), m16bit(convertTo16bit), mOK(true)
    {
        mRef = new dc::NetParameter;
    }
    virtual ~CaffeWeightFactory() { }

    bool is16bit() const
    {
        return m16bit;
    }

    std::vector<void*>& getTmpAllocs()
    {
        return mTmpAllocs;
    }

    virtual Weights operator()(const std::string& layerName, WeightType weightType)
    {
        int numLayers = mMsg.layer_size();

        const dc::BlobProto* blobMsg;

        if (numLayers > 0)
        {
            int i = 0;
            for (; i < mMsg.layer_size(); i++)
            {
                std::string n = mMsg.layer(i).name();
                if (mMsg.layer(i).name() == layerName) {
                    break;
                }
            }

            int index = static_cast<int>(weightType);
            blobMsg = &mMsg.layer(i).blobs(index);
        }
        else
        {
            int i = 0;
            for (; i < mMsg.layers_size(); i++)
            {
                std::string n = mMsg.layers(i).name();
                if (mMsg.layers(i).name() == layerName) {
                    break;
                }
            }

            int index = static_cast<int>(weightType);
            blobMsg = &mMsg.layers(i).blobs(index);
        }


        if (!m16bit)
        {
            if (blobMsg->data_size() >0)
            {
                mOK &= checkForNans<float>(blobMsg->data().data(), int(blobMsg->data_size()), layerName);
                return Weights(DataType::FLOAT, blobMsg->data().data(), NvS64(blobMsg->data_size()));
            }
            std::cerr << layerName << ": ERROR - 32-bit weights not found for 32-bit model" << std::endl;
            mOK = false;
            return Weights(DataType::FLOAT, NULL, 0);
        }


        size_t count;
        float16* data;
        if (blobMsg->half_data_size() > 0)
        {
            count = blobMsg->half_data().size();
            data = (float16*)blobMsg->half_data().data();
            for (int i = 0; i < blobMsg->half_data().size(); i++) {
                // 'cos the fp16 data is stored in uint32, luvverly.
                data[i] = data[i * 2];
            }
        }
        else
        {
            count = blobMsg->data().size();
            data = reinterpret_cast<float16*>(malloc(count*sizeof(float16)));
            mTmpAllocs.push_back(data);
            float* data32 = (float*)blobMsg->data().data();
            for (size_t i = 0; i < count; i++)
            {
                if (data32[i]>std::numeric_limits<float16>::max() ||
                    data32[i] < -std::numeric_limits<float16>::max())
                {
                    std::cerr << "error:" << layerName << ": - weights are out"
                        " of range for 16-bit conversion" << std::endl;
                    mOK = false;
                }
                data[i] = data32[i];

            }
        }


        mOK &= checkForNans<float16>(data, count, layerName);
        return Weights(DataType::HALF, data, NvS64(count));
    }

    bool isOK()
    {
        return mOK;
    }

private:
    template<typename T> bool checkForNans(const void* values, int count, const std::string& layerName)
    {
        const T* v = reinterpret_cast<const T*>(values);
        for (int i = 0; i < count; i++)
        {
            if (std::isnan(float(v[i])))
            {
                NVDLA_UNUSED(layerName);
                // std::cout << layerName << ": Nan detected in weights" << std::endl;
                return false;
            }
        }
        return true;
    }

    const dc::NetParameter& mMsg;
    dc::NetParameter * mRef;
    std::vector<void*>& mTmpAllocs;
    bool m16bit;

    bool mOK;
};

static ILayer* parseConvolution(INetwork *network, const dc::LayerParameter& msg,
                                       CaffeWeightFactory& weightFactory,
                                       IBlobNameToTensor* tensors)
{
    const dc::ConvolutionParameter& p = msg.convolution_param();
    int numOutputs = p.num_output();
    int numGroups  = p.has_group()? p.group() : 1;
    ILayer* layer = NULL;

    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);

    int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
    int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

    int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
    int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

    int dilationW = p.dilation_size() > 0 ? p.dilation(0) : 1;
    int dilationH = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;

    BiasMode biasMode = BiasMode::bNONE;

    // TODO: cross-correlation vs convolution
    Weights kernelWeights = weightFactory(msg.name(), /*WeightType::*/kGENERIC);
    Weights biasWeights =
        (!p.has_bias_term() || p.bias_term()) ?
        weightFactory(msg.name(), /*WeightType::*/kBIAS) :
        Weights(DataType::FLOAT, NULL, 0);

    if ( biasWeights.count == 0 )
    {
        biasMode = BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = BiasMode::bm_ELEMENTWISE;
    }

    Dims2 tlPadding = Dims2(padH, padW);
    Dims2 brPadding = Dims2(padH, padW);
    Dims2 stride    = Dims2(strideH, strideW);
    Dims2 dilation  = Dims2(dilationH, dilationW);
    Dims2 kernelSize= Dims2(kernelH, kernelW);

    // TODO: cross-correlation vs convolution
    layer = network->addConvolution((*tensors)[msg.bottom(0)], numOutputs, 0,
                                    kernelSize, tlPadding, brPadding, stride, dilation,
                                    kernelWeights, biasWeights, biasMode, numGroups);

    return layer;
}

static ILayer* parsePooling(INetwork* network, const dc::LayerParameter&msg,
                                   CaffeWeightFactory& /*weightFactory*/,
                                   IBlobNameToTensor * tensors)
{
    const dc::PoolingParameter& p = msg.pooling_param();
    if (p.pool() != dc::PoolingParameter::MAX && p.pool() != dc::PoolingParameter::AVE)
    {
        gLogError << "only AVE and MAX pool operations are supported" << std::endl;
        return 0;
    }


    // mandatory
    int kernelH, kernelW;
    if (p.has_global_pooling() && p.global_pooling())
    {
        Dims4 dims = (*tensors)[msg.bottom(0)]->getDimensions();
        kernelH = dims.h;
        kernelW = dims.w;
    }
    else
    {
        // mandatory
        kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size();
        kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size();
    }

    int strideH = p.has_stride_h() ? p.stride_h() : p.has_stride() ? p.stride() : 1;
    int strideW = p.has_stride_w() ? p.stride_w() : p.has_stride() ? p.stride() : 1;

    int padH = p.has_pad_h() ? p.pad_h() : p.has_pad() ? p.pad() : 0;
    int padW = p.has_pad_w() ? p.pad_w() : p.has_pad() ? p.pad() : 0;

    Dims2 windowSize = Dims2(kernelH, kernelW);
    Dims2 stride     = Dims2(strideH, strideW);
    Dims2 tlPadding  = Dims2(padH, padW);
    Dims2 brPadding  = Dims2(padH, padW);

    PoolingType type = p.has_pool() && p.pool() ==
        dc::PoolingParameter::AVE ? PoolingType::kAVERAGE : PoolingType::kMAX;

    ILayer *layer = network->addPooling((*tensors)[msg.bottom(0)], type,
                                        windowSize, stride, tlPadding, brPadding);

    if (layer)
    {
        layer->setName(msg.name().c_str());
        if (p.has_torch_pooling() ? p.torch_pooling() : false) {
            static_cast<CaffeParserPoolingDimsCallback &>
                (network->getPoolingOutputDimensionsFormula()).mHasTorchPooling.insert(msg.name());
        }

        (*tensors)[msg.top(0)] = layer->getOutput(0);
    }
    return layer;
}

static ILayer* parseInnerProduct(INetwork* network, const dc::LayerParameter&msg,
                                        CaffeWeightFactory& weightFactory,
                                        IBlobNameToTensor * tensors)
{
    const dc::InnerProductParameter& p = msg.inner_product_param();
    int numOutputs = p.num_output();

    Weights kernelWeights = weightFactory(msg.name(), /*WeightType::*/kGENERIC);
    Weights biasWeights = !p.has_bias_term() || p.bias_term() ? weightFactory(msg.name(), /*WeightType::*/kBIAS) : Weights(DataType::FLOAT, NULL, 0);
    BiasMode biasMode = BiasMode::bNONE;

    if ( biasWeights.count == 0 )
    {
        biasMode = BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = BiasMode::bm_ELEMENTWISE;
    }

    return network->addFullyConnected((*tensors)[msg.bottom(0)], numOutputs,
                                      kernelWeights, biasWeights, biasMode);

}


static ILayer* parseReLU(INetwork* network, const dc::LayerParameter&msg,
                            CaffeWeightFactory& /*weightFactory*/,
                            IBlobNameToTensor * tensors)
{
    return network->addActivation((*tensors)[msg.bottom(0)], /*ActivationType::*/kRELU);
}

static ILayer* parseSoftMax(INetwork * network, const dc::LayerParameter&msg,
                                   CaffeWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    return network->addSoftMax((*tensors)[msg.bottom(0)]);
}

static ILayer* parseLRN(INetwork * network, const dc::LayerParameter&msg,
                               CaffeWeightFactory& /*weightFactory*/, IBlobNameToTensor* tensors)
{
    const dc::LRNParameter& p = msg.lrn_param();
    int localSize = p.has_local_size() ? p.local_size() : 5;
    float alpha = p.has_alpha() ? p.alpha() : 1;
    float beta = p.has_beta() ? p.beta() : 5;
    float k = p.has_k() ? p.k() : 1;

    return network->addLRN((*tensors)[msg.bottom(0)], localSize, alpha, beta, k);
}


static ILayer* parsePower(INetwork * network, const dc::LayerParameter&msg,
                                 CaffeWeightFactory& weightFactory, IBlobNameToTensor *tensors)
{
    const dc::PowerParameter& p = msg.power_param();

    float shift = p.has_shift() ? p.shift() : 0.0f;
    float scale = p.has_scale() ? p.scale() : 1.0f;
    float power = p.has_power() ? p.power() : 1.0f;

    if (power != 1.0f || shift != 0.0f)
    {
        //std::cout << "Caffe Parser: shift and power not supported in scale layers" << std::endl;
        return 0;
    }

    bool is16bit = weightFactory.is16bit();
    Weights wShift, wScale, wPower;
    if (is16bit)
    {
        float16* t = reinterpret_cast<float16*>(malloc(3 * sizeof(float16)));
        t[0] = float16(shift), t[1] = float16(scale), t[2] = float16(power);
        wShift = Weights(DataType::HALF, &t[0], 1);
        wScale = Weights(DataType::HALF, &t[1], 1);
        wPower = Weights(DataType::HALF, &t[2], 1);
        weightFactory.getTmpAllocs().push_back(t);
    }
    else
    {
        float* t = reinterpret_cast<float*>(malloc(3 * sizeof(float)));
        t[0] = shift, t[1] = scale, t[2] = power;
        wShift = Weights(DataType::FLOAT, &t[0], 1);
        wScale = Weights(DataType::FLOAT, &t[1], 1);
        wPower = Weights(DataType::FLOAT, &t[2], 1);
        weightFactory.getTmpAllocs().push_back(t);
    }


    return network->addScale((*tensors)[msg.bottom(0)], /*ScaleMode::*/sUNIFORM, wShift, wScale, wPower);
}


static ILayer* parseEltwise(INetwork * network, const dc::LayerParameter&msg,
                                   CaffeWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    const dc::EltwiseParameter& p = msg.eltwise_param();

    ElementWiseOperation op = /*ElementWiseOperation::*/kSUM;
    switch (p.operation())
    {
    case dc::EltwiseParameter_EltwiseOp_SUM: op = /*ElementWiseOperation::*/kSUM; break;
    case dc::EltwiseParameter_EltwiseOp_PROD: op = /*ElementWiseOperation::*/kPROD; break;
    case dc::EltwiseParameter_EltwiseOp_MAX: op = /*ElementWiseOperation::*/ew_kMAX; break;
    }

    return network->addElementWise((*tensors)[msg.bottom(0)], (*tensors)[msg.bottom(1)], op);
}


static ILayer* parseConcat(INetwork * network, const dc::LayerParameter&msg,
                                  CaffeWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    //const dc::ConcatParameter& p = msg.concat_param(); // TODO: unused

    std::vector<ITensor*> ptrs;
    for (unsigned int i = 0, n = msg.bottom_size(); i < n; i++) {
        ptrs.push_back((*tensors)[msg.bottom().Get(i)]);
    }

    return network->addConcatenation(&ptrs[0], msg.bottom_size());
}


static ILayer* parseDeconvolution(INetwork * network, const dc::LayerParameter& msg,
                                         CaffeWeightFactory& weightFactory, IBlobNameToTensor * tensors)
{
    const dc::ConvolutionParameter& p = msg.convolution_param();
    int numOutputs = p.num_output();

    BiasMode biasMode = BiasMode::bNONE;

    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);

    int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
    int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

    int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
    int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

    int dilationW = p.dilation_size() > 0 ? p.dilation(0) : 1;
    int dilationH = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;

    int numGroups = p.has_group()? p.group() : 1;

    Weights kernelWeights = weightFactory(msg.name(), /*WeightType::*/kGENERIC);
    Weights biasWeights =
        !p.has_bias_term() || p.bias_term() ?
        weightFactory(msg.name(), /*WeightType::*/kBIAS) :
        Weights(DataType::FLOAT, NULL, 0);

    if ( biasWeights.count == 0 )
    {
        biasMode = BiasMode::bNONE;
    }
    else if ( biasWeights.count == 1 )
    {
        biasMode = BiasMode::bUNIFORM;
    }
    else if ( biasWeights.count == numOutputs )
    {
        biasMode = BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = BiasMode::bm_ELEMENTWISE;
    }

    Dims2 stride = Dims2(strideH, strideW);
    Dims2 dilation  = Dims2(dilationH, dilationW);
    Dims2 tlPadding = Dims2(padH, padW);
    Dims2 brPadding = Dims2(padH, padW);
    Dims2 kernelSize = Dims2(kernelH, kernelW);

    ILayer *layer = network->addDeconvolution((*tensors)[msg.bottom(0)], numOutputs, 0,
                                              kernelSize, tlPadding, brPadding, stride, dilation,
                                              kernelWeights, biasWeights, biasMode, numGroups);

    if (numGroups != 1)
    {
        // std::cout << "Deconvolution layer: groups not supported" << std::endl;
        return 0;
    }

    return layer;
}

static ILayer* parseSigmoid(INetwork * network, const dc::LayerParameter&msg,
                                   CaffeWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    return network->addActivation((*tensors)[msg.bottom(0)], /*ActivationType::*/kSIGMOID);
}

static ILayer* parseTanH(INetwork * network, const dc::LayerParameter&msg,
                                   CaffeWeightFactory& /*weightFactory*/, IBlobNameToTensor * tensors)
{
    return network->addActivation((*tensors)[msg.bottom(0)], /*ActivationType::*/kTANH);
}

static ILayer* parseBatchNormalization(INetwork * network, const dc::LayerParameter &msg,
                                              CaffeWeightFactory& weightFactory, IBlobNameToTensor *tensors)
{
    const dc::BatchNormParameter& p = msg.batch_norm_param();

    Weights mean = weightFactory(msg.name(), /*WeightType::*/kMEAN);
    Weights variance = weightFactory(msg.name(), /*WeightType::*/kVARIANCE);
    Weights movingAverage = weightFactory(msg.name(), /*WeightType::*/kMOVING_AVERAGE);
    float eps = p.eps();
    float scaleFactor = 1.0f;
    float average = 0.0f;
    int i;

    average = *(static_cast<const float*>(movingAverage.values));
    if ( average == 0.0f )
    {
        gLogError << "Batch Normalization moving average is zero " << std::endl;
        return 0;
    }
    scaleFactor /= average;

    if (mean.count != variance.count)
    {
        gLogError << "Mean and variance have differing number of elements " << mean.count << " & " << variance.count << std::endl;
        return 0;
    }

    float *meanBlob = (float *)mean.values;
    float *varianceBlob = (float *)variance.values;

    Dims4 inputDims = (*tensors)[msg.bottom(0)]->getDimensions();
    BatchNormMode mode;

    if (mean.count == 1)
    {
        mode = BatchNormMode::bnUNIFORM;
        meanBlob[0] = meanBlob[0] * scaleFactor;
        varianceBlob[0] = varianceBlob[0] * scaleFactor;
    }
    else if (mean.count == inputDims.c)
    {
        mode = BatchNormMode::bnm_CHANNEL;
        for (i = 0; i < mean.count; i++)
        {
            meanBlob[i] = meanBlob[i] * scaleFactor;
            varianceBlob[i] = varianceBlob[i] * scaleFactor;
        }
    }
    else
    {
        gLogError << "Unknown batch norm mode" << std::endl;
        return 0;
    }

    /* DLA hardware expects mean and variance and not scale and shift */
    return network->addBatchNorm((*tensors)[msg.bottom(0)], mode, mean, variance, eps);
}

static ILayer* parseScale(INetwork* network, const dc::LayerParameter& msg,
                   CaffeWeightFactory& weightFactory, IBlobNameToTensor* tensors)
{
    const dc::ScaleParameter& p = msg.scale_param();

    Weights scale = weightFactory(msg.name(), WeightType::kGENERIC);
    Weights shift = p.has_bias_term() ? weightFactory(msg.name(), WeightType::kBIAS) : Weights(scale.type, NULL, 0);
    Weights power = Weights(scale.type, NULL, 0);
    Dims4 inputDims = (*tensors)[msg.bottom(0)]->getDimensions();
    ScaleMode mode;

    if (msg.bottom_size() > 1)
    {
        gLogError << "Parser can't handle more than 1 inputs to scale op" << std::endl;
        return 0;
    }

    if ( scale.count == 1 )
    {
        mode = ScaleMode::sUNIFORM;
    }
    else if ( scale.count == inputDims.c )
    {
        mode = ScaleMode::sCHANNEL;
    }
    else if ( scale.count == (inputDims.c * inputDims.h * inputDims.w) )
    {
        mode = ScaleMode::sm_ELEMENTWISE;
    }
    else
    {
        gLogError << "Unknown scale mode" << std::endl;
        return 0;
    }

    if ( shift.count > 0 )
    {
        if ( shift.count != scale.count )
        {
            gLogError << "Bias dims not same as scale dims" << std::endl;
            return 0;
        }
    }

    return network->addScale((*tensors)[msg.bottom(0)], mode, shift, scale, power);
}


typedef ILayer*(*LayerParseFn)(INetwork *,
                                      const dc::LayerParameter&,
                                      CaffeWeightFactory&,
                                      IBlobNameToTensor *);


typedef std::map<std::string, LayerParseFn> LayerParseFnMap;

LayerParseFnMap::value_type gParseTableData[] =
    {
        LayerParseFnMap::value_type("Convolution", parseConvolution),
        LayerParseFnMap::value_type("Pooling", parsePooling),
        LayerParseFnMap::value_type("InnerProduct", parseInnerProduct),
        LayerParseFnMap::value_type("ReLU", parseReLU),
        LayerParseFnMap::value_type("Softmax", parseSoftMax),
        LayerParseFnMap::value_type("SoftmaxWithLoss", parseSoftMax),
        LayerParseFnMap::value_type("LRN", parseLRN),
        LayerParseFnMap::value_type("Power", parsePower),
        LayerParseFnMap::value_type("Eltwise", parseEltwise),
        LayerParseFnMap::value_type("Concat", parseConcat),
        LayerParseFnMap::value_type("Deconvolution", parseDeconvolution),
        LayerParseFnMap::value_type("Sigmoid", parseSigmoid),
        LayerParseFnMap::value_type("TanH", parseTanH),
        LayerParseFnMap::value_type("BatchNorm", parseBatchNormalization),
        LayerParseFnMap::value_type("Scale", parseScale)
    };
const int nelems = sizeof gParseTableData / sizeof gParseTableData[0];
LayerParseFnMap gParseTable( gParseTableData, gParseTableData + nelems);




CaffeParser::~CaffeParser()
{

    std::vector<void*>::iterator v;

    for (v = mTmpAllocs.begin(); v!= mTmpAllocs.end(); v++) {
        free(*v);
    }

    delete mBlobNameToTensor;
}

const IBlobNameToTensor* CaffeParser::parse(const char* deployFile,
                                            const char* modelFile,
                                            INetwork * network)
{

    CHECK_NULL_RET_NULL(deployFile);
    CHECK_NULL_RET_NULL(modelFile);
    assert(mDimsCallback == 0);

    if (!mDimsCallback) {
        mDimsCallback = new CaffeParserPoolingDimsCallback;
    }
    network->setPoolingOutputDimensionsFormula(mDimsCallback);

    // this is used to deal with dropout layers which have different input and output
    mModel = new dc::NetParameter();
    if (!readBinaryProto(mModel/*.get()*/, modelFile, mProtobufBufferSize))
    {
        gLogError << "Could not parse model file" << std::endl;
        return 0;
    }

    mDeploy = new dc::NetParameter();
    if (!readTextProto(mDeploy/*.get()*/, deployFile))
    {
        gLogError << "Could not parse deploy file" << std::endl;
        return 0;
    }

    bool ok = true;
    CaffeWeightFactory weights(*mModel/**mModel.get()*/,
                               false /*weightType == DataType::kHALF*/, mTmpAllocs);

    mBlobNameToTensor = new BlobNameToTensor();

    for (int i = 0; i < mDeploy->input_size(); i++)
    {
        Dims4 dims;
        if (mDeploy->input_shape_size()) {
            dims.n = (int)mDeploy->input_shape().Get(i).dim().Get(0);
            dims.c = (int)mDeploy->input_shape().Get(i).dim().Get(1);
            dims.h = (int)mDeploy->input_shape().Get(i).dim().Get(2);
            dims.w = (int)mDeploy->input_shape().Get(i).dim().Get(3);
        }
        else { // deprecated, but still used in a lot of networks
            dims.n = (int)mDeploy->input_dim().Get(i * 4 + 0);
            dims.c = (int)mDeploy->input_dim().Get(i * 4 + 1);
            dims.h = (int)mDeploy->input_dim().Get(i * 4 + 2);
            dims.w = (int)mDeploy->input_dim().Get(i * 4 + 3);
        }

        ITensor* tensor = network->addInput(mDeploy->input().Get(0).c_str(), dims);
        mBlobNameToTensor->add(mDeploy->input().Get(0), tensor);

    }

    for (int i = 0; i < mDeploy->layer_size() && ok; i++)
    {
        const dc::LayerParameter& layerMsg = mDeploy->layer(i);
        if (layerMsg.has_phase() && layerMsg.phase() == dc::TEST) {
            continue;
        }

        if (layerMsg.type() == "Dropout")
        {
            mBlobNameToTensor->add(layerMsg.top().Get(0),
                                   mBlobNameToTensor->find(layerMsg.bottom().Get(0).c_str()));
            continue;
        }

        if (layerMsg.type() == "Input")
        {
            const dc::InputParameter& p = layerMsg.input_param();
            for (int i = 0; i < layerMsg.top_size(); i++)
            {
                const dc::BlobShape& shape = p.shape().Get(i);
                Dims4 dims(shape.dim().Get(0), shape.dim().Get(1), shape.dim().Get(2), shape.dim().Get(3));
                ITensor* tensor = network->addInput(layerMsg.top(i).c_str(), dims);
                mBlobNameToTensor->add(layerMsg.top().Get(i), tensor);
            }
            continue;
        }
        if (layerMsg.type() == "Flatten")
        {
            ITensor* tensor = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = tensor;
            std::cout << "Warning: Flatten layer ignored." << std::endl;
            continue;
        }

        LayerParseFnMap::iterator v = gParseTable.find(layerMsg.type());

        if (v == gParseTable.end())
        {
            gLogError << "could not parse layer type " << layerMsg.type() << std::endl;
            ok = false;
        }
        else
        {
            ILayer* layer = (*v->second)(network, layerMsg, weights, mBlobNameToTensor);

            if (layer == 0)
            {
                gLogError << "error: parsing layer type " << layerMsg.type() <<
                    " index " << i << std::endl;
                ok = false;
            }
            else
            {
                layer->setName(layerMsg.name().c_str());
                mBlobNameToTensor->add(layerMsg.top(0), layer->getOutput(0));
            }
        }
    }

    mBlobNameToTensor->setTensorNames();
    return ok && weights.isOK() ? mBlobNameToTensor : 0;
}

int CaffeParser::identifyOutputs(INetwork * network)
{
    std::set< ITensor* > outputTensors;
    std::set< ITensor* > inputTensors;

    for (int l = 0; l < network->getNumLayers(); ++l)
    {
        ILayer* layer = network->getLayer(l);
        if (!layer)
            return -1;

        for (int ii = 0; ii < layer->getNumInputs(); ++ii) {
            inputTensors.insert(layer->getInput(ii));
        }

        for (int oo = 0; oo < layer->getNumOutputs(); ++oo)
        {
            outputTensors.insert(layer->getOutput(oo));
        }
    }

    for (std::set<ITensor*>::iterator oi = outputTensors.begin(); oi != outputTensors.end(); ++oi)
    {
        // an output tensor which is not an input to any other layers is a network output tensor
        if (inputTensors.find(*oi) == inputTensors.end())
        {
            network->markOutput(*oi);
            gLogInfo << "mark " << (*oi)->getName() << std::endl;
        }
    }

    return network->getNumOutputs();
}

BinaryProtoBlob::BinaryProtoBlob(void* memory, DataType type, Dims4 dimensions) :
    mMemory(memory), mDataType(type), mDimensions(dimensions)
{
}

Dims4 BinaryProtoBlob::getDimensions()
{
    return mDimensions;
}

const void* BinaryProtoBlob::getData()
{
    return mMemory;
}

void BinaryProtoBlob::destroy()
{
    delete this;
}

BinaryProtoBlob::~BinaryProtoBlob()
{
    free(mMemory);
}

} // nvdla::caffe::priv
} // nvdla::caffe
} // nvdla::
