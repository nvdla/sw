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

#include <algorithm>
#include <map>
#include <sstream>
#include <tgmath.h>

#include "priv/CanonicalAST.h"
#include "priv/Check.h"
#include "priv/Layer.h"
#include "priv/Network.h"
#include "priv/Tensor.h"
#include "priv/Wisdom.h"
#include "priv/WisdomContainer.h"

using std::map;
using std::vector;
using std::string;
using std::endl;
using std::stringstream;

namespace nvdla {

INetwork::INetwork() { }
INetwork::~INetwork() { }

INetwork *createNetwork()
{
    priv::NetworkFactory::NetworkPrivPair n = priv::NetworkFactory::newNetwork();
    return n.i();
}

NvDlaError destroyNetwork(INetwork *network)
{
    NvDlaError e = NvDlaSuccess;
    PROPAGATE_ERROR_FAIL(priv::NetworkFactory::deleteNetwork(network));

fail:
    return e;
}

// TBD: why is this not Network
Dims2 INetwork::NetworkDefaultConvolutionFormula::compute(Dims2 input, Dims2 kernel,
                                                          Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const
{
    return Dims2((input.h + tlPadding.h + brPadding.h - kernel.h) / stride.h + 1,
                 (input.w + tlPadding.w + brPadding.w - kernel.w) / stride.w + 1);
}

Dims2 INetwork::NetworkDefaultConvolutionFormula::compute(Dims2 input, Dims2 kernel,
                                                          Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char*) const
{
    NvS32 dilatedH = (kernel.h - 1)*dilation.h + 1;
    NvS32 dilatedW = (kernel.w - 1)*dilation.w + 1;
    return Dims2((input.h + tlPadding.h + brPadding.h - dilatedH) / stride.h + 1,
                 (input.w + tlPadding.w + brPadding.w - dilatedW) / stride.w + 1);
}

Dims2 INetwork::NetworkDefaultDeconvolutionFormula::compute(Dims2 input, Dims2 kernel,
                                                            Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const
{
    // exact inverse of the computation for convolution forward
    return Dims2((input.h - 1) * stride.h + kernel.h - (tlPadding.h + brPadding.h),
                 (input.w - 1) * stride.w + kernel.w - (tlPadding.w + brPadding.w));
}

Dims2 INetwork::NetworkDefaultDeconvolutionFormula::compute(Dims2 input, Dims2 kernel,
                                                            Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char*) const
{
    NvS32 dilatedH = (kernel.h - 1)*dilation.h + 1;
    NvS32 dilatedW = (kernel.w - 1)*dilation.w + 1;
    // exact inverse of the computation for convolution forward
    return Dims2((input.h - 1) * stride.h + dilatedH - (tlPadding.h + brPadding.h),
                 (input.w - 1) * stride.w + dilatedW - (tlPadding.w + brPadding.w));
}

Dims2 INetwork::NetworkDefaultPoolingFormula::compute(Dims2 input, Dims2 kernel,
                                                      Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const
{
    int pooledH, pooledW;
    pooledH = static_cast<int>
                (ceil(static_cast<float>(input.h + tlPadding.h + brPadding.h - kernel.h) / stride.h)) + 1;
    pooledW = static_cast<int>
                (ceil(static_cast<float>(input.w + tlPadding.w + brPadding.w - kernel.w) / stride.w)) + 1;


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

namespace priv {



static INetwork::NetworkDefaultConvolutionFormula   sDefaultConvDims;
static INetwork::NetworkDefaultDeconvolutionFormula sDefaultDeconvDims;
static INetwork::NetworkDefaultPoolingFormula       sDefaultPoolingDims;

NetworkFactory::NetworkPrivPair NetworkFactory::newNetwork()
{
    INetwork *network;
    Network *network_priv;
    network = network_priv = new priv::Network();
    if (network) {
        s_priv.insert(network, network_priv);
        s_self.insert(network, network);
    }
    return NetworkPrivPair(network, network_priv);
}

NvDlaError NetworkFactory::deleteNetwork(INetwork *network)
{
    if (network != NULL) {
        Network *network_priv = priv(network);
        if (network_priv != NULL) {
            delete network_priv;
        }

        s_priv.remove(network);
        s_self.remove(network);
    }

    return NvDlaSuccess;
}

Network *NetworkFactory::priv(INetwork *network)
{
    BiMap<INetwork *, Network *>::left_iterator f = s_priv.find_left(network);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

INetwork *NetworkFactory::i(Network *network)
{
    BiMap<INetwork *, Network *>::right_iterator f = s_priv.find_right(network);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

INetwork *NetworkFactory::self(void *s)
{
    BiMap<void *, INetwork *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}


INetwork *NetworkFactory::deserializeFrom(WisdomContainerEntry *entry)
{
    //     gLogError << __func__ << endl;
    bool ok = true;
    NVDLA_UNUSED(ok);
    INetwork *network = NULL;

    if ( entry->type() != IWisdomContainerEntry::ENTRY_TYPE_OBJECT ) {
        gLogError << __func__ << " container should be of object type" << endl;
        goto done;
    }

    // only one type of network right now (INetwork/Network)...
    //WisdomContainerEntry factory_type_entry;
    //    NetworkTypeEnum factory_type;
    //    NvU32 v;
    //    ok = entry->getEntry("factory_type", IWisdomContainerEntry::ENTRY_TYPE_UINT32, &factory_type_entry);
    //    ok = ok && factory_type_entry.readUInt32(v);
    //    if ( !ok ) {
    //	goto done;
    //    }
    //factory_type = LayerTypeEnum::underlying_type(v);
    //ok = factory_type.valid();
    //if ( !ok ) {
    //		goto done;
    //	}

    //	switch ( factory_type.e() )
    //	{
    //	case NetworkFactoryType::canonical_ast:
    network = deserializeNetwork(entry);
    //default:
    // but, shouldn't be possible since l_type.valid() is true...
    // ok = false;
    //		goto done;
    //	}


 done:
    return network;
}

BiMap<INetwork *, Network*> NetworkFactory::s_priv;
BiMap<void *, INetwork*> NetworkFactory::s_self;

// there's only one type of "Tensor" for now. so only one of these... so it looks
// silly.  see the same paths in "LayerFactory::deserialize*" for why it makes sense
// to organize this way preemptively.
INetwork *NetworkFactory::deserializeNetwork(WisdomContainerEntry *entry)
{
    //    gLogError << __func__ << endl;
    NetworkFactory::NetworkPrivPair n = NetworkFactory::newNetwork();
    if ( !n ) {
        gLogError << __func__ << " error allocating new network" << endl;
        return NULL;
    }
    n.priv()->deserializeFrom(entry);
    return n.i();
}


Network::Network() :
    mConvDims(&sDefaultConvDims),
    mDeconvDims(&sDefaultDeconvDims),
    mPoolDims(&sDefaultPoolingDims)
{

}

NvU16 Network::getFactoryType() const
{
    return 0; // only one type of network so far, not complicated by factory splits
}


IConvolutionLayer* Network::addConvolution(ITensor* inputTensor, int numOutputChannels, int paddingValue,
                                           Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                           Weights kernelWeights,
                                           Weights biasWeights, BiasMode biasMode, int numGroups)
{
    API_CHECK_NULL_RET_NULL(inputTensor);
    API_CHECK_RETVAL(numOutputChannels >= 1 && numOutputChannels < MAX_OUTPUT_MAPS, 0 );
    API_CHECK_RETVAL(kernelSize.h > 0, 0);
    API_CHECK_RETVAL(kernelSize.w > 0, 0);
    API_CHECK_RETVAL((kernelSize.h * kernelSize.w) < MAX_KERNEL_DIMS_PRODUCT, 0);
    API_CHECK_WEIGHTS_RETVAL(kernelWeights, 0);
    API_CHECK_WEIGHTS0_RETVAL(biasWeights, 0);
    API_CHECK_ENUM_RANGE_RETVAL(BiasMode, biasMode, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    ConvolutionLayerDiamond d =
        LayerFactory::newConvolutionLayer(this, name,
                                          inputTensor,
                                          output, numOutputChannels, paddingValue,
                                          kernelSize, tlPadding, brPadding, stride, dilation,
                                          kernelWeights, biasWeights, biasMode, numGroups);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


IFullyConnectedLayer* Network::addFullyConnected(ITensor* inputTensor, int outputSize,
                                                 Weights kernelWeights, Weights biasWeights, BiasMode biasMode)
{
    API_CHECK_NULL_RET_NULL(inputTensor);
    API_CHECK_RETVAL(outputSize >= 1 && outputSize < MAX_OUTPUT_MAPS, 0);
    API_CHECK_WEIGHTS_RETVAL(kernelWeights, 0);
    API_CHECK_WEIGHTS0_RETVAL(biasWeights, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    FullyConnectedLayerDiamond d =
        LayerFactory::newFullyConnectedLayer(this, name,
                                             inputTensor,
                                             output, outputSize,
                                             kernelWeights, biasWeights, biasMode);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


IActivationLayer* Network::addActivation(ITensor* inputTensor, ActivationType type)
{
    API_CHECK_NULL_RET_NULL(inputTensor);
    API_CHECK_ENUM_RANGE_RETVAL(ActivationType, type, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    ActivationLayerDiamond d =
        LayerFactory::newActivationLayer(this, name, inputTensor, output, type);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


IPoolingLayer* Network::addPooling(ITensor* input, PoolingType type,
                                   Dims2 windowSize, Dims2 stride,
                                   Dims2 tlPadding, Dims2 brPadding)
{
    API_CHECK_NULL_RET_NULL(input);
    API_CHECK_RETVAL(type.v() <= EnumMax<PoolingType>(), 0);
    API_CHECK_RETVAL(windowSize.h > 0, 0);
    API_CHECK_RETVAL(windowSize.w > 0, 0);
    API_CHECK_RETVAL((windowSize.h*windowSize.w) < MAX_KERNEL_DIMS_PRODUCT, 0);
    API_CHECK_RETVAL((stride.h + stride.w) < MAX_STRIDE_SUM, 0);
    API_CHECK_RETVAL(tlPadding.w < windowSize.w, 0);
    API_CHECK_RETVAL(tlPadding.h < windowSize.h, 0);
    API_CHECK_RETVAL(brPadding.w < windowSize.w, 0);
    API_CHECK_RETVAL(brPadding.h < windowSize.h, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    PoolingLayerDiamond d =
        LayerFactory::newPoolingLayer(this, name, input, output,
                                      type,
                                      windowSize, stride,
                                      tlPadding, brPadding);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


ILRNLayer* Network::addLRN(ITensor* input, int lrnWindow, float alpha, float beta, float k)
{
    API_CHECK_NULL_RET_NULL(input);
    API_CHECK_RETVAL(lrnWindow >= ILRNLayer::Parameters::minWindowSize() &&
                     lrnWindow <= ILRNLayer::Parameters::maxWindowSize(), 0);
    API_CHECK_RETVAL(fabsf(alpha) <= ILRNLayer::Parameters::maxAbsAlpha(), 0);
    API_CHECK_RETVAL(beta >= ILRNLayer::Parameters::minBeta() &&
                     beta <= ILRNLayer::Parameters::maxBeta(), 0);
    API_CHECK_RETVAL(k >= ILRNLayer::Parameters::minK() &&
                     k <= ILRNLayer::Parameters::maxK(), 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    LRNLayerDiamond d =
        LayerFactory::newLRNLayer(this, name, input, output,
                                  lrnWindow, alpha, beta, k);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


IScaleLayer* Network::addScale(ITensor* inputTensor, ScaleMode mode,
                               Weights shift, Weights scale, Weights power)
{
    API_CHECK_NULL_RET_NULL(inputTensor);
    API_CHECK_ENUM_RANGE_RETVAL(ScaleMode, mode, 0);
    API_CHECK_RETVAL(scale.type == shift.type && shift.type == power.type, 0);
    API_CHECK_WEIGHTS0_RETVAL(shift, 0);
    API_CHECK_WEIGHTS0_RETVAL(scale, 0);
    API_CHECK_WEIGHTS0_RETVAL(power, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    ScaleLayerDiamond d =
        LayerFactory::newScaleLayer(this, name, inputTensor, output,
                                    mode, shift, scale, power);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}

IBatchNormLayer* Network::addBatchNorm(ITensor* inputTensor, BatchNormMode mode,
                                       Weights mean, Weights variance, float epsilon)
{
    API_CHECK_NULL_RET_NULL(inputTensor);
    API_CHECK_ENUM_RANGE_RETVAL(BatchNormMode, mode, 0);
    API_CHECK_RETVAL(mean.type == variance.type, 0);
    API_CHECK_WEIGHTS0_RETVAL(mean, 0);
    API_CHECK_WEIGHTS0_RETVAL(variance, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());

    BatchNormLayerDiamond d =
            LayerFactory::newBatchNormLayer(this, name, inputTensor, output,
                                            mode, mean, variance, epsilon);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );

    mLayers.push_back(d.base().i());
    return d.derived().i();
}

ISoftMaxLayer* Network::addSoftMax(ITensor* input)
{
    API_CHECK_NULL_RET_NULL(input);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    SoftMaxLayerDiamond d =
        LayerFactory::newSoftMaxLayer(this, name, input, output);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


IConcatenationLayer *Network::addConcatenation(ITensor * const * inputs, int numInputs )
{
    API_CHECK_RETVAL(numInputs > 0 && numInputs < MAX_CONCAT_INPUTS, 0);
    API_CHECK_NULL_RET_NULL(inputs);
    for (int j = 0; j < numInputs; j++) {
        API_CHECK_NULL_RET_NULL(inputs[j]);
    }

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    ConcatenationLayerDiamond d =
        LayerFactory::newConcatenationLayer(this, name,
                                            inputs, numInputs, output);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}


ISliceLayer *Network::addSlice(ITensor* input, int numOutputs)
{
    API_CHECK_RETVAL(numOutputs > 0 && numOutputs < MAX_SLICE_OUTPUTS, 0);
    API_CHECK_NULL_RET_NULL(input);

    string name = newLayerName();

    ITensor* outputs[numOutputs];
    for (int ii=0; ii<numOutputs; ii++)
    {
        outputs[ii] = addTensor(newTensorName());
    }

    SliceLayerDiamond d =
        LayerFactory::newSliceLayer(this, name, input, outputs, numOutputs);

    for (int ii=0; ii<numOutputs; ii++)
    {
        outputs[ii]->setDimensions(d.derived().priv()->getOutputDimensions());
    }

    mLayers.push_back(d.base().i());
    return d.derived().i();
}


IDeconvolutionLayer*
Network::addDeconvolution
(
 ITensor* input,
 int     numOutputs, int paddingValue,
 Dims2   kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
 Weights kernelWeights,
 Weights biasWeights, BiasMode biasMode, int numGroups
 )
{
    API_CHECK_NULL_RET_NULL(input);
    API_CHECK_RETVAL(numOutputs > 0 && numOutputs < MAX_OUTPUT_MAPS, 0);
    API_CHECK_RETVAL(kernelSize.h > 0, 0);
    API_CHECK_RETVAL(kernelSize.w > 0, 0);
    API_CHECK_RETVAL((kernelSize.h * kernelSize.w) < MAX_KERNEL_DIMS_PRODUCT, 0);
    API_CHECK_WEIGHTS_RETVAL(kernelWeights, 0);
    API_CHECK_WEIGHTS0_RETVAL(biasWeights, 0);
    API_CHECK_ENUM_RANGE_RETVAL(BiasMode, biasMode, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    DeconvolutionLayerDiamond d =
        LayerFactory::newDeconvolutionLayer(this, name,
                                            input, output, numOutputs, paddingValue,
                                            kernelSize, tlPadding, brPadding, stride, dilation,
                                            kernelWeights, biasWeights, biasMode, numGroups);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );

    mLayers.push_back(d.base().i());
    return d.derived().i();
}

IElementWiseLayer*
Network::addElementWise(ITensor* input1, ITensor* input2, ElementWiseOperation op)
{
    API_CHECK_NULL_RET_NULL(input1);
    API_CHECK_NULL_RET_NULL(input2);
    API_CHECK_ENUM_RANGE_RETVAL(ElementWiseOperation, op, 0);

    string name = newLayerName();

    ITensor* output = addTensor(newTensorName());
    Tensor*  output_priv = TensorFactory::priv(output);
    NVDLA_UNUSED(output_priv);

    ITensor* inputs[2] = { input1, input2 };

    ElementWiseLayerDiamond d =
        LayerFactory::newElementWiseLayer(this, name, inputs, output, op);

    output->setDimensions( d.derived().priv()->getOutputDimensions() );


    mLayers.push_back(d.base().i());
    return d.derived().i();
}

ITensor* Network::addInput(const char* name, Dims4 dims)
{
    API_CHECK_NULL_RET_NULL(name);
    API_CHECK_DIMS4_TENSOR_RETVAL(dims, 0);
    ITensor* tensor = addTensor(string(name));
    tensor->setDimensions(dims);
    mInputs.push_back(tensor);
    return tensor;
}


bool Network::markInput(ITensor *tensor)
{
    API_CHECK_NULL_RETVAL(tensor, false);
    API_CHECK_DIMS3_TENSOR_RETVAL(tensor->getDimensions(), false);
    //TBD: check that this isn't already marked.
    mInputs.push_back(tensor);
    return true;
}

void Network::markOutput(ITensor* tensor)
{
    API_CHECK_NULL(tensor);
    if (std::find(mOutputs.begin(), mOutputs.end(), tensor) == mOutputs.end())
    {
        mOutputs.push_back(tensor);
    }
}

int Network::getNumLayers() const
{
    return static_cast<int>(mLayers.size());
}

ILayer* Network::getLayer(int i) const
{
    if (i < 0 || i >= int(mLayers.size())) {
        return 0;
    }
    return mLayers[i];
}

void Network::destroy()
{
    delete this;
}

Network::~Network()
{

}

const ILayer* Network::findLayer(const string& name) const
{
    vector< ILayer * >::const_iterator it;
    for (it = mLayers.begin(); it != mLayers.end(); it++ ) {
        if ( (*it)->getName() == name)
            return *it;
    }

    return 0;
}

string Network::newTensorName() const
{
    stringstream s;
    s << "tensor-anon-" << mTensors.size();
    return s.str();
}

string Network::newLayerName() const
{
    stringstream s;
    s << "layer-anon-" << mLayers.size();
    return s.str();
}

void Network::setPoolingOutputDimensionsFormula(INetwork::OutputDimensionsFormula* callback)
{
    mPoolDims = callback ? callback : &sDefaultPoolingDims;
}

void Network::setConvolutionOutputDimensionsFormula(INetwork::OutputDimensionsFormula* callback)
{
    mConvDims = callback ? callback : &sDefaultConvDims;
}

void Network::setDeconvolutionOutputDimensionsFormula(INetwork::OutputDimensionsFormula* callback)
{
    mDeconvDims = callback ? callback : &sDefaultDeconvDims;
}




int Network::getNumInputs() const
{
    return (int)mInputs.size();
}

int Network::getNumOutputs() const
{
    return (int)mOutputs.size();
}

ITensor* Network::getOutput(int index) const
{
    if (index < 0 || index >= int(mOutputs.size())) {
        return 0;
    }
    return mOutputs[index];
}

ITensor* Network::getInput(int index) const
{
    if (index < 0 || index >= int(mInputs.size())) {
        return 0;
    }
    return mInputs[index];
}

INetwork::OutputDimensionsFormula& Network::getPoolingOutputDimensionsFormula() const
{
    return *mPoolDims;
}

INetwork::OutputDimensionsFormula& Network::getConvolutionOutputDimensionsFormula() const
{
    return *mConvDims;
}

INetwork::OutputDimensionsFormula& Network::getDeconvolutionOutputDimensionsFormula() const
{
    return *mDeconvDims;
}

const vector<ITensor*>& Network::getInputs()  const
{
    return mInputs;
}
const vector< ILayer * >& Network::getLayers()  const
{
    return mLayers;
}
const vector<ITensor *>& Network::getOutputs() const
{
    return mOutputs;
}

ITensor* Network::addTensor(const string &s)
{
    TensorFactory::TensorPrivPair t = TensorFactory::newTensor();
    if ( !t ) {
        return NULL;
    }
    t.priv()->setNetwork(this);
    t.priv()->setName(s.c_str());
    mTensors.push_back(t.i());
    return t.i();
}

bool Network::assignSymbols(Wisdom *wisdom)
{
    bool ok = true;

    for (size_t l = 0; l < mLayers.size(); l++ ) {
        Layer *layer = LayerFactory::priv(mLayers[l]);
        if ( !layer ) {
            gLogError << "missing layer " << l << " in network?" << endl;
            continue;
        }

        string sym;
        ok = wisdom->findLayerSymbol(layer, sym);
        if ( ! ok ) {
            ok = wisdom->assignLayerSymbol(layer, sym);
            if ( !ok ) {
                gLogError << "unable to assign symbol name to layer " << layer->getName() << " ?" << endl;
                goto done;
            }
        }

        // tell the layer to assign symbols for whatever it references...
        ok = layer->assignSymbols(wisdom);
        if ( !ok ) {
            gLogError << "unable to assign symbols for layer " << layer->getName() << endl;
            goto done;
        }
    }

 done:

    return ok;
}


bool Network::serializeTo(WisdomContainerEntry *e) const
{
    vector<Layer *> layers;
    Wisdom *wisdom;
    WisdomContainerEntry inputs_entry, outputs_entry, layers_entry;
    map<Tensor *, bool> gather_tensors;

    bool ok = e && e->container_priv() && e->container_priv()->wisdom();
    if ( !ok ) {
        gLogError << "can't serialize a network without a working wisdom context." << endl;
        goto done;
    }

    wisdom = e->container_priv()->wisdom_priv();

    // gLogError << "serializing network with " << getNumLayers() << " layers, " << getNumInputs() << " inputs, and " << getNumOutputs() << " outputs." << endl;

    ok = ok && e->writeUInt32("factory_type", getFactoryType());
    ok = ok && e->writeUInt32("num_inputs",   getNumInputs());
    ok = ok && e->writeUInt32("num_outputs",  getNumOutputs());
    ok = ok && e->writeUInt32("num_layers",   getNumLayers());

    ok = ok && e->insertEntryIfNotPresent("inputs", IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &inputs_entry);
    ok = ok && e->insertEntryIfNotPresent("outputs",IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &outputs_entry);
    ok = ok && e->insertEntryIfNotPresent("layers", IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &layers_entry);


    for (size_t l = 0; l < mLayers.size(); l++ ) {
        ILayer *ilayer = mLayers[l];
        Layer *layer = LayerFactory::priv(ilayer);

        if ( !(ilayer && layer) ) {
            gLogError << "missing layer " << l << " in network?" << endl;
            ok = false;
            goto done;
        }
        layers.push_back(layer);

        // be sure there's a symbol associated with the layer so it can be referred
        // to by the network later during deserialization.
        string sym;
        ok = wisdom->findLayerSymbol(layer, sym);

        // layer symbols should already have been assigned if needed.
        if ( ! ok ) {
            gLogError << "unassigned layer " << layer->getName() << " ?" << endl;
            goto done;
        }

        ok = ok && layers_entry.writeString(toString(l), sym);
        // gLogError << "writing symbol=[" << sym << "] to layers_entry=[" << ss.str() << "]" << endl;
        if ( !ok ) {
            gLogError << "failed to write symbol to layers_entry index" << endl;
            goto done;
        }
    }

    //
    // gather up all tensors referred to by the layers and set them
    //
    for (size_t l = 0; l < layers.size(); l++ ) {
        int num_inputs = layers[l]->getNumInputs();
        int num_outputs  = layers[l]->getNumOutputs();
        for(int i = 0; i < num_inputs; i++) {
            gather_tensors[TensorFactory::priv(layers[l]->getInput(i))] = true;
        }
        for(int o = 0; o < num_outputs; o++) {
            gather_tensors[TensorFactory::priv(layers[l]->getOutput(o))] = true;
        }
    }

    for ( map<Tensor *, bool>::iterator ti = gather_tensors.begin();
          ok && (ti != gather_tensors.end());
          ti++ ) {
        Tensor *t = ti->first;
        ok = ok && wisdom->setTensor(t);
    }

    if ( !ok ) {
        gLogError << __func__ << " failed to serialize one or more tensors" << endl;
        goto done;
    }


    //
    // now set the layers
    //
    for (size_t l = 0; l < layers.size(); l++ ) {
        ok = ok && wisdom->setLayer(layers[l]);
    }

    if ( !ok ) {
        gLogError << __func__ << " failed to serialize one or more layers" << endl;
        goto done;
    }

    //
    // record the input and output tensors
    //
    for (size_t i = 0; i < mInputs.size(); i++ ) {
        string sym;
        ok = ok && wisdom->findITensorSymbol(getInput(i), sym);
        ok = ok && inputs_entry.writeString(toString(i), sym);
    }
    if ( !ok ) {
        gLogError << __func__ << " failed to serialize one or more inputs" << endl;
        goto done;
    }

    for (size_t o = 0; o < mOutputs.size(); o++ ) {
        string sym;
        ok = ok && wisdom->findITensorSymbol(getOutput(o), sym);
        ok = ok && outputs_entry.writeString(toString(o), sym);
    }
    if ( !ok ) {
        gLogError << __func__ << " failed to serialize one or more outputs" << endl;
        goto done;
    }

 done:

    return ok;

}

//
// read from the wisdom container entry and deserialize
// all the objects/data necessary to bring the network
// into live memory.
//
bool Network::deserializeFrom(WisdomContainerEntry *e)
{
    canonical_ast::Graph *graph;
    NVDLA_UNUSED(graph);

    //    gLogError << __func__ << " 1 " << endl;
    Wisdom *wisdom;
    WisdomContainerEntry inputs_entry, outputs_entry, layers_entry;
    NvU32 num_inputs, num_outputs, num_layers;
    map<string, ITensor *> tensor_symbols;
    vector<ILayer *> layers;

    bool ok = true;

    wisdom = e->container_priv()->wisdom_priv();
    ok = NULL != wisdom;
    if ( !ok ) {
        gLogError << __func__ << "missing Wisdom" << endl;
        goto done;
    }

    ok = ok && e->getEntry(string("num_inputs"),  IWisdomContainerEntry::ENTRY_TYPE_UINT32, &inputs_entry);
    ok = ok && e->getEntry(string("num_outputs"), IWisdomContainerEntry::ENTRY_TYPE_UINT32, &outputs_entry);
    ok = ok && e->getEntry(string("num_layers"),  IWisdomContainerEntry::ENTRY_TYPE_UINT32, &layers_entry);

    ok = ok && inputs_entry. readUInt32(num_inputs);
    ok = ok && outputs_entry.readUInt32(num_outputs);
    ok = ok && layers_entry. readUInt32(num_layers);

    if ( !ok ) {
        gLogError << __func__ << " failed to get all num_* entries" << endl;
        goto done;
    }

    //XXX upper bounds check?
    if ( num_inputs == 0 || num_outputs == 0 || num_layers == 0 ) {
        //ok = false;
        gLogError << __func__ << " invalid network deserialization data?" << endl;
        gLogError << __func__ << " inputs=" << num_inputs << " outputs=" << num_outputs << " layers=" <<
            num_layers << endl;
        //	goto done;
    }

    // note re-use of the *_entry locals from above...
    ok = ok && e->getEntry(string("inputs"),  IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &inputs_entry);
    ok = ok && e->getEntry(string("outputs"), IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &outputs_entry);
    ok = ok && e->getEntry(string("layers"),  IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &layers_entry);
    if ( !ok ) {
        gLogError << __func__ << " failed to get inputs, outputs and layers entries" << endl;
        goto done;
    }

    //
    // Gather up the layers referenced by the network.  For each, ascertain
    // whether or not it has been deserialized (check with the Wisdom).
    // If not, do so.
    //
    // The set of layers used by the network is stored as an array of layer symbols.
    //

    layers.clear();

    for ( size_t l = 0; ok && (l < num_layers); l++ ) {
        WisdomContainerEntry layer_index_entry;
        string layer_symbol;
        ILayer *layer;

        ok = ok && layers_entry.getEntry(toString(l),
                                         IWisdomContainerEntry::ENTRY_TYPE_STRING,
                                         &layer_index_entry);
        if ( !ok ) {
            gLogError << "couldn't get layers entry for " << toString(l) << endl;
            goto done;
        }
        ok = ok && layer_index_entry.readString(layer_symbol);
        if ( !ok ) {
            gLogError << "couldn't read layer index symbol string? " << toString(l) << endl;
            goto done;
            break;
        }

        layer = wisdom->getLayerFromSymbol(layer_symbol);

        ok = (NULL != layer);
        if ( ok ) {
            layers.push_back(layer);
        } else {
            gLogError << "couldn't get layer from symbol=[" << layer_symbol << "]" << endl;
            goto done;
        }

        mLayers.push_back(layer);
    }
    if ( !ok ) {
        gLogError << __func__ << " failed to find or instantiate (some) network layers" << endl;
        goto done;
    }


    // go through the input and output tensors and mark them as such.
    // they should have all been deserialized by way of layer references.
    // so if they aren't found something is really wrong.



    for ( size_t i = 0; ok && (i < num_inputs); i++ ) {

        WisdomContainerEntry input_index_entry;
        string input_symbol;
        ITensor *tensor;

        ok = ok && inputs_entry.getEntry(toString(i),
                                         IWisdomContainerEntry::ENTRY_TYPE_STRING,
                                         &input_index_entry);
        if ( !ok ) {
            gLogError << "couldn't get input entry for " << toString(i) << endl;
            goto done;
        }
        ok = ok && input_index_entry.readString(input_symbol);
        if ( !ok ) {
            gLogError << "couldn't read input index symbol string? " << toString(i) << endl;
            goto done;
        }
        tensor = wisdom->findTensorSymbol(input_symbol);
        if ( !tensor ) {
            ok = false;
            gLogError << " couldn't find input tensor sym=[" << input_symbol << "]" << endl;
            goto done;
        }

        ok = markInput(tensor);
        if ( !ok ) {
            gLogError << " problem marking tensor sym=[" << input_symbol << "] as a network input." << endl;
            goto done;
        }
    }

    for ( size_t o = 0; ok && (o < num_outputs); o++ ) {
        WisdomContainerEntry output_index_entry;
        string output_symbol;
        ITensor *tensor;

        ok = ok && outputs_entry.getEntry(toString(o),
                                          IWisdomContainerEntry::ENTRY_TYPE_STRING,
                                          &output_index_entry);
        if ( !ok ) {
            gLogError << "couldn't get output entry for " << toString(o) << endl;
            goto done;
        }
        ok = ok && output_index_entry.readString(output_symbol);
        if ( !ok ) {
            gLogError << "couldn't read output index symbol string? " << toString(o) << endl;
            goto done;
        }
        tensor = wisdom->findTensorSymbol(output_symbol);
        if ( !tensor ) {
            ok = false;
            gLogError << " couldn't find output tensor sym=[" << output_symbol << "]" << endl;
            goto done;
        }
        markOutput(tensor);
    }

 done:
    return ok;

}



} // nvdla::priv
} // nvdla::
