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

#include <sstream>

#include "priv/Type.h"

#include "priv/Network.h"
#include "priv/Layer.h"
#include "priv/Check.h"

#include "priv/Wisdom.h"
#include "priv/WisdomContainer.h"

#include <math.h>

using std::stringstream;
using std::string;
using std::endl;

namespace nvdla {

ILayer::ILayer() { }
ILayer::~ILayer() { }
IConvolutionLayer::~IConvolutionLayer() { }
IFullyConnectedLayer::~IFullyConnectedLayer() { }
IActivationLayer::~IActivationLayer() { }
IPoolingLayer::~IPoolingLayer() { }
ILRNLayer::~ILRNLayer() { }
IScaleLayer::~IScaleLayer() { }
IBatchNormLayer::~IBatchNormLayer() { }
ISoftMaxLayer::~ISoftMaxLayer() { }
IConcatenationLayer::~IConcatenationLayer() { }
ISliceLayer::~ISliceLayer() { }
IDeconvolutionLayer::~IDeconvolutionLayer() { }
IElementWiseLayer::~IElementWiseLayer() { }

namespace priv {


ENUM_PARAMETER_STATIC(LayerTypeEnum, LAYER_FACTORY_TYPE_ENUMS, "LayerTypeEnum")

//----------------------------------------------------------------------
// Layer Factory
//----------------------------------------------------------------------
//!
//! Layers have two sorts of data which need to be deserialized.
//! The first are simple numeric or enumerant parameters.  The second,
//! more complicated case are the Tensor references: the inputs and outputs.
//! The inputs and outputs are serialized as symbols which are read
//! back here and looked up in the Wisdom.
//!
ILayer *LayerFactory::deserializeFrom(WisdomContainerEntry *entry)
{
    bool ok = false;
    ILayer *layer = NULL;
    NvU32 v;
    WisdomContainerEntry l_type_entry;

    LayerTypeEnum l_type;

    if ( entry->type() != IWisdomContainerEntry::ENTRY_TYPE_OBJECT ) {
        goto done;
    }

    ok = entry->getEntry("factory_type", IWisdomContainerEntry::ENTRY_TYPE_UINT32, &l_type_entry);

    ok = ok && l_type_entry.readUInt32(v);
    if ( !ok ) {
        gLogError << __func__ << " no factory type entry" << endl;
        goto done;
    }
    l_type = LayerTypeEnum::underlying_type(v);

    ok = l_type.valid();
    if ( !ok ) {
        gLogError << __func__ << " invalid factory type entry" << endl;
        goto done;
    }

    switch ( l_type.e() )
    {
    case LayerFactoryType::CONVOLUTION:
        layer = deserializeLayer<ConvolutionLayerDiamond>(entry);
        break;
    case LayerFactoryType::FULLY_CONNECTED:
        layer = deserializeLayer<FullyConnectedLayerDiamond>(entry);
        break;
    case LayerFactoryType::ACTIVATION:
        layer = deserializeLayer<ActivationLayerDiamond>(entry);
        break;
    case LayerFactoryType::POOLING:
        layer = deserializeLayer<PoolingLayerDiamond>(entry);
        break;
    case LayerFactoryType::LRN:
        layer = deserializeLayer<LRNLayerDiamond>(entry);
        break;
    case LayerFactoryType::SCALE:
        layer = deserializeLayer<ScaleLayerDiamond>(entry);
        break;
    case LayerFactoryType::BATCH_NORM:
        layer = deserializeLayer<BatchNormLayerDiamond>(entry);
        break;
    case LayerFactoryType::SOFT_MAX:
        layer = deserializeLayer<SoftMaxLayerDiamond>(entry);
        break;
    case LayerFactoryType::CONCATENATION:
        layer = deserializeLayer<ConcatenationLayerDiamond>(entry);
        break;
    case LayerFactoryType::DECONVOLUTION:
        layer = deserializeLayer<DeconvolutionLayerDiamond>(entry);
        break;
    case LayerFactoryType::ELEMENT_WISE :
        layer = deserializeLayer<ElementWiseLayerDiamond>(entry);
        break;

    default:
        // but, shouldn't be possible since l_type.valid() is true...
        // ok = false;
        goto done;
    }

 done:
    return layer;

}

template <class D>
class BasePrivDiamondMap
{
public:
    static void insert(D p)
    {
        s_base_priv_diamond_map[p.base().priv()] = p;
    }
    static D find(typename D::BasePrivType *base_priv)
    {
        typename std::map<typename D::BasePrivType *, D>::iterator f =
            s_base_priv_diamond_map.find(base_priv);
        if ( f == s_base_priv_diamond_map.end() ) {
            return D(0, 0, 0, 0);
        }
        return f->second;
    }
protected:
    static std::map<typename D::BasePrivType *, D> s_base_priv_diamond_map;
};


template <class D>
typename D::DerivedPrivType * LayerFactory::derivedPriv(typename D::BasePrivType *base_priv)
{
    D d  = BasePrivDiamondMap<D>::find(base_priv);
    return d.derived().priv();
}

template <class D> D LayerFactory::newLayer()
{
    typename D::BaseInterfaceType *base_i;       typename D::BasePrivType *base_priv;
    typename D::DerivedInterfaceType *derived_i; typename D::DerivedPrivType *derived_priv;

    base_priv = derived_priv = new typename D::DerivedPrivType();
    base_i = derived_i = derived_priv;

    D d(base_i, base_priv, derived_i, derived_priv);

    BasePrivDiamondMap<D>::insert(d);

    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);

    return d;
}

template <class D>
typename D::BaseInterfaceType * LayerFactory::deserializeLayer(WisdomContainerEntry *e)
{
    D conv = LayerFactory::newLayer<D>();
    if ( !conv ) {
        gLogError << __func__ << " base i NULL!" << endl;
        return NULL;
    }
    conv.derived().priv()->deserializeFrom(e);
    return conv.base().i();
}

//!
//! Factories implement a poor-man's replacement for dynamic (priv down) casting.
//! MISRA's ok with legit uses of dynamic_cast but we've decided to avoid RTTI, so...
//!
//! note: concurrent access to map is... well, whatever std::map allows.
//!

BiMap<ILayer *, Layer *> LayerFactory::s_priv;
BiMap<void *, ILayer *> LayerFactory::s_self;

Layer *LayerFactory::priv(ILayer *layer)
{
    // gLogError << __func__ << " looking up priv for base_i=" << layer << endl;
    BiMap<ILayer *, Layer *>::left_iterator f = s_priv.find_left(layer);
    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ILayer *LayerFactory::i(Layer *layer)
{
    BiMap<ILayer *, Layer *>::right_iterator f = s_priv.find_right(layer);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

ILayer *LayerFactory::self(void *s)
{
    BiMap<void *, ILayer *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}

ConvolutionLayerDiamond
LayerFactory::newConvolutionLayer(INetwork * network,
                                  const std::string & name,
                                  ITensor * input,
                                  ITensor * output,
                                  int numOutputMaps,
                                  int paddingValue,
                                  Dims2 kernelSize,
                                  Dims2 tlPadding,
                                  Dims2 brPadding,
                                  Dims2 stride,
                                  Dims2 dilation,
                                  Weights kernelWeights,
                                  Weights biasWeights,
                                  BiasMode biasMode,
                                  int numGroups)
{
    ILayer     *base_i;    Layer     *base_priv;
    IConvolutionLayer *derived_i; ConvolutionLayer *derived_priv;

    base_priv = derived_priv = new ConvolutionLayer(network, name,
                                                    input, output,
                                                    numOutputMaps, paddingValue,
                                                    kernelSize, tlPadding, brPadding, stride, dilation,
                                                    kernelWeights, biasWeights, biasMode, numGroups);
    base_i = derived_i = derived_priv;
    ConvolutionLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<ConvolutionLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;

}

FullyConnectedLayerDiamond
LayerFactory::newFullyConnectedLayer(INetwork* network,
                                     const std::string& name,
                                     ITensor* input,
                                     ITensor* output,
                                     int numOutputChannels,
                                     Weights kernelWeights,
                                     Weights biasWeights,
                                     BiasMode biasMode)
{
    ILayer        *base_i;    Layer         *base_priv;
    IFullyConnectedLayer *derived_i;  FullyConnectedLayer *derived_priv;


    base_priv = derived_priv = new FullyConnectedLayer(network, name,
                                                       input, output,
                                                       numOutputChannels,
                                                       kernelWeights, biasWeights, biasMode);
    base_i = derived_i = derived_priv;
    FullyConnectedLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<FullyConnectedLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;

}

ActivationLayerDiamond
LayerFactory::newActivationLayer(INetwork* network,
                                 const std::string& name,
                                 ITensor* input,
                                 ITensor* output,
                                 ActivationType activationType)
{
    ILayer    *base_i;    Layer     *base_priv;
    IActivationLayer *derived_i; ActivationLayer *derived_priv;

    base_priv = derived_priv = new ActivationLayer(network, name, input, output, activationType);
    base_i = derived_i = derived_priv;
    ActivationLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<ActivationLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

PoolingLayerDiamond
LayerFactory::newPoolingLayer(INetwork* network,
                              const std::string& name,
                              ITensor* input,
                              ITensor* output,
                              PoolingType type,
                              Dims2 windowSize,
                              Dims2 stride,
                              Dims2 tlPadding,
                              Dims2 brPadding)
{
    ILayer  *base_i;   Layer *base_priv;
    IPoolingLayer *derived_i; PoolingLayer *derived_priv;

    base_priv = derived_priv = new PoolingLayer(network, name, input, output,
                                                type, windowSize, stride, tlPadding, brPadding);
    base_i = derived_i = derived_priv;
    PoolingLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<PoolingLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

LRNLayerDiamond
LayerFactory::newLRNLayer(INetwork* network,
                          const std::string& name,
                          ITensor* input,
                          ITensor* output,
                          int windowSize,
                          float alpha,
                          float beta,
                          float k)
{
    ILayer *base_i;    Layer *base_priv;
    ILRNLayer     *derived_i; LRNLayer     *derived_priv;

    base_priv = derived_priv = new LRNLayer(network, name, input, output,
                                            windowSize, alpha, beta, k);
    base_i = derived_i = derived_priv;
    LRNLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<LRNLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

ScaleLayerDiamond
LayerFactory::newScaleLayer(INetwork* network,
                            const std::string& name,
                            ITensor* input,
                            ITensor* output,
                            ScaleMode mode,
                            Weights shift,
                            Weights scale,
                            Weights power)
{
    ILayer *base_i;    Layer *base_priv;
    IScaleLayer   *derived_i; ScaleLayer   *derived_priv;

    base_priv = derived_priv = new ScaleLayer(network, name, input, output,
                                              mode, shift, scale, power);
    base_i = derived_i = derived_priv;
    ScaleLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<ScaleLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

BatchNormLayerDiamond
LayerFactory::newBatchNormLayer(INetwork* network,
                                const std::string& name,
                                ITensor* input,
                                ITensor* output,
                                BatchNormMode mode,
                                Weights mean,
                                Weights variance,
                                float eps)
{
    ILayer *base_i;     Layer *base_priv;
    IBatchNormLayer *derived_i; BatchNormLayer *derived_priv;

    base_priv = derived_priv = new BatchNormLayer(network, name, input, output,
                                                  mode, mean, variance, eps);
    base_i = derived_i = derived_priv;
    BatchNormLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<BatchNormLayerDiamond>::insert(d);
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

SoftMaxLayerDiamond
LayerFactory::newSoftMaxLayer(INetwork* network,
                              const std::string& name,
                              ITensor* input,
                              ITensor* output)
{
    ILayer *base_i;    Layer *base_priv;
    ISoftMaxLayer *derived_i; SoftMaxLayer *derived_priv;

    base_priv = derived_priv = new SoftMaxLayer(network, name, input, output);
    base_i = derived_i = derived_priv;
    SoftMaxLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<SoftMaxLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

ConcatenationLayerDiamond
LayerFactory::newConcatenationLayer(INetwork* network,
                                    const std::string& name,
                                    ITensor*const * inputs,
                                    int numInputs,
                                    ITensor* output)
{
    ILayer         *base_i;  Layer       *base_priv;
    IConcatenationLayer *derived_i; ConcatenationLayer *derived_priv;

    base_priv = derived_priv = new ConcatenationLayer(network, name, inputs, numInputs, output);
    base_i = derived_i = derived_priv;
    ConcatenationLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<ConcatenationLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

SliceLayerDiamond
LayerFactory::newSliceLayer(INetwork* network,
                            const std::string& name,
                            ITensor* input,
                            ITensor* const* outputs,
                            int numOutputs)
{
    ILayer         *base_i;  Layer       *base_priv;
    ISliceLayer *derived_i; SliceLayer *derived_priv;

    base_priv = derived_priv = new SliceLayer(network, name, input, outputs, numOutputs);
    base_i = derived_i = derived_priv;
    SliceLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<SliceLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

DeconvolutionLayerDiamond
LayerFactory::newDeconvolutionLayer(INetwork* network,
                                    const std::string& name,
                                    ITensor* input,
                                    ITensor* output,
                                    int numOutputMaps,
                                    int paddingValue,
                                    Dims2 kernelSize,
                                    Dims2 tlPadding,
                                    Dims2 brPadding,
                                    Dims2 stride,
                                    Dims2 dilation,
                                    Weights kernelWeights,
                                    Weights biasWeights,
                                    BiasMode biasMode,
                                    int numGroups)
{
    ILayer         *base_i;   Layer      *base_priv;
    IDeconvolutionLayer *derived_i; DeconvolutionLayer *derived_priv;

    base_priv = derived_priv = new DeconvolutionLayer(network, name,
                                                      input, output,
                                                      numOutputMaps, paddingValue,
                                                      kernelSize, tlPadding, brPadding, stride, dilation,
                                                      kernelWeights, biasWeights, biasMode, numGroups);
    base_i = derived_i = derived_priv;
    DeconvolutionLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<DeconvolutionLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}

ElementWiseLayerDiamond
LayerFactory::newElementWiseLayer(INetwork* network,
                                  const std::string& name,
                                  ITensor*const *inputs,
                                  ITensor* output,
                                  ElementWiseOperation op)
{
    ILayer     *base_i;    Layer     *base_priv;
    IElementWiseLayer *derived_i; ElementWiseLayer *derived_priv;

    base_priv = derived_priv = new ElementWiseLayer(network, name, inputs, output, op);
    base_i = derived_i = derived_priv;
    ElementWiseLayerDiamond d(base_i, base_priv, derived_i, derived_priv);
    BasePrivDiamondMap<ElementWiseLayerDiamond>::insert( d );
    s_priv.insert(base_i, base_priv);
    s_self.insert(base_i, base_i);
    return d;
}


//----------------------------------------------------------------------
// Layer
//----------------------------------------------------------------------

Layer::Layer(INetwork *n, LayerType type,
                           const std::string& name,
                           ITensor* input,
                           ITensor* output) :
    mNetwork(n), mType(type), mName(name)
{
    mInputs.push_back(input);
    mOutputs.push_back(output);
}

Layer::Layer(INetwork* n, LayerType type,
                           const std::string& name,
                           ITensor*const * inputs,
                           int numInputs,
                           ITensor*const * outputs,
                           int numOutputs)
    : mNetwork(n), mType(type), mName(name)
{
    for (int i = 0; i < numInputs; i++)
        mInputs.push_back(inputs[i]);
    for (int i = 0; i < numOutputs; i++)
        mOutputs.push_back(outputs[i]);
}

Layer::~Layer()
{

}

bool Layer::deserializeFrom(WisdomContainerEntry *entry)
{
    bool ok = true;
    std::string name;

    WisdomContainerEntry name_entry;
    WisdomContainerEntry num_inputs_entry, num_outputs_entry;
    WisdomContainerEntry inputs_entry, outputs_entry;
    NvU32 num_inputs, num_outputs;
    std::vector<ITensor *> inputs;
    std::vector<ITensor *> outputs;


    ok = ok && entry->getEntry("name",        IWisdomContainerEntry::ENTRY_TYPE_STRING, &name_entry);
    ok = ok && entry->getEntry("num_inputs",  IWisdomContainerEntry::ENTRY_TYPE_UINT32, &num_inputs_entry);
    ok = ok && entry->getEntry("num_outputs", IWisdomContainerEntry::ENTRY_TYPE_UINT32, &num_outputs_entry);

    if ( !ok ) {
        gLogError << __func__ << " missing name, num_inputs and/or num_outputs entries" << endl;
        goto done;
    }

    ok = ok && name_entry.readString(name);
    ok = ok && num_inputs_entry.readUInt32(num_inputs);
    ok = ok && num_outputs_entry.readUInt32(num_outputs);

    if ( !ok ) {
        gLogError << __func__ << " missing name, num_inputs and/or num_outputs" << endl;
        goto done;
    }

    // XXX: need upper limit checks on these...
    ok = ok && ( 0 < num_inputs && 0 < num_outputs );

    if ( !ok ) {
        gLogError << __func__ << " missing in or outputs" << endl;
        goto done;
    }

    setName(name.c_str());
    inputs.resize(num_inputs, NULL);
    outputs.resize(num_outputs, NULL);


    ok = ok && entry->getEntry(string("inputs"),  IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &inputs_entry);
    ok = ok && entry->getEntry(string("outputs"), IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &outputs_entry);
    if ( !ok ) {
        gLogError << __func__ << " failed to get inputs and outputs entries" << endl;
        goto done;
    }


    for (size_t i = 0; ok && (i < num_inputs); i++ ) {
        string symbol;
        WisdomContainerEntry tensor_index_entry;
        string tensor_symbol;
        ok = ok && inputs_entry.getEntry(toString(i),
                                         IWisdomContainerEntry::ENTRY_TYPE_STRING,
                                         &tensor_index_entry);
        ok = ok && tensor_index_entry.readString(tensor_symbol);
        if ( !ok ) {
            gLogError << __func__ << " couldn't read (input) tensor index symbol string? " << toString(i) << endl;
            break;
        }
        ITensor *t = entry->container_priv()->wisdom_priv()->getTensorFromSymbol(tensor_symbol);
        ok = (NULL != t);
        if ( !ok ) {
            gLogError << "couldn't get (input) tensor from symbol=[" << tensor_symbol << "]" << endl;
        } else {
            setInput(i, t);
        }
    }

    for (size_t o = 0; ok && o < num_outputs; o++ ) {
        string symbol;
        WisdomContainerEntry tensor_index_entry;
        string tensor_symbol;
        ok = ok && outputs_entry.getEntry(toString(o),
                                         IWisdomContainerEntry::ENTRY_TYPE_STRING,
                                         &tensor_index_entry);
        ok = ok && tensor_index_entry.readString(tensor_symbol);
        if ( !ok ) {
            gLogError << __func__ << " couldn't read (output) tensor index symbol string? " << toString(o) << endl;
            break;
        }
        ITensor *t = entry->container_priv()->wisdom_priv()->getTensorFromSymbol(tensor_symbol);
        ok = NULL != t;
        if ( !ok ) {
            gLogError << "couldn't get (output) tensor from symbol=[" << tensor_symbol << "]" << endl;
        } else {
            setOutput(o, t);
        }
    }

 done:
    return ok;
}


bool Layer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = true;
    // scribble out which tensors were referred to as inputs and outputs.
    WisdomContainerEntry factory_type_entry, inputs_entry, outputs_entry;

    ok = ok && e->writeString("name", getName());
    ok = ok && e->writeUInt32("factory_type", getFactoryType());
    ok = ok && e->writeUInt32("num_inputs", getNumInputs());
    ok = ok && e->writeUInt32("num_outputs", getNumOutputs());

    ok = ok && e->insertEntryIfNotPresent("inputs", IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &inputs_entry);
    ok = ok && e->insertEntryIfNotPresent("outputs", IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &outputs_entry);

    for ( size_t i = 0; ok && (i < mInputs.size()); i++ ) {
        string sym;
        ok = e->container_priv()->wisdom_priv()->findITensorSymbol(getInput(i), sym);
        ok = ok && inputs_entry.writeString(toString(i), sym);
    }

    for ( size_t o = 0; ok && (o < mOutputs.size()); o++ ) {
        string sym;
        ok = e->container_priv()->wisdom_priv()->findITensorSymbol(getOutput(o), sym);
        ok = ok && outputs_entry.writeString(toString(o), sym);
    }

    return ok;
}


//
// internally facing Layer:: interfaces
//
std::string Layer::getInputSymbol(int i) const
{
    return mInputSymbols[i];
}

std::string Layer::getOutputSymbol(int i) const
{
    return mOutputSymbols[i];
}

void Layer::setInput(int i, ITensor *t)
{
    mInputs[i] = t;
}

void Layer::setOutput(int i, ITensor *t)
{
    mOutputs[i] = t;
}
// Layer is virtual so it has no typeid
#if 0
NvU16 Layer::getFactoryType() const
{
    return ~0; // only one so far, not complicated by factory splits
}
#endif

bool Layer::assignSymbols(Wisdom *wisdom)
{
    bool ok = true;
    std::string sym;
    Tensor *tensor;
    std::vector<Tensor *>tensors;

    // gLogError << __func__ << " inputs=" << mInputs.size() << " outputs=" << mOutputs.size() << endl;

    // inputs, then outputs
    for (size_t i = 0; i < mInputs.size(); i++ ) {
        tensor = TensorFactory::priv(mInputs[i]);
        ok = NULL != tensor;
        if ( !tensor ) {
            gLogError << "missing input tensor " << i << " in layer " << mName << endl;
            goto done;
        }
        tensors.push_back(tensor);
    }
    for (size_t o = 0; o < mOutputs.size(); o++) {
        tensor = TensorFactory::priv(mOutputs[o]);
        ok = NULL != tensor;
        if ( !ok ) {
            gLogError << "missing output tensor " << o << " in layer" << mName << endl;
            goto done;
        }
        tensors.push_back(tensor);
    }
    for (size_t t = 0; t < tensors.size(); t++ ) {
        tensor = tensors[t];
        ok = wisdom->findTensorSymbol(tensor, sym);
        if ( !ok ) {
            ok = wisdom->assignTensorSymbol(tensor, sym);
            if ( !ok ) {
                gLogError << "unable to assign tensor symbol " << t << " in layer " << mName << endl;
                goto done;
            }
        }
    }

 done:
    return ok;
}

//
// externally facing
//

void Layer::setName(const char* name)
{
    API_CHECK_NULL(name);
    mName = name;
}

const char* Layer::getName() const
{
    return mName.c_str();
}

LayerType Layer::getType() const
{
    return mType;
}

int Layer::getNumInputs() const
{
    return (int)(mInputs.size());
}

ITensor* Layer::getInput(int i) const
{
    return (size_t)i < mInputs.size() ? mInputs[i] : 0;
}

int Layer::getNumOutputs() const
{
    return (int)(mOutputs.size());
}

ITensor* Layer::getOutput(int i) const
{
    return (size_t)i < mOutputs.size() ? mOutputs[i] : 0;
}

void Layer::setKernelSize(Dims2 kernelSize)
{
    NVDLA_UNUSED(kernelSize);
}

Dims2 Layer::getKernelSize() const
{
    return Dims2(1, 1);
}

void Layer::setNumOutputMaps(int /*numOutputMaps*/)
{

}

int Layer::getNumOutputMaps() const
{
    return 0;
}

void Layer::setStride(Dims2 /*stride*/)
{

}

Dims2 Layer::getStride() const
{
    return Dims2(1, 1);
}

void Layer::setDilation(Dims2 /*dilation*/)
{

}

Dims2 Layer::getDilation() const
{
    return Dims2(1, 1);
}

void Layer::setTopLeftPadding(Dims2 /*padding*/)
{

}

Dims2 Layer::getTopLeftPadding() const
{
    return Dims2(0, 0);
}

void Layer::setBottomRightPadding(Dims2 /*padding*/)
{

}

Dims2 Layer::getBottomRightPadding() const
{
    return Dims2(0, 0);
}

int Layer::getPaddingValue() const
{
    return 0;
}

void Layer::setPaddingValue(int /*padding*/)
{

}

void Layer::setNumGroups(int /*numGroups*/)
{

}

int Layer::getNumGroups() const
{
    return 0;
}

void Layer::setKernelWeights(Weights /*weights*/)
{

}

Weights Layer::getKernelWeights() const
{
    return Weights(DataType::FLOAT, NULL, 0);
}

void Layer::setBiasWeights(Weights /*weights*/)
{

}

Weights Layer::getBiasWeights() const
{
    return Weights(DataType::FLOAT, NULL, 0);
}

NvDlaError Layer::supportsPrecision(nvdla::DataType prec)
{
    NvDlaError e = NvDlaSuccess;

    if (prec.v() > EnumMax<DataType>())
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Precision %d is beyond max allowed(%d)\n",
                prec.v(), EnumMax<DataType>());
    }

    switch(prec.v())
    {
        case nvdla::DataType::HALF:
        case nvdla::DataType::INT8:
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Precision %d is not supported\n",
                    prec.v());
    }

fail:
    return e;
}


//----------------------------------------------------------------------
// ConvolutionLayer
//----------------------------------------------------------------------
ConvolutionLayer::ConvolutionLayer() : Layer(NULL, LayerType::kCONVOLUTION, "", NULL, NULL) { }
ConvolutionLayer::ConvolutionLayer(INetwork* network, const std::string& name,
                                   ITensor* input,
                                   ITensor* output,
                                   int numOutputMaps,
                                   Dims2 kernelSize,
                                   Weights kernelWeights,
                                   Weights biasWeights,
                                   BiasMode biasMode,
                                   int numGroups)
    : Layer(network, LayerType::kCONVOLUTION, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(numOutputMaps > 0 && numOutputMaps < MAX_OUTPUT_MAPS);
    API_CHECK(kernelSize.h > 0 && kernelSize.w > 0 && kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT);
    API_CHECK_WEIGHTS(kernelWeights);
    API_CHECK_WEIGHTS0(biasWeights);
    API_CHECK_ENUM_RANGE(BiasMode, biasMode);
    mParams.kernelSize = kernelSize;
    mParams.numOutputMaps = numOutputMaps;
    mParams.topLeftPadding = Dims2(0, 0);
    mParams.bottomRightPadding = Dims2(0, 0);
    mParams.paddingValue = 0;
    mParams.stride = Dims2(1, 1);
    mParams.kernelWeights = kernelWeights;
    mParams.biasWeights = biasWeights;
    mParams.biasMode = biasMode;
    mParams.numGroups = numGroups;
}
ConvolutionLayer::ConvolutionLayer(INetwork* network, const std::string& name,
                                   ITensor* input,
                                   ITensor* output,
                                   int numOutputMaps,
                                   int paddingValue,
                                   Dims2 kernelSize,
                                   Dims2 tlPadding,
                                   Dims2 brPadding,
                                   Dims2 stride,
                                   Dims2 dilation,
                                   Weights kernelWeights,
                                   Weights biasWeights,
                                   BiasMode biasMode,
                                   int numGroups)
    : Layer(network, LayerType::kCONVOLUTION, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(numOutputMaps > 0 && numOutputMaps < MAX_OUTPUT_MAPS);
    API_CHECK(kernelSize.h > 0 && kernelSize.w > 0 && kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT);
    API_CHECK_WEIGHTS(kernelWeights);
    API_CHECK_WEIGHTS0(biasWeights);
    API_CHECK_ENUM_RANGE(BiasMode, biasMode);
    mParams.kernelSize = kernelSize;
    mParams.numOutputMaps = numOutputMaps;
    mParams.topLeftPadding = tlPadding;
    mParams.bottomRightPadding = brPadding;
    mParams.paddingValue = paddingValue;
    mParams.stride = stride;
    mParams.dilation = dilation;
    mParams.kernelWeights = kernelWeights;
    mParams.biasWeights = biasWeights;
    mParams.biasMode = biasMode;
    mParams.numGroups = numGroups;
}

ConvolutionLayer::~ConvolutionLayer() { }

NvU16 ConvolutionLayer::getFactoryType() const
{
    return LayerFactoryType::CONVOLUTION;
}

bool ConvolutionLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeDims2("kernel_size",          mParams.kernelSize);
    ok = ok && e->writeDims2("top_left_padding",     mParams.topLeftPadding);
    ok = ok && e->writeDims2("bottom_right_padding", mParams.bottomRightPadding);
    ok = ok && e->writeDims2("stride",               mParams.stride);
    ok = ok && e->writeDims2("dilation",             mParams.dilation);
    ok = ok && e->writeInt32("num_output_maps",      mParams.numOutputMaps);
    ok = ok && e->writeInt32("num_groups",           mParams.numGroups);
    ok = ok && e->writeInt32("padding_value",        mParams.paddingValue);
    ok = ok && e->writeWeights("kernel_weights",     mParams.kernelWeights);
    ok = ok && e->writeWeights("bias_weights",       mParams.biasWeights);
    ok = ok && e->writeUInt8Enum("bias_mode",        mParams.biasMode);
    ok = ok && e->writeInt32("groups",               mParams.numGroups);
    return ok;
}

bool ConvolutionLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvU8 v = BiasMode::bUNIFORM;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readDims2("kernel_size",          mParams.kernelSize);
    ok = ok && e->readDims2("top_left_padding",     mParams.topLeftPadding);
    ok = ok && e->readDims2("bottom_right_padding", mParams.bottomRightPadding);
    ok = ok && e->readDims2("stride",               mParams.stride);
    ok = ok && e->readDims2("dilation",             mParams.dilation);
    ok = ok && e->readInt32("num_output_maps",      mParams.numOutputMaps);
    ok = ok && e->readInt32("num_groups",           mParams.numGroups);
    ok = ok && e->readInt32("padding_value",        mParams.paddingValue);
    ok = ok && e->readWeights("kernel_weights",     mParams.kernelWeights);
    ok = ok && e->readWeights("bias_weights",       mParams.biasWeights);
    ok = ok && e->readUInt8Enum("bias_mode",        v);
    ok = ok && e->readInt32("groups",               mParams.numGroups);
    mParams.biasMode = BiasMode(v);

    return ok;
}

#define CurrentScope ConvolutionLayer

ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, kernelSize, KernelSize,
                            (kernelSize.h > 0 && kernelSize.w > 0 &&
                             kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, topLeftPadding, TopLeftPadding,
                            (topLeftPadding.h >= 0 && topLeftPadding.w >= 0 &&
                             topLeftPadding.h + topLeftPadding.w < MAX_PADDING_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, bottomRightPadding, BottomRightPadding,
                         (bottomRightPadding.h >= 0 && bottomRightPadding.w >= 0 &&
                          bottomRightPadding.h + bottomRightPadding.w < MAX_PADDING_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, stride, Stride,
                            (stride.h > 0 && stride.w > 0 &&
                             stride.h + stride.w < MAX_STRIDE_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, dilation, Dilation,
                             (dilation.h > 0 && dilation.w > 0 &&
                              dilation.h + dilation.w < MAX_DILATION_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, paddingValue, PaddingValue,
                             (paddingValue >= 0 && paddingValue < MAX_PADDING_VALUE) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, numGroups, NumGroups,
                            (numGroups > 0 && numGroups < MAX_GROUPS) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, numOutputMaps, NumOutputMaps,
                            (numOutputMaps > 0 && numOutputMaps < MAX_OUTPUT_MAPS) )
ACCESSOR_MUTATOR_CHECK_ENUM(BiasMode, biasMode, BiasMode)
ACCESSOR_MUTATOR_WEIGHTS(Weights, kernelWeights, KernelWeights)
ACCESSOR_MUTATOR_WEIGHTS(Weights, biasWeights, BiasWeights)
#undef CurrentScope


Dims4 ConvolutionLayer::getOutputDimensions() const
{
    Dims4 inputDims = mInputs[0]->getDimensions();
    // assert(inputDims.c > 0 && inputDims.h > 0 && inputDims.w > 0);
    Dims2 cd = mNetwork->getConvolutionOutputDimensionsFormula().
            compute(Dims2(inputDims.h, inputDims.w),
                    mParams.kernelSize, mParams.stride, mParams.topLeftPadding, mParams.bottomRightPadding, mParams.dilation, getName());
    return Dims4(mParams.numOutputMaps, cd.h, cd.w);
}

const ConvolutionLayer::Parameters& ConvolutionLayer::getParams() const
{
    return mParams;
}

//----------------------------------------------------------------------
// FullyConnectedLayer
//----------------------------------------------------------------------
FullyConnectedLayer::FullyConnectedLayer() : Layer(NULL, LayerType::kFULLY_CONNECTED, "", NULL, NULL) { }
FullyConnectedLayer::FullyConnectedLayer(INetwork* network,
                                         const std::string& name,
                                         ITensor* input,
                                         ITensor* output,
                                         int numOutputChannels,
                                         Weights kernelWeights,
                                         Weights biasWeights,
                                         BiasMode biasMode) :
    Layer(network, LayerType::kFULLY_CONNECTED, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(numOutputChannels > 0 && numOutputChannels < MAX_OUTPUT_MAPS);
    API_CHECK_WEIGHTS(kernelWeights);
    API_CHECK_WEIGHTS0(biasWeights);
    API_CHECK_ENUM_RANGE(BiasMode, biasMode);
    mParams.numOutputChannels = numOutputChannels;
    mParams.kernelWeights = kernelWeights;
    mParams.biasWeights = biasWeights;
    mParams.biasMode = biasMode;
}

bool FullyConnectedLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeWeights("kernel_weights", mParams.kernelWeights);
    ok = ok && e->writeWeights("bias_weights",   mParams.biasWeights);
    ok = ok && e->writeUInt8Enum("bias_mode",    mParams.biasMode);
    return ok;
}

bool FullyConnectedLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvU8 v = BiasMode::bUNIFORM;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readWeights("kernel_weights", mParams.kernelWeights);
    ok = ok && e->readWeights("bias_weights",   mParams.biasWeights);
    ok = ok && e->readUInt8Enum("bias_mode",    v);
    mParams.biasMode = BiasMode(v);
    return ok;
}

NvU16 FullyConnectedLayer::getFactoryType() const
{
    return LayerFactoryType::FULLY_CONNECTED;
}


#define CurrentScope FullyConnectedLayer
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, kernelSize, KernelSize,
                            (kernelSize.h > 0 && kernelSize.w > 0 &&
                             kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, numOutputChannels, NumOutputChannels,
                            (numOutputChannels > 0 && numOutputChannels < MAX_OUTPUT_MAPS) );
ACCESSOR_MUTATOR_CHECK_ENUM(BiasMode, biasMode, BiasMode);
ACCESSOR_MUTATOR_WEIGHTS(Weights, kernelWeights, KernelWeights)
ACCESSOR_MUTATOR_WEIGHTS(Weights, biasWeights, BiasWeights)
#undef CurrentScope

Dims4 FullyConnectedLayer::getOutputDimensions() const
{
    return Dims4(mParams.numOutputChannels, 1, 1);
}

const FullyConnectedLayer::Parameters& FullyConnectedLayer::getParams() const
{
    return mParams;
}

int FullyConnectedLayer::getNumOutputMaps() const
{
    return getNumOutputChannels();
}

void FullyConnectedLayer::setNumOutputMaps(int maps)
{
    setNumOutputChannels(maps);
}

FullyConnectedLayer::~FullyConnectedLayer()
{

}


//----------------------------------------------------------------------
// ActivationLayer
//----------------------------------------------------------------------
ActivationLayer::ActivationLayer() : Layer(NULL, LayerType::kACTIVATION, "", NULL, NULL) { }
ActivationLayer::ActivationLayer(INetwork* network,
                                 const std::string& name,
                                 ITensor* input,
                                 ITensor* output,
                                 ActivationType activationType)
    : Layer(network, LayerType::kACTIVATION, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK_ENUM_RANGE(ActivationType, activationType);
    mParams.activationType = activationType;
}

bool ActivationLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeInt32("activation_type", mParams.activationType);
    return ok;
}

bool ActivationLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvS32 v = kRELU;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readInt32("activation_type", v);
    mParams.activationType = ActivationType(v);
    return ok;
}

ActivationLayer::~ActivationLayer() {}

NvU16 ActivationLayer::getFactoryType() const
{
    return LayerFactoryType::ACTIVATION;
}

#define CurrentScope ActivationLayer
ACCESSOR_MUTATOR_CHECK_ENUM(ActivationType, activationType, ActivationType)
//ACCESSOR_MUTATOR_CHECK_ENUM(DataType, arithmeticType, ArithmeticType)
#undef CurrentScope
Dims4 ActivationLayer::getOutputDimensions() const
{
    return mInputs[0]->getDimensions();
}

const ActivationLayer::Parameters& ActivationLayer::getParams() const
{
    return mParams;
}


//----------------------------------------------------------------------
// PoolingLayer
//----------------------------------------------------------------------
PoolingLayer::PoolingLayer() :	Layer(NULL, LayerType::kPOOLING, "", NULL, NULL) { }
PoolingLayer::PoolingLayer(INetwork* network,
                           const std::string& name,
                           ITensor* input,
                           ITensor* output,
                           PoolingType type,
                           Dims2 windowSize)
    : Layer(network, LayerType::kPOOLING, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(type.v() <= EnumMax<PoolingType>());
    API_CHECK(windowSize.h > 0 && windowSize.w > 0 &&
              windowSize.h*windowSize.w < MAX_KERNEL_DIMS_PRODUCT);
    API_CHECK(windowSize.h > 0 && windowSize.w > 0 &&
              windowSize.h + windowSize.w < MAX_STRIDE_SUM);

    mParams.poolingType = type;
    mParams.windowSize = windowSize;
    mParams.stride = windowSize;
    mParams.topLeftPadding = Dims2(0, 0);
    mParams.bottomRightPadding = Dims2(0, 0);
}

PoolingLayer::PoolingLayer(INetwork* network,
                           const std::string& name,
                           ITensor* input,
                           ITensor* output,
                           PoolingType type,
                           Dims2 windowSize,
                           Dims2 stride,
                           Dims2 tlPadding,
                           Dims2 brPadding)
    : Layer(network, LayerType::kPOOLING, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(type.v() <= EnumMax<PoolingType>());
    API_CHECK(windowSize.h > 0 && windowSize.w > 0 &&
              windowSize.h * windowSize.w < MAX_KERNEL_DIMS_PRODUCT);
    API_CHECK(stride.h > 0 && stride.w > 0 &&
              stride.h + stride.w < MAX_STRIDE_SUM);

    mParams.poolingType = type;
    mParams.windowSize  = windowSize;
    mParams.stride      = stride;
    mParams.topLeftPadding = tlPadding;
    mParams.bottomRightPadding = brPadding;
}
PoolingLayer::~PoolingLayer() { }

bool PoolingLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeDims2("window_size",          mParams.windowSize);
    ok = ok && e->writeDims2("top_left_padding",     mParams.topLeftPadding);
    ok = ok && e->writeDims2("bottom_right_padding", mParams.bottomRightPadding);
    ok = ok && e->writeDims2("stride",               mParams.stride);
    ok = ok && e->writeInt32("pooling_type",         mParams.poolingType);
    return ok;
}

bool PoolingLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvS32 v = PoolingType::kMAX;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readDims2("window_size",          mParams.windowSize);
    ok = ok && e->readDims2("top_left_padding",     mParams.topLeftPadding);
    ok = ok && e->readDims2("bottom_right_padding", mParams.bottomRightPadding);
    ok = ok && e->readDims2("stride",               mParams.stride);
    ok = ok && e->readInt32("pooling_type",         v);
    mParams.poolingType = PoolingType(v);
    return ok;
}

NvU16 PoolingLayer::getFactoryType() const
{
    return LayerFactoryType::POOLING;
}

#define CurrentScope PoolingLayer

ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, windowSize, WindowSize,
                            (windowSize.h > 0 && windowSize.w > 0 &&
                             windowSize.h*windowSize.w < MAX_KERNEL_DIMS_PRODUCT) );
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, topLeftPadding, TopLeftPadding,
                            (topLeftPadding.h >= 0 && topLeftPadding.w >= 0 &&
                             topLeftPadding.h + topLeftPadding.w < MAX_PADDING_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, bottomRightPadding, BottomRightPadding,
                         (bottomRightPadding.h >= 0 && bottomRightPadding.w >= 0 &&
                          bottomRightPadding.h + bottomRightPadding.w < MAX_PADDING_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, stride, Stride,
                            (stride.h > 0 && stride.w > 0 &&
                             stride.h + stride.w < MAX_STRIDE_SUM) );
#undef CurrentScope

Dims4 PoolingLayer::getOutputDimensions() const
{
    Dims4 inputDims = mInputs[0]->getDimensions();
    //assert(inputDims.c > 0 && inputDims.h > 0 && inputDims.w > 0);
    Dims2 poolDims = mNetwork->getPoolingOutputDimensionsFormula().
            compute(Dims2(inputDims.h, inputDims.w), mParams.windowSize,
                    mParams.stride, mParams.topLeftPadding, mParams.bottomRightPadding, getName());
    return Dims4(inputDims.c, poolDims.h, poolDims.w);
}

const PoolingLayer::Parameters& PoolingLayer::getParams() const { return mParams; }

PoolingType PoolingLayer::getPoolingType() const { return mParams.poolingType; }
void PoolingLayer::setPoolingType(PoolingType pt) { mParams.poolingType = pt; }

//----------------------------------------------------------------------
// LRNLayer
//----------------------------------------------------------------------
LRNLayer::LRNLayer() :	Layer(NULL, LayerType::kLRN, "", NULL, NULL) { }
LRNLayer::LRNLayer(INetwork* network, const std::string& name,
                   ITensor* input, ITensor* output,
                   int windowSize, float alpha, float beta, float k)
    : Layer(network, LayerType::kLRN, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(windowSize >= ILRNLayer::Parameters::minWindowSize() &&
              windowSize <= ILRNLayer::Parameters::maxWindowSize());
    API_CHECK(fabsf(alpha) < ILRNLayer::Parameters::maxAbsAlpha());
    API_CHECK(beta >= ILRNLayer::Parameters::minBeta() &&
              beta <= ILRNLayer::Parameters::maxBeta());
    API_CHECK(k >= ILRNLayer::Parameters::minK() &&
              k <= ILRNLayer::Parameters::maxK());
    mParams.windowSize = windowSize;
    mParams.alpha = alpha;
    mParams.beta = beta;
    mParams.k = k;
}
LRNLayer::~LRNLayer() {}

bool LRNLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeInt32("window_size", mParams.windowSize);
    ok = ok && e->writeFloat32("alpha",     mParams.alpha);
    ok = ok && e->writeFloat32("beta",      mParams.beta);
    ok = ok && e->writeFloat32("k",         mParams.k);
    return ok;
}

bool LRNLayer::deserializeFrom(WisdomContainerEntry *e)
{
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readInt32("window_size", mParams.windowSize);
    ok = ok && e->readFloat32("alpha",     mParams.alpha);
    ok = ok && e->readFloat32("beta",      mParams.beta);
    ok = ok && e->readFloat32("k",         mParams.k);
    return ok;
}

NvU16 LRNLayer::getFactoryType() const
{
    return LayerFactoryType::LRN;
}

#define CurrentScope LRNLayer
ACCESSOR_MUTATOR_CHECK_EXPR(float, alpha, Alpha,
                            (fabsf(alpha) < ILRNLayer::Parameters::maxAbsAlpha()) );
ACCESSOR_MUTATOR_CHECK_EXPR(float, beta, Beta,
                            (beta >= ILRNLayer::Parameters::minBeta() &&
                             beta <= ILRNLayer::Parameters::maxBeta()) );
ACCESSOR_MUTATOR_CHECK_EXPR(float, k, K,
                            (k >= ILRNLayer::Parameters::minK() &&
                             k <= ILRNLayer::Parameters::maxK()) );
ACCESSOR_MUTATOR_CHECK_EXPR(int, windowSize, WindowSize,
                            (windowSize >= ILRNLayer::Parameters::minWindowSize() &&
                             windowSize <= ILRNLayer::Parameters::maxWindowSize()) );
#undef CurrentScope

Dims4 LRNLayer::getOutputDimensions() const
{
    return mInputs[0]->getDimensions();
}

const LRNLayer::Parameters& LRNLayer::getParams() const { return mParams; }


//----------------------------------------------------------------------
// ScaleLayer
//----------------------------------------------------------------------
ScaleLayer::ScaleLayer() : Layer(NULL, LayerType::kSCALE, "", NULL, NULL) { }
ScaleLayer::ScaleLayer(INetwork* network, const std::string& name,
                       ITensor* input,
                       ITensor* output,
                       ScaleMode mode,
                       Weights shift,
                       Weights scale,
                       Weights power)
    : Layer(network, LayerType::kSCALE, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK_ENUM_RANGE(ScaleMode, mode);
    API_CHECK_WEIGHTS0(shift);
    API_CHECK_WEIGHTS0(scale);
    API_CHECK_WEIGHTS0(power);
    mParams.mode = mode;
    mParams.shift = shift;
    mParams.scale = scale;
    mParams.power = power;
}
ScaleLayer::~ScaleLayer() {}


bool ScaleLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeWeights("shift", mParams.shift);
    ok = ok && e->writeWeights("scale", mParams.scale);
    ok = ok && e->writeWeights("power", mParams.power);
    ok = ok && e->writeUInt8Enum("mode",  mParams.mode);
    return ok;
}

bool ScaleLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvU8 v = ScaleMode::sUNIFORM;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readWeights("shift", mParams.shift);
    ok = ok && e->readWeights("scale", mParams.scale);
    ok = ok && e->readWeights("power", mParams.power);
    ok = ok && e->readUInt8Enum("mode", v);
    mParams.mode = ScaleMode(v);
    return ok;
}

NvU16 ScaleLayer::getFactoryType() const
{
    return LayerFactoryType::SCALE;
}


#define CurrentScope ScaleLayer
ACCESSOR_MUTATOR_WEIGHTS(Weights, scale, Scale);
ACCESSOR_MUTATOR_WEIGHTS(Weights, shift, Shift);
ACCESSOR_MUTATOR_WEIGHTS(Weights, power, Power);
ACCESSOR_MUTATOR_CHECK_ENUM(ScaleMode, mode, Mode);
//ACCESSOR_MUTATOR_CHECK_ENUM(DataType, arithmeticType, ArithmeticType)
#undef CurrentScope

Dims4 ScaleLayer::getOutputDimensions()  const
{
    return mInputs[0]->getDimensions();
}

const ScaleLayer::Parameters& ScaleLayer::getParams() const { return mParams; }

//----------------------------------------------------------------------
// BatchNormLayer
//----------------------------------------------------------------------
BatchNormLayer::BatchNormLayer() : Layer(NULL, LayerType::kBATCH_NORM, "", NULL, NULL) { }
BatchNormLayer::BatchNormLayer(INetwork* network, const std::string& name,
                       ITensor* input,
                       ITensor* output,
                       BatchNormMode mode,
                       Weights mean,
                       Weights variance,
                       float epsilon)
    : Layer(network, LayerType::kBATCH_NORM, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK_ENUM_RANGE(BatchNormMode, mode);
    API_CHECK_WEIGHTS0(mean);
    API_CHECK_WEIGHTS0(variance);
    mParams.mode = mode;
    mParams.mean = mean;
    mParams.variance = variance;
    mParams.epsilon  = epsilon;
}
BatchNormLayer::~BatchNormLayer() {}


bool BatchNormLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeWeights("mean", mParams.mean);
    ok = ok && e->writeWeights("variance", mParams.variance);
    ok = ok && e->writeUInt8Enum("mode",  mParams.mode);
    ok = ok && e->writeFloat32("epsilon", mParams.epsilon);
    return ok;
}

bool BatchNormLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvU8 v = BatchNormMode::bnUNIFORM;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readWeights("mean", mParams.mean);
    ok = ok && e->readWeights("variance", mParams.variance);
    ok = ok && e->readUInt8Enum("mode", v);
    mParams.mode = BatchNormMode(v);
    ok = ok && e->readFloat32("epsilon", mParams.epsilon);
    return ok;
}

NvU16 BatchNormLayer::getFactoryType() const
{
    return LayerFactoryType::BATCH_NORM;
}


#define CurrentScope BatchNormLayer
ACCESSOR_MUTATOR_WEIGHTS(Weights, mean, Mean);
ACCESSOR_MUTATOR_WEIGHTS(Weights, variance, Variance);
ACCESSOR_MUTATOR_CHECK_ENUM(BatchNormMode, mode, Mode);
ACCESSOR_MUTATOR(float, epsilon, Epsilon);
#undef CurrentScope

Dims4 BatchNormLayer::getOutputDimensions()  const
{
    return mInputs[0]->getDimensions();
}

const BatchNormLayer::Parameters& BatchNormLayer::getParams() const { return mParams; }


//----------------------------------------------------------------------
// SoftMaxLayer
//----------------------------------------------------------------------
SoftMaxLayer::SoftMaxLayer() :	Layer(NULL, LayerType::kSOFTMAX, "", NULL, NULL) { }
SoftMaxLayer::SoftMaxLayer(INetwork* network,
                           const std::string& name,
                           ITensor* input,
                           ITensor* output)
    : Layer(network, LayerType::kSOFTMAX, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
}
SoftMaxLayer::~SoftMaxLayer() {}

bool SoftMaxLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    return ok;
}
bool SoftMaxLayer::deserializeFrom(WisdomContainerEntry *e)
{
    bool ok = Layer::deserializeFrom(e);
    return ok;
}

Dims4 SoftMaxLayer::getOutputDimensions() const
{
    return mInputs[0]->getDimensions();
}

NvU16 SoftMaxLayer::getFactoryType() const
{
    return LayerFactoryType::SOFT_MAX;
}

#define CurrentScope SoftMaxLayer
//ACCESSOR_MUTATOR_CHECK_ENUM(DataType, arithmeticType, ArithmeticType)
#undef CurrentScope

const SoftMaxLayer::Parameters& SoftMaxLayer::getParams() const { return mParams; }


//----------------------------------------------------------------------
// ConcatenationLayer
//----------------------------------------------------------------------
ConcatenationLayer::ConcatenationLayer() :	Layer(NULL, LayerType::kCONCATENATION, "", NULL, NULL) { }
ConcatenationLayer::ConcatenationLayer(INetwork* network,
                                       const std::string& name,
                                       ITensor*const * inputs,
                                       int numInputs,
                                       ITensor* output)
    : Layer(network, LayerType::kCONCATENATION, name, inputs, numInputs, &output, 1)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(inputs);
    API_CHECK(numInputs < MAX_NUM_INPUT_LAYERS);
    API_CHECK_NULL(output);
    for (int j = 0; j < numInputs; j++)
        API_CHECK_NULL(inputs[j]);
}
ConcatenationLayer::~ConcatenationLayer() {}

bool ConcatenationLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    return ok;
}

bool ConcatenationLayer::deserializeFrom(WisdomContainerEntry *e)
{
    bool ok = Layer::deserializeFrom(e);
    return ok;
}

NvU16 ConcatenationLayer::getFactoryType() const
{
    return LayerFactoryType::CONCATENATION;
}

Dims4 ConcatenationLayer::getOutputDimensions() const
{
    int h = 0, w = 0, c = 0;
    for (unsigned i = 0; i < mInputs.size(); i++)
    {
        Dims4 dims = mInputs[i]->getDimensions();
        if (i == 0) {
            h = dims.h;
            w = dims.w;
        }
        //assert(h == dims.h); // TODO: error message
        //assert(w == dims.w);
        c += dims.c;
    }
    return Dims4(c, h, w);
}

const ConcatenationLayer::Parameters& ConcatenationLayer::getParams() const
{
    return mParams;
}


//----------------------------------------------------------------------
// SliceLayer
//----------------------------------------------------------------------
SliceLayer::SliceLayer() :  Layer(NULL, LayerType::kSLICE, "", NULL, NULL) { }
SliceLayer::SliceLayer(INetwork* network,
                       const std::string& name,
                       ITensor* input,
                       ITensor* const* outputs,
                       int numOutputs)
    : Layer(network, LayerType::kSLICE, name, &input, 1, outputs, numOutputs)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK(numOutputs < MAX_NUM_OUTPUT_LAYERS);
    API_CHECK_NULL(outputs);
    for (int j = 0; j < numOutputs; j++)
        API_CHECK_NULL(outputs[j]);
}
SliceLayer::~SliceLayer() {}

bool SliceLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    return ok;
}

bool SliceLayer::deserializeFrom(WisdomContainerEntry *e)
{
    bool ok = Layer::deserializeFrom(e);
    return ok;
}

NvU16 SliceLayer::getFactoryType() const
{
    return LayerFactoryType::SLICE;
}

Dims4 SliceLayer::getOutputDimensions() const
{
    Dims4 inputDims = mInputs[0]->getDimensions();

    if (inputDims.c % mOutputs.size() != 0)
        std::cerr << "channel dimension [" << inputDims.c << "] not divisible by [" << mOutputs.size() << "]" << std::endl;

    // TODO: Allow slices along H and W dimensions
    Dims4 outputDims = inputDims;
    outputDims.c = inputDims.c / mOutputs.size();

    return outputDims;
}

const SliceLayer::Parameters& SliceLayer::getParams() const
{
    return mParams;
}


//----------------------------------------------------------------------
// DeconvolutionLayer
//----------------------------------------------------------------------
DeconvolutionLayer::DeconvolutionLayer() :	Layer(NULL, LayerType::kDECONVOLUTION, "", NULL, NULL) { }
DeconvolutionLayer::DeconvolutionLayer(INetwork* network,
                                       const std::string& name,
                                       ITensor* input,
                                       ITensor* output,
                                       int numOutputMaps,
                                       Dims2 kernelSize,
                                       Weights kernelWeights,
                                       Weights biasWeights)
    : Layer(network, LayerType::kDECONVOLUTION, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(numOutputMaps > 0 && numOutputMaps < MAX_OUTPUT_MAPS);
    API_CHECK(kernelSize.h > 0 && kernelSize.w > 0 &&
              kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT);
    API_CHECK_WEIGHTS(kernelWeights);
    API_CHECK_WEIGHTS0(biasWeights);
    mParams.kernelSize = kernelSize;
    mParams.numOutputMaps = numOutputMaps;
    mParams.topLeftPadding = Dims2(0, 0); // set via accessors
    mParams.bottomRightPadding = Dims2(0, 0); // set via accessors
    mParams.stride = Dims2(1, 1);
    mParams.numGroups = 1;
    mParams.paddingValue = 0;
    mParams.kernelWeights = kernelWeights;
    mParams.biasWeights = biasWeights;
}
DeconvolutionLayer::DeconvolutionLayer(INetwork* network, const std::string& name,
                                       ITensor* input,
                                       ITensor* output,
                                       int numOutputMaps,
                                       int paddingValue,
                                       Dims2 kernelSize,
                                       Dims2 tlPadding,
                                       Dims2 brPadding,
                                       Dims2 stride,
                                       Dims2 dilation,
                                       Weights kernelWeights,
                                       Weights biasWeights,
                                       BiasMode biasMode,
                                       int numGroups)
    : Layer(network, LayerType::kDECONVOLUTION, name, input, output)
{
    API_CHECK_NULL(network);
    API_CHECK_NULL(input);
    API_CHECK_NULL(output);
    API_CHECK(numOutputMaps > 0 && numOutputMaps < MAX_OUTPUT_MAPS);
    API_CHECK(kernelSize.h > 0 && kernelSize.w > 0 && kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT);
    API_CHECK_WEIGHTS(kernelWeights);
    API_CHECK_WEIGHTS0(biasWeights);
    API_CHECK_ENUM_RANGE(BiasMode, biasMode);
    mParams.kernelSize = kernelSize;
    mParams.numOutputMaps = numOutputMaps;
    mParams.topLeftPadding = tlPadding;
    mParams.bottomRightPadding = brPadding;
    mParams.paddingValue = paddingValue;
    mParams.stride = stride;
    mParams.dilation = dilation;
    mParams.kernelWeights = kernelWeights;
    mParams.biasWeights = biasWeights;
    mParams.biasMode = biasMode;
    mParams.numGroups = numGroups;
}

DeconvolutionLayer::~DeconvolutionLayer() {}

bool DeconvolutionLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeDims2("kernel_size",          mParams.kernelSize);
    ok = ok && e->writeDims2("top_left_padding",     mParams.topLeftPadding);
    ok = ok && e->writeDims2("bottom_right_padding", mParams.bottomRightPadding);
    ok = ok && e->writeDims2("stride",               mParams.stride);
    ok = ok && e->writeDims2("dilation",             mParams.dilation);
    ok = ok && e->writeInt32("num_output_maps",      mParams.numOutputMaps);
    ok = ok && e->writeInt32("num_groups",           mParams.numGroups);
    ok = ok && e->writeInt32("padding_value",        mParams.paddingValue);
    ok = ok && e->writeWeights("kernel_weights",     mParams.kernelWeights);
    ok = ok && e->writeWeights("bias_weights",       mParams.biasWeights);
    ok = ok && e->writeUInt8Enum("bias_mode",        mParams.biasMode);
    ok = ok && e->writeInt32("groups",               mParams.numGroups);
    return ok;
}

bool DeconvolutionLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvU8 v = BiasMode::bUNIFORM;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readDims2("kernel_size",          mParams.kernelSize);
    ok = ok && e->readDims2("top_left_padding",     mParams.topLeftPadding);
    ok = ok && e->readDims2("bottom_right_padding", mParams.bottomRightPadding);
    ok = ok && e->readDims2("stride",               mParams.stride);
    ok = ok && e->readDims2("dilation",             mParams.dilation);
    ok = ok && e->readInt32("num_output_maps",      mParams.numOutputMaps);
    ok = ok && e->readInt32("num_groups",           mParams.numGroups);
    ok = ok && e->readInt32("padding_value",        mParams.paddingValue);
    ok = ok && e->readWeights("kernel_weights",     mParams.kernelWeights);
    ok = ok && e->readWeights("bias_weights",       mParams.biasWeights);
    ok = ok && e->readUInt8Enum("bias_mode",        v);
    ok = ok && e->readInt32("groups",               mParams.numGroups);
    mParams.biasMode = BiasMode(v);

    return ok;
}

NvU16 DeconvolutionLayer::getFactoryType() const
{
    return LayerFactoryType::DECONVOLUTION;
}

#define CurrentScope DeconvolutionLayer
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, kernelSize, KernelSize,
                            (kernelSize.h > 0 && kernelSize.w > 0 &&
                             kernelSize.h*kernelSize.w < MAX_KERNEL_DIMS_PRODUCT) );
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, topLeftPadding, TopLeftPadding,
                            (topLeftPadding.h >= 0 && topLeftPadding.w >= 0 &&
                             topLeftPadding.h + topLeftPadding.w < MAX_PADDING_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, bottomRightPadding, BottomRightPadding,
                         (bottomRightPadding.h >= 0 && bottomRightPadding.w >= 0 &&
                          bottomRightPadding.h + bottomRightPadding.w < MAX_PADDING_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, paddingValue, PaddingValue,
                             (paddingValue >= 0 && paddingValue < MAX_PADDING_VALUE) )
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, stride, Stride,
                            (stride.h > 0 && stride.w > 0 &&
                             stride.h + stride.w < MAX_STRIDE_SUM) );
ACCESSOR_MUTATOR_CHECK_EXPR(Dims2, dilation, Dilation,
                             (dilation.h > 0 && dilation.w > 0 &&
                              dilation.h + dilation.w < MAX_DILATION_SUM) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, numGroups, NumGroups,
                            (numGroups > 0 && numGroups < MAX_GROUPS) )
ACCESSOR_MUTATOR_CHECK_EXPR(int, numOutputMaps, NumOutputMaps,
                            (numOutputMaps > 0 && numOutputMaps < MAX_OUTPUT_MAPS ) )
//ACCESSOR_MUTATOR_CHECK_ENUM(DataType, arithmeticType, ArithmeticType)
ACCESSOR_MUTATOR_CHECK_ENUM(BiasMode, biasMode, BiasMode)
ACCESSOR_MUTATOR_WEIGHTS(Weights, kernelWeights, KernelWeights)
ACCESSOR_MUTATOR_WEIGHTS(Weights, biasWeights, BiasWeights)
#undef CurrentScope

Dims4 DeconvolutionLayer::getOutputDimensions() const
{

    Dims4 inputDims = mInputs[0]->getDimensions();
    //		assert(inputDims.c > 0 && inputDims.h > 0 && inputDims.w > 0);
    Dims2 convDims = mNetwork->getDeconvolutionOutputDimensionsFormula().
            compute(Dims2(inputDims.h, inputDims.w),
                    mParams.kernelSize, mParams.stride, mParams.topLeftPadding, mParams.bottomRightPadding, mParams.dilation, getName());
    return Dims4(mParams.numOutputMaps, convDims.h, convDims.w);
}

const DeconvolutionLayer::Parameters& DeconvolutionLayer::getParams() const
{
    return mParams;
}


//----------------------------------------------------------------------
// ElementWiseLayer
//----------------------------------------------------------------------
ElementWiseLayer::ElementWiseLayer() :	Layer(NULL, LayerType::kELEMENTWISE, "", NULL, NULL) { }
ElementWiseLayer::ElementWiseLayer(INetwork* network, const std::string& name,
                                   ITensor*const *inputs, ITensor* output,
                                   ElementWiseOperation op)
    : Layer(network, LayerType::kELEMENTWISE, name, inputs, 2, &output, 1)
{
    API_CHECK_NULL(network);
    API_CHECK_ENUM_RANGE(ElementWiseOperation, op);
    API_CHECK_NULL(inputs);
    API_CHECK_NULL(inputs[0]);
    API_CHECK_NULL(inputs[1]);
    API_CHECK_NULL(output);
    mParams.operation = op;
}

ElementWiseLayer::~ElementWiseLayer() {}


bool ElementWiseLayer::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = Layer::serializeTo(e);
    ok = ok && e->writeInt32("operation", mParams.operation);
    return ok;
}

bool ElementWiseLayer::deserializeFrom(WisdomContainerEntry *e)
{
    NvS32 v = kSUM;
    bool ok = Layer::deserializeFrom(e);
    ok = ok && e->readInt32("operation", v);
    mParams.operation = ElementWiseOperation(v);
    return ok;
}

NvU16 ElementWiseLayer::getFactoryType() const
{
    return LayerFactoryType::ELEMENT_WISE;
}

#define CurrentScope ElementWiseLayer
ACCESSOR_MUTATOR_CHECK_ENUM(ElementWiseOperation, operation, Operation)
//ACCESSOR_MUTATOR_CHECK_ENUM(DataType, arithmeticType, ArithmeticType)
#undef CurrentScope

Dims4 ElementWiseLayer::getOutputDimensions() const
{
    return mInputs[0]->getDimensions();
}

const ElementWiseLayer::Parameters& ElementWiseLayer::getParams() const
{
    return mParams;
}


//----------------------------------------------------------------------
// These shenanigans are required to explicitly instantiate the static
// members of the various DiamondMaps
//----------------------------------------------------------------------
template <>
std::map<Layer *, ConvolutionLayerDiamond> BasePrivDiamondMap<ConvolutionLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, ConvolutionLayerDiamond>();

template <>
std::map<Layer *, FullyConnectedLayerDiamond> BasePrivDiamondMap<FullyConnectedLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, FullyConnectedLayerDiamond>();

template <>
std::map<Layer *, ActivationLayerDiamond> BasePrivDiamondMap<ActivationLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, ActivationLayerDiamond>();

template <>
std::map<Layer *, PoolingLayerDiamond> BasePrivDiamondMap<PoolingLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, PoolingLayerDiamond>();

template <>
std::map<Layer *, LRNLayerDiamond> BasePrivDiamondMap<LRNLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, LRNLayerDiamond>();

template <>
std::map<Layer *, ScaleLayerDiamond> BasePrivDiamondMap<ScaleLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, ScaleLayerDiamond>();

template <>
std::map<Layer *, BatchNormLayerDiamond> BasePrivDiamondMap<BatchNormLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, BatchNormLayerDiamond>();

template <>
std::map<Layer *, SoftMaxLayerDiamond> BasePrivDiamondMap<SoftMaxLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, SoftMaxLayerDiamond>();

template <>
std::map<Layer *, ConcatenationLayerDiamond> BasePrivDiamondMap<ConcatenationLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, ConcatenationLayerDiamond>();

template <>
std::map<Layer *, SliceLayerDiamond> BasePrivDiamondMap<SliceLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, SliceLayerDiamond>();

template <>
std::map<Layer *, DeconvolutionLayerDiamond> BasePrivDiamondMap<DeconvolutionLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, DeconvolutionLayerDiamond>();

template <>
std::map<Layer *, ElementWiseLayerDiamond> BasePrivDiamondMap<ElementWiseLayerDiamond>::s_base_priv_diamond_map =
    std::map<Layer *, ElementWiseLayerDiamond>();


//----------------------------------------------------------------------
// must explicitly instance these...
//----------------------------------------------------------------------

template ConvolutionLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<ConvolutionLayerDiamond>(ConvolutionLayerDiamond::BasePrivType *base_priv);

template FullyConnectedLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<FullyConnectedLayerDiamond>(FullyConnectedLayerDiamond::BasePrivType *base_priv);

template ActivationLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<ActivationLayerDiamond>(ActivationLayerDiamond::BasePrivType *base_priv);

template PoolingLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<PoolingLayerDiamond>(PoolingLayerDiamond::BasePrivType *base_priv);

template LRNLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<LRNLayerDiamond>(LRNLayerDiamond::BasePrivType *base_priv);

template ScaleLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<ScaleLayerDiamond>(ScaleLayerDiamond::BasePrivType *base_priv);

template BatchNormLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<BatchNormLayerDiamond>(BatchNormLayerDiamond::BasePrivType *base_priv);

template SoftMaxLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<SoftMaxLayerDiamond>(SoftMaxLayerDiamond::BasePrivType *base_priv);

template ConcatenationLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<ConcatenationLayerDiamond>(ConcatenationLayerDiamond::BasePrivType *base_priv);

template SliceLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<SliceLayerDiamond>(SliceLayerDiamond::BasePrivType *base_priv);

template DeconvolutionLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<DeconvolutionLayerDiamond>(DeconvolutionLayerDiamond::BasePrivType *base_priv);

template ElementWiseLayerDiamond::DerivedPrivType *
LayerFactory::derivedPriv<ElementWiseLayerDiamond>(ElementWiseLayerDiamond::BasePrivType *base_priv);





} // nvdla::priv
} // ::nvdla
