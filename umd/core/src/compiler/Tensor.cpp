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

#include <cmath>  // for fabs

#include "priv/Check.h"

#include "priv/Tensor.h"
#include "priv/WisdomContainer.h"


using std::string;
using std::endl;

namespace nvdla {


ITensor::ITensor() { }
ITensor::~ITensor() { }


namespace priv {



TensorFactory::TensorPrivPair TensorFactory::newTensor()
{
    ITensor *tensor;
    Tensor *tensor_priv;
    tensor = tensor_priv = new priv::Tensor();
    if (tensor) {
        s_priv.insert(tensor,tensor_priv);
        s_self.insert(tensor,tensor);
    }
    return TensorPrivPair(tensor, tensor_priv);
}

Tensor *TensorFactory::priv(ITensor *tensor)
{
    // gLogError << __func__ << " looking up priv for base_i=" << tensor << endl;
    BiMap<ITensor *, Tensor *>::left_iterator f = s_priv.find_left(tensor);
    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ITensor *TensorFactory::i(Tensor *tensor)
{
    BiMap<ITensor *, Tensor *>::right_iterator f = s_priv.find_right(tensor);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}


ITensor *TensorFactory::self(void *s)
{
    BiMap<void *, ITensor *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}


ITensor *TensorFactory::deserializeFrom(WisdomContainerEntry *entry)
{
    bool ok = false;
    ITensor *tensor = NULL;

    // only one type of tensor right now (ITensor/Tensor)... but go through the motions
    WisdomContainerEntry factory_type_entry;
    // TensorType factory_type;
    NvU32 v;

    if ( entry->type() != IWisdomContainerEntry::ENTRY_TYPE_OBJECT ) {
        goto done;
    }

    ok = entry->getEntry("factory_type", IWisdomContainerEntry::ENTRY_TYPE_UINT32, &factory_type_entry);

    ok = ok && factory_type_entry.readUInt32(v);
    if ( !ok ) {
        goto done;
    }

    tensor = deserializeTensor(entry);

 done:
    return tensor;
}

BiMap<ITensor *, Tensor*> TensorFactory::s_priv;
BiMap<void *, ITensor*> TensorFactory::s_self;

// there's only one type of "Tensor" for now. so only one of these... so it looks
// silly.  see the same paths in "LayerFactory::deserialize*" for why it makes sense
// to organize this way preemptively.
ITensor *TensorFactory::deserializeTensor(WisdomContainerEntry *entry)
{
    TensorFactory::TensorPrivPair t = newTensor();
    if ( !t ) {
        return NULL;
    }
    t.priv()->deserializeFrom(entry);
    return t.i();
}


NvU16 Tensor::getFactoryType() const
{
    return 0; // only one type of tensor so far, not complicated by factory splits
}


Tensor::~Tensor() { }

bool Tensor::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = true;
    WisdomContainerEntry name_entry;

    // gLogError << __func__ << " name=[" << getName() << "]" << endl;

    ok = ok && e->writeUInt32("factory_type", getFactoryType());
    ok = ok && e->writeString("name", getName());
    ok = ok && e->writeDims4("dimensions", getDimensions());
    ok = ok && e->writeInt32("data_format", getDataFormat().v());
    ok = ok && e->writeInt32("data_type", getDataType().v());
    ok = ok && e->writeInt32("tensor_type", (int) getTensorType());

    return ok;
}

bool Tensor::deserializeFrom(WisdomContainerEntry *e)
{
    WisdomContainerEntry name_entry;
    string name;
    Dims4 dims;
    NvS32 data_format, data_type, tensor_type;

    bool ok = e->getEntry(string("name"), IWisdomContainerEntry::ENTRY_TYPE_STRING, &name_entry);
    ok = ok && name_entry.readString(name);
    if ( ok ) {
        setName(name.c_str());
    }
    ok = ok && e->readDims4("dimensions", dims);
    if (ok) {
        setDimensions(dims);
    }

    ok = ok && e->readInt32("data_format", data_format);
    if ( ok ) {
        setDataFormat((DataFormat)data_format);
    }

    ok = ok && e->readInt32("data_type", data_type);
    if ( ok ) {
        setDataType((DataType)data_type);
    }

    ok = ok && e->readInt32("tensor_type", tensor_type);
    if ( ok ) {
        setTensorType((TensorType)tensor_type);
    }


    return ok;
}


void Tensor::setDimensions(Dims4 dimensions)
{
    //API_CHECK_DIMS3_TENSOR(dimensions);

    mDimensions = dimensions;
}

Dims4 Tensor::getDimensions() const
{
    return mDimensions;
}

bool Tensor::isNetworkInput() const
{
    API_CHECK_NULL_RETVAL(mNetwork, false);

    for (int i = 0; i < mNetwork->getNumInputs(); i++) {
        if (mNetwork->getInput(i) == this) {
            return true;
        }
    }
    return false;
}
bool Tensor::isNetworkOutput() const
{
    API_CHECK_NULL_RETVAL(mNetwork, false);

    for (int i = 0; i < mNetwork->getNumOutputs(); i++) {
        if (mNetwork->getOutput(i) == this) {
            return true;
        }
    }
    return false;
}


Tensor::Tensor(INetwork* network, const string name) :
    mDimensions({0,0,0}),
    mNetwork(network),
    mName(name),
    mDataFormat(DataFormat::NCHW),
    mDataType(DataType::FLOAT)
{
    API_CHECK_NULL(network);
}

const char* Tensor::getName() const
{
    return mName.c_str();
}

void Tensor::setName(const char* n)
{
    API_CHECK_NULL(n); mName = n;
}

INetwork *Tensor::getNetwork() const
{
    return mNetwork;
}

void Tensor::setNetwork(INetwork *network)
{
    mNetwork = network;
}

DataFormat Tensor::getDataFormat() const
{
    return mDataFormat;
}

void Tensor::setDataFormat(DataFormat f)
{
    mDataFormat = f;
}

DataType Tensor::getDataType() const
{
    return mDataType;
}

void Tensor::setDataType(DataType t)
{
    mDataType = t;
}

TensorType Tensor::getTensorType() const
{
    return mTensorType;
}

void Tensor::setTensorType(TensorType t)
{
    mTensorType = t;
}

NvDlaError Tensor::setChannelDynamicRange(NvS32 chnlIndx, NvF32 min, NvF32 max)
{
    NvDlaError e = NvDlaSuccess;
    NvF32 scaleFactor = std::max<NvF32>(std::fabs(min), std::fabs(max))/127.0;

    if (chnlIndx >= mDimensions.c)
    {
        e = NvDlaError_BadParameter;
        goto fail;
    }
    else if (chnlIndx == -1)
    {
        // clear existing scales and adjust vector capacity if need be before inserting
        mChnlScales.clear();
        mChnlScales.reserve(mDimensions.c);
        for (NvU32 cc = 0; cc < static_cast<NvU32>(mDimensions.c); ++cc)
        {
            mChnlScales.push_back(scaleFactor);
        }
    }
    else
    {
        // adjust vector capacity before inserting
        if (mChnlScales.capacity() < (size_t)mDimensions.c)
        {
            mChnlScales.reserve(mDimensions.c);
        }
        mChnlScales.insert(mChnlScales.begin() + chnlIndx, scaleFactor);
    }

fail:
    return e;
}

NvDlaError Tensor::setChannelOffset(NvS32 chnlIndx, NvF32 offset)
{
    return NvDlaError_NotSupported;
}

} // nvdla::priv

} // nvdla
