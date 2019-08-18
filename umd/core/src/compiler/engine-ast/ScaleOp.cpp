/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>

#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "priv/LowPrecision.h"
#include "priv/WeightTranslationUnit.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

engine_ast::SDPNode* engine_ast::SDPScaleOpNode::addSDPBiasOpNode
(
    canonical_ast::Node* orig_can_node
)
{
    Tensor* origOutputTensor;
    Tensor* ioTensor                     = NULL;
    engine_ast::Edge* ioEdge             = NULL;
    engine_ast::SDPBiasOpNode* biasNode  = NULL;
    canonical_ast::Graph* canGraph       = orig_can_node->graph();
    NVDLA_UNUSED(ioEdge);

    biasNode = engine_ast::NodeFactory::newSDPBiasOpNode(orig_can_node, graph());
    biasNode->params().setHasBiasReduction(false);   // if scale-op has a shift term, the shift data is 'added' to scaled data

    if ( !biasNode ) {
        goto done;
    }

    // cache the dimensions of output tensor of parent node
    origOutputTensor = canGraph->downstreamEdges(orig_can_node).at(0)->originalTensor();

    ioTensor = origOutputTensor->clone();
    ioTensor->setTensorType(TensorType::kIO);

    ioEdge   = graph()->addDataEdge((canonical_ast::Edge*)0, this, biasNode, ioTensor);

done:
    return biasNode;
}

/**
 * Analogous to captureCanonicalParams
 *
 * This function is useful when no canonical node is available and
 * to derive a new set of params with newly created unit
 * scale data.
 **/
NvDlaError engine_ast::SDPScaleOpNode::populateWithUnitScaleParams
(
    engine_ast::SDPMode scaleMode,
    Dims4               scaleDims
)
{
    NvDlaError e = NvDlaSuccess;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();
    Weights unitScaleData = Weights(nvdla::DataType::FLOAT, NULL, 0);

    Tensor* scaleDataTensor;
    engine_ast::Edge* scaleDataEdge  = NULL;
    NVDLA_UNUSED(scaleDataEdge);

    /* Creates a new unit scale data */
    PRECISION_SWITCH(nvdla::DataType::FLOAT, computePrecision.v(), unitScaleData,
                                                                    WeightTrns::createUnitScaleData,
                                                                    scaleMode,
                                                                    scaleDims);

    params().x1Params().setEnabled(true);
    params().x1Params().setMode(scaleMode);
    params().setScaleDims(scaleDims);
    params().setRawScaleData(unitScaleData);
    params().setDLAScaleData(Weights(DataType::FLOAT, NULL, 0));

    setUnitScale(true);

    /* adds aux tensor and an edge corresponding to it */
    scaleDataTensor = graph()->addAuxTensor(graph()->newAuxTensorName(), params().scaleDims(), TensorType::kSCALE);
    scaleDataEdge = graph()->addDataEdge((canonical_ast::Edge*)0, 0, this, scaleDataTensor);

    return e;
}

NvDlaError engine_ast::SDPScaleOpNode::captureCanonicalScaleData()
{
    NvDlaError e = NvDlaSuccess;
    Tensor* rawSclDataTensor;
    engine_ast::Edge* rawSclDataEdge  = NULL;
    NVDLA_UNUSED(rawSclDataEdge);

    rawSclDataTensor = graph()->addAuxTensor(graph()->newAuxTensorName(), params().scaleDims(), TensorType::kSCALE);
    rawSclDataEdge = graph()->addDataEdge((canonical_ast::Edge*)0, 0, this, rawSclDataTensor);

    return e;
}

void engine_ast::SDPScaleOpNode::captureCanonicalParams()
{
    NvDlaError e = NvDlaSuccess;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();
    nvdla::QuantizationMode quantizationMode =  graph()->profile()->quantizationMode();

    params().x1Params().setEnabled(true);
    params().x1Params().setAluType(SDPALUTypeEnum::SDP_ALU_TYPE_SUM);
    params().x1Params().setOpType(SDPOpTypeEnum::SDP_OP_TYPE_MUL);

    /**
     * Cautiously captures canonical params into engine params.
     * Convert per-layer scaling data to per-channel scaling data if
     *  [1] per-filter quantization mode is enabled and operated at INT8 precision
     *  [2] SDP scale mode is PER_LAYER.
     * This conversion is done in advance to avoid possible complex computation at later stages,
     * when rescaling factors of convolution combine with SDP scale data.
     **/
    if (computePrecision.v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 &&
        quantizationMode.v() == nvdla::QuantizationMode::PER_FILTER &&
        canonicalNode()->params().mode() == ScaleMode::sUNIFORM)
    {
        Dims4 trnsScaleDims;
        Weights orignalScaleBlob = canonicalNode()->params().scale();
        Weights trnsScaleBlob = Weights(DataType::FLOAT, NULL, 0);

        if (canonicalNode()->inputEdges().size() == 0)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No input edges available for %s", canonicalNode()->name().c_str());
        }

        Tensor *inputTensor = canonicalNode()->inputEdges().at(0)->originalTensor();
        if (inputTensor == NULL)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Missing input tensor for %s", canonicalNode()->name().c_str());
        }

        // Update scale channels to input edge channels.
        trnsScaleDims = canonicalNode()->params().scaleDims();
        trnsScaleDims.c = inputTensor->getDimensions().c;

        // conversion from per-layer to per-channel
        PRECISION_SWITCH(orignalScaleBlob.type.v(), computePrecision.v(), trnsScaleBlob,
                                                            WeightTrns::translatePLScaleToPCScale,
                                                            orignalScaleBlob,
                                                            trnsScaleDims.c);

        params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL);
        params().setScaleDims(trnsScaleDims);
        params().setRawScaleData(trnsScaleBlob);
    }
    else{
        switch(canonicalNode()->params().mode())
        {
            case ScaleMode::sUNIFORM:        params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_LAYER); break;
            case ScaleMode::sCHANNEL:        params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL); break;
            case ScaleMode::sm_ELEMENTWISE:  params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_ELEMENT); break;
            default: params().x1Params().setMode(SDPModeEnum::SDP_MODE_UNKNOWN);
        }
        params().setScaleDims(canonicalNode()->params().scaleDims());
        params().setRawScaleData(canonicalNode()->params().scale());
    }
    params().setDLAScaleData(Weights(DataType::FLOAT, NULL, 0));
    PROPAGATE_ERROR_FAIL(captureCanonicalScaleData());

    if ( graph()->debugWeights() )
    {
        Weights rawData = params().rawScaleData();
        gLogInfo << "raw weights of " << name() << ": ";
        for (ssize_t ii = 0; ii < rawData.count; ++ii)
            gLogInfo << reinterpret_cast<NvF32*>(const_cast<void*>(rawData.values))[ii] << ", ";
        gLogInfo << endl;
    }
fail:
    return;
}

void engine_ast::SDPScaleOpNode::inheritParams(Node* otherNode)
{
    // inherit the parameters that make sense (keep adding later)
    SDPScaleOpNode* otherScale = NodeFactory::nodeCast<SDPScaleOpNode*>(otherNode);
    params().setX1Params(otherScale->params().x1Params());
    params().setScaleDims(otherScale->params().scaleDims());
    params().setRawScaleData(otherScale->params().rawScaleData());
    params().setDLAScaleData(otherScale->params().DLAScaleData());
    params().setConvMode(otherScale->params().convMode());
    params().setWinogradParams(otherScale->params().winogradParams());
    params().setNumGroups(otherScale->params().numGroups());
    params().setOutCVT(otherScale->params().outCVT());
}


//!<  take reciprocal of raw caffe scale-data
template <typename MP, typename CP>
Weights engine_ast::SDPScaleOpNode::inverseScaleData
(
    engine_ast::SDPMode  scaleMode,      // per-layer/channel/elementwise
    Dims4                scaleDims,      // dims of orig caffe scale-data blob
    Weights&             srcScaleData    // ptr to orig caffe scale blob
)
{
    Weights invSclData = Weights(nvdla::DataType::FLOAT, NULL, 0);

    API_CHECK_WEIGHTS_RETVAL(srcScaleData, invSclData);

    MP* pSrcScale = reinterpret_cast<MP*>(const_cast<void*>(srcScaleData.values));
    MP* pDestScale = (MP*)engine_ast::MemoryCollector::getInstance()->allocateMemory(srcScaleData.count* sizeof(MP));
    memset(pDestScale, 0, srcScaleData.count * sizeof(MP));

    invSclData.type   = srcScaleData.type;
    invSclData.count  = srcScaleData.count;
    invSclData.values = NULL;

    // Scale data can be of 3 types: per-layer/per-channel/per-element
    // Per-Layer:
    if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        pDestScale[0] = 1/pSrcScale[0];
    }
    // Per-Channel:
    else if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL)
    {
        int c = 0;
        for ( ; c < scaleDims.c; ++c)
        {
            pDestScale[c] = 1/pSrcScale[c];
        }
    }
    else
    {
        int i = 0;
        for ( ; i < (scaleDims.c * scaleDims.w * scaleDims.h); ++i)
        {
            pDestScale[i] = 1/pSrcScale[i];
        }
    }

    invSclData.values = pDestScale;

    return invSclData;
}

/*-----------------------Merge Similar Math Ops------------------------------*/
engine_ast::Node* engine_ast::SDPScaleOpNode::mergeWithSDPOp(SDPNode* nextSDP)
{
    Node* removableNode = NULL;

    // If activation is already enabled, do not allow any math op fusions
    if (params().x1Params().actType().v() != SDP_ACT_TYPE_UNKNOWN
        && params().x1Params().actType().v() != SDP_ACT_TYPE_NONE)
    {
        if (graph()->debugMathOptz())
        {
            gLogInfo << "Merge not feasible: " << this->name() << " activation: ";
            gLogInfo << NvU16(params().x1Params().actType().v()) << endl;
        }
        goto fail;
    }

    // fixme: limit the scale math op fusion with only relu for now
    if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_ACTIVATION)
    {
        removableNode = tryToMergeWithActOp(nextSDP);
    }

fail:
    return removableNode;
}

/**
 * Utility function to convert input scale data into a float vector (trnsFp32Scale)
 **/
NvDlaError engine_ast::SDPScaleOpNode::getFp32ScaleData
(
    const Weights data,
    std::vector<NvF32>& trnsFp32Scale
)
{
    NvDlaError e = NvDlaSuccess;

    if (data.count <= 0 || data.values == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Invalid scale data provided");
    }

    for (NvU32 ii = 0; ii < data.count; ii++)
    {
        NvF32 scaleValue = 0;
        switch(data.type)
        {
            case nvdla::DataType::FLOAT:
                scaleValue = static_cast<NvF32>(reinterpret_cast<NvF32*>(const_cast<void*>(data.values))[ii]);
                break;
            case nvdla::DataType::HALF:
                scaleValue = static_cast<NvF32>(reinterpret_cast<half_float::half*>(const_cast<void*>(data.values))[ii]);
                break;
            case nvdla::DataType::INT16:
                scaleValue = static_cast<NvF32>(reinterpret_cast<NvS16*>(const_cast<void*>(data.values))[ii]);
                break;
            case nvdla::DataType::INT8:
                scaleValue = static_cast<NvF32>(reinterpret_cast<NvS8*>(const_cast<void*>(data.values))[ii]);
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported,
                        "Can't convert scale data which is not FLOAT/HALF/INT16/INT8 for %s",
                        name().c_str());
        }
        trnsFp32Scale.push_back(scaleValue);
    }

fail:
    return e;
}

/**
 * Converts raw scale data to NvF32 and rescales.
 * Rescaling applicable if the following conditions statisfied.
 *  [1] If there exists a fused convolution node with SDP scale
 *  [2] Quantization mode: PER_FILTER operating at INT8 precision
 *
 * Usecase
 * -------
 *         B = A * W op C
 * S_B * Q_B = (S_A * Q_A) * (S_W[k] * Q_W) op C
 *       Q_B = (Q_A * Q_W) * (((S_A * S_W[k]) op C) /S_B)
 *       Q_B = (Q_A * Q_W) * (2 ^ -m * n_i16[k'])
 *
 * Where Q_A, quantized int8 input to conv
 *       Q_B, rescaled int8 output from SDP
 *       C,   actual scale data - PER_CHANNEL or PER_ELEMENT (k')
 *
 * rescaleScaleDataForPerFilter computes ((S_A * S_W[k]) op C)/S_B in fp32 precision
 * and overwrites existing raw scale data with it.
 **/
NvDlaError engine_ast::SDPScaleOpNode::rescaleScaleDataForPerFilter()
{
    NvDlaError e = NvDlaSuccess;

    ConvCoreNode* fusedConv;

    /* Different scalars */
    std::vector<NvF32> filterScales;
    std::vector<NvF32> inTensorScales;
    std::vector<NvF32> outTensorScales;

    NvF32 perTensorInTensorScale;
    NvF32 perTensorOutTensorScale;

    /* Original data of scale */
    engine_ast::SDPMode scaleMode;
    Dims4 origScaleDims;
    Weights origScaleBlob;

    std::vector<NvF32> trnsFp32Scale;

    Weights trnsScaleBlob = Weights(nvdla::DataType::FLOAT, NULL, 0);
    NvF32 *trnsScaleData = NULL;
    NvU32 trnsCnt = 0;

    fusedConv = NodeFactory::nodeCast<ConvCoreNode*>(dependencyParams(0).fusedNode(IODirectionEnum::INPUT));
    if (fusedConv == NULL)
    {
        // Rescale only if fused conv available.
        goto fail;
    }

    filterScales = fusedConv->params().filterScales();
    inTensorScales = fusedConv->inputEdges().at(0)->originalTensor()->getChannelScales();
    outTensorScales = outputEdges().at(0)->originalTensor()->getChannelScales();

    perTensorInTensorScale = inTensorScales.at(0);
    perTensorOutTensorScale = outTensorScales.at(0);

    origScaleBlob = params().rawScaleData();
    origScaleDims = params().scaleDims();
    scaleMode = params().x1Params().mode();

    // Preliminary checks
    ASSERT(filterScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);

    ASSERT(inTensorScales.size() == (size_t)fusedConv->inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvF32 its = 1; its < inTensorScales.size(); ++its)
    {
        if ( perTensorInTensorScale != inTensorScales[its] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", fusedConv->name().c_str());
        }
    }

    ASSERT(outTensorScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvF32 ots = 1; ots < outTensorScales.size(); ++ots)
    {
        if ( perTensorOutTensorScale != outTensorScales[ots] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for output of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
        }
    }

    // With PER_FILTER quantization mode, scale data can never be PER_LAYER.
    if (scaleMode.v() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Invalid SDP mode: %s when PER_FILTER is ON",
                            scaleMode.c_str());
    }

    // Convert scale data into float and get it as vector
    trnsFp32Scale.clear();
    PROPAGATE_ERROR_FAIL(getFp32ScaleData(origScaleBlob, trnsFp32Scale));
    ASSERT(trnsFp32Scale.size() == (NvU32)origScaleBlob.count);

    trnsScaleData =
        reinterpret_cast<NvF32*>(
            engine_ast::MemoryCollector::getInstance()->allocateMemory(origScaleBlob.count * sizeof(NvF32))
        );

    // Rescale based on filter values. Considering it to NCHW format.
    for (NvS32 cc = 0; cc < origScaleDims.c; cc++)
    {
        NvF32 perChannelScale = filterScales.at(cc);
        for (NvS32 hh = 0; hh < origScaleDims.h; hh++)
        {
            for (NvS32 ww = 0; ww < origScaleDims.w; ww++)
            {
                NvU32 offset =  ww +
                                origScaleDims.w * ( hh +
                                origScaleDims.h * (cc) );
                trnsScaleData[offset] =
                    (((trnsFp32Scale.at(offset) * perChannelScale) * perTensorInTensorScale) / perTensorOutTensorScale);
                trnsCnt++;
            }
        }
    }

    trnsScaleBlob.type = nvdla::DataType::FLOAT;
    trnsScaleBlob.count = trnsCnt;
    trnsScaleBlob.values = trnsScaleData;

    params().setRawScaleData(trnsScaleBlob);

fail:
    return e;
}

/**
 * Converts raw scale data to NvF32 and rescales.
 * Rescaling applicable if the following conditions statisfied.
 *  [1] If there exists a fused convolution node with SDP scale
 *  [2] Quantization mode: PER_KERNEL operating at INT8 precision
 *
 * Usecase
 * -------
 *         B = A * W op C
 * S_B * Q_B = (S_A * Q_A) * (S_W[1] * Q_W) op C
 *       Q_B = (Q_A * Q_W) * (((S_A * S_W[1]) op C) /S_B)
 *       Q_B = (Q_A * Q_W) * (2 ^ -m * n_i16[k'])
 *
 * Where Q_A, quantized int8 input to conv
 *       Q_B, rescaled int8 output from SDP
 *       C,   actual scale data - PER_LAYER, PER_CHANNEL, PER_ELEMENT (k')
 *
 * rescaleScaleDataForPerKernel computes ((S_A * S_W[1]) op C)/S_B in fp32 precision
 * and overwrites existing raw scale data with it.
 **/
NvDlaError engine_ast::SDPScaleOpNode::rescaleScaleDataForPerKernel()
{
    NvDlaError e = NvDlaSuccess;

    ConvCoreNode* fusedConv;

    std::vector<NvF32> filterScales;
    std::vector<NvF32> inTensorScales;
    std::vector<NvF32> outTensorScales;
    NvF32 perKernelScale;
    NvF32 perTensorInTensorScale;
    NvF32 perTensorOutTensorScale;

    Weights origScaleBlob;
    Dims4 origScaleDims;

    std::vector<NvF32> trnsFp32Scale;

    Weights trnsScaleBlob = Weights(nvdla::DataType::FLOAT, NULL, 0);
    NvF32 *trnsScaleData = NULL;
    NvU32 trnsCnt = 0;

    fusedConv = NodeFactory::nodeCast<ConvCoreNode*>(dependencyParams(0).fusedNode(IODirectionEnum::INPUT));
    if (fusedConv == NULL)
    {
        // Rescale only if fused conv available.
        goto fail;
    }

    filterScales = fusedConv->params().filterScales();
    perKernelScale = filterScales.at(0);
    inTensorScales = fusedConv->inputEdges().at(0)->originalTensor()->getChannelScales();
    perTensorInTensorScale = inTensorScales.at(0);
    outTensorScales = outputEdges().at(0)->originalTensor()->getChannelScales();
    perTensorOutTensorScale = outTensorScales.at(0);

    origScaleBlob = params().rawScaleData();
    origScaleDims = params().scaleDims();

    // Preliminary checks
    ASSERT(filterScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvU32 ff = 1; ff < filterScales.size(); ++ff)
    {
        if ( perKernelScale != filterScales[ff] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Filter scales should be same for %s when PER_KERNEL "
                                    "quantization is ON", fusedConv->name().c_str());
        }
    }

    ASSERT(inTensorScales.size() == (size_t)fusedConv->inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvF32 its = 1; its < inTensorScales.size(); ++its)
    {
        if ( perTensorInTensorScale != inTensorScales[its] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", fusedConv->name().c_str());
        }
    }

    ASSERT(outTensorScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvF32 ots = 1; ots < outTensorScales.size(); ++ots)
    {
        if ( perTensorOutTensorScale != outTensorScales[ots] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for output of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
        }
    }

    // Convert scale data into float and get it as vector
    trnsFp32Scale.clear();
    PROPAGATE_ERROR_FAIL(getFp32ScaleData(origScaleBlob, trnsFp32Scale));
    ASSERT(trnsFp32Scale.size() == (NvU32)origScaleBlob.count);

    trnsScaleData =
        reinterpret_cast<NvF32*>(
            engine_ast::MemoryCollector::getInstance()->allocateMemory(origScaleBlob.count * sizeof(NvF32))
        );

    // Rescale based on filter values. Considering it to NCHW format.
    for (NvS32 cc = 0; cc < origScaleDims.c; cc++)
    {
        for (NvS32 hh = 0; hh < origScaleDims.h; hh++)
        {
            for (NvS32 ww = 0; ww < origScaleDims.w; ww++)
            {
                NvU32 offset =  ww +
                                origScaleDims.w * ( hh +
                                origScaleDims.h * (cc) );
                trnsScaleData[offset] =
                    (((trnsFp32Scale.at(offset) * perKernelScale) * perTensorInTensorScale) / perTensorOutTensorScale);
                trnsCnt++;
            }
        }
    }

    trnsScaleBlob.type = nvdla::DataType::FLOAT;
    trnsScaleBlob.count = trnsCnt;
    trnsScaleBlob.values = trnsScaleData;

    params().setRawScaleData(trnsScaleBlob);

fail:
    return e;
}

/**
 * Converts raw scale data to NvF32 and rescales.
 * Rescaling applicable when sdp exists as standalone.
 *
 * Usecase
 * -------
 *         B = A op C
 * S_B * Q_B = (S_A * Q_A) op C
 *       Q_B = (Q_A * Q_W) * ((S_A op C) /S_B)
 *       Q_B = (Q_A * Q_W) * (2 ^ -m * n_i16[k'])
 *
 * Where Q_A, quantized int8 input to SDP
 *       Q_B, rescaled int8 output from SDP
 *       C,   actual scale data - PER_LAYER, PER_CHANNEL, PER_ELEMENT
 *
 * rescaleScaleDataForNoFusedConv computes (S_A op C)/S_B in fp32 precision
 * and overwrites existing raw scale data with it.
 **/
NvDlaError engine_ast::SDPScaleOpNode::rescaleScaleDataForNoFusedConv()
{
    NvDlaError e = NvDlaSuccess;

    std::vector<NvF32> trnsFp32Scale;

    std::vector<NvF32> inTensorScales = inputEdges().at(0)->originalTensor()->getChannelScales();
    std::vector<NvF32> outTensorScales = outputEdges().at(0)->originalTensor()->getChannelScales();
    NvF32 perTensorInTensorScale = inTensorScales.at(0);
    NvF32 perTensorOutTensorScale = outTensorScales.at(0);

    Weights origScaleBlob = params().rawScaleData();
    Dims4 origScaleDims = params().scaleDims();

    Weights trnsScaleBlob = Weights(nvdla::DataType::FLOAT, NULL, 0);
    NvF32 *trnsScaleData = NULL;
    NvU32 trnsCnt = 0;

    // Preliminary checks
    ASSERT(inTensorScales.size() == (size_t)inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvF32 its = 1; its < inTensorScales.size(); ++its)
    {
        if ( perTensorInTensorScale != inTensorScales[its] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
        }
    }

    ASSERT(outTensorScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c);
    for (NvF32 ots = 1; ots < outTensorScales.size(); ++ots)
    {
        if ( perTensorOutTensorScale != outTensorScales[ots] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for output of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
        }
    }

    // Convert scale data into float and get it as vector
    trnsFp32Scale.clear();
    PROPAGATE_ERROR_FAIL(getFp32ScaleData(origScaleBlob, trnsFp32Scale));
    ASSERT(trnsFp32Scale.size() == (NvU32)origScaleBlob.count);

    trnsScaleData =
        reinterpret_cast<NvF32*>(
            engine_ast::MemoryCollector::getInstance()->allocateMemory(origScaleBlob.count * sizeof(NvF32))
        );

    for (NvS32 cc = 0; cc < origScaleDims.c; cc++)
    {
        for (NvS32 hh = 0; hh < origScaleDims.h; hh++)
        {
            for (NvS32 ww = 0; ww < origScaleDims.w; ww++)
            {
                NvU32 offset =  ww +
                                origScaleDims.w * ( hh +
                                origScaleDims.h * (cc) );
                trnsScaleData[offset] = ((trnsFp32Scale.at(offset) * perTensorInTensorScale) / perTensorOutTensorScale);
                trnsCnt++;
                if (graph()->debugQuantization())
                {
                    gLogInfo << name() << " rawScl * Si / So: " << trnsFp32Scale.at(offset)
                             << " * " << perTensorInTensorScale << " / "
                             << perTensorOutTensorScale << " = " << trnsScaleData[offset] << endl;
                }
            }
        }
    }

    trnsScaleBlob.type = nvdla::DataType::FLOAT;
    trnsScaleBlob.count = trnsCnt;
    trnsScaleBlob.values = trnsScaleData;

    params().setRawScaleData(trnsScaleBlob);

fail:
    return e;
}

static bool absCompare(NvF32 a, NvF32 b)
{
    return (std::fabs(a) < std::fabs(b));
}

/**
 * Converts scale data -> SDP Mul(Int16) + truncate.
 **/
NvDlaError engine_ast::SDPScaleOpNode::scaleDataToInt16()
{
    NvDlaError e = NvDlaSuccess;

    Weights origScaleBlob;

    NvF32 maxScaleValue;
    std::pair< NvS16, NvU8 > maxScaleAndShift;
    std::vector<std::pair< NvS16, NvU8 >> scaleAndShift;

    std::vector<NvF32> trnsFp32Scale;
    NvS16 *trnsScaleI16Data;
    Weights trnsScaleI16Blob;

    origScaleBlob = params().rawScaleData();

    // Convert scale data into float and get it as vector
    PROPAGATE_ERROR_FAIL(getFp32ScaleData(origScaleBlob, trnsFp32Scale));
    ASSERT(trnsFp32Scale.size() == (NvU32)origScaleBlob.count);

    // Compute m and n where each transformed scale value can be represented as s = 2^-m . n
    maxScaleValue = *std::max_element(trnsFp32Scale.begin(), trnsFp32Scale.end(), absCompare);
    e = calculateScaleAndShiftFromScalar<NvS16, NvU8>(maxScaleValue, &maxScaleAndShift);
    if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, " Couldn't converge on `2^(x) * y` which could "
                                "safely represent %f within acceptable tolerance for %s\n",
                                maxScaleValue, name().c_str());
    }
    e = factorizeScalars<NvS16, NvU8>(trnsFp32Scale, &scaleAndShift, maxScaleAndShift.second);
    if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't factorize scalars for %s for int8 into scale+truncate pairs",
                                               name().c_str());
    }

    // reset raw scaling data and truncate values
    trnsScaleI16Data =
        reinterpret_cast<NvS16*>(engine_ast::MemoryCollector::getInstance()->allocateMemory(origScaleBlob.count * sizeof(NvS16)));
    for (NvU32 ii = 0; ii < origScaleBlob.count; ii++)
    {
        trnsScaleI16Data[ii] = scaleAndShift.at(ii).first;
        if (graph()->debugQuantization())
        {
            gLogInfo << name() << " i16Scale: " << trnsFp32Scale.at(ii) << " = " << scaleAndShift.at(ii).first
                     << " * 2^- " << (int)maxScaleAndShift.second << endl;
        }
    }

    trnsScaleI16Blob = Weights(nvdla::DataType::INT16, trnsScaleI16Data, origScaleBlob.count);

    params().setRawScaleData(trnsScaleI16Blob);
    params().x1Params().setTruncate(maxScaleAndShift.second);
fail:
    return e;
}

NvDlaError engine_ast::SDPScaleOpNode::handleLowPrecisionConversions()
{
    NvDlaError e = NvDlaSuccess;
    ConvCoreNode* fusedConv;

    if (graph()->profile()->computePrecision().v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        // Handles only for low precision INT8
        goto fail;
    }

    /**
     * Rescales scale data (in fp32 precision) based on different use cases.
     * Three usecases may arise,
     *  [1] With fusedConv (PER_KERNEL quantization)+ sdp scale. This use case arises because
     *          (a) No profile other than fast-math supports mathematical fusion.
     *          (b) If not other sdp node immediately follows SDPScaleOpNode, no math fusion.
     *  [2] With fusedConv (PER_FILTER quantiation) + sdp scale. Similar to [1]
     *  [2] SDP scale without conv.
     **/
    fusedConv = NodeFactory::nodeCast<ConvCoreNode*>(dependencyParams(0).fusedNode(IODirectionEnum::INPUT));
    if (fusedConv != NULL)
    {
        if (graph()->profile()->quantizationMode().v() == nvdla::QuantizationMode::PER_KERNEL)
        {
            PROPAGATE_ERROR_FAIL(rescaleScaleDataForPerKernel());
        }
        else if (graph()->profile()->quantizationMode().v() == nvdla::QuantizationMode::PER_FILTER)
        {
            PROPAGATE_ERROR_FAIL(rescaleScaleDataForPerFilter());
        }
    }
    else
    {
        PROPAGATE_ERROR_FAIL(rescaleScaleDataForNoFusedConv());
    }

    /**
     * converts rescaled scale data to int16 precision (S_fp = 2^-m . n_i16)
     **/
    if (auxEdges()[0]->tensorSurfaceDesc()->surfaceFormat().f().precision() ==
        surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16)
    {
        PROPAGATE_ERROR_FAIL(scaleDataToInt16());
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported,
            "Unsupported handling of scale data with %s precision",
            auxEdges()[0]->tensorSurfaceDesc()->surfaceFormat().f().precision().c_str());
    }

fail:
    return e;
}

std::vector<surface::SurfaceFormat> engine_ast::SDPScaleOpNode::suggestAuxSurfaceFormats(engine_ast::Edge* xEdge)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    std::vector<surface::SurfaceFormat> supportedAuxSFs = supportedAuxSurfFormats();
    std::vector<surface::SurfaceFormat> suggestedAuxSFs;
    std::vector<surface::SurfaceFormat>::iterator auxSFItr;
    surface::SurfacePrecision compPrec = graph()->profile()->computePrecision();

    if (supportedAuxSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported aux surface formats for %s", name().c_str());
    }

    switch(compPrec.v())
    {
        case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16: {
            surface::IsSurfacePrecisionDifferent desiredSP(surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16);
            supportedAuxSFs.erase(std::remove_if(supportedAuxSFs.begin(), supportedAuxSFs.end(), desiredSP), supportedAuxSFs.end());
        } break;
        case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:
        case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8: {
            surface::IsSurfacePrecisionDifferent desiredSP(surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16);
            supportedAuxSFs.erase(std::remove_if(supportedAuxSFs.begin(), supportedAuxSFs.end(), desiredSP), supportedAuxSFs.end());
        } break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support compute precision: %s for %s\n",
                                    compPrec.c_str(), name().c_str());
    }

    suggestedAuxSFs = supportedAuxSFs;

fail:
    return suggestedAuxSFs;
}

/* Configure SDP SuperOp SubEngine with Scale Op */
NvDlaError engine_ast::SDPScaleOpNode::configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN)
{
    NvDlaError e = NvDlaSuccess;

    if (xN == SDP_ENGINE_X1)
    {
        sdpSuperOp->params().setX1Params(params().x1Params());
        sdpSuperOp->params().setConvMode(params().convMode());
        sdpSuperOp->params().setWinogradParams(params().winogradParams());
    }
    else
    {
        sdpSuperOp->params().setX2Params(params().x1Params());
    }

    sdpSuperOp->params().setAuxDataType(xN, TensorType::kSCALE);

    sdpSuperOp->params().setMultiplierDims(xN, params().scaleDims());
    sdpSuperOp->params().setDLADataDims(xN, params().scaleDims());

    sdpSuperOp->params().setRawMultiplierData(xN, params().rawScaleData());
    sdpSuperOp->params().setDLAData(xN, params().DLAScaleData());

    sdpSuperOp->params().setAuxSurfaceFormats(xN, suggestAuxSurfaceFormats());

    if ( graph()->debugFuseSubEngineOps() )
    {
        gLogInfo << "configureSDPSuperOpSubEngine: " << this->name() << " in ";
        gLogInfo << sdpSuperOp->name() << " x" << (NvU16)xN.e()+1 << endl;
    }

    return e;
}

/********************************Aux Data Translation***************************/
NvDlaError engine_ast::SDPScaleOpNode::translateAuxData()
{
    NvDlaError e = NvDlaSuccess;

    engine_ast::Edge* auxEdge;
    Weights trnsSclData;
    Weights rawSclData = params().rawScaleData();
    surface::SurfacePrecision computePrecision;
    surface::SurfacePrecision srcPrecision;
    NvU32 channelsPerGroup = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    auxEdge = auxEdges()[0];
    computePrecision = auxEdge->tensorSurfaceDesc()->surfaceFormat().f().precision();
    srcPrecision = inputEdges()[0]->tensorSurfaceDesc()->surfaceFormat().f().precision();

    if ( graph()->debugWeights() )
    {
        gLogInfo << "translating weights for " << name() << "scale-dims = " <<
                auxEdge->tensorSurfaceDesc()->dimensions().n << "," <<
                auxEdge->tensorSurfaceDesc()->dimensions().c << "," <<
                auxEdge->tensorSurfaceDesc()->dimensions().h << "," <<
                auxEdge->tensorSurfaceDesc()->dimensions().w << "," <<
                "and size= " << rawSclData.count << endl;
    }

    if (srcPrecision == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        channelsPerGroup = graph()->target_config()->atomicKSize();
    }

    {
        WeightTrns::WeightDims sclDims (rawSclData.count,
                                        auxEdge->tensorSurfaceDesc()->dimensions().n,
                                        auxEdge->tensorSurfaceDesc()->dimensions().c,
                                        auxEdge->tensorSurfaceDesc()->dimensions().w,
                                        auxEdge->tensorSurfaceDesc()->dimensions().h,
                                        1,   //strides dont matter for Scale
                                        1);

        PRECISION_SWITCH(rawSclData.type.v(), computePrecision.v(), trnsSclData, WeightTrns::translateDataForScale,
                                                                                 params().x1Params().mode(),
                                                                                 sclDims,
                                                                                 rawSclData,
                                                                                 channelsPerGroup);

        if (trnsSclData.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Scale data trnaslation failed for node '%s'", name().c_str());
        }

        params().setDLAScaleData(trnsSclData);
    }

fail:
    return e;
}
NvDlaError engine_ast::SDPScaleOpNode::emitOp(Graph *g,
                                           DLAInterface *target_dla,
                                           NvU32 op_slot, NvU32 batch_id,
                                           DLACommonOpDescAccessor       dep,
                                           DLAOperationContainerAccessor op,
                                           DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;
    DLASDPOpDescAccessor  sdp_op = op.sdpOpDescAccessor(0);
    DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
    DLASDPOpAccessor x1_op_acc = sdp_op.x1OpAccessor();
    DLASDPOpAccessor x2_op_acc = sdp_op.x2OpAccessor();
    DLASDPOpAccessor y_op_acc  = sdp_op.yOpAccessor();
    DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
    DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
    DLADataCubeAccessor y_data_acc     = surf_acc.yDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(x2_data_acc);
    NVDLA_UNUSED(y_data_acc);
    NVDLA_UNUSED(fused_acc);

    surface::TensorSurfaceDesc *src_tsd  = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd  = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *scale_tsd = g->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    *sdp_op.srcPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, src_tsd->surfaceFormat().precision());
    *sdp_op.dstPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, dst_tsd->surfaceFormat().precision());
    *sdp_op.LUTIndex()       = -1;
    *sdp_op.batchNum()       = 1;
    *sdp_op.batchStride()    = 0;

    *out_cvt_acc.scale()     = 1;
    *out_cvt_acc.truncate()  = 0;
    *out_cvt_acc.offset()    = 0;
    *out_cvt_acc.enable()    = 1;

    *x1_op_acc.enable()      = 1;
    *x1_op_acc.ALUType()     = x1_op_acc.ALUType_Sum();
    *x1_op_acc.type()        = x1_op_acc.type_Mul();
    *x1_op_acc.mode()        = ASTToDLAInterface::getSDPMode(target_dla,
                                                    params(batch_id).x1Params().mode());
    *x1_op_acc.act()         = ASTToDLAInterface::getSDPActType(target_dla, params(batch_id).x1Params().actType());
    *x1_op_acc.shiftValue()  = 0;
    *x1_op_acc.ALUOperand()  = 0;

    if (params(batch_id).x1Params().mode().e() ==
            engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        Weights scaleData = params().DLAScaleData();

        if (scaleData.count > 1)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                "More than one data available"
                                "in per-layer mode(#data = %u)",
                                scaleData.count);
        }
        switch (scaleData.type)
        {
            case DataType::HALF:
            case DataType::INT16:
                *x1_op_acc.MulOperand() =
                    *reinterpret_cast<const NvS16 *>(scaleData.values);
                break;
            case DataType::INT8:
                *x1_op_acc.MulOperand() =
                    *reinterpret_cast<const NvS8 *>(scaleData.values);
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                    "Unexpected data type %s",
                                    scaleData.type.c_str());
        }
    }
    else
    {
        *x1_op_acc.MulOperand() = 1;
    }
    *x1_op_acc.truncate()    = params().x1Params().truncate();;
    *x1_op_acc.precision()   = ASTToDLAInterface::getSDPPrecision(target_dla, scale_tsd->surfaceFormat().precision());

    *x2_op_acc.enable() = 0;
    *y_op_acc.enable()  = 0;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    if (params(batch_id).x1Params().mode().e() !=
        engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER) {
        setDataCubeAccessor(x1_data_acc, scale_tsd, IODirectionEnum::UNKNOWN,
                            batch_id);
    }
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);

    if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD )
    {
        *sdp_op.convMode() = sdp_op.convMode_Winograd();

        *src_data_acc.width() = params().winogradParams().ioDims.w;
        *src_data_acc.height() = params().winogradParams().ioDims.h;

        *dst_data_acc.width() = params().winogradParams().ioDims.w;
        *dst_data_acc.height() = params().winogradParams().ioDims.h;
    }
    else if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_DIRECT )
    {
        *sdp_op.convMode() = sdp_op.convMode_Direct();
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Conv mode for %s", name().c_str());
    }


    if ( g->debugOps() )
    {
        gLogInfo << "SDP scale node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc precision " << (int)*sdp_op.srcPrecision() << endl;
        gLogInfo << "\tdst precision " << (int)*sdp_op.dstPrecision() << endl;
        gLogInfo << "\tx1 enable " << (int)*x1_op_acc.enable() << endl;
        if (*x1_op_acc.enable())
        {
            gLogInfo << "\tx1 precision " << (int)*x1_op_acc.precision() << endl;
            gLogInfo << "\tx1 aluType " << (int)*x1_op_acc.ALUType() << endl;
            gLogInfo << "\tx1 type " << (int)*x1_op_acc.type() << endl;
            gLogInfo << "\tx1 mode " << (int)*x1_op_acc.mode() << endl;
            gLogInfo << "\tx1 act " << (int)*x1_op_acc.act() << endl;
            gLogInfo << "\tx1 shiftValue " << (int)*x1_op_acc.shiftValue() << endl;
            gLogInfo << "\tx1 aluOperand " << (int)*x1_op_acc.ALUOperand() << endl;
            gLogInfo << "\tx1 mulOperand " << (int)*x1_op_acc.MulOperand() << endl;
            gLogInfo << "\tx1 truncate " << (int)*x1_op_acc.truncate() << endl;
        }
        gLogInfo << "\tx2 enable " << (int)*x2_op_acc.enable() << endl;
        if (*x2_op_acc.enable())
        {
            gLogInfo << "\tx2 precision " << (int)*x2_op_acc.precision() << endl;
            gLogInfo << "\tx2 aluType " << (int)*x2_op_acc.ALUType() << endl;
            gLogInfo << "\tx2 type " << (int)*x2_op_acc.type() << endl;
            gLogInfo << "\tx2 mode " << (int)*x2_op_acc.mode() << endl;
            gLogInfo << "\tx2 act " << (int)*x2_op_acc.act() << endl;
            gLogInfo << "\tx2 shiftValue " << (int)*x2_op_acc.shiftValue() << endl;
            gLogInfo << "\tx2 aluOperand " << (int)*x2_op_acc.ALUOperand() << endl;
            gLogInfo << "\tx2 mulOperand " << (int)*x2_op_acc.MulOperand() << endl;
            gLogInfo << "\tx2 truncate " << (int)*x2_op_acc.truncate() << endl;
        }
        gLogInfo << "\ty enable " << (int)*y_op_acc.enable() << endl;
        if (*y_op_acc.enable())
        {
            gLogInfo << "\ty precision " << (int)*y_op_acc.precision() << endl;
            gLogInfo << "\ty aluType " << (int)*y_op_acc.ALUType() << endl;
            gLogInfo << "\ty type " << (int)*y_op_acc.type() << endl;
            gLogInfo << "\ty mode " << (int)*y_op_acc.mode() << endl;
            gLogInfo << "\ty act " << (int)*y_op_acc.act() << endl;
            gLogInfo << "\ty shiftValue " << (int)*y_op_acc.shiftValue() << endl;
            gLogInfo << "\ty aluOperand " << (int)*y_op_acc.ALUOperand() << endl;
            gLogInfo << "\ty mulOperand " << (int)*y_op_acc.MulOperand() << endl;
            gLogInfo << "\ty truncate " << (int)*y_op_acc.truncate() << endl;
        }
        gLogInfo << "\tsrc tsd:" << src_tsd->id() << endl;
        gLogInfo << "\tdst tsd:" << dst_tsd->id() << "/" << dst_tsd->tensorBufferDesc()->id()
                                                         << ":off= " << dst_tsd->bufferOffset() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tconv_mode " << (int)*sdp_op.convMode() << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.address() << "[" << *src_data_acc.offset() << "]" << endl;
        gLogInfo << "\tsrc type=" << (int)*src_data_acc.type() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tscale addr=" << *x1_data_acc.address() << "[" << *x1_data_acc.offset() << "]" << endl;
        gLogInfo << "\tscale type=" << (int)*x1_data_acc.type() << endl;
        gLogInfo << "\tscale size " << *x1_data_acc.size()    << endl;
        gLogInfo << "\tscale width " << *x1_data_acc.width()   << endl;
        gLogInfo << "\tscale height " << *x1_data_acc.height()   << endl;
        gLogInfo << "\tscale channel " << *x1_data_acc.channel()  << endl;
        gLogInfo << "\tscale linestride " << *x1_data_acc.lineStride() << endl;
        gLogInfo << "\tscale surfstride " << *x1_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << "[" << *dst_data_acc.offset() << "]" << endl;
        gLogInfo << "\tdst type=" << (int)*dst_data_acc.type() << endl;
        gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
        gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
        gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
        gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
    }

fail:
    return e;
}

void engine_ast::CPUScaleOpNode::captureCanonicalParams()
{
    params().setPower(canonicalNode()->params().power());
    params().setScale(canonicalNode()->params().scale());
    params().setShift(canonicalNode()->params().shift());
}

NvDlaError engine_ast::CPUScaleOpNode::emitOp(Graph *g,
                                           EMUInterface *emu_if,
                                           NvU32 op_slot, NvU32 batch_id,
                                           EMUOperationContainerAccessor op,
                                           EMUOperationBufferContainerAccessor buf)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 mem_atomic_size = graph()->target_config()->memoryAtomicSize();
    EMUPowerOpDescAccessor power_op = op.powerOpDescAccessor(0);
    EMUCommonOpDescAccessor power_op_common = power_op.commonOpDescAccessor();

    surface::TensorSurfaceDesc *src_tsd = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    *power_op_common.op_type() = 0; //<-- EMU_OP_POWER
    if (graph()->profile()->computePrecision().v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        *power_op_common.input_scale_factor()  = inputEdges().at(0)->originalTensor()->getChannelScales().at(0);
        *power_op_common.output_scale_factor() = outputEdges().at(0)->originalTensor()->getChannelScales().at(0);
    }
    else
    {
        *power_op_common.input_scale_factor()  = 1.0f;
        *power_op_common.output_scale_factor() = 1.0f;
    }

    *power_op.power() = *reinterpret_cast<NvF32*>(const_cast<void*>(this->params().power().values));
    *power_op.scale() = *reinterpret_cast<NvF32*>(const_cast<void*>(this->params().scale().values));
    *power_op.shift() = *reinterpret_cast<NvF32*>(const_cast<void*>(this->params().shift().values));
    *power_op.scale() = 1.0f;       //fixme: hack for current mnist fp16 test

    EMUPowerBufferDescsAccessor power_buffer = buf.powerBufferDescsAccessor(0);
    EMUBufferDescAccessor src_data_acc = power_buffer.srcDataAccessor();
    EMUBufferDescAccessor dst_data_acc = power_buffer.dstDataAccessor();

    NvS16 src_id, dst_id;

    src_id = src_tsd->addressId(batch_id);
    dst_id = dst_tsd->addressId(batch_id);

    *src_data_acc.addressIndex() = src_id;
    *src_data_acc.addressIndexOffset() = src_tsd->addressIdOffset(batch_id);
    *src_data_acc.size()       = (NvU32)src_tsd->size();    //fixme: 64b -> 32b
    *src_data_acc.format()     = ASTToEMUInterface::getDataFormat(emu_if, src_tsd->surfaceFormat(), mem_atomic_size);
    *src_data_acc.width()      = (NvU16)src_tsd->dimensions().w; //fixme: 32b -> 16b
    *src_data_acc.height()     = (NvU16)src_tsd->dimensions().h; //fixme: 32b -> 16b
    *src_data_acc.channel()    = (NvU16)src_tsd->dimensions().c; //fixme: 32b -> 16b
    if ( src_tsd->bindable() ) {
        NvS16 addrId = src_tsd->addressId(batch_id);

        uintptr_t lineOffs = uintptr_t(src_data_acc.lineStride());// - uintptr_t(power_buffer.struct_base());
        uintptr_t surfOffs = uintptr_t(src_data_acc.surfStride());// - uintptr_t(power_buffer.struct_base());
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, lineOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Line));
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, surfOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Surf));
    }
    *src_data_acc.lineStride() = src_tsd->lineStride();
    *src_data_acc.surfStride() = src_tsd->surfaceStride();

    *dst_data_acc.addressIndex() = dst_id;
    *dst_data_acc.addressIndexOffset() = dst_tsd->addressIdOffset(batch_id);
    *dst_data_acc.size()       = (NvU32)dst_tsd->size();    //fixme: 64b -> 32b
    *dst_data_acc.format()     = ASTToEMUInterface::getDataFormat(emu_if, dst_tsd->surfaceFormat(), mem_atomic_size);
    *dst_data_acc.width()      = (NvU16)dst_tsd->dimensions().w; //fixme: 32b -> 16b
    *dst_data_acc.height()     = (NvU16)dst_tsd->dimensions().h; //fixme: 32b -> 16b
    *dst_data_acc.channel()    = (NvU16)dst_tsd->dimensions().c; //fixme: 32b -> 16b
    if ( dst_tsd->bindable() ) {
        NvS16 addrId = dst_tsd->addressId(batch_id);
        uintptr_t lineOffs = uintptr_t(dst_data_acc.lineStride());// - uintptr_t(power_buffer.struct_base());
        uintptr_t surfOffs = uintptr_t(dst_data_acc.surfStride());// - uintptr_t(power_buffer.struct_base());
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, lineOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Line));
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, surfOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Surf));
    }
    *dst_data_acc.lineStride() = dst_tsd->lineStride();
    *dst_data_acc.surfStride() = dst_tsd->surfaceStride();

    if ( g->debugOps() )
    {
        gLogInfo << "Power node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc tsd:batch= " << src_tsd->id() << " addr= " << *src_data_acc.addressIndex() << "[" << *src_data_acc.addressIndexOffset() << "]" << endl;
        gLogInfo << "\tdst tsd:batch= " << dst_tsd->id() << " addr= " << *dst_data_acc.addressIndex() << "[" << *dst_data_acc.addressIndexOffset() << "]" << endl;
        gLogInfo << "\tinput scale factor " << *power_op_common.input_scale_factor() << endl;
        gLogInfo << "\toutput scale factor " << *power_op_common.output_scale_factor() << endl;

        gLogInfo << "\tsrc size=" << *src_data_acc.size() << endl;
        gLogInfo << "\tsrc format=" << *src_data_acc.format() << endl;
        gLogInfo << "\tsrc width=" << *src_data_acc.width() << endl;
        gLogInfo << "\tsrc height=" << *src_data_acc.height() << endl;
        gLogInfo << "\tsrc channel=" << *src_data_acc.channel() << endl;

        gLogInfo << "\tdst size=" << *dst_data_acc.size() << endl;
        gLogInfo << "\tdst format=" << *dst_data_acc.format() << endl;
        gLogInfo << "\tdst width=" << *dst_data_acc.width() << endl;
        gLogInfo << "\tdst height=" << *dst_data_acc.height() << endl;
        gLogInfo << "\tdst channel=" << *dst_data_acc.channel() << endl;

    }

    return e;
}

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
static nvdla_prototest_interface::SDPOp_SDPOpMode opMode2InfOpMode(engine_ast::SDPMode om)
{
    nvdla_prototest_interface::SDPOp_SDPOpMode iom = nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL;
    switch(om.v())
    {
        case engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL: iom =
                nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL; break;
        case engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT: iom =
                nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_POINT; break;
        case engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER: iom =
                nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_LAYER; break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown op mode: %s", om.c_str());
    }

    return iom;
}

static nvdla_prototest_interface::SDPActivation actType2InfActType(engine_ast::SDPActType at)
{
    nvdla_prototest_interface::SDPActivation iat = nvdla_prototest_interface::SDPActivation::ACT_NONE;
    switch(at.v())
    {
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_NONE: iat =
                nvdla_prototest_interface::SDPActivation::ACT_NONE; break;
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_RELU: iat =
                nvdla_prototest_interface::SDPActivation::ACT_RELU; break;
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_TANH:
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_SIGMOID:iat =
                nvdla_prototest_interface::SDPActivation::ACT_LUT; break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown act type: %s", at.c_str());
    }

    return iat;
}

NvDlaError engine_ast::SDPScaleOpNode::emitOp(NvU32 op_slot, NvU32 batch_id,
                                           DLAInterface* target_dla,
                                           DLACommonOpDescAccessor&        dep,
                                           DLAOperationContainerAccessor&  op,
                                           DLASurfaceContainerAccessor&    surf,
                                           nvdla_prototest_interface::Layer* protoLayer)
{
    NvDlaError e = NvDlaSuccess;
    NvU8 numConsumers = 0;

    DLASDPOpDescAccessor  sdp_op    = op.sdpOpDescAccessor(0);
    DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
    DLASDPOpAccessor x1_op_acc      = sdp_op.x1OpAccessor();
    DLASDPOpAccessor x2_op_acc      = sdp_op.x2OpAccessor();
    DLASDPOpAccessor y_op_acc       = sdp_op.yOpAccessor();
    DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
    DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
    DLADataCubeAccessor y_data_acc      = surf_acc.yDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(out_cvt_acc);
    NVDLA_UNUSED(x2_data_acc);
    NVDLA_UNUSED(y_data_acc);
    NVDLA_UNUSED(batch_id);

    surface::TensorSurfaceDesc *src_tsd     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *scale_tsd    = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());


    nvdla_prototest_interface::SDPOpDesc* protoSDPOpDesc        = protoLayer->mutable_op_config()->mutable_sdp_op();
    nvdla_prototest_interface::SDPSurfaceDesc* protoSDPSurfDesc = protoLayer->mutable_surface()->mutable_sdp_surface();
    nvdla_prototest_interface::SDPOp*          protoSDPX1OpDesc = protoSDPOpDesc->mutable_x1_op();
    nvdla_prototest_interface::SDPOp*          protoSDPX2OpDesc = protoSDPOpDesc->mutable_x2_op();
    nvdla_prototest_interface::SDPOp*          protoSDPYOpDesc  = protoSDPOpDesc->mutable_y_op();
    nvdla_prototest_interface::DataCube* protoSrcDataCube       = protoSDPSurfDesc->mutable_src_data();
    nvdla_prototest_interface::DataCube* protoDstDataCube       = protoSDPSurfDesc->mutable_dst_data();
    nvdla_prototest_interface::DataCube* protoX1DataCube        = protoSDPSurfDesc->mutable_x1_data();
    nvdla_prototest_interface::DataPrecision protoSrcPrec, protoDstPrec;

    protoLayer->set_index(op_slot);
    protoLayer->set_roi_index(0);
    protoLayer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP);
    protoLayer->set_dependency_count(*dep.dependencyCount());

    /* consumers */
    for (size_t c = 0; c < EngineType::num_elements(); c++)
    {
        NvS8 fw_op_index = ASTToDLAInterface::getEngineType(target_dla, c);
        if ( fw_op_index < 0 )
        {
            continue;
        }

        DLAConsumerAccessor cons_acc = dep.consumerAccessor(fw_op_index);
        if (*cons_acc.index() != -1) {
            numConsumers++;
            nvdla_prototest_interface::Consumer* protoConsumer = protoLayer->add_bottom();
            protoConsumer->set_index(*cons_acc.index());
            switch(c) {
                case EngineTypeEnum::BDMA : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_BDMA); break;
                case EngineTypeEnum::CONVOLUTION : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
                case EngineTypeEnum::SDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP); break;
                case EngineTypeEnum::PDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_PDP); break;
                case EngineTypeEnum::CDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CDP); break;
                case EngineTypeEnum::RUBIK: protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_RUBIK); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer");
            }
            switch(dependencyParams().consumer(c).opEvent().v()) {
                case OperationEventTypeEnum::OP_CDMA_WEIGHT_DONE : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_CDMA_WT_DONE); break;
                case OperationEventTypeEnum::OP_CDMA_DATA_DONE   : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_CDMA_DT_DONE); break;
                case OperationEventTypeEnum::OP_COMPLETED        : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_COMPLETED); break;
                case OperationEventTypeEnum::OP_ENABLED          : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_ENABLED); break;
                case OperationEventTypeEnum::OP_PROGRAMMED       : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_PROGRAMMED); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer event");
            }
        }
    }

    /* fused node */
    if (dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT))
    {
        nvdla_prototest_interface::Consumer* protoFusedConsumer = protoLayer->mutable_fused();
        protoFusedConsumer->set_index(*fused_acc.index());
        protoFusedConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_ENABLED);
        switch(dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT)->engineType().v()) {
            case EngineTypeEnum::CONVOLUTION: protoFusedConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
            default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "SDP can have only Conv op as its fused partner on input side");
        }
    }

    switch(src_tsd->surfaceFormat().precision().v()) {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : protoSrcPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : protoSrcPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : protoSrcPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized src precision");
    }

    switch(dst_tsd->surfaceFormat().precision().v()) {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : protoDstPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : protoDstPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : protoDstPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized dst precision");
    }

    protoSDPOpDesc->set_src_precision(protoSrcPrec);
    protoSDPOpDesc->set_dst_precision(protoDstPrec);
    protoSDPOpDesc->set_lut_index(-1);

    protoSDPOpDesc->mutable_out_cvt()->set_enable(1);
    protoSDPOpDesc->mutable_out_cvt()->set_offset(0);
    protoSDPOpDesc->mutable_out_cvt()->set_scale(1);
    protoSDPOpDesc->mutable_out_cvt()->set_truncate(0);

    protoSDPOpDesc->set_conv_mode(nvdla_prototest_interface::ConvMode::DIRECT);
    protoSDPOpDesc->set_batch_num(1);
    protoSDPOpDesc->set_batch_stride(0);

    protoSDPX1OpDesc->set_enable(*x1_op_acc.enable());
    protoSDPX1OpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPX1OpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_MUL);
    protoSDPX1OpDesc->set_mode(opMode2InfOpMode(params().x1Params().mode()));
    protoSDPX1OpDesc->set_act(actType2InfActType(params().x1Params().actType()));
    protoSDPX1OpDesc->set_shift_value(*x1_op_acc.shiftValue());
    protoSDPX1OpDesc->set_alu_operand(*x1_op_acc.ALUOperand());
    protoSDPX1OpDesc->set_mul_operand(*x1_op_acc.MulOperand());
    protoSDPX1OpDesc->set_truncate(*x1_op_acc.truncate());
    protoSDPX1OpDesc->set_precision(protoSrcPrec);
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*x1_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*x1_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*x1_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*x1_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*x1_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*x1_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*x1_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*x1_op_acc.cvt().mulCVTAccessor().offset());

    protoSDPX2OpDesc->set_enable(*x2_op_acc.enable());
    protoSDPX2OpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPX2OpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
    protoSDPX2OpDesc->set_mode(nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL);
    protoSDPX2OpDesc->set_act(nvdla_prototest_interface::SDPActivation::ACT_NONE);
    protoSDPX2OpDesc->set_shift_value(*x2_op_acc.shiftValue());
    protoSDPX2OpDesc->set_alu_operand(*x2_op_acc.ALUOperand());
    protoSDPX2OpDesc->set_mul_operand(*x2_op_acc.MulOperand());
    protoSDPX2OpDesc->set_truncate(*x2_op_acc.truncate());
    protoSDPX2OpDesc->set_precision(protoSrcPrec);
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*x2_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*x2_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*x2_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*x2_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*x2_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*x2_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*x2_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*x2_op_acc.cvt().mulCVTAccessor().offset());


    protoSDPYOpDesc->set_enable(*y_op_acc.enable());
    protoSDPYOpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPYOpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
    protoSDPYOpDesc->set_mode(nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL);
    protoSDPYOpDesc->set_act(nvdla_prototest_interface::SDPActivation::ACT_NONE);
    protoSDPYOpDesc->set_shift_value(*y_op_acc.shiftValue());
    protoSDPYOpDesc->set_alu_operand(*y_op_acc.ALUOperand());
    protoSDPYOpDesc->set_mul_operand(*y_op_acc.MulOperand());
    protoSDPYOpDesc->set_truncate(*y_op_acc.truncate());
    protoSDPYOpDesc->set_precision(protoSrcPrec);
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*y_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*y_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*y_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*y_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*y_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*y_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*y_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*y_op_acc.cvt().mulCVTAccessor().offset());


    if (*fused_acc.index() != -1) {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_HW);
    } else {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    }
    protoSrcDataCube->set_address(*src_data_acc.address());
    protoSrcDataCube->set_size(src_tsd->tensorBufferDesc()->size() - src_tsd->bufferOffset());
    protoSrcDataCube->set_width(*src_data_acc.width());
    protoSrcDataCube->set_height(*src_data_acc.height());
    protoSrcDataCube->set_channel(*src_data_acc.channel());
    protoSrcDataCube->set_line_stride(*src_data_acc.lineStride());
    protoSrcDataCube->set_surf_stride(*src_data_acc.surfStride());
    protoSrcDataCube->set_plane_stride(*src_data_acc.planeStride());
    protoSrcDataCube->mutable_mem_info()->set_mem_id(src_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoSrcDataCube->mutable_mem_info()->set_mem_size(src_tsd->tensorBufferDesc()->size());
    protoSrcDataCube->mutable_mem_info()->set_offset(src_tsd->bufferOffset());

    protoDstDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoDstDataCube->set_address(*dst_data_acc.address());
    protoDstDataCube->set_size(dst_tsd->tensorBufferDesc()->size() - dst_tsd->bufferOffset());
    protoDstDataCube->set_width(*dst_data_acc.width());
    protoDstDataCube->set_height(*dst_data_acc.height());
    protoDstDataCube->set_channel(*dst_data_acc.channel());
    protoDstDataCube->set_line_stride(*dst_data_acc.lineStride());
    protoDstDataCube->set_surf_stride(*dst_data_acc.surfStride());
    protoDstDataCube->set_plane_stride(*dst_data_acc.planeStride());
    protoDstDataCube->mutable_mem_info()->set_mem_id(dst_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoDstDataCube->mutable_mem_info()->set_mem_size(dst_tsd->tensorBufferDesc()->size());
    protoDstDataCube->mutable_mem_info()->set_offset(dst_tsd->bufferOffset());
    if (numConsumers == 0) {
        protoDstDataCube->mutable_mem_info()->set_fill_type(nvdla_prototest_interface::FillerType::FILL_NONE);
        protoDstDataCube->mutable_mem_info()->set_flag(nvdla_prototest_interface::MemFlag::DLA_MEM_OUTPUT);
        protoDstDataCube->mutable_mem_info()->set_precision(protoDstPrec);
    }

    protoX1DataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoX1DataCube->set_address(*x1_data_acc.address());
    protoX1DataCube->set_size(scale_tsd->tensorBufferDesc()->size());
    protoX1DataCube->set_width(*x1_data_acc.width());
    protoX1DataCube->set_height(*x1_data_acc.height());
    protoX1DataCube->set_channel(*x1_data_acc.channel());
    protoX1DataCube->set_line_stride(*x1_data_acc.lineStride());
    protoX1DataCube->set_surf_stride(*x1_data_acc.surfStride());
    protoX1DataCube->set_plane_stride(*x1_data_acc.planeStride());
    protoX1DataCube->mutable_mem_info()->set_mem_id(scale_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoX1DataCube->mutable_mem_info()->set_mem_size(scale_tsd->tensorBufferDesc()->size());
    protoX1DataCube->mutable_mem_info()->set_offset(scale_tsd->bufferOffset());
fail:
    return e;
}
#endif


};  // nvdla::priv::
};  // nvdla::
