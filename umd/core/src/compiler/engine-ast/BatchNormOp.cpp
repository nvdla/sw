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

#include "priv/Type.h"
#include "priv/EngineAST.h"
#include "priv/LowPrecision.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "priv/Compiler.h"
#include "priv/WeightTranslationUnit.h"

#include "ErrorMacros.h"

using std::endl;
using std::max;

namespace nvdla
{
namespace priv
{

NvDlaError engine_ast::SDPBatchNormOpNode::captureCanonicalBatchNormData()
{
    NvDlaError e = NvDlaError_Success;
    Dims4             bnDims;
    Tensor*           bnDataTensor = NULL;
    //    engine_ast::Edge* bnDataEdge   = NULL;

    if (params().meanDims() != params().varianceDims())
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Mismatching dims for mean and variance");
    }

    bnDims = params().batchNormDims();

    bnDataTensor = graph()->addAuxTensor(graph()->newAuxTensorName(), bnDims, TensorType::kBATCH_NORM);

    /*bnDataEdge = */graph()->addDataEdge((canonical_ast::Edge*)0, (engine_ast::Node*)0, this, bnDataTensor);

fail:
    return e;
}

void engine_ast::SDPBatchNormOpNode::captureCanonicalParams()
{
    NvDlaError e = NvDlaSuccess;

    params().x1Params().setEnabled(true);
    switch(canonicalNode()->params().mode())
    {
        case BatchNormMode::bnUNIFORM:   params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_LAYER); break;
        case BatchNormMode::bnm_CHANNEL: params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL); break;
        default: params().x1Params().setMode(SDPModeEnum::SDP_MODE_UNKNOWN);
    }
    params().x1Params().setAluType(SDPALUTypeEnum::SDP_ALU_TYPE_SUM);
    params().x1Params().setOpType(SDPOpTypeEnum::SDP_OP_TYPE_BOTH);
    params().setEpsilon(canonicalNode()->params().epsilon());
    params().setMeanDims(canonicalNode()->params().meanDims());
    params().setVarianceDims(canonicalNode()->params().varianceDims());
    if (params().meanDims() != params().varianceDims())
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Mean dims and Variance dims don't match for %s", name().c_str());
    }
    else
    {
        params().setBatchNormDims(params().meanDims());
    }
    params().setRawMeanData(canonicalNode()->params().mean());
    params().setRawVarianceData(canonicalNode()->params().variance());
    params().setDLABatchNormData(Weights(DataType::FLOAT, NULL, 0));
    PROPAGATE_ERROR_FAIL(captureCanonicalBatchNormData());

fail:
    return;
}

void engine_ast::SDPBatchNormOpNode::inheritParams(Node* otherNode)
{
    // inherit the parameters that make sense (keep adding later)
    SDPBatchNormOpNode* otherBN = NodeFactory::nodeCast<SDPBatchNormOpNode*>(otherNode);
    params().setX1Params(otherBN->params().x1Params());
    params().setEpsilon(otherBN->params().epsilon());
    params().setMeanDims(otherBN->params().meanDims());
    params().setVarianceDims(otherBN->params().varianceDims());
    params().setBatchNormDims(otherBN->params().batchNormDims());
    params().setRawMeanData(otherBN->params().rawMeanData());
    params().setRawVarianceData(otherBN->params().rawVarianceData());
    params().setDLABatchNormData(otherBN->params().DLABatchNormData());
    params().setOutCVT(otherBN->params().outCVT());

    SDPNode* asSdpNode = NodeFactory::nodeCast<SDPNode*>(this);
    SDPNode* otherBNAsSdpNode = NodeFactory::nodeCast<SDPNode*>(otherNode);
    asSdpNode->params().setConvMode(otherBNAsSdpNode->params().convMode());
    asSdpNode->params().setWinogradParams(otherBNAsSdpNode->params().winogradParams());
    asSdpNode->params().setNumGroups(otherBNAsSdpNode->params().numGroups());
}

std::vector<surface::SurfaceFormat> engine_ast::SDPBatchNormOpNode::suggestAuxSurfaceFormats(engine_ast::Edge* xEdge)
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

template<typename MP, typename CP>
NvU32 engine_ast::SDPBatchNormOpNode::factorsPerEntity
(
    MP rawValue,
    AuxDataType auxType,
    MP* factorizedValue
)
{
    NvU32 numFactors = 1;   // default

    MP processedValue  = rawValue;
    MP tempFactors     = rawValue;

    if (auxType == MEAN_DATA)
    {
        processedValue = (-1) * rawValue;
        //FIXME: add logic to handle over/under-shooting mean
    }
    else if (auxType == VARIANCE_DATA)
    {
        rawValue += params().epsilon();
        processedValue = MP(1/sqrtf(rawValue));
    }


    tempFactors = processedValue;

    // FIXME: determine strategy to find limits on negative side
    if (processedValue > std::numeric_limits<CP>::max())
    {
        NvF32 nthRoot       = 2.0;   // start with sqrt min

        do {
            tempFactors = powf(processedValue, 1/nthRoot);
            nthRoot++;
            numFactors++;
        } while(tempFactors > std::numeric_limits<CP>::max());

        if ( debugFactorization() )
        {
            gLogInfo << processedValue << " > " << float(std::numeric_limits<CP>::max())
                     << " Factorized to " << float(tempFactors) << " after taking " << (nthRoot -1) << "th root!! "
                     << " Updating " << rawValue << " to " << tempFactors << endl;
        }
    }

    if (factorizedValue != NULL)
    {
        *factorizedValue = tempFactors;
    }

    return numFactors;
}

template<typename MP, typename CP>
NvDlaError engine_ast::SDPBatchNormOpNode::maxFactorsPerEntity
(
    Weights& auxData,
    AuxDataType auxType,
    NvU32* numFactors
)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Edge* auxEdge;

    MP* auxBlob = reinterpret_cast<MP*>(const_cast<void*>(auxData.values));

    PROPAGATE_ERROR_FAIL(nodeAuxEdge(&auxEdge));

    if (params().x1Params().mode().v() == SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        *numFactors = factorsPerEntity<MP, CP>(auxBlob[0], auxType);
    }
    else if (params().x1Params().mode().v() == SDPModeEnum::SDP_MODE_PER_CHANNEL)
    {
        NvS32 chnls = auxEdge->tensorSurfaceDesc()->dimensions().c;
        for (int cc = 0; cc < chnls; ++cc)
        {
            *numFactors = max(factorsPerEntity<MP, CP>(auxBlob[cc], auxType), *numFactors);
        }
    }

fail:
    return e;
}

template<typename MP, typename CP>
NvDlaError engine_ast::SDPBatchNormOpNode::processMeanAndVar
(
    Weights& rawMeanData,
    Weights& processedMeanData,
    Weights& rawVarData,
    std::vector<Weights>& processedVarData
)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Edge* auxEdge;

    MP* rawMeanBlob = reinterpret_cast<MP*>(const_cast<void*>(rawMeanData.values));
    MP* rawVarBlob = reinterpret_cast<MP*>(const_cast<void*>(rawVarData.values));
    MP* procMeanBlob;
    MP* procVarBlob;

    PROPAGATE_ERROR_FAIL(nodeAuxEdge(&auxEdge));

    // prepare mean and variance structs
    processedMeanData.type   = rawMeanData.type;
    processedMeanData.count  = rawMeanData.count;
    procMeanBlob = (MP*)engine_ast::MemoryCollector::getInstance()->allocateMemory(rawMeanData.count * sizeof(MP));
    processedMeanData.values = procMeanBlob;

    for (size_t vv = 0; vv < processedVarData.size(); ++vv)
    {
        processedVarData[vv].type   = rawMeanData.type;
        processedVarData[vv].count  = rawMeanData.count;
        procVarBlob = (MP*)engine_ast::MemoryCollector::getInstance()->allocateMemory(rawVarData.count * sizeof(MP));
        processedVarData[vv].values = procVarBlob;
    }

    // process mean and variance
    if (params().x1Params().mode().v() == SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        procMeanBlob[0] = (-1) * rawMeanBlob[0];
        for (size_t vv = 0; vv < processedVarData.size(); ++vv)
        {
            MP procVarValue;
            procVarBlob = reinterpret_cast<MP*>(const_cast<void*>(processedVarData[vv].values));

            // if no need to factorize, 1st blob should contain the processed data
            // and subsequent blobs should contain '1'
            if (factorsPerEntity<MP, CP>(rawVarBlob[0], VARIANCE_DATA, &procVarValue) == 1)
            {
                procVarBlob[0] = (vv == 0) ? procVarValue : 1;
            }
        }
    }
    else if (params().x1Params().mode().v() == SDPModeEnum::SDP_MODE_PER_CHANNEL)
    {
        NvS32 chnls = auxEdge->tensorSurfaceDesc()->dimensions().c;
        for (int cc = 0; cc < chnls; ++cc)
        {
            procMeanBlob[cc] = (-1) * rawMeanBlob[cc];
            for (size_t vv = 0; vv < processedVarData.size(); ++vv)
            {
                MP procVarValue;
                procVarBlob = reinterpret_cast<MP*>(const_cast<void*>(processedVarData[vv].values));

                // if no need to factorize, 1st blob should contain the processed data
                // and subsequent blobs should contain '1'
                if (factorsPerEntity<MP, CP>(rawVarBlob[cc], VARIANCE_DATA, &procVarValue) == 1)
                {
                    procVarBlob[cc] = (vv == 0) ? procVarValue : 1;
                }
                else
                {
                    procVarBlob[cc] = procVarValue;
                }
            }
        }
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support SDP mode %s", params().x1Params().mode().c_str());
    }

fail:
    return e;
}

/*---------------------Scale Mean Data to INT16------------------------------*/
static bool absCompare(NvS32 a, NvS32 b)
{
    return (std::abs(a) < std::abs(b));
}

NvDlaError engine_ast::SDPBatchNormOpNode::scaleMeanToInt16
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    std::vector<NvF32>& inTensorScales
)
{
    NvDlaError e = NvDlaSuccess;

    NvF32 perTensorInTensorScl = inTensorScales.at(0);
    bool isPerKernelQtz  = graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL;
    bool isScalingDone   = false;

    NvS32 maxMean;
    NvS32 numBits;
    NvU32 meanShift;
    std::vector< NvS32 > rescaledMean32;

    NvF32 scaledFP32Mean = 0.0f;
    NvS16 int16Mean      = 0;
    Weights origMeanBlob = params().rawMeanData();
    Weights int16MeanBlob;
    NvS16* pInt16Mean = (NvS16*)std::malloc(origMeanBlob.count * sizeof(NvS16));
    NvU32 numMeanData = origMeanBlob.count;

    if ( fusedConv )
    {
        NvU32 numFilters = filterScales.size();
        ASSERT ( filterScales.size() == (size_t)auxEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvU32 ff = 0; ff < numFilters; ++ff)
        {
            if ( isPerKernelQtz && (filterScales[0] != filterScales[ff]) )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Filter scales should be same for %s when PER_KERNEL "
                                    "quantization is ON", fusedConv->name().c_str());
            }
        }

        ASSERT ( inTensorScales.size() == (size_t)fusedConv->inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvF32 its = 1; its < inTensorScales.size(); ++its)
        {
            if ( perTensorInTensorScl != inTensorScales[its] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", fusedConv->name().c_str());
            }
        }

        ASSERT( numFilters == origMeanBlob.count );
    }
    else
    {
        ASSERT ( inTensorScales.size() == (size_t)inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvF32 its = 1; its < inTensorScales.size(); ++its)
        {
            if ( perTensorInTensorScl != inTensorScales[its] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input, when PER_TENSOR scaling is ON");
            }
        }
    }

    for ( NvU32 bb = 0; bb < numMeanData; ++bb)
    {
        switch (origMeanBlob.type)
        {
            case nvdla::DataType::FLOAT: {
                NvF32 fp32Mean = reinterpret_cast<NvF32*>(const_cast<void*>(origMeanBlob.values))[bb];
                NvF32 meanRescaleFactor = perTensorInTensorScl;
                if ( fusedConv )
                {
                    meanRescaleFactor = perTensorInTensorScl * filterScales[bb];
                }
                scaledFP32Mean = NvF32(fp32Mean / meanRescaleFactor);
                NvS32 int32Mean = std::floor(scaledFP32Mean + 0.5f);
                rescaledMean32.push_back(int32Mean);
            } break;
            case nvdla::DataType::HALF: {
                NvF32 fp16Mean = reinterpret_cast<half_float::half*>(const_cast<void*>(origMeanBlob.values))[bb];
                NvF32 meanRescaleFactor = perTensorInTensorScl;
                if ( fusedConv )
                {
                    meanRescaleFactor = perTensorInTensorScl * filterScales[bb];
                }
                scaledFP32Mean = NvF32(fp16Mean / meanRescaleFactor);
                NvS32 int32Mean = std::floor(scaledFP32Mean + 0.5f);
                rescaledMean32.push_back(int32Mean);
            } break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Can't scale mean data which is not FLOAT / HALF "
                                        "precision for %s\n", name().c_str());
        }
    }

    maxMean = *std::max_element(rescaledMean32.begin(), rescaledMean32.end(), absCompare);
    numBits = ceil(log(abs(maxMean))/log(2)) + 1;
    meanShift = std::min(SDP_LEFT_SHIFT_MAX_PLACES, std::max(0, numBits - 16));

    do
    {
        isScalingDone = true;
        for ( NvU32 bb = 0; bb < numMeanData; ++bb)
        {
            int16Mean = static_cast<NvS16>(rescaledMean32[bb] >> meanShift);
            pInt16Mean[bb] = int16Mean;

            if ( graph()->debugQuantization() )
            {
                if (fusedConv)
                {
                    gLogInfo << "rawMean/Si*Sw "
                             << reinterpret_cast<NvF32*>(const_cast<void*>(origMeanBlob.values))[bb] << " / "
                             << " ( " << perTensorInTensorScl << " * " << filterScales[bb] << " ) = "
                             << (reinterpret_cast<NvF32*>(const_cast<void*>(origMeanBlob.values))[bb]/(perTensorInTensorScl * filterScales[bb]))
                             << " -> " << rescaledMean32[bb] << " -> " << (int)int16Mean << "*2^-" << meanShift << endl;
                }
                else
                {
                    gLogInfo << "rawMean/Si "
                             << reinterpret_cast<NvF32*>(const_cast<void*>(origMeanBlob.values))[bb] << " / "
                             << perTensorInTensorScl << " = "
                             << (reinterpret_cast<NvF32*>(const_cast<void*>(origMeanBlob.values))[bb]/perTensorInTensorScl)
                             << " -> " << rescaledMean32[bb] << " -> " << (int)int16Mean << "*2^-" << meanShift << endl;
                }
            }
            if (int16Mean == 0 && meanShift > 0)
            {
                meanShift--;
                isScalingDone = false;
                break;
            }
        }
    } while (!isScalingDone);

    params().x1Params().setShiftValue(meanShift);

    int16MeanBlob.type   = nvdla::DataType::INT16;
    int16MeanBlob.values = pInt16Mean;
    int16MeanBlob.count  = origMeanBlob.count;

    // set scaled mean
    params().setRawMeanData(int16MeanBlob);

fail:
    return e;
}

/********************************Process Aux Data*****************************/
/*
 * BN has mean and variance data to deal with. These need pre-processing before
 * going to SDP such that the mean-subtraction becomes (-mean) addition and
 * variance-divison becomes (1/variance) multiplication.
 * This API also pre-fetches if any of these offline math operation could
 * create data that over/under shoots INT8/INT16/FP16 range. If so, it
 * breaks the super-small/super-big data and delegates it to an adjunct
 * scale/bias operation
 */
NvDlaError engine_ast::SDPBatchNormOpNode::preProcessAuxData()
{
    NvDlaError e = NvDlaSuccess;

    NvU32 numMeanBlobs = 0;
    NvU32 numVarBlobs = 0;
    Weights rawMeanData = params().rawMeanData();
    Weights rawVarData  = params().rawVarianceData();

    nvdla::DataType modelPrec             = rawMeanData.type == rawVarData.type ?
                                            rawMeanData.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();

    /* Batch-Norm:-> y = (x - mean) / sqrt(variance+eps)
     * SDP can do ADD and MUL, not SUB and DIV
     */

    // this step calculates requirement and number of adjunct_operations
    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), e, maxFactorsPerEntity, rawMeanData, MEAN_DATA, &numMeanBlobs)
    if (numMeanBlobs > 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue , "FIXME: Cannot handle Mean values beyond computePrecision limits for now");
    }
    else if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(e);
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), e, maxFactorsPerEntity, rawVarData, VARIANCE_DATA, &numVarBlobs)
    if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(e);
    }

    // now process the data to convert mean to (-mean) and var to (1/sqrt(var)), meanwhile factorizing
    // those processed entities which don't fit in the dynamic range of compute precision
    {
        Weights processedMeanBlob;
        std::vector<Weights> processedVarBlobs(numVarBlobs);

        PRECISION_SWITCH(modelPrec.v(), computePrec.v(), e, processMeanAndVar, rawMeanData, processedMeanBlob,
                                                                               rawVarData, processedVarBlobs);

        /* Override raw data from parser to the pre-processed data.
         * NOTE: that they are still don't have the suitable layout for DLA;
         *       so they are still 'raw' for the engine
         */
        params().setRawMeanData(processedMeanBlob);
        params().setRawVarianceData(processedVarBlobs[0]);

        /* If the varData (FIXME: mean) was factorized, append a scale op post this BN,
         * to multiply the subsequent processed variance factors to the result of this BN
         */
        if (numVarBlobs > 1)
        {
            Tensor* ioTensor;
            engine_ast::Edge* origOutEdge;
            engine_ast::Edge* ioEdge;
            NVDLA_UNUSED(ioEdge);

            std::vector<engine_ast::SDPScaleOpNode*> adjunctScaleOps = std::vector<engine_ast::SDPScaleOpNode*>();
            std::vector<engine_ast::SDPScaleOpNode*>::iterator scaleIter;

            PROPAGATE_ERROR_FAIL(nodeDataEdge(supportedOutSurfCategories(), ast::EdgeSideEnum::FIRST, &origOutEdge));

            for (NvU32 vv = 1; vv < numVarBlobs; ++vv)
            {
                engine_ast::SDPScaleOpNode* newScaleNode = engine_ast::NodeFactory::newSDPScaleOpNode(NULL, graph());
                if ( !newScaleNode)
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't create adjunct scale op");
                }

                newScaleNode->params().x1Params().setMode(params().x1Params().mode());
                newScaleNode->params().setScaleDims(params().batchNormDims());
                newScaleNode->params().setRawScaleData(processedVarBlobs[vv]);
                newScaleNode->params().setDLAScaleData(Weights(DataType::FLOAT, NULL, 0));
                PROPAGATE_ERROR_FAIL(newScaleNode->captureCanonicalScaleData());

                adjunctScaleOps.push_back(newScaleNode);
            }
            //delegate the orig BN's output edge to the last of scale ops
            graph()->replaceEdgeNodes(origOutEdge, ast::EdgeSideEnum::FIRST, this, adjunctScaleOps.back());

            // connect the newly added scale nodes with edges
            for(scaleIter = adjunctScaleOps.begin(); scaleIter != adjunctScaleOps.end(); ++scaleIter)
            {
                Tensor* origOutTensor = origOutEdge->originalTensor();
                engine_ast::SDPScaleOpNode* scaleNode = (*scaleIter);
                engine_ast::Node* edgeSrcNode;

                ioTensor = origOutTensor->clone();
                ioTensor->setTensorType(TensorType::kIO);
                if (scaleIter == adjunctScaleOps.begin())
                {
                    edgeSrcNode = this;
                }
                else
                {
                    edgeSrcNode = *(scaleIter-1);
                }

                ioEdge = graph()->addDataEdge((canonical_ast::Edge*)0, edgeSrcNode, scaleNode, ioTensor);
            }
        }
    }
fail:
    return e;
}

/*--------------------Quantize Aux Data-----------------------------*/
NvDlaError engine_ast::SDPBatchNormOpNode::quantizeAuxData()
{
    NvDlaError e = NvDlaSuccess;

    std::vector<NvF32> filterScales = {0.0f};
    std::vector<NvF32> inTensorScales;

    ConvCoreNode* fusedConv = NULL;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (graph()->profile()->computePrecision().v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        // No need to quantize data for FP16
        goto fail;
    }

    if ( graph()->profile()->tensorScalingMode().v() != nvdla::TensorScalingMode::PER_TENSOR )
    {
       // don't support any other scaling mode than PER_TENSOR
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support tensor scaling mode: %s\n",
                                graph()->profile()->tensorScalingMode().c_str());
    }

    fusedConv = NodeFactory::nodeCast<ConvCoreNode*>(graph()->upstreamDataNodes(this).size() ?
                                                     graph()->upstreamDataNodes(this).at(0) :
                                                     NULL);
    if ( fusedConv )
    {
        PROPAGATE_ERROR_FAIL( fusedConv->verifyEdgePorts() );
        filterScales = fusedConv->params().filterScales();
        inTensorScales  = fusedConv->inputEdges().at(0)->originalTensor()->getChannelScales();
    }
    else
    {
        inTensorScales  = inputEdges().at(0)->originalTensor()->getChannelScales();
    }
    /*
     * StandAlone INT8 Batch Norm op is represented as:
     *
     *      O  = (I - mean) / sigma
     *    QoSo = (QiSi - mean) / sigma
     *      Qo = (Qi - (mean/Si)) * (Si/ (So * sigma))
     *
     *   Do mean/Si in preprocess aux data and scale it to INT16
     *   Set meanData to this new value
     *
     *   Do (Si/ (So * sigma)) in handleLowPrecision and convert it to 2^(-m) * n equation
     *   where n becomes the variance data and m values does the right shift
     *
     * CONV + BN Unfused
     *
     *      O = ((I * W) - mean ) / sigma
     *   QoSo = ((QiSi * QwSw) - mean) / sigma
     *     Qo = (Qi * Qw - (mean / (Si * Sw)) * ((Si * Sw) / (So * Sigma))
     *
     *   Do mean / (Si * Sw) in preprocess aux data and scale it to INT16
     *   Set meanData to this new value
     *
     *   Do (Si * Sw) / (So * Sigma) in handleLowPrecision and convert it to 2^(-m) * n equation
     *   where n becomes the variance data and m values does the right shift
     */
    if ( graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL ||
                 graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_FILTER )
    {
         if (auxEdges().at(0)->tensorSurfaceDesc()->surfaceFormat().f().precision().v() ==
                    surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16)
        {
            PROPAGATE_ERROR_FAIL( scaleMeanToInt16(fusedConv, filterScales, inTensorScales) );
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support precision: %s\n",
                        auxEdges().at(0)->tensorSurfaceDesc()->surfaceFormat().f().precision().v());
        }
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support quantization mode: %s\n",
                                graph()->profile()->quantizationMode().c_str());
    }

fail:
    return e;
}

/*------------------Low Precision Conversions for Batch Norm--------------------------*/
template<typename MP, typename CP>
NvDlaError engine_ast::SDPBatchNormOpNode::performPerLayerRescaling
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    Weights& rawVarData,
    std::vector<NvF32>& inTensorScales,
    std::vector<NvF32>& outTensorScales
)
{
    NvDlaError e = NvDlaSuccess;

    NvS16 perTensorScl  = 0;
    NvU8  perTensorShft = 0;
    NvF32 outputRescale = 0.0f;
    std::pair<NvS16, NvU8> scaleAndShift;

    NvS16* procVarBlob;
    Weights procVarData;
    MP* rawVarBlob = reinterpret_cast<MP*>(const_cast<void*>(rawVarData.values));

    procVarData.type   = rawVarData.type;
    procVarData.count  = rawVarData.count;
    procVarBlob = reinterpret_cast<NvS16*>(std::malloc(rawVarData.count * sizeof(NvS16)));
    procVarData.values = procVarBlob;

    NvU32 numFilters = 0;
    NvF32 perKernelScale = 0.0f;
    NvF32 perTensorInTensorScl  = inTensorScales.at(0);
    NvF32 perTensorOutTensorScl = outTensorScales.at(0);

    if ( fusedConv )
    {
        numFilters            = filterScales.size();
        perKernelScale        = filterScales.at(0);
        ASSERT ( filterScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvU32 ff = 1; ff < numFilters; ++ff)
        {
            if ( perKernelScale != filterScales[ff] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Filter scales should be same for %s when PER_KERNEL "
                                    "quantization is ON", fusedConv->name().c_str());
            }
        }

        ASSERT ( inTensorScales.size() == (size_t)fusedConv->inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvF32 its = 1; its < inTensorScales.size(); ++its)
        {
            if ( perTensorInTensorScl != inTensorScales[its] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", fusedConv->name().c_str());
            }
        }
    }
    else
    {
        ASSERT ( inTensorScales.size() == (size_t)inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvF32 its = 1; its < inTensorScales.size(); ++its)
        {
            if ( perTensorInTensorScl != inTensorScales[its] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
            }
        }
    }

    ASSERT ( outTensorScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
    for (NvF32 ots = 1; ots < outTensorScales.size(); ++ots)
    {
        if ( perTensorOutTensorScl != outTensorScales[ots] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for output of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
        }
    }

    outputRescale = ( perTensorInTensorScl * rawVarBlob[0] ) / perTensorOutTensorScl;
    if ( fusedConv )
    {
        outputRescale = outputRescale * perKernelScale;
    }
    e = calculateScaleAndShiftFromScalar<NvS16, NvU8>(outputRescale, &scaleAndShift);
    if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, " Couldn't converge on `2^(x) * y` which could "
                                "safely represent %f within acceptable tolerance for %s\n",
                                outputRescale, name().c_str());
    }

    perTensorScl  = scaleAndShift.first;
    procVarBlob[0] = perTensorScl;
    perTensorShft = scaleAndShift.second;

    if (graph()->debugQuantization())
    {
        gLogInfo << name() << " Si * V[i] / So = " << perTensorInTensorScl << " * " << rawVarBlob[0] << " / " << perTensorOutTensorScl
                 << " = " << perTensorScl << "* 2^-" << (int)perTensorShft << endl;
    }

    params().x1Params().setTruncate(perTensorShft);
    params().setRawVarianceData(procVarData);

fail:
    return e;
}

template<typename MP, typename CP>
NvDlaError engine_ast::SDPBatchNormOpNode::performPerChannelRescaling
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    Weights& rawVarData,
    std::vector<NvF32>& inTensorScales,
    std::vector<NvF32>& outTensorScales
)
{
    NvDlaError e = NvDlaSuccess;

    NvF32 maxRescaleFactor = 0.0f;
    std::vector< NvF32 > outputRescales;
    std::pair< NvS16, NvU8 > maxRescaleFactorScaleAndShift;
    std::vector< std::pair<NvS16, NvU8> > scalesAndShifts;

    Weights rescaleWtsBlob;

    NvU32 numVarData;
    NvF32 perTensorInTensorScl  = inTensorScales.at(0);
    NvF32 perTensorOutTensorScl = outTensorScales.at(0);
    MP* rawVarBlob = reinterpret_cast<MP*>(const_cast<void*>(rawVarData.values));
    numVarData = rawVarData.count;


    if ( fusedConv )
    {
        ASSERT ( filterScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        ASSERT ( inTensorScales.size() == (size_t)fusedConv->inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvF32 its = 1; its < inTensorScales.size(); ++its)
        {
            if ( perTensorInTensorScl != inTensorScales[its] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", fusedConv->name().c_str());
            }
        }
    }
    else
    {
        ASSERT ( inTensorScales.size() == (size_t)inputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
        for (NvF32 its = 1; its < inTensorScales.size(); ++its)
        {
            if ( perTensorInTensorScl != inTensorScales[its] )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for input of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
            }
        }
    }

    ASSERT ( outTensorScales.size() == (size_t)outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
    for (NvF32 ots = 1; ots < outTensorScales.size(); ++ots)
    {
        if ( perTensorOutTensorScl != outTensorScales[ots] )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Channel scales should be same for output of %s when PER_TENSOR "
                                    "scaling is ON", name().c_str());
        }
    }

    for (NvU32 cc = 0; cc < numVarData; ++cc)
    {
        NvF32 rescaleData = ( perTensorInTensorScl * rawVarBlob[cc] ) / perTensorOutTensorScl;
        if ( fusedConv )
        {
            rescaleData = rescaleData * filterScales[cc];
        }
        outputRescales.push_back(rescaleData);
    }

    maxRescaleFactor = *std::max_element(outputRescales.begin(), outputRescales.end());

    // find the shifter value for the max of the rescale factors, since we can use only 1 shifter
    e = calculateScaleAndShiftFromScalar<NvS16, NvU8>(maxRescaleFactor, &maxRescaleFactorScaleAndShift);
    if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, " Couldn't converge on `2^(x) * y` which could "
                                "safely represent %f within acceptable tolerance for %s\n",
                                maxRescaleFactor, name().c_str());
    }

    // pass the common shifter to determine int16 scalars for each rescale factors
    e = factorizeScalars<NvS16, NvU8>(outputRescales, &scalesAndShifts, maxRescaleFactorScaleAndShift.second);
    if (e != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't factorize scalars for %s for int8 into scale+truncate pairs",
                                               name().c_str());
    }

    {
        // fixme: assert that the existing aux data is also of type INT16
        rescaleWtsBlob.type = nvdla::DataType::INT16;
        rescaleWtsBlob.count = numVarData;
        NvS16* pINT16Rescalars = reinterpret_cast<NvS16*>(std::malloc(numVarData * sizeof(NvS16)));
        for (NvU32 cc = 0; cc < numVarData; ++cc)
        {
            pINT16Rescalars[cc] = scalesAndShifts[cc].first;
            if (graph()->debugQuantization())
            {
                gLogInfo << name() << " Si * V[k] / So = " << perTensorInTensorScl << " * " << rawVarBlob[cc] << " / " << perTensorOutTensorScl
                         << " = " << scalesAndShifts[cc].first << "* 2^-" << (int)scalesAndShifts[cc].second << endl;
            }
        }
        rescaleWtsBlob.values = pINT16Rescalars;
    }

    params().x1Params().setTruncate(maxRescaleFactorScaleAndShift.second);
    params().setRawVarianceData(rescaleWtsBlob);

fail:
    return e;
}

NvDlaError engine_ast::SDPBatchNormOpNode::handleLowPrecisionConversions()
{
    NvDlaError e = NvDlaSuccess;

    ConvCoreNode* fusedConv;

    std::vector<NvF32> inTensorScales;
    std::vector<NvF32> outTensorScales;
    std::vector<NvF32> filterScales = {0.0f};
    Weights rawVarData = params().rawVarianceData();

    nvdla::DataType modelPrec             = rawVarData.type;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (graph()->profile()->computePrecision().v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        // nop
        goto fail;
    }

    if ( graph()->profile()->tensorScalingMode().v() != nvdla::TensorScalingMode::PER_TENSOR )
    {
        // don't support any other scaling mode than PER_TENSOR
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support tensor scaling mode: %s\n",
                                graph()->profile()->tensorScalingMode().c_str());
    }

    fusedConv = NodeFactory::nodeCast<ConvCoreNode*>(dependencyParams(0).fusedNode(IODirectionEnum::INPUT));
    if ( fusedConv )
    {
        PROPAGATE_ERROR_FAIL( fusedConv->verifyEdgePorts() );
        filterScales = fusedConv->params().filterScales();
        inTensorScales  =fusedConv->inputEdges().at(0)->originalTensor()->getChannelScales();
    }
    else
    {
        inTensorScales  = inputEdges().at(0)->originalTensor()->getChannelScales();
    }

    outTensorScales = outputEdges().at(0)->originalTensor()->getChannelScales();

    if ( graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL ||
         graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_FILTER )
    {
        if ( params().x1Params().mode().v() == SDPModeEnum::SDP_MODE_PER_LAYER )
        {
            PRECISION_SWITCH(modelPrec.v(), computePrec.v(), e, performPerLayerRescaling,
                                                                fusedConv,
                                                                filterScales,
                                                                rawVarData,
                                                                inTensorScales,
                                                                outTensorScales);
            if (e != NvDlaSuccess)
            {
                ORIGINATE_ERROR_FAIL(e, "Could not rescale the per layer variance data");
            }
        }
        else if ( params().x1Params().mode().v() == SDPModeEnum::SDP_MODE_PER_CHANNEL )
        {
            PRECISION_SWITCH(modelPrec.v(), computePrec.v(), e, performPerChannelRescaling,
                                                                fusedConv,
                                                                filterScales,
                                                                rawVarData,
                                                                inTensorScales,
                                                                outTensorScales);
            if (e != NvDlaSuccess)
            {
                ORIGINATE_ERROR_FAIL(e, "Could not rescale the per channel variance data");
            }
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support quantization mode: %s\n",
                                graph()->profile()->quantizationMode().c_str());
        }
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support quantization mode: %s\n",
                                graph()->profile()->quantizationMode().c_str());
    }
fail:
    return e;
}

/*-------------------------Merge SDP Operations------------------------------*/
/*
 * Sometimes BN and scale ops are adjacent in the graph. It may be likely to combine
 * the variance factors of the BN op with the scale factors of the adjacent scale op.
 * This api attempts to combine them if the product doesn't overshoot the dynamic range
 * of the prevalent compute precision
 */
engine_ast::Node* engine_ast::SDPBatchNormOpNode::mergeWithSDPOp(SDPNode* nextSDP)
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

    // fixme: limit the batchnorm math op fusion with only sdp-scale, sdp-bias and relu for now
    if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_SCALE)
    {
        removableNode = tryToMergeWithScaleOp(nextSDP);
    }
    else if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_BIAS)
    {
        removableNode = tryToMergeWithBiasOpInplace(nextSDP);
        if (!removableNode)
        {
            removableNode = tryToMergeWithBiasOp(nextSDP);
        }
    }
    else if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_ACTIVATION)
    {
        if (NodeFactory::nodeCast<SDPActivationOpNode*>(nextSDP)->params(/*batch_id*/0).x1Params().actType().v() ==
                SDPActTypeEnum::SDP_ACT_TYPE_RELU)
        {
            removableNode = tryToMergeWithActOp(nextSDP);
        }
    }

fail:
    return removableNode;
}

engine_ast::Node* engine_ast::SDPBatchNormOpNode::tryToMergeWithScaleOp(SDPNode* SDPSclOp)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode     = NULL;
    SDPScaleOpNode* scaleOp = NodeFactory::nodeCast<SDPScaleOpNode*>(SDPSclOp);
    Weights rawVarData      = params().rawVarianceData();
    Weights rawScaleData    = scaleOp->params().rawScaleData();
    Dims4 commonDims        = params().varianceDims();
    Weights combinedVarAndScaleData;
    WeightTrns wtTrns;
    NVDLA_UNUSED(e);

    nvdla::DataType modelPrec             = rawVarData.type == rawScaleData.type ?
                                            rawVarData.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();

    WeightTrns::WeightDims wtTrnsDims (rawVarData.count, commonDims.n, commonDims.c,
                                       commonDims.w, commonDims.h, 1, 1);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if ( params().x1Params().mode().e() != scaleOp->params().x1Params().mode().e() ||
         params().varianceDims() != scaleOp->params().scaleDims())
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't combine Variance and Scale factors "
                        << "of " << name() <<  " and " << scaleOp->name()
                        << " since they operate in different modes"<< endl;
        }
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedVarAndScaleData,
                                                     wtTrns.combineMultiplicationFactors,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     rawVarData,
                                                     rawScaleData);

    if (combinedVarAndScaleData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Variance and Scale factors of "
                        << name() << " and " << scaleOp->name() << endl;
        }
        goto fail;
    }

    /* Override the variance data with the combined blob.
     * NOTE: that it still doesn't have the suitable layout for DLA;
     *       so its still 'raw' variance data for the engine
     */
    params().setRawVarianceData(combinedVarAndScaleData);

    // activation from next SDP node
    params().x1Params().setActType(scaleOp->params().x1Params().actType());

    /* Since Scale factors are folded into BN, we can safely remove
     * the Scale Op from engine AST
     */
    removableNode = SDPSclOp;

    if ( graph()->debugMathOptz() )
    {
        gLogInfo << "Fuse " << name() << " and " << SDPSclOp->name() << " into " << name() << endl;
    }

fail:
    return removableNode;
}

/*
 * BatchNorm preceded by a Convolution/Scale Op and followed by a Bias op is a good oppotunity
 * for multi-level fusion
 *
 * Conv + BN + Bias : y = [((d * w) + m)*(1/v)] + b
 *                      = d*w/v + m/v + b
 *                      = d*w' + b'
 *                      = Conv + Bias
 *                      => 3 ops reduced to 2 and
 *                         2 SDPs combined in the process to hold up only single SDP-X sub-engine
 *
 * Scl + BN + Bias :  y = [((x*s) + m)*(1/v)] + b
 *                      = x*s/b + m/v + b
 *                      = x*s' + b'
 *                      = Scl + Bias
 *                      => 3 ops reduced to 2 and
 *                         further optimization can collapse (Scl + Bias) into a new BN;
 *                         so eventually reduced to 1 op
 *
 * BN + Bias :        y = ((x + m)*(1/v)) + b
 *                      = x/v + m/v + b
 *                      = x*(1/v) + b'
 *                      = Scl + Bias
 *                      => 2 ops trnasform into 2 different ops
 *                         however further optimization can collapse this (Scl + Bias) into a new BN;
 *                         so eventually reduced to 1 op
 *
 * From all above scenarios, its evident that whenever you encounter BN + Bias, then convert the BN into Scl
 * and delegate the Variance data to it and the Mean data to the  succeeding Bias
 * and in future optimizations let:
 *      - Conv,Scl combine into Conv
 *      - Bias,Bias combine to Bias
 *      - Scl, Bias combine to BN again
 */
engine_ast::Node* engine_ast::SDPBatchNormOpNode::tryToMergeWithBiasOp(SDPNode* SDPBiasOp)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode     = NULL;
    SDPBiasOpNode* biasOp   = NodeFactory::nodeCast<SDPBiasOpNode*>(SDPBiasOp);
    Weights rawMeanData     = params().rawMeanData();
    Weights rawVarData      = params().rawVarianceData();
    Weights rawBiasData     = biasOp->params().rawBiasData();
    Dims4 bnDims            = params().batchNormDims();
    Weights combinedMeanAndVarData;
    Weights combinedMeanVarAndBiasData;
    WeightTrns wtTrns;
    NVDLA_UNUSED(e);

    nvdla::DataType modelPrec             = rawVarData.type == rawMeanData.type ?
                                            rawVarData.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();
    WeightTrns::WeightDims wtTrnsDims (rawVarData.count, bnDims.n, bnDims.c, bnDims.w, bnDims.h, 1, 1);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if ( params().x1Params().mode().e() != biasOp->params().x1Params().mode().e() ||
         params().batchNormDims() != biasOp->params().biasDims())
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't combine Mean, Variance with Bias factors "
                        << "of " << name() <<  " and " << biasOp->name()
                        << " since they operate in different modes"<< endl;
        }
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedMeanAndVarData,
                                                     wtTrns.combineMultiplicationFactors,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     rawMeanData,
                                                     rawVarData);

    if (combinedMeanAndVarData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Variance and Scale factors of "
                        << name() << " and " << biasOp->name() << endl;
        }
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedMeanVarAndBiasData,
                                                     wtTrns.combineAdditionFactors,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     combinedMeanAndVarData,
                                                     rawBiasData);

    if (combinedMeanVarAndBiasData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Mean/Variance with Bias factors of "
                        << name() << " and " << biasOp->name() << endl;
        }
        goto fail;
    }

    /* BN + Bias : y = ((x+m)*(1/v)) + b
     *               = x/v + m/v + b
     * If each of the m/v factors and (m/v + b) factors didn't over/undershoot
     * the compute precision limits, then go ahead and replace the BN node with
     * a brand new Scl node - delegating the (1/v) factors of the original BN node
     * to the new Scl node and delegate the (m/v + b) combined factors computed to the
     * existing Bias node
     */
    {
        NodeSequence substituteNodes;
        SDPScaleOpNode* newSclReplaceNode = NULL;

        /* Step-1: Substitute BN with Scl node */
        newSclReplaceNode = NodeFactory::newSDPScaleOpNode(NULL, graph());
        if ( !newSclReplaceNode)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't create new Scl op for replacing %s",
                    name().c_str());
        }
        newSclReplaceNode->params().x1Params().setMode(params().x1Params().mode());
        newSclReplaceNode->params().setScaleDims(bnDims);
        newSclReplaceNode->params().setRawScaleData(rawVarData);
        newSclReplaceNode->params().setDLAScaleData(Weights(DataType::FLOAT, NULL, 0));
        PROPAGATE_ERROR_FAIL(newSclReplaceNode->captureCanonicalScaleData());

        /* Step-2: Transfer the (m/v + b) factors to the existing Bias node */
        biasOp->params().setRawBiasData(combinedMeanVarAndBiasData);

        /* Step-3: Substitute the BN node with Scl node */
        substituteNodes.push_back(newSclReplaceNode);
        PROPAGATE_ERROR_FAIL(graph()->substituteNodeInAST(this, substituteNodes));

        /* Step-4: Since Scl is replacing BN, inherit the op mode from the BN node before its removed*/
        newSclReplaceNode->params().setConvMode(params().convMode());
        newSclReplaceNode->params().setWinogradParams(params().winogradParams());

        /* Step-5: Finally remove the BN node */
        removableNode = this;

        if ( graph()->debugMathOptz() )
        {
            gLogInfo << "Replace " << name() << " with " << newSclReplaceNode->name() << endl;
        }
    }

fail:
    return removableNode;
}

/*
 * BatchNorm followed by a Bias op can be squashed into BatchNorm.
 *
 * BN + Bias :        y = ((x + m)*(1/v)) + b
 *                      = x/v + m/v + b
 *                      = (x + (m + b*v)) * 1/v
 *                      = (x + m') * 1/v
 * Here, 2 ops fused into 1, if m' fits in the format precision range
 * Generally, bias values are much higher than kernel weights,
 * so batch-norm + bias -> batch-norm, optimization may perform well
 * in accuracy.
 *
 */
engine_ast::Node* engine_ast::SDPBatchNormOpNode::tryToMergeWithBiasOpInplace(SDPNode* SDPBiasOp)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode     = NULL;
    SDPBiasOpNode* biasOp   = NodeFactory::nodeCast<SDPBiasOpNode*>(SDPBiasOp);
    Weights rawMeanData     = params().rawMeanData();
    Weights rawVarData      = params().rawVarianceData();
    Weights rawBiasData     = biasOp->params().rawBiasData();
    Dims4 bnDims            = params().batchNormDims();
    Weights invertVarData;
    Weights combinedBiasAndVarData;
    Weights combinedMeanVarAndBiasData;
    WeightTrns wtTrns;
    NVDLA_UNUSED(e);

    nvdla::DataType modelPrec             = rawVarData.type == rawMeanData.type ?
                                            rawVarData.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();
    WeightTrns::WeightDims wtTrnsDims (rawVarData.count, bnDims.n, bnDims.c, bnDims.w, bnDims.h, 1, 1);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if ( params().x1Params().mode().e() != biasOp->params().x1Params().mode().e() ||
         params().batchNormDims() != biasOp->params().biasDims())
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't combine Mean, Variance with Bias factors "
                        << "of " << name() <<  " and " << biasOp->name()
                        << " since they operate in different modes"<< endl;
        }
        goto fail;
    }

    /* rawVarData stores 1/variance values,
     * Since we need variance, take a reciprocal.
     * Since reciprocal blob is temporary, no precision switch required.
     */
    PRECISION_SWITCH(modelPrec.v(), modelPrec.v(), invertVarData,
                                                     wtTrns.invertDataBlob,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     rawVarData);

    if (invertVarData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't invert variance during fusing of "
                        << name() << " and " << biasOp->name() << endl;
        }
        goto fail;
    }

    /* bias * variance */
    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedBiasAndVarData,
                                                     wtTrns.combineMultiplicationFactors,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     rawBiasData,
                                                     invertVarData);

    if (combinedBiasAndVarData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Variance and Scale factors of "
                        << name() << " and " << biasOp->name() << endl;
        }
        goto fail;
    }

    /* mean' =  mean + (bias * variance) */
    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedMeanVarAndBiasData,
                                                     wtTrns.combineAdditionFactors,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     combinedBiasAndVarData,
                                                     rawMeanData);

    if (combinedMeanVarAndBiasData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Mean/Variance with Bias factors of "
                        << name() << " and " << biasOp->name() << endl;
        }
        goto fail;
    }
    params().setRawMeanData(combinedMeanVarAndBiasData);

    // activation from next SDP node
    params().x1Params().setActType(biasOp->params().x1Params().actType());

    removableNode = SDPBiasOp;

fail:
    if (combinedBiasAndVarData.values)
    {
        engine_ast::MemoryCollector::getInstance()->freeMemory(const_cast<void*>(combinedBiasAndVarData.values));
    }
    if (invertVarData.values)
    {
        engine_ast::MemoryCollector::getInstance()->freeMemory(const_cast<void*>(invertVarData.values));
    }
    return removableNode;
}

/* Configure SDP SuperOp SubEngine with BatchNorm Op */
NvDlaError engine_ast::SDPBatchNormOpNode::configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN)
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

    sdpSuperOp->params().setAuxDataType(xN, TensorType::kBATCH_NORM);

    sdpSuperOp->params().setMultiplierDims(xN, params().varianceDims());
    sdpSuperOp->params().setAdderDims(xN, params().meanDims());
    sdpSuperOp->params().setDLADataDims(xN, params().batchNormDims());

    sdpSuperOp->params().setRawMultiplierData(xN, params().rawVarianceData());
    sdpSuperOp->params().setRawAdderData(xN, params().rawMeanData());
    sdpSuperOp->params().setDLAData(xN, params().DLABatchNormData());

    sdpSuperOp->params().setAuxSurfaceFormats(xN, suggestAuxSurfaceFormats());

    if ( graph()->debugFuseSubEngineOps() )
    {
        gLogInfo << "configureSDPSuperOpSubEngine: " << this->name() << " in ";
        gLogInfo << sdpSuperOp->name() << " x" << (NvU16)xN.e()+1 << endl;
    }

    return e;
}

/********************************Aux Data Translation***************************/
NvDlaError engine_ast::SDPBatchNormOpNode::translateAuxData()
{
    NvDlaError e = NvDlaSuccess;

    engine_ast::Edge* auxEdge;

    Weights trnsBNData;
    surface::SurfacePrecision computePrecision;
    Weights meanData = params().rawMeanData();
    Weights varData  = params().rawVarianceData();

    // find the aux data edge
    PROPAGATE_ERROR_FAIL(nodeAuxEdge(&auxEdge));

    computePrecision = auxEdge->tensorSurfaceDesc()->surfaceFormat().f().precision();

    if ( graph()->debugWeights() )
    {
        gLogInfo << "translating weights for " << name() << "bn-dims = " <<
                auxEdge->tensorSurfaceDesc()->dimensions().n << "," <<
                auxEdge->tensorSurfaceDesc()->dimensions().c << "," <<
                auxEdge->tensorSurfaceDesc()->dimensions().h << "," <<
                auxEdge->tensorSurfaceDesc()->dimensions().w << "," <<
                "and size= " << meanData.count << " + " << varData.count << endl;
    }

    {
        WeightTrns::WeightDims bnDims (meanData.count,
                                       auxEdge->tensorSurfaceDesc()->dimensions().n,
                                       auxEdge->tensorSurfaceDesc()->dimensions().c,
                                       auxEdge->tensorSurfaceDesc()->dimensions().w,
                                       auxEdge->tensorSurfaceDesc()->dimensions().h,
                                       1,   //strides dont matter for BN
                                       1);

        PRECISION_SWITCH(meanData.type.v(), computePrecision.v(), trnsBNData, WeightTrns::translateDataForBatchNorm,
                                                                              params().x1Params().mode(),
                                                                              bnDims,
                                                                              meanData,
                                                                              varData);

        if (trnsBNData.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "BN data trnaslation failed for node '%s'", name().c_str());
        }

        params().setDLABatchNormData(trnsBNData);

    }

fail:
    return e;
}

NvDlaError engine_ast::SDPBatchNormOpNode::emitOp(Graph *g,
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
    surface::TensorSurfaceDesc *bn_tsd   = g->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

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
    *x1_op_acc.type()        = x1_op_acc.type_Both();
    *x1_op_acc.mode()        = ASTToDLAInterface::getSDPMode(target_dla, params(batch_id).x1Params().mode());
    *x1_op_acc.act()         = ASTToDLAInterface::getSDPActType(target_dla, params(batch_id).x1Params().actType());
    *x1_op_acc.shiftValue()  = params(batch_id).x1Params().shiftValue();;
    if (params(batch_id).x1Params().mode().e() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        Weights bnData = params().DLABatchNormData();

        if (bnData.count > 2)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                "More than one data available"
                                "in per-layer mode(#data = %u)",
                                bnData.count);
        }
        switch (bnData.type)
        {
            case DataType::HALF:
            case DataType::INT16:
                *x1_op_acc.ALUOperand() =
                    reinterpret_cast<const NvS16 *>(bnData.values)[SDP_ADDER_DATA_INDEX];
                *x1_op_acc.MulOperand() =
                    reinterpret_cast<const NvS16 *>(bnData.values)[SDP_MULTIPLIER_DATA_INDEX];
                break;
            case DataType::INT8:
                *x1_op_acc.ALUOperand() =
                    reinterpret_cast<const NvS8 *>(bnData.values)[SDP_ADDER_DATA_INDEX];
                *x1_op_acc.MulOperand() =
                    reinterpret_cast<const NvS8 *>(bnData.values)[SDP_MULTIPLIER_DATA_INDEX];
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                    "Unexpected data type %s",
                                    bnData.type.c_str());
        }
    }
    else
    {
        *x1_op_acc.ALUOperand() = 0;
        *x1_op_acc.MulOperand() = 1;
    }
    *x1_op_acc.truncate()    = params(batch_id).x1Params().truncate();;
    *x1_op_acc.precision()   = ASTToDLAInterface::getSDPPrecision(target_dla, bn_tsd->surfaceFormat().precision());

    *x2_op_acc.enable() = 0;
    *y_op_acc.enable()  = 0;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);

    if (params(batch_id).x1Params().mode().e() != engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        setDataCubeAccessor(x1_data_acc, bn_tsd, IODirectionEnum::UNKNOWN, batch_id);
    }

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
        gLogInfo << "SDP BatchNorm node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
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
        gLogInfo << "\tdst tsd:" << dst_tsd->id() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tconv_mode " << (int)*sdp_op.convMode() << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.address() << endl;
        gLogInfo << "\tsrc type=" << (int)*src_data_acc.type() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tbn addr=" << *x1_data_acc.address() << endl;
        gLogInfo << "\tbn type=" << (int)*x1_data_acc.type() << endl;
        gLogInfo << "\tbn size " << *x1_data_acc.size()    << endl;
        gLogInfo << "\tbn width " << *x1_data_acc.width()   << endl;
        gLogInfo << "\tbn height " << *x1_data_acc.height()   << endl;
        gLogInfo << "\tbn channel " << *x1_data_acc.channel()  << endl;
        gLogInfo << "\tbn linestride " << *x1_data_acc.lineStride() << endl;
        gLogInfo << "\tbn surfstride " << *x1_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
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

NvDlaError engine_ast::SDPBatchNormOpNode::emitOp(NvU32 op_slot, NvU32 batch_id,
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
    protoSDPX1OpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_BOTH);
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

};  // nvdla::priv

};  // nvdla::
