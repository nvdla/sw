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
#include "priv/LowPrecision.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "priv/Compiler.h"
#include "priv/WeightTranslationUnit.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

NvDlaError engine_ast::SDPBiasOpNode::captureCanonicalBiasData()
{
    NvDlaError e = NvDlaError_Success;
    Tensor* bias_data_tensor;

    bias_data_tensor = graph()->addAuxTensor(graph()->newAuxTensorName(), params().biasDims(), TensorType::kBIAS);

    graph()->addDataEdge((canonical_ast::Edge*)0, 0, this, bias_data_tensor);

    return e;
}

void engine_ast::SDPBiasOpNode::captureCanonicalParams()
{
    NvDlaError e = NvDlaSuccess;

    params().x1Params().setEnabled(true);
    switch(canonicalNode()->canonicalOpType().v())
    {
        case canonical_ast::CanonicalOpTypeEnum::SCALE: {
             canonical_ast::ScaleNode* canScaleNode = canonical_ast::NodeFactory::nodeCast<canonical_ast::ScaleNode*>(canonicalNode());
             params().setBiasDims(canScaleNode->params().shiftDims());
             params().setRawBiasData(canScaleNode->params().shift());
             switch(canScaleNode->params().mode())
             {
                 case ScaleMode::sUNIFORM:        params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_LAYER); break;
                 case ScaleMode::sCHANNEL:        params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL); break;
                 case ScaleMode::sm_ELEMENTWISE:  params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_ELEMENT); break;
                 default: params().x1Params().setMode(SDPModeEnum::SDP_MODE_UNKNOWN);
             }
        } break;
        case canonical_ast::CanonicalOpTypeEnum::CONVOLUTION: {
             canonical_ast::ConvolutionNode* canConvNode = canonical_ast::NodeFactory::nodeCast<canonical_ast::ConvolutionNode*>(canonicalNode());
             params().setBiasDims(canConvNode->params().biasDims());
             params().setRawBiasData(canConvNode->params().biasData());
             params().setNumGroups(canConvNode->params().numGroups());
             switch(canConvNode->params().biasMode())
             {
                 case BiasMode::bUNIFORM:         params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_LAYER); break;
                 case BiasMode::bCHANNEL:         params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL); break;
                 case BiasMode::bm_ELEMENTWISE:   params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_ELEMENT); break;
                 default: params().x1Params().setMode(SDPModeEnum::SDP_MODE_UNKNOWN);
             }
        } break;
        case canonical_ast::CanonicalOpTypeEnum::FULLY_CONNECTED: {
             canonical_ast::FullyConnectedNode* canFCNode = canonical_ast::NodeFactory::nodeCast<canonical_ast::FullyConnectedNode*>(canonicalNode());
             params().setBiasDims(canFCNode->params().biasDims());
             params().setRawBiasData(canFCNode->params().biasData());
             switch(canFCNode->params().biasMode())
             {
                 case BiasMode::bUNIFORM:         params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_LAYER); break;
                 case BiasMode::bCHANNEL:         params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL); break;
                 case BiasMode::bm_ELEMENTWISE:   params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_ELEMENT); break;
                 default: params().x1Params().setMode(SDPModeEnum::SDP_MODE_UNKNOWN);
             }
        } break;
        case canonical_ast::CanonicalOpTypeEnum::DECONVOLUTION: {
             canonical_ast::DeconvolutionNode* canDeconvNode = canonical_ast::NodeFactory::nodeCast<canonical_ast::DeconvolutionNode*>(canonicalNode());
             params().setBiasDims(canDeconvNode->params().biasDims());
             params().setRawBiasData(canDeconvNode->params().biasData());
             params().setNumGroups(canDeconvNode->params().numGroups());
             switch(canDeconvNode->params().biasMode())
             {
                 case BiasMode::bUNIFORM:         params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_LAYER); break;
                 case BiasMode::bCHANNEL:         params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_CHANNEL); break;
                 case BiasMode::bm_ELEMENTWISE:   params().x1Params().setMode(SDPModeEnum::SDP_MODE_PER_ELEMENT); break;
                 default: params().x1Params().setMode(SDPModeEnum::SDP_MODE_UNKNOWN);
             }
        } break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized canonical op type: %s", canonicalNode()->canonicalOpType().c_str());
    }
    params().x1Params().setAluType(SDPALUTypeEnum::SDP_ALU_TYPE_SUM);
    params().x1Params().setOpType(SDPOpTypeEnum::SDP_OP_TYPE_ADD);
    params().setDLABiasData(Weights(DataType::FLOAT, NULL, 0));
    PROPAGATE_ERROR_FAIL(captureCanonicalBiasData());

    if ( graph()->debugWeights() )
    {
        Weights rawData = params().rawBiasData();
        gLogInfo << "raw weights of " << name() << ": ";
        for (ssize_t ii = 0; ii < rawData.count; ++ii)
            gLogInfo << reinterpret_cast<NvF32*>(const_cast<void*>(rawData.values))[ii] << ", ";
        gLogInfo << endl;
    }

fail:
    return;
}

void engine_ast::SDPBiasOpNode::inheritParams(Node* otherNode)
{
    // inherit the parameters that make sense (keep adding later)
    SDPBiasOpNode* otherBias = NodeFactory::nodeCast<SDPBiasOpNode*>(otherNode);
    params().setX1Params(otherBias->params().x1Params());
    params().setBiasDims(otherBias->params().biasDims());
    params().setRawBiasData(otherBias->params().rawBiasData());
    params().setDLABiasData(otherBias->params().DLABiasData());
    params().setHasBiasReduction(otherBias->params().hasBiasReduction());
    params().setAxis(otherBias->params().axis());
    params().setConvMode(otherBias->params().convMode());
    params().setWinogradParams(otherBias->params().winogradParams());
    params().setNumGroups(otherBias->params().numGroups());
    params().setOutCVT(otherBias->params().outCVT());
}

std::vector<surface::SurfaceFormat> engine_ast::SDPBiasOpNode::suggestAuxSurfaceFormats(engine_ast::Edge* xEdge)
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

engine_ast::Node* engine_ast::SDPBiasOpNode::mergeWithSDPOp(SDPNode* nextSDP)
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

    // fixme: limit the bias math op fusion with only sdp-bias, sdp-batch-norm and relu types for now
    if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_BIAS)
    {
        removableNode = tryToMergeWithBiasOp(nextSDP);
    }
    if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_SCALE)
    {
        removableNode = tryToMergeWithScaleOp(nextSDP);
    }
    else if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_BATCH_NORM)
    {
        removableNode = tryToMergeWithBatchNormOp(nextSDP);
    }
    else if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_ACTIVATION)
    {
        removableNode = tryToMergeWithActOp(nextSDP);
    }

fail:
    return removableNode;
}


/*
 * 2 adjacent bias ops can be combined into a single op if the combined bias factors
 * don't over/undershoot the compute precision of the pipeline
 */
engine_ast::Node* engine_ast::SDPBiasOpNode::tryToMergeWithBiasOp(SDPNode* SDPBiasOp)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode       = NULL;
    SDPBiasOpNode* currBiasOp = this;
    SDPBiasOpNode* nextBiasOp = NodeFactory::nodeCast<SDPBiasOpNode*>(SDPBiasOp);
    Weights rawCurrBiasData   = currBiasOp->params().rawBiasData();
    Weights rawNextBiasData   = nextBiasOp->params().rawBiasData();
    Dims4 biasDims            = currBiasOp->params().biasDims();
    Weights combinedBiasData;
    WeightTrns wtTrns;
    NVDLA_UNUSED(e);

    nvdla::DataType modelPrec             = rawCurrBiasData.type == rawNextBiasData.type ?
                                            rawCurrBiasData.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();

    WeightTrns::WeightDims wtTrnsDims (rawCurrBiasData.count, biasDims.n, biasDims.c,
                                       biasDims.w, biasDims.h, 1, 1);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if ( currBiasOp->params().x1Params().mode().e() != nextBiasOp->params().x1Params().mode().e() ||
         currBiasOp->params().biasDims() != nextBiasOp->params().biasDims())
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't combine Bias factors of "
                        << currBiasOp->name() <<  " and " << nextBiasOp->name()
                        << " since they operate in different modes"<< endl;
        }
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedBiasData,
                                                     wtTrns.combineAdditionFactors,
                                                     currBiasOp->params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     rawCurrBiasData,
                                                     rawNextBiasData);
    if (combinedBiasData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Bias factors of "
                        << currBiasOp->name() << " and " << nextBiasOp->name() << endl;
        }
        goto fail;
    }

    /*
     * Since bias factors of 2 adjacent bias ops could be combined, inherit the combined
     * factors into the existing bias op and remove the next one
     */
    params().setRawBiasData(combinedBiasData);
    removableNode = nextBiasOp;

fail:
    return removableNode;
}

/*
 * A bias op and an adjacent scale op can be combined into a single batch-norm op
 * by treating the bias data as the mean and the scale data as the variance of BN
 */
engine_ast::Node* engine_ast::SDPBiasOpNode::tryToMergeWithScaleOp(SDPNode* SDPScaleOp)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode     = NULL;
    SDPScaleOpNode* scaleOp = NodeFactory::nodeCast<SDPScaleOpNode*>(SDPScaleOp);
    Weights rawBiasData     = params().rawBiasData();
    Weights rawScaleData    = scaleOp->params().rawScaleData();
    Dims4 commonDims        = scaleOp->params().scaleDims();
    Weights rawMeanData;
    Weights rawVarianceData;
    NVDLA_UNUSED(e);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if ( params().x1Params().mode().e() != scaleOp->params().x1Params().mode().e() ||
         params().biasDims() != scaleOp->params().scaleDims() ||
         params().x1Params().mode().e() == engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't combine Bias and Scale factors "
                        << "of " << scaleOp->name() <<  " and " << name()
                        << " which operate in different modes"<< endl;
        }
        goto fail;
    }

    /* replace the 2 operations with a brand new BN op in the engine AST.
     */
    {
        /* Step-1: Substitute Bias with BN node */
        NodeSequence substituteNodes;
        SDPBatchNormOpNode* newBNReplaceNode = NodeFactory::newSDPBatchNormOpNode(NULL, graph());
        if ( !newBNReplaceNode)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't create new BN op for replacing %s and %s",
                    name().c_str(), name().c_str());
        }
        newBNReplaceNode->params().x1Params().setMode(params().x1Params().mode());
        newBNReplaceNode->params().setEpsilon(0);
        newBNReplaceNode->params().setMeanDims(commonDims);
        newBNReplaceNode->params().setVarianceDims(commonDims);
        newBNReplaceNode->params().setBatchNormDims(commonDims);

        // activation from next SDP node
        newBNReplaceNode->params().x1Params().setActType(scaleOp->params().x1Params().actType());

        //  Orig set of operations: (bias + scale): y = (x + b) * s
        //  New operation: (batch-norm): y = (x + m)*v
        //                               mean: m
        //                               var: v
        newBNReplaceNode->params().setRawMeanData(rawBiasData);
        newBNReplaceNode->params().setRawVarianceData(rawScaleData);
        newBNReplaceNode->params().setDLABatchNormData(Weights(DataType::FLOAT, NULL, 0));
        PROPAGATE_ERROR_FAIL(newBNReplaceNode->captureCanonicalBatchNormData());

        substituteNodes.push_back(newBNReplaceNode);
        PROPAGATE_ERROR_FAIL(graph()->substituteNodeInAST(this, substituteNodes));

        /* Step-2: Remove the scale node by disconnecting it from the input side */
        PROPAGATE_ERROR_FAIL(graph()->removeNodeFromAST(scaleOp, IODirectionEnum::INPUT));

        /* since BN is replacing Bias + Scl, inherit the op mode from the Bias node before its removed*/
        newBNReplaceNode->params().setConvMode(params().convMode());
        newBNReplaceNode->params().setWinogradParams(params().winogradParams());

        /* Step-3: The compiler will eventually remove this bias node */
        removableNode = this;
    }

fail:
    return removableNode;
}

/*
 * Bias followed by BN can be folded into a single BN if the bias factors of the Bias Op
 * and mean factors of the BN Op can be combined into 1 without over/undershooting the
 * compute precision limits
 *
 * Bias + BN : y = ([x + b] + m)*(1/v)
 *               = (x + m')*(1/v)
 *               = BN
 *               => 2 ops folded into 1
 */
engine_ast::Node* engine_ast::SDPBiasOpNode::tryToMergeWithBatchNormOp(SDPNode* SDPBNOp)
{
    NvDlaError e = NvDlaSuccess;

    Node* removableNode = NULL;
    SDPBatchNormOpNode* bnOp = NodeFactory::nodeCast<SDPBatchNormOpNode*>(SDPBNOp);
    Weights rawBiasData   = params().rawBiasData();
    Weights rawMeanData   = bnOp->params().rawMeanData();
    Dims4 commonDims      = params().biasDims();
    Weights combinedBiasAndMeanData;
    WeightTrns wtTrns;
    NVDLA_UNUSED(e);

    nvdla::DataType modelPrec             = rawBiasData.type == rawMeanData.type ?
                                            rawBiasData.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();

    WeightTrns::WeightDims wtTrnsDims (rawBiasData.count, commonDims.n, commonDims.c,
                                       commonDims.w, commonDims.h, 1, 1);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if ( params().x1Params().mode().e() != bnOp->params().x1Params().mode().e() ||
         params().biasDims() != bnOp->params().meanDims())
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't combine Bias and Mean factors of "
                        << name() <<  " and " << bnOp->name()
                        << " since they operate in different modes"<< endl;
        }
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedBiasAndMeanData,
                                                     wtTrns.combineAdditionFactors,
                                                     params().x1Params().mode(),
                                                     wtTrnsDims,
                                                     rawBiasData,
                                                     rawMeanData);
    if (combinedBiasAndMeanData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Bias and Mean factors of "
                        << name() << " and " << bnOp->name() << endl;
        }
        goto fail;
    }

    /*
     * Since bias and mean factors of the Bias and BN ops could be combined, inherit the combined
     * factors into the existing BN op and remove the bias node
     */
    bnOp->params().setRawMeanData(combinedBiasAndMeanData);
    removableNode = this;

fail:
    return removableNode;
}

/* Configure SDP SuperOp SubEngine with Bios Op */
NvDlaError engine_ast::SDPBiasOpNode::configureSDPSuperOpSubEngine(SDPSuperOpNode* sdpSuperOp, SDPSubEngineType xN)
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

    sdpSuperOp->params().setAuxDataType(xN, TensorType::kBIAS);

    sdpSuperOp->params().setAdderDims(xN, params().biasDims());
    sdpSuperOp->params().setDLADataDims(xN, params().biasDims());

    sdpSuperOp->params().setRawAdderData(xN, params().rawBiasData());
    sdpSuperOp->params().setDLAData(xN, params().DLABiasData());

    sdpSuperOp->params().setAuxSurfaceFormats(xN, suggestAuxSurfaceFormats());

    if ( graph()->debugFuseSubEngineOps() )
    {
        gLogInfo << "configureSDPSuperOpSubEngine: " << this->name() << " in ";
        gLogInfo << sdpSuperOp->name() << " x" << (NvU16)xN.e()+1 << endl;
    }

    return e;
}

/*---------------------Quantize Bias Data------------------------------*/
NvDlaError engine_ast::SDPBiasOpNode::quantizeBiasToInt8
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    std::vector<NvF32>& inTensorScales
)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 numFilters           = filterScales.size();
    NvF32 perTensorInTensorScl = inTensorScales.at(0);

    bool isPerKernelQtz  = graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL;

    std::pair< NvS8, NvU8 > scaleAndShift;
    Weights int8BiasBlob;
    Weights origBiasBlob = params().rawBiasData();
    NvS8* pInt8Bias      = (NvS8*)std::malloc(origBiasBlob.count * sizeof(NvS8));
    NvU8 commonShift     = 0;
    NvF32 scaledFP32Bias = 0.0f;

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

    ASSERT( numFilters == origBiasBlob.count );
    for ( NvU32 bb = 0; bb < numFilters; ++bb)
    {
        switch(origBiasBlob.type)
        {
            case nvdla::DataType::FLOAT: {
                NvF32 fp32Bias = reinterpret_cast<NvF32*>(const_cast<void*>(origBiasBlob.values))[bb];
                scaledFP32Bias = NvF32(fp32Bias / (filterScales[bb] * perTensorInTensorScl));
            } break;
            case nvdla::DataType::HALF: {
                half_float::half fp16Bias = reinterpret_cast<half_float::half*>(const_cast<void*>(origBiasBlob.values))[bb];
                scaledFP32Bias = NvF32(fp16Bias / (filterScales[bb] * perTensorInTensorScl));
            } break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Can't quantize bias data which is not FLOAT / HALF "
                                        "precision for %s\n", name().c_str());
        }

        e = calculateScaleAndShiftFromScalar<NvS8, NvU8>(scaledFP32Bias, &scaleAndShift, commonShift);
        if (e != NvDlaSuccess)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, " Couldn't converge on `2^(x) * y` which could "
                                    "safely represent %f within acceptable tolerance for %s\n",
                                    scaledFP32Bias, name().c_str());
        }

        pInt8Bias[bb] = scaleAndShift.first;
        if (bb == 0)
            commonShift = scaleAndShift.second;
    }

    int8BiasBlob.type   = nvdla::DataType::INT8;
    int8BiasBlob.values = pInt8Bias;
    int8BiasBlob.count  = origBiasBlob.count;

    // set quantized bias and common left shifter for all biases
    params().setRawBiasData(int8BiasBlob);
    params().x1Params().setShiftValue(commonShift);

fail:
    return e;
}

static bool absCompare(NvS32 a, NvS32 b)
{
    return (std::abs(a) < std::abs(b));
}

NvDlaError engine_ast::SDPBiasOpNode::scaleBiasToInt16
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    std::vector<NvF32>& inTensorScales
)
{
    NvDlaError e = NvDlaSuccess;

    NvS32 maxBias;
    NvS32 numBits;
    NvU32 biasShift;
    std::vector< NvS32 > rescaledBias32;

    NvF32 perTensorInTensorScl = inTensorScales.at(0);

    bool isPerKernelQtz  = graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL;

    NvF32 scaledFP32Bias = 0.0f;
    NvS16 int16Bias      = 0;
    Weights origBiasBlob = params().rawBiasData();
    Weights int16BiasBlob;
    NvS16* pInt16Bias = (NvS16*)std::malloc(origBiasBlob.count * sizeof(NvS16));
    NvU32 numBiasData = origBiasBlob.count;

    if (fusedConv)
    {
        NvU32 numFilters = filterScales.size();
        ASSERT ( filterScales.size() == (size_t)fusedConv->outputEdges().at(0)->tensorSurfaceDesc()->dimensions().c );
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

    for ( NvU32 bb = 0; bb < numBiasData; ++bb)
    {
        switch (origBiasBlob.type)
        {
            case nvdla::DataType::FLOAT: {
                NvF32 fp32Bias = reinterpret_cast<NvF32*>(const_cast<void*>(origBiasBlob.values))[bb];
                NvF32 biasRescaleFactor = perTensorInTensorScl;
                if ( fusedConv )
                {
                    biasRescaleFactor = perTensorInTensorScl * filterScales[bb];
                }
                scaledFP32Bias = NvF32(fp32Bias / biasRescaleFactor);
                NvS32 int32Bias = static_cast<NvS32>(std::floor(scaledFP32Bias + 0.5f));
                rescaledBias32.push_back(int32Bias);
            } break;
            case nvdla::DataType::HALF: {
                NvF32 fp16Bias = reinterpret_cast<half_float::half*>(const_cast<void*>(origBiasBlob.values))[bb];
                NvF32 biasRescaleFactor = perTensorInTensorScl;
                if ( fusedConv )
                {
                    biasRescaleFactor = perTensorInTensorScl * filterScales[bb];
                }
                scaledFP32Bias = NvF32(fp16Bias / biasRescaleFactor);
                NvS32 int32Bias = static_cast<NvS32>(std::floor(scaledFP32Bias + 0.5f));
                rescaledBias32.push_back(int32Bias);
            } break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Can't scale bias data which is not FLOAT / HALF "
                                        "precision for %s\n", name().c_str());
        }
    }

    maxBias = *std::max_element(rescaledBias32.begin(), rescaledBias32.end(), absCompare);
    numBits = ceil(log(abs(maxBias))/log(2)) + 1;
    biasShift = std::min(SDP_LEFT_SHIFT_MAX_PLACES, std::max(0, numBits - 16));

    params().x1Params().setShiftValue(biasShift);

    for ( NvU32 bb = 0; bb < numBiasData; ++bb)
    {
	    int16Bias = static_cast<NvS16>(rescaledBias32[bb] >> biasShift);
        pInt16Bias[bb] = int16Bias;

        if ( graph()->debugQuantization() )
        {
            if (fusedConv)
            {
                gLogInfo << "rawBias/Si*Sw "
                         << reinterpret_cast<NvF32*>(const_cast<void*>(origBiasBlob.values))[bb] << " / "
                         << " ( " << perTensorInTensorScl << " * " << filterScales[bb] << " ) = "
                         << (reinterpret_cast<NvF32*>(const_cast<void*>(origBiasBlob.values))[bb]/(perTensorInTensorScl * filterScales[bb]))
                         << " -> " << rescaledBias32[bb] << " -> " << (int)int16Bias << "*2^-" << biasShift << endl;
            }
            else
            {
                gLogInfo << "rawBias/Si "
                         << reinterpret_cast<NvF32*>(const_cast<void*>(origBiasBlob.values))[bb] << " / "
                         << perTensorInTensorScl << " = "
                         << (reinterpret_cast<NvF32*>(const_cast<void*>(origBiasBlob.values))[bb]/perTensorInTensorScl)
                         << " -> " << rescaledBias32[bb] << " -> " << (int)int16Bias << "*2^-" << biasShift << endl;
            }
        }
    }

    int16BiasBlob.type   = nvdla::DataType::INT16;
    int16BiasBlob.values = pInt16Bias;
    int16BiasBlob.count  = origBiasBlob.count;

    // set scaled bias
    params().setRawBiasData(int16BiasBlob);

fail:
    return e;
}

/*--------------------Quantize Aux Data-----------------------------*/
NvDlaError engine_ast::SDPBiasOpNode::quantizeAuxData()
{
    NvDlaError e = NvDlaSuccess;

    std::vector<NvF32> filterScales;
    std::vector<NvF32> inTensorScales;

    ConvCoreNode* fusedConv = NULL;

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

    fusedConv = NodeFactory::nodeCast<ConvCoreNode*>(graph()->upstreamDataNodes(this).size() ?
                                                     graph()->upstreamDataNodes(this).at(0) :
                                                     NULL);
    if ( fusedConv )
    {
        PROPAGATE_ERROR_FAIL( fusedConv->verifyEdgePorts() );
        filterScales    = fusedConv->params().filterScales();
        inTensorScales  = fusedConv->inputEdges().at(0)->originalTensor()->getChannelScales();
    }
    else
    {
        inTensorScales  = inputEdges().at(0)->originalTensor()->getChannelScales();
    }

    /*
     * Convolution with bias op is represented as:
     *
     *      O  = I * W + b
     *    QoSo = QiSi * QwSw + b
     *      Qo = (Qi * Qw + b/Si*Sw) * (Si*Sw/So)
     *
     * Per-kernel equation looks like:
     *      Qo = (Qi * Qw + b[k]/Si*Sw) * (Si*Sw/So)
     * where-as per-filter equation looks like:
     *      Qo = (Qi * Qw + b[k]/Si*Sw[k]) * (Si*Sw[k]/So)
     *
     * For per-kernel scaled bias addition: (pure int8 path)
     *      - find per-output channel, scale and shift combination which approximately represents (b[k]/Si*Sw)
     *      - represented as scaled bias: `2^(ls) * m[k]`
     *      - where m[k] : per-filter quantized bias
     *      - ls : common left shift for scaling all (just quantized) biases
     *
     * For per-kernel scaled bias addition: (int16 path)
     *      - represent (b[k]/Si*Sw) as a int16 number
     *
     * Again per-filter equation looks like:
     *      Qo = (Qi * Qw + b[k]/Si*Sw[k]) * (Si*Sw[k]/So)
     *
     * For per-filter scaled bias addition: (pure int8 path)
     *      - find per-output channel, scale and shift combination which approximately represents (b[k]/Si*Sw[k])
     *      - represented as scaled bias: `2^(ls) * m[k]`
     *      - where m[k] : per-filter quantized bias
     *      - ls : common left shift for scaling all (just quantified) biases
     *
     * For per-filter scaled bias addition: (int16 path)
     *      - find per-output channel, scaled bias which represents fp32(b[k]/Si*Sw[k]) as a i16 value
     *      - scaled bias represented as : b'[k] in i16
     *      - feed the scaled bias b'[k] to the ALU unit of SDP-X
     */
    if ( graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL ||
         graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_FILTER )
    {
        if (auxEdges().at(0)->tensorSurfaceDesc()->surfaceFormat().f().precision().v() ==
            surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
        {
            PROPAGATE_ERROR_FAIL( quantizeBiasToInt8(fusedConv, filterScales, inTensorScales) );
        }
        else if (auxEdges().at(0)->tensorSurfaceDesc()->surfaceFormat().f().precision().v() ==
            surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16)
        {
            PROPAGATE_ERROR_FAIL( scaleBiasToInt16(fusedConv, filterScales, inTensorScales) );
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

/*------------------Low Precision Conversions--------------------------*/
NvDlaError engine_ast::SDPBiasOpNode::performPerKernelRescaling
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    std::vector<NvF32>& inTensorScales,
    std::vector<NvF32>& outTensorScales,
    PrecisionCVTParams& outCvt
)
{
    NvDlaError e = NvDlaSuccess;

    NvS16 perTensorScl  = 0;
    NvU8  perTensorShft = 0;
    NvF32 outputRescale = 0.0f;
    std::pair<NvS16, NvU8> scaleAndShift;

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

    outputRescale = perTensorInTensorScl / perTensorOutTensorScl;
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
    perTensorShft = scaleAndShift.second;

    if (graph()->debugQuantization())
    {
        if (fusedConv)
        {
            gLogInfo << name() << " Si * Sw / So = " << perTensorInTensorScl << " * " << perKernelScale << " / " << perTensorOutTensorScl
                << " = " << perTensorScl << "* 2^-" << (int)perTensorShft << endl;
        }
        else
        {
            gLogInfo << name() << " Si / So = " << perTensorInTensorScl << " / " << perTensorOutTensorScl
                << " = " << perTensorScl << "* 2^-" << (int)perTensorShft << endl;
        }
    }

    outCvt.setEnable(1);
    outCvt.setOffset(0);
    outCvt.setScale(perTensorScl);
    outCvt.setTruncate(perTensorShft);

fail:
    return e;
}

NvDlaError engine_ast::SDPBiasOpNode::performPerChannelRescaling
(
    ConvCoreNode* fusedConv,
    std::vector<NvF32>& filterScales,
    std::vector<NvF32>& inTensorScales,
    std::vector<NvF32>& outTensorScales,
    PrecisionCVTParams& outCvt
)
{
    NvDlaError e = NvDlaSuccess;

    NvF32 maxRescaleFactor = 0.0f;
    std::vector< NvF32 > outputRescales;
    std::pair< NvS16, NvU8 > maxRescaleFactorScaleAndShift;
    std::vector< std::pair<NvS16, NvU8> > scalesAndShifts;

    Weights rescaleWtsBlob;

    Weights rawBiasData = params().rawBiasData();
    NvU32 numBiasData = rawBiasData.count;
    NvF32 perTensorInTensorScl  = inTensorScales.at(0);
    NvF32 perTensorOutTensorScl = outTensorScales.at(0);

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

    for (NvU32 cc = 0; cc < numBiasData; ++cc)
    {
        NvF32 rescaleData = perTensorInTensorScl / perTensorOutTensorScl;
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
        rescaleWtsBlob.count = numBiasData;
        NvS16* pINT16Rescalars = reinterpret_cast<NvS16*>(std::malloc(numBiasData * sizeof(NvS16)));
        for (NvU32 cc = 0; cc < numBiasData; ++cc)
        {
            pINT16Rescalars[cc] = scalesAndShifts[cc].first;
            if (graph()->debugQuantization())
            {
                if (fusedConv)
                {
                     gLogInfo << name() << " Si * Sw[k] / So = " << perTensorInTensorScl << " * " << filterScales[cc] << " / " << perTensorOutTensorScl
                         << " = " << scalesAndShifts[cc].first << "* 2^-" << (int)maxRescaleFactorScaleAndShift.second << endl;
                }
                else
                {
                    gLogInfo << name() << " Si / So = " << perTensorInTensorScl << " / " << perTensorOutTensorScl
                         << " = " << scalesAndShifts[cc].first << "* 2^-" << (int)maxRescaleFactorScaleAndShift.second << endl;
                }
            }
        }
        rescaleWtsBlob.values = pINT16Rescalars;
        setRescaleData(rescaleWtsBlob);
        params().x1Params().setINT8Rescaling(true);
        params().x1Params().setOpType(SDPOpTypeEnum::SDP_OP_TYPE_BOTH);
    }


    outCvt.setEnable(1);
    outCvt.setOffset(0);
    outCvt.setScale(1);
    outCvt.setTruncate(0);

    // fixme: move the m_truncate to sdpengine params
    switch(engineOpType().v())
    {
        case EngineOpTypeEnum::SDP_BIAS:
            NodeFactory::nodeCast<SDPBiasOpNode*>(this)->params().x1Params().setTruncate(maxRescaleFactorScaleAndShift.second);
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support per-channel scaling in conjunction with %s\n", name().c_str());
    }

fail:
    return e;
}

NvDlaError engine_ast::SDPBiasOpNode::handleLowPrecisionConversions()
{
    NvDlaError e = NvDlaSuccess;

    PrecisionCVTParams outCvt;
    ConvCoreNode* fusedConv;

    std::vector<NvF32> inTensorScales;
    std::vector<NvF32> outTensorScales;
    std::vector<NvF32> filterScales;

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
        filterScales    = fusedConv->params().filterScales();
        inTensorScales  = fusedConv->inputEdges().at(0)->originalTensor()->getChannelScales();
    }
    else
    {
        inTensorScales  = inputEdges().at(0)->originalTensor()->getChannelScales();
    }

    outTensorScales = outputEdges().at(0)->originalTensor()->getChannelScales();

    /*
     * Convolution with bias op is represented as:
     *
     *      O  = I * W + b
     *    QoSo = QiSi * QwSw + b
     *      Qo = (Qi * Qw + b/Si*Sw) * (Si*Sw/So)
     *
     * Per-kernel equation looks like:
     *      Qo = (Qi * Qw + b[k]/Si*Sw) * (Si*Sw/So)
     * wehre-as per-filter equation looks like:
     *      Qo = (Qi * Qw + b[k]/Si*Sw[k]) * (Si*Sw[k]/So)
     *
     * For per-kernel rescaling:
     *      - find a scale and truncate combination which approximately represents (Si*Sw/So)
     *      - represented as `2^(-t) * s`
     *      - program this in the output CVT of SDP
     */
    if ( graph()->profile()->quantizationMode().v() == nvdla::QuantizationMode::PER_KERNEL )
    {
        PROPAGATE_ERROR_FAIL( performPerKernelRescaling(fusedConv,
                                                        filterScales,
                                                        inTensorScales,
                                                        outTensorScales,
                                                        outCvt) );
    }
    /*
     * Again per-filter equation looks like:
     *      Qo = (Qi * Qw + b[k]/Si*Sw[k]) * (Si*Sw[k]/So)
     *
     * For per-filter rescaling:
     *      - find per-output channel, scale and truncate combinations which approximately represent (Si*Sw[k]/So)
     *      - represented as `2^(-t) * s[k]`
     *      - where s[k] : per-channel scale represented as i16
     *      - t : common truncate that can safely represent all per-channel rescale factors
     *      - feed s[k] to the MUL unit of SDP-X and program 't' in the truncate operand
     */
    else if ( graph()->profile()->quantizationMode().v() == nvdla::QuantizationMode::PER_FILTER )
    {
        PROPAGATE_ERROR_FAIL( performPerChannelRescaling(fusedConv,
                                                         filterScales,
                                                         inTensorScales,
                                                         outTensorScales,
                                                         outCvt) );
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support quantization mode: %s\n",
                                graph()->profile()->quantizationMode().c_str());
    }

    params().setOutCVT(outCvt);

fail:
    return e;
}

/*----------------------Weight Translation ----------------------------*/
NvDlaError engine_ast::SDPBiasOpNode::translateAuxData()
{
    NvDlaError e = NvDlaError_Success;
    engine_ast::Edge* auxEdge;
    surface::SurfacePrecision computePrecision;
    surface::SurfacePrecision srcPrecision;
    NvU32 channelsPerGroup = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    auxEdge = auxEdges()[0];
    computePrecision = auxEdge->tensorSurfaceDesc()->surfaceFormat().f().precision();
    srcPrecision = inputEdges()[0]->tensorSurfaceDesc()->surfaceFormat().f().precision();

    {
        Weights trnsBiasData;
        Weights translateBiasData;
        Weights translateScaleData;
        Weights rawBiasData = params().rawBiasData();

        if (srcPrecision == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
        {
            channelsPerGroup = graph()->target_config()->atomicKSize();
        }

        if ( graph()->debugWeights() )
        {
            gLogInfo << "translating weights for " << this->id() << " bias-dims kcrs = " <<
                    auxEdge->tensorSurfaceDesc()->dimensions().n << "," <<
                    auxEdge->tensorSurfaceDesc()->dimensions().c << "," <<
                    auxEdge->tensorSurfaceDesc()->dimensions().h << "," <<
                    auxEdge->tensorSurfaceDesc()->dimensions().w << "" <<
                " and size= " << rawBiasData.count << endl;
        }

        WeightTrns::WeightDims biasDims (rawBiasData.count,
                                         auxEdge->tensorSurfaceDesc()->dimensions().n,
                                         auxEdge->tensorSurfaceDesc()->dimensions().c,
                                         auxEdge->tensorSurfaceDesc()->dimensions().w,
                                         auxEdge->tensorSurfaceDesc()->dimensions().h,
                                         1,   //FIXME: grab correct strides
                                         1);

        if (rawBiasData.count != auxEdge->tensorSurfaceDesc()->dimensions().n *
                                 auxEdge->tensorSurfaceDesc()->dimensions().c *
                                 auxEdge->tensorSurfaceDesc()->dimensions().h *
                                 auxEdge->tensorSurfaceDesc()->dimensions().w)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "bias dims dont match bias size");
        }

        if (params().x1Params().isINT8Rescaling())
        {
            // interlay bias data with rescale factors
            Weights rawRescaleData = rescaleData();
            ASSERT(rawBiasData.type == rawRescaleData.type);

            auxEdge->tensorSurfaceDesc()->setAlignLineStride(true);

            PRECISION_SWITCH(rawBiasData.type.v(), computePrecision.v(), translateBiasData, WeightTrns::translateDataForBias,
                                                                                       params().x1Params().mode(),
                                                                                       biasDims,
                                                                                       rawBiasData,
                                                                                       channelsPerGroup);

            PRECISION_SWITCH(rawRescaleData.type.v(), computePrecision.v(), translateScaleData, WeightTrns::translateDataForScale,
                                                                                       params().x1Params().mode(),
                                                                                       biasDims,
                                                                                       rawRescaleData,
                                                                                       channelsPerGroup);

            PRECISION_SWITCH(rawBiasData.type.v(), computePrecision.v(), trnsBiasData, WeightTrns::interlayDataBlobs,
                                                                                       translateBiasData,
                                                                                       translateScaleData);
        }
        else
        {
            // only translate bias data to dla friendly layout
            PRECISION_SWITCH(rawBiasData.type.v(), computePrecision.v(), trnsBiasData, WeightTrns::translateDataForBias,
                                                                                       params().x1Params().mode(),
                                                                                       biasDims,
                                                                                       rawBiasData,
                                                                                       channelsPerGroup);
        }

        if (trnsBiasData.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Bias Wt trnaslation failed for node '%s'", name().c_str());
        }

        params().setDLABiasData(trnsBiasData);
    }

fail:
    return e;
}

NvDlaError engine_ast::SDPBiasOpNode::emitOp(Graph *g,
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
    surface::TensorSurfaceDesc *bias_tsd = g->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    *sdp_op.srcPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, src_tsd->surfaceFormat().precision());
    *sdp_op.dstPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, dst_tsd->surfaceFormat().precision());
    *sdp_op.LUTIndex()       = -1;
    *sdp_op.batchNum()       = 1;
    *sdp_op.batchStride()    = 0;

    *out_cvt_acc.scale()     = params().outCVT().scale();
    *out_cvt_acc.truncate()  = params().outCVT().truncate();
    *out_cvt_acc.offset()    = params().outCVT().offset();
    *out_cvt_acc.enable()    = static_cast<NvU8>(params().outCVT().isEnable());

    *x1_op_acc.enable()      = 1;
    *x1_op_acc.ALUType()     = x1_op_acc.ALUType_Sum();
    *x1_op_acc.type()        = ASTToDLAInterface::getSDPOpType(target_dla, params(batch_id).x1Params().opType());
    *x1_op_acc.mode()        = ASTToDLAInterface::getSDPMode(target_dla, params(batch_id).x1Params().mode());
    *x1_op_acc.act()         = ASTToDLAInterface::getSDPActType(target_dla, params(batch_id).x1Params().actType());
    *x1_op_acc.shiftValue()  = params().x1Params().shiftValue();
    *x1_op_acc.truncate()    = params().x1Params().truncate();

    if (params(batch_id).x1Params().mode().e() == engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        Weights biasData = params().DLABiasData();

        if (biasData.count > 1)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                "More than one data available"
                                "in per-layer mode(#data = %u)",
                                biasData.count);
        }
        switch (biasData.type)
        {
            case DataType::HALF:
            case DataType::INT16:
                *x1_op_acc.ALUOperand() =
                    reinterpret_cast<const NvS16 *>(biasData.values)[SDP_ADDER_DATA_INDEX];
                *x1_op_acc.MulOperand() = params().x1Params().isINT8Rescaling() ?
                    reinterpret_cast<const NvS16 *>(biasData.values)[SDP_MULTIPLIER_DATA_INDEX] :
                    1;
                break;
            case DataType::INT8:
                *x1_op_acc.ALUOperand() =
                    reinterpret_cast<const NvS8 *>(biasData.values)[SDP_ADDER_DATA_INDEX];
                *x1_op_acc.MulOperand() = params().x1Params().isINT8Rescaling() ?
                    reinterpret_cast<const NvS8 *>(biasData.values)[SDP_MULTIPLIER_DATA_INDEX] :
                    1;
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                    "Unexpected data type %s",
                                    biasData.type.c_str());
        }
    }
    else
    {
        *x1_op_acc.ALUOperand() = 0;
        *x1_op_acc.MulOperand() = 1;
    }
    *x1_op_acc.precision() = *sdp_op.srcPrecision(); // precision of engine = precision of its input tensor
    *x1_op_acc.precision() = ASTToDLAInterface::getSDPPrecision(target_dla, bias_tsd->surfaceFormat().precision());

    *x2_op_acc.enable() = 0;
    *y_op_acc.enable()  = 0;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);
    if (params(batch_id).x1Params().mode().e() != engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER)
    {
        setDataCubeAccessor(x1_data_acc, bias_tsd, IODirectionEnum::UNKNOWN, batch_id);
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
        gLogInfo << "SDP bias node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
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
        gLogInfo << "\tbias tsd:" << bias_tsd->id() << endl;
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
        gLogInfo << "\tbias addr=" << *x1_data_acc.address() <<endl;
        gLogInfo << "\tbias type=" << (int)*x1_data_acc.type() << endl;
        gLogInfo << "\tbias size " << *x1_data_acc.size()    << endl;
        gLogInfo << "\tbias width " << *x1_data_acc.width()   << endl;
        gLogInfo << "\tbias height " << *x1_data_acc.height()   << endl;
        gLogInfo << "\tbias channel " << *x1_data_acc.channel()  << endl;
        gLogInfo << "\tbias linestride " << *x1_data_acc.lineStride() << endl;
        gLogInfo << "\tbias surfstride " << *x1_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
        gLogInfo << "\tdst type=" << (int)*dst_data_acc.type() << endl;
        gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
        gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
        gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
        gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
        gLogInfo << "\tout_cvt enable " << (int)*out_cvt_acc.enable() << endl;
        gLogInfo << "\tout_cvt scale " << (int)*out_cvt_acc.scale() << endl;
        gLogInfo << "\tout_cvt offset " << (int)*out_cvt_acc.offset() << endl;
        gLogInfo << "\tout_cvt truncate " << (int)*out_cvt_acc.truncate() << endl;
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

NvDlaError engine_ast::SDPBiasOpNode::emitOp(NvU32 op_slot, NvU32 batch_id,
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
    surface::TensorSurfaceDesc *bias_tsd    = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());


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

    protoSDPOpDesc->mutable_out_cvt()->set_enable(*out_cvt_acc.enable());
    protoSDPOpDesc->mutable_out_cvt()->set_offset(*out_cvt_acc.offset());
    protoSDPOpDesc->mutable_out_cvt()->set_scale(*out_cvt_acc.scale());
    protoSDPOpDesc->mutable_out_cvt()->set_truncate(*out_cvt_acc.truncate());

    protoSDPOpDesc->set_conv_mode(nvdla_prototest_interface::ConvMode::DIRECT);
    protoSDPOpDesc->set_batch_num(1);
    protoSDPOpDesc->set_batch_stride(0);

    protoSDPX1OpDesc->set_enable(*x1_op_acc.enable());
    protoSDPX1OpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPX1OpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
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
    protoX1DataCube->set_size(bias_tsd->tensorBufferDesc()->size());
    protoX1DataCube->set_width(*x1_data_acc.width());
    protoX1DataCube->set_height(*x1_data_acc.height());
    protoX1DataCube->set_channel(*x1_data_acc.channel());
    protoX1DataCube->set_line_stride(*x1_data_acc.lineStride());
    protoX1DataCube->set_surf_stride(*x1_data_acc.surfStride());
    protoX1DataCube->set_plane_stride(*x1_data_acc.planeStride());
    protoX1DataCube->mutable_mem_info()->set_mem_id(bias_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoX1DataCube->mutable_mem_info()->set_mem_size(bias_tsd->tensorBufferDesc()->size());
    protoX1DataCube->mutable_mem_info()->set_offset(bias_tsd->bufferOffset());
fail:
    return e;
}
#endif

};  // nvdla::priv::
};  // nvdla::
