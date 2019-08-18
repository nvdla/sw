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

#include <algorithm>
#include <iostream>
#include <math.h>   // ceilf

#include "priv/EngineAST.h"
#include "priv/Tensor.h"
#include "priv/Profile.h"
#include "priv/TargetConfig.h"
#include "priv/WeightTranslationUnit.h"
#include "ErrorMacros.h"

using std::endl;
using std::min;
using std::vector;

namespace nvdla
{
namespace priv
{

engine_ast::SDPNode* engine_ast::ConvCoreNode::addSDPJointOpNode
(
    canonical_ast::Node* origCanNode
)
{
    NvDlaError e = NvDlaSuccess;
    Tensor* streamTensor;
    engine_ast::SDPNode* sdpJointNode = NULL;
    canonical_ast::Graph* canGraph    = origCanNode->graph();

    if (origCanNode->params().hasBiasTerm())
    {
        sdpJointNode = engine_ast::NodeFactory::newSDPBiasOpNode(origCanNode, graph());
        engine_ast::NodeFactory::nodeCast<engine_ast::SDPBiasOpNode*>(sdpJointNode)->params().setHasBiasReduction(false);
    }
    else if (graph()->profile()->computePrecision().v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        Dims4 scaleDims(1,1,1,1);
        NvF32* procScaleBlob;
        Weights rawScaleData;
        SDPMode scaleMode;

        if ( graph()->profile()->quantizationMode().v() == nvdla::QuantizationMode::PER_KERNEL )
        {
            rawScaleData.count  = 1;
            scaleMode = engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER;
        }
        else if ( graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_FILTER )
        {
            Tensor* inTensor = NULL;
            inTensor = origCanNode->outputEdges().at(0)->originalTensor();
            scaleDims.c = inTensor->getDimensions().c;
            rawScaleData.count  = inTensor->getDimensions().c;
            scaleMode = engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL;
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support quantization mode: %s\n",
                                graph()->profile()->quantizationMode().c_str());
        }

        procScaleBlob = (NvF32*)engine_ast::MemoryCollector::getInstance()->allocateMemory(rawScaleData.count * sizeof(NvF32));
        memset(procScaleBlob, 0.0, rawScaleData.count * sizeof(NvF32));
        rawScaleData.values = procScaleBlob;
        rawScaleData.type   = nvdla::DataType::FLOAT;

        for (int i = 0; i < rawScaleData.count; i++)
        {
            procScaleBlob[i] = 1.0f;
        }

        // no canonical scale ancestor
        sdpJointNode = engine_ast::NodeFactory::newSDPScaleOpNode(NULL, graph());

        engine_ast::NodeFactory::nodeCast<engine_ast::SDPScaleOpNode*>(sdpJointNode)->params().x1Params().setMode(scaleMode);
        engine_ast::NodeFactory::nodeCast<engine_ast::SDPScaleOpNode*>(sdpJointNode)->params().setScaleDims(scaleDims);
        engine_ast::NodeFactory::nodeCast<engine_ast::SDPScaleOpNode*>(sdpJointNode)->params().setRawScaleData(rawScaleData);
        engine_ast::NodeFactory::nodeCast<engine_ast::SDPScaleOpNode*>(sdpJointNode)->params().setDLAScaleData(Weights(DataType::FLOAT, NULL, 0));

        PROPAGATE_ERROR_FAIL(engine_ast::NodeFactory::nodeCast<engine_ast::SDPScaleOpNode*>(sdpJointNode)->captureCanonicalScaleData());
    }
    else
    {
        sdpJointNode = engine_ast::NodeFactory::newSDPNOPNode(origCanNode, graph());
    }

    if ( !sdpJointNode )
    {
        goto fail;
    }

    // cache the dimensions of output tensor of parent node
    streamTensor = canGraph->downstreamEdges(origCanNode).at(0)->originalTensor()->clone();
    streamTensor->setTensorType(TensorType::kSTREAM);

    // connect either of the sdp nodes to the conv node, using an edge which
    // represents a tensor on wire, we call it a stream tensor, whose dims are same as
    // the orig o/p tensor of parent conv node, although no buffers would be reserved
    // for it during runtime
    graph()->addDataEdge((canonical_ast::Edge*)0, this, sdpJointNode, streamTensor);

fail:
    return sdpJointNode;
}

engine_ast::SDPNode* engine_ast::ConvCoreNode::addSDPJointOpNode
(
    SDPNode* copyFromSDP
)
{
    NvDlaError e = NvDlaSuccess;
    Edge* sdpInEdge;
    Tensor *streamTensor;
    engine_ast::SDPNode* sdpJointNode  = NULL;
    NVDLA_UNUSED(e);

    switch(copyFromSDP->engineOpType().v())
    {
        case EngineOpTypeEnum::SDP_ACTIVATION:
            sdpJointNode = NodeFactory::newSDPActivationOpNode(
                canonical_ast::NodeFactory::nodeCast<canonical_ast::ActivationNode*>(copyFromSDP->canonicalNode()),
                copyFromSDP->graph());
            NodeFactory::nodeCast<SDPActivationOpNode*>(sdpJointNode)->inheritParams(copyFromSDP);
            break;
        case EngineOpTypeEnum::SDP_BATCH_NORM:
            sdpJointNode = NodeFactory::newSDPBatchNormOpNode(
                canonical_ast::NodeFactory::nodeCast<canonical_ast::BatchNormNode*>(copyFromSDP->canonicalNode()),
                copyFromSDP->graph());
            NodeFactory::nodeCast<SDPBatchNormOpNode*>(sdpJointNode)->inheritParams(copyFromSDP);
            break;
        case EngineOpTypeEnum::SDP_BIAS:
            sdpJointNode = NodeFactory::newSDPBiasOpNode(copyFromSDP->canonicalNode(), copyFromSDP->graph());
            NodeFactory::nodeCast<SDPBiasOpNode*>(sdpJointNode)->inheritParams(copyFromSDP);
            break;
        case EngineOpTypeEnum::SDP_NOP:
            sdpJointNode = NodeFactory::newSDPNOPNode(copyFromSDP->canonicalNode(), copyFromSDP->graph());
            NodeFactory::nodeCast<SDPNOPNode*>(sdpJointNode)->inheritParams(copyFromSDP);
            break;
        case EngineOpTypeEnum::SDP_SCALE:
            sdpJointNode = NodeFactory::newSDPScaleOpNode(
                canonical_ast::NodeFactory::nodeCast<canonical_ast::ScaleNode*>(copyFromSDP->canonicalNode()),
                copyFromSDP->graph());
            NodeFactory::nodeCast<SDPScaleOpNode*>(sdpJointNode)->inheritParams(copyFromSDP);
            break;
        case EngineOpTypeEnum::SDP_SUPER:
            //FIXME
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support SDP node type %s as joint op of CONV %s",
                    copyFromSDP->engineOpType().c_str(), name().c_str());
            //engine_ast::Graph::printGraph(graph(), true, "addSDPJointOpNode");
            //sdpJointNode = NodeFactory::newSDPSuperOpNode(copyFromSDP->canonicalNode(), copyFromSDP->graph());
            //NodeFactory::nodeCast<SDPSuperOpNode*>(sdpJointNode)->inheritParams(copyFromSDP);
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support SDP node type %s as joint op of CONV %s",
                    copyFromSDP->engineOpType().c_str(), name().c_str());
    }

    if ( !sdpJointNode )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory);
    }

    copyFromSDP->nodeDataEdge(TensorType::kSTREAM, ast::EdgeSideEnum::SECOND, &sdpInEdge);
    streamTensor = sdpInEdge->originalTensor()->clone();
    streamTensor->setTensorType(TensorType::kSTREAM);

    graph()->addDataEdge((canonical_ast::Edge*)0, this, sdpJointNode, streamTensor);

fail:
    return sdpJointNode;
}


NvDlaError engine_ast::ConvCoreNode::nodeAuxEdge(engine_ast::Edge **ret_edge)
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(nodeDataEdge(TensorType::kWEIGHT, ast::EdgeSideEnum::SECOND, ret_edge));

fail:
    return e;
}

std::vector<surface::SurfaceFormat> engine_ast::ConvCoreNode::suggestAuxSurfaceFormats(engine_ast::Edge* auxEdge)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    // dumb suggestion logic for now
    surface::SurfaceFormat inputSF;
    surface::SurfaceCategory inputCategory;
    surface::SurfacePrecision compPrec = graph()->profile()->computePrecision();
    std::vector<surface::SurfaceFormat> supportedAuxSFs = supportedAuxSurfFormats();
    std::vector<surface::SurfaceFormat> suggestedAuxSFs;
    std::vector<surface::SurfaceFormat>::iterator auxSFIter;

    /* if input surface format is not yet registered, chose a default */
    inputSF = inputEdges()[0]->tensorSurfaceDesc() ?
              inputEdges()[0]->tensorSurfaceDesc()->surfaceFormat() :
              surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT;
    inputCategory = inputSF.v() != surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT ?
                    inputSF.category() :
                    surface::SurfaceCategoryEnum::FEATURE_DATA;

    if (supportedAuxSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported aux surface formats for %s", name().c_str());
    }
    else if (supportedAuxSFs.size() == 1)
    {
        suggestedAuxSFs = supportedAuxSFs;
        goto fail;
    }

    for (auxSFIter = supportedAuxSFs.begin(); auxSFIter != supportedAuxSFs.end(); ++auxSFIter)
    {
        if ((*auxSFIter).precision().v() != compPrec.v())
        {
            continue;
        }

        if (params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD)
        {
            if (std::string((*auxSFIter).c_str()).find("WG") == std::string::npos)
            {
                continue;
            }
            else
            {
                suggestedAuxSFs.push_back(*auxSFIter);
            }
        }
        else
        {
            if (inputCategory.v() == surface::SurfaceCategoryEnum::IMG &&
                    std::string((*auxSFIter).c_str()).find("IMG") == std::string::npos)
            {
                continue;
            }
            else if (inputCategory.v() == surface::SurfaceCategoryEnum::FEATURE_DATA &&
                    std::string((*auxSFIter).c_str()).find("DC") == std::string::npos)
            {
                continue;
            }
            else
            {
                suggestedAuxSFs.push_back(*auxSFIter);
            }
        }
    }


    if (suggestedAuxSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No suggested aux surface formats for node:%s", name().c_str());
    }

fail:
    return suggestedAuxSFs;
}

/*------------------Suggest Dims/Strides/Size/Buffer-Offset------------------*/
/*
 * DLA-CONV engine works on IMG and FEATURE_DATA input in either
 * DC or Winograd mode; whereas DC has no special stride/size alignment
 * requirements, Winograd mode needs perfect x(4x4) input and output
 * surfaces. This means tweaking input/output surface dims whenever
 * we want to chose WG.
 * But, code emission layer finally takes care of adjusting dims, the compiler
 * here should only accommodate for the larger stride/size requirements.
 */
Dims4 engine_ast::ConvCoreNode::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    bool isSrcTSD = false;
    bool isAuxTSD = false;
    bool isDstTSD = false;
    Dims4 suggestedDims(-1,-1,-1,-1);

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isAuxTSD = auxEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (isSrcTSD)
    {
        // src tensor for dc/wg has no special requirements;
        // inherit the suggested dims from upstream node if any
        Node* srcNode = graph()->upstreamNodes(inputEdges()[0]).size() ?
                        graph()->upstreamNodes(inputEdges()[0])[0] : NULL;
        if (srcNode)
        {
            suggestedDims = srcNode->suggestSurfaceDims(inputEdges()[0]->tensorSurfaceDesc());
        }
        else
        {
            suggestedDims = tsd->dimensions();
        }
    }
    else
    {
        if (params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD)
        {
            if (isAuxTSD)
            {
                // aux tensor for wg has special requirements
                suggestedDims = params().winogradParams().auxDims;
            }
            else if (isDstTSD)
            {
                // dst tensor for wg has special requirements
                suggestedDims = params().winogradParams().outDims;
            }
        }
        else if (params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT)
        {
            suggestedDims = tsd->dimensions();
        }
        else
        {
            REPORT_ERROR(NvDlaError_BadValue, "Unknown conv mode for %s", name().c_str());
        }
    }

fail:
    return suggestedDims;
}

NvU32 engine_ast::ConvCoreNode::suggestLineStride(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 lineStride = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDLineStride.find(tsd) != m_nodeTSDLineStride.end())
    {
        lineStride = m_nodeTSDLineStride[tsd];
        goto fail;
    }

    {
        surface::TensorSurfaceDesc probeTSD = *tsd;
        Dims4 surfDims = suggestSurfaceDims(tsd);
        probeTSD.setDimensions(surfDims);
        probeTSD.resetLineStride();
        lineStride = probeTSD.lineStride();
    }

    m_nodeTSDLineStride[tsd] = lineStride;

fail:
    return lineStride;
}

NvU32 engine_ast::ConvCoreNode::suggestSurfaceStride(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 surfaceStride = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDSurfaceStride.find(tsd) != m_nodeTSDSurfaceStride.end())
    {
        surfaceStride = m_nodeTSDSurfaceStride[tsd];
        goto fail;
    }

    {
        surface::TensorSurfaceDesc probeTSD = *tsd;
        Dims4 surfDims = suggestSurfaceDims(tsd);
        probeTSD.setDimensions(surfDims);
        probeTSD.resetSurfaceStride();
        surfaceStride = probeTSD.surfaceStride();
    }

    m_nodeTSDSurfaceStride[tsd] = surfaceStride;

fail:
    return surfaceStride;
}

NvU64 engine_ast::ConvCoreNode::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 size = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDSurfaceSize.find(tsd) != m_nodeTSDSurfaceSize.end())
    {
        size = m_nodeTSDSurfaceSize[tsd];
        goto fail;
    }

    {
        surface::TensorSurfaceDesc probeTSD = *tsd;
        Dims4 surfDims = suggestSurfaceDims(tsd);
        probeTSD.setDimensions(surfDims);
        probeTSD.resetSize();
        size = probeTSD.size();
    }

    m_nodeTSDSurfaceSize[tsd] = size;

fail:
    return size;
}

NvU64 engine_ast::ConvCoreNode::suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 offset = 0;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    bool isAuxTSD = false;
    Node* srcNode = NULL;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDSurfaceOffsetInBuffer.find(tsd) != m_nodeTSDSurfaceOffsetInBuffer.end())
    {
        offset = m_nodeTSDSurfaceOffsetInBuffer[tsd];
        goto fail;
    }

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;
    isAuxTSD = auxEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (isAuxTSD)
    {
        offset = 0;
    }
    else if (isSrcTSD)
    {
        srcNode = graph()->upstreamDataNodes(this).size() ? graph()->upstreamDataNodes(this)[0] : NULL;
        if (srcNode && srcNode->engineType().v() == EngineTypeEnum::SPLIT)
        {
            SplitNode* splitNode = NodeFactory::nodeCast<SplitNode*>(srcNode);
            // fixme: this kind of diving catches can be avoided if slice Ids are maintained inside tsd
            switch(splitNode->params().splitAxis().v())
            {
                case SplitAxisEnum::SPLIT_ALONG_W:
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support conv split-w mode for %s",
                            name().c_str());
                    break;
                case SplitAxisEnum::SPLIT_ALONG_H:
                    // in case of split along Height direction, use the calculated info
                    offset = splitDataInfo().inputBufferOffset;
                    break;
                case SplitAxisEnum::SPLIT_ALONG_C:
                    // in case of split along Chnl direction, let the split node decide suitable surface offset
                    offset = splitNode->suggestSurfaceOffsetInBuffer(tsd);
                    break;
                case SplitAxisEnum::SPLIT_ALONG_NONE:
                    // split-along-none is a pass through
                    offset = 0;
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown split type for %s",
                            splitNode->name().c_str());
            }
        }
        else
        {
            offset = 0;
        }
    }
    else if (isDstTSD)
    {
        offset = splitDataInfo().outputBufferOffset;
    }

    m_nodeTSDSurfaceOffsetInBuffer[tsd] = offset;

fail:
    return offset;
}

NvDlaError engine_ast::ConvCoreNode::verifySurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU16 bpe;
    Dims4 auxDims;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    bool isAuxTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    surface::TensorSurfaceDesc* auxTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;
    isAuxTSD = auxEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    bpe = tsd->surfaceFormat().bytesPerElement();
    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();
    auxTSD = isAuxTSD ? tsd : auxEdges()[0]->tensorSurfaceDesc();
    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    auxDims = suggestSurfaceDims(auxTSD);

    if (isSrcTSD || isAuxTSD)
    {
        if (params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD)
        {
            NvU32 wtCForWG = auxDims.c;
            if (((auxDims.c * bpe) % 32) != 0 )
            {
                wtCForWG = auxDims.c + (32 - ((auxDims.c * bpe) % 32))/bpe;
            }

            if (wtCForWG != (NvU32)params().winogradParams().auxDims.c)
            {
                PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) #Channels determind is wrong: %d != %d",
                                                        name().c_str(),
                                                        wtCForWG,
                                                        params().winogradParams().auxDims.c);
            }
            else if (wtCForWG != (NvU32)params().winogradParams().inDims.c)
            {
                PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Aux and Input tensors should have the same number of channels: %d != %d",
                                                        name().c_str(),
                                                        wtCForWG,
                                                        params().winogradParams().inDims.c);
            }
        }
        else if (params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT)
        {
            if (srcTSD->surfaceFormat().category().v() == surface::SurfaceCategoryEnum::FEATURE_DATA)
            {
                NvS32 auxChnls = params().numGroups() * auxTSD->dimensions().c;
                NvS32 dataChnls = srcTSD->dimensions().c;
                if  (auxChnls != dataChnls)
                {
                    PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Aux and Input tensors should have the same number of channels: %d != %d",
                                                        name().c_str(),
                                                            auxChnls,
                                                            dataChnls);
                }
            }
            /* Note that:
             *  for IMG input, aux_data is re-arranged with mandatory pre-chnl-extension so that
             *  (C_ext = C_orig * W_orig) and  (W_ext = 1)
             *  ...there's no way to get back C_orig from C_ext since W and H can be different (rectangular kernels)
             */
        }
        else
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown conv mode for %s", name().c_str());
        }
    }
    else
    {
        if  (auxTSD->dimensions().n != dstTSD->dimensions().c) {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Aux and Output tensors should have the same number of kernels: %d != %d",
                                                    name().c_str(),
                                                    auxTSD->dimensions().n,
                                                    dstTSD->dimensions().c);
        }
    }

fail:
    return e;
}

// uncomment for debugging
#if 0
static bool printKCRS(Weights kernel, Dims4 kernelDims, NvS32 kernelIdx)
{
    bool printAll = kernelIdx == -1;

    for (unsigned kk = (printAll ? 0 : kernelIdx); kk <= (unsigned)(printAll ? kernelDims.n - 1 : kernelIdx); ++kk)
    {
        for (unsigned cc = 0; cc < (unsigned)kernelDims.c; ++cc)
        {
            NvDlaDebugPrintf("[%d, %d]:\n",kk, cc);
            for (unsigned rr = 0; rr < (unsigned)kernelDims.h; ++rr)
            {
                for (unsigned ss = 0; ss < (unsigned)kernelDims.w; ++ss)
                {
                    if (kernel.type == nvdla::DataType::FLOAT)
                    {
                        float* pWts = reinterpret_cast<float*>(const_cast<void*>(kernel.values));
                        float val = pWts[ss +
                                        kernelDims.w*(rr +
                                        kernelDims.h*(cc +
                                        kernelDims.c*(kk)))];
                        NvDlaDebugPrintf("%.4f, ", val);
                    }
                    else if (kernel.type == nvdla::DataType::INT8)
                    {
                        NvS8* pWts = reinterpret_cast<NvS8*>(const_cast<void*>(kernel.values));
                        NvS8 val = pWts[ss +
                                       kernelDims.w*(rr +
                                       kernelDims.h*(cc +
                                       kernelDims.c*(kk)))];
                        NvDlaDebugPrintf("%4d, ", val);
                    }
                    else
                    {
                        NvDlaDebugPrintf("FIXME: add support to print %d\n", (int)kernel.type);
                    }
                }
                NvDlaDebugPrintf("\n");
            }
        }
    }
    return true;
}
#endif

NvDlaError engine_ast::ConvCoreNode::quantizeAuxData()
{
    NvDlaError e = NvDlaSuccess;

    Edge* auxEdge = NULL;
    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();
    Weights origWtsBlob = params().rawWeights();
    Weights quantizedWtsBlob;
    NvS8* quantizedWts = NULL;
    std::vector<NvF32> filterScales;

    NvU32 G = params().numGroups();
    NvU32 K = params().weightDims().n / G;
    NvU32 C = params().weightDims().c / G;
    NvU32 RS = params().weightDims().h * params().weightDims().w; // per-group values
    NvU32 kStride = C * RS;
    NvU32 cStride = RS;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    auxEdge = auxEdges().at(0);

    // quantize weights iff computing in low precision
    if (computePrecision.v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        // nop
        goto fail;
    }
    // if caffe weights are already int8, then there's no need for quantization
    else if (params().rawWeights().type == nvdla::DataType::INT8)
    {
        // nop
        goto fail;
    }
    // if surface precision for aux edge is not int8, then there's no need for quantization
    else if (auxEdge->tensorSurfaceDesc()->surfaceFormat().f().precision().v() !=
             surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        goto fail;
    }
    // if already quantized, return
    else if ( params().filterScales().size() )
    {
        // might be already quantized and rearranged with pre/post Chnl Ext for IMG Conv,
        // no more quantization needed
        goto fail;
    }

    // not yet support weight quantization for group convolutions
    if (params().numGroups() != 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support weight quantization for group convolutions yet for %s\n",
                                name().c_str());
    }

    quantizedWts = reinterpret_cast<NvS8*>(std::malloc(origWtsBlob.count * sizeof(NvS8)));

    if (graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_KERNEL)
    {
        PRECISION_SWITCH(origWtsBlob.type.v(), computePrecision.v(), filterScales, WeightTrns::perKernelQuantizeWts,
                                                                                   origWtsBlob,
                                                                                   G, K, C, RS, kStride, cStride,
                                                                                   quantizedWts);
        if ( filterScales.size() )
        {
            params().setFilterScales(filterScales);
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Something went wrong with wt quantization for %s\n",
                                    name().c_str());
        }
    }
    else if (graph()->profile()->quantizationMode() == nvdla::QuantizationMode::PER_FILTER)
    {
        PRECISION_SWITCH(origWtsBlob.type.v(), computePrecision.v(), filterScales, WeightTrns::perFilterQuantizeWts,
                                                                                   origWtsBlob,
                                                                                   G, K, C, RS, kStride, cStride,
                                                                                   quantizedWts);
        if ( filterScales.size() )
        {
            params().setFilterScales(filterScales);
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Something went wrong with wt quantization for %s\n",
                                    name().c_str());
        }
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Quantization mode %s not supported",
                                graph()->profile()->quantizationMode().c_str());
    }

    quantizedWtsBlob.type   = DataType::INT8;
    quantizedWtsBlob.values = quantizedWts;
    quantizedWtsBlob.count  = origWtsBlob.count;

    params().setRawWeights(quantizedWtsBlob);

fail:
    return e;
}

/*
 * Mandatory pre-channel extension for IMG Convolution
 */
NvDlaError engine_ast::ConvCoreNode::mandatoryChnlExtForIMG()
{
    NvDlaError e = NvDlaSuccess;

    Edge* auxEdge       = NULL;
    Edge* dataInputEdge = NULL;

    Dims4 rawWtDims;
    Dims4 preChnlExtDims;
    Dims4 zeroPadDims;
    Dims4 dataInputDims;
    Weights rawWts;
    Weights zeroPadWts;
    Weights preChnlExtWts;
    WeightTrns::WeightDims rawWtTrnsDims;
    WeightTrns::WeightDims zeroPadWtTrnsDims;
    WeightTrns::WeightDims preChnlExtWtTrnsDims;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    dataInputEdge = inputEdges().at(0);
    auxEdge       = auxEdges().at(0);

    rawWts          = params().rawWeights();
    dataInputDims   = dataInputEdge->tensorSurfaceDesc()->dimensions();
    rawWtDims       = auxEdge->tensorSurfaceDesc()->dimensions();
    rawWtTrnsDims   = WeightTrns::WeightDims(rawWts.count,
                                            rawWtDims.n, rawWtDims.c, rawWtDims.w, rawWtDims.h,
                                            (int)params().stride().w, (int)params().stride().h);

    // default initialization so that it doesn't fail in Chnl Extn stage if Zero-padding was not required
    zeroPadWts        = rawWts;
    zeroPadDims       = rawWtDims;
    zeroPadWtTrnsDims = rawWtTrnsDims;

    /* Step-1: Zero padding */
    if (rawWtDims.c != dataInputDims.c)
    {
        zeroPadWtTrnsDims = WeightTrns::WeightDims(rawWts.count * dataInputDims.c / rawWtDims.c,
                                                    rawWtDims.n, dataInputDims.c, rawWtDims.w, rawWtDims.h,
                                                    (int)params().stride().w, (int)params().stride().h);
        PRECISION_SWITCH(rawWts.type.v(), computePrecision.v(), zeroPadWts, WeightTrns::zeroPadWtsForIMG,
                                                                            rawWtTrnsDims,
                                                                            zeroPadWtTrnsDims,
                                                                            rawWts);
        if (zeroPadWts.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Zero padding failed for weights of node '%s'", name().c_str());
        }
        zeroPadDims = Dims4(zeroPadWtTrnsDims.numKernels, zeroPadWtTrnsDims.numChannels,
                            zeroPadWtTrnsDims.height, zeroPadWtTrnsDims.width);
        auxEdge->tensorSurfaceDesc()->setDimensions(zeroPadDims);

        // update raw weights to the zero padded weights
        params().setRawWeights(zeroPadWts);
    }

    /* Step-2: Mandatory pre-channel extension */
    preChnlExtWtTrnsDims = WeightTrns::WeightDims(zeroPadWts.count,
                                                  zeroPadDims.n, (zeroPadDims.c * zeroPadDims.w), 1, zeroPadDims.h,
                                                  (int)params().stride().w, (int)params().stride().h);
    PRECISION_SWITCH(rawWts.type.v(), computePrecision.v(), preChnlExtWts, WeightTrns::preChnlExtWtsForIMG,
                                                                            zeroPadWtTrnsDims,
                                                                            preChnlExtWtTrnsDims,
                                                                            zeroPadWts);

    if (preChnlExtWts.values == NULL)
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "IMG channel pre-extension failed for weights of node '%s'", name().c_str());
    }

    preChnlExtDims = Dims4(preChnlExtWtTrnsDims.numKernels, preChnlExtWtTrnsDims.numChannels,
                            preChnlExtWtTrnsDims.height, preChnlExtWtTrnsDims.width);
    auxEdge->tensorSurfaceDesc()->setDimensions(preChnlExtDims);

    // update raw weights to the pre-channel-extended weights
    params().setRawWeights(preChnlExtWts);

fail:
    return e;
}

/*
 * Optional post-channel extension for IMG convolution for higher MAC utilization
 */
NvDlaError engine_ast::ConvCoreNode::optionalChnlExtForIMG()
{
    NvDlaError e = NvDlaSuccess;

    Edge* auxEdge       = NULL;
    Edge* dataInputEdge = NULL;
    Dims4 preChnlExtDims;
    Dims4 dataInputDims;

    NvU32 postExtFactor     = 0;
    NvU32 preChnlExtWtChnls = 0;
    NvU32 zeroPaddedWtChnls = 0;
    NvU32 origWtWidth       = params().weightDims().w;
    NvU32 convXStride       = params().stride().w;

    Weights preChnlExtWts = params().rawWeights();
    Weights postChnlExtWts;

    WeightTrns::WeightDims preChnlExtWtTrnsDims;
    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();
    ConvCoreConvolutionOpNode* imgConvNode     = NodeFactory::nodeCast<ConvCoreConvolutionOpNode*>(this);
    NvU32 atomicCSize = graph()->target_config()->atomicCSize();
    NvU32 atomicKSize = graph()->target_config()->atomicKSize();
    NvU32 cbufWidth   = graph()->target_config()->bufEntryWidth();

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    dataInputEdge   = inputEdges().at(0);
    auxEdge         = auxEdges().at(0);

    dataInputDims   = dataInputEdge->tensorSurfaceDesc()->dimensions();
    preChnlExtDims  = auxEdge->tensorSurfaceDesc()->dimensions();

    preChnlExtWtChnls = preChnlExtDims.c;
    zeroPaddedWtChnls = dataInputDims.c;    // after pre-chnl extension, wts are padded with extra chnls to match data-in chnls

    if ( convXStride > origWtWidth )
    {
        gLogWarning << "Weight Post-Chnl Extension not possible when conv_x_stride (" << convXStride
                    << ") > orig_kernel_width (" << origWtWidth << ")" << endl;
        imgConvNode->params().setPostExtension(0);
        goto fail;
    }

    if ( (zeroPaddedWtChnls * convXStride * 3 + preChnlExtWtChnls) <= atomicCSize &&
            preChnlExtWtChnls <= atomicCSize/4 )
    {
        postExtFactor = 4;
    }
    else if ( (zeroPaddedWtChnls * convXStride + preChnlExtWtChnls) <= atomicCSize &&
                preChnlExtWtChnls <= atomicCSize/2 )
    {
        postExtFactor = 2;
    }
    else
    {
        postExtFactor = 0;
    }

    if ( postExtFactor != 0 )
    {
        bool postChnlExtWtsSuccess = false;
        preChnlExtWtTrnsDims = WeightTrns::WeightDims(preChnlExtWts.count,
                                                      preChnlExtDims.n,
                                                      preChnlExtDims.c,
                                                      1,
                                                      preChnlExtDims.h,
                                                      (int)params().stride().w, (int)params().stride().h);
        PRECISION_SWITCH(preChnlExtWts.type.v(), computePrecision.v(), postChnlExtWts, WeightTrns::postChnlExtWtsForIMG,
                                                                                       preChnlExtWtTrnsDims,
                                                                                       preChnlExtWts,
                                                                                       postExtFactor,
                                                                                       postChnlExtWtsSuccess,
                                                                                       atomicKSize,
                                                                                       atomicCSize,
                                                                                       cbufWidth);
        if (!postChnlExtWtsSuccess)
        {
            gLogWarning << "Unable to do IMG channel post-extension for weights of node '"
                        << name() << "', proceed without channel post-extension" << endl;
        }
        else
        {
            // update DLA weights to the post-channel-extended weights; no more remapping needed
            params().setDLAWeights(postChnlExtWts);
            imgConvNode->params().setPostExtension(postExtFactor);
        }
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::processWtsForIMG()
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL( mandatoryChnlExtForIMG() );

    if ( graph()->profile()->canIMGPostChnlExtend() )
    {
        PROPAGATE_ERROR_FAIL( optionalChnlExtForIMG() );
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::squashWeightGroups()
{
    NvDlaError e = NvDlaSuccess;

    // group conv is implemented by splitting kernel groups into separate kernels
    // change kernel channel size to input data channel size
    // change num of kernels to num of groups
    NvU32 numGroups = params().numGroups();

    bool isIMGConv      = false;
    Edge* dataInputEdge = NULL;
    Edge* auxEdge       = NULL;

    Weights origWts;
    Weights groupedWts;
    WeightTrns::WeightDims origWtTrnsDims;
    WeightTrns::WeightDims groupedWtDims;

    Dims4 origWtDims;
    Dims4 groupedDims;
    Dims4 dataInputDims;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    dataInputEdge = inputEdges()[0];
    auxEdge       = auxEdges()[0];
    dataInputDims = dataInputEdge->tensorSurfaceDesc()->dimensions();
    isIMGConv     = dataInputEdge->tensorSurfaceDesc()->surfaceFormat().category() == surface::SurfaceCategoryEnum::IMG;

    origWts = params().rawWeights();
    origWtDims = auxEdge->tensorSurfaceDesc()->dimensions();
    origWtTrnsDims = WeightTrns::WeightDims(origWts.count, origWtDims.n, origWtDims.c, origWtDims.w, origWtDims.h,
                                        (int)params().stride().w, (int)params().stride().h);

    if (isIMGConv)
    {
        /*
            * For image convolution, DLA supports channel size of 3 or 4, if actual
            * input channel size is 2 or 3 then application will pad 0s to make
            * channel size as 4. But number of groups are not updated, it requires
            * updating number of groups too.
            *
            * 1. num channels 1, remains channel 1, not group conv NA
            * 2. num channels 2, extended to channels 4
            *    - case 1 : num channels for weight = 1, no issue here
            *    - case 2 : num channels for weight = 2, not group conv, NA
            * 3. num channels 3, extended to channels 4
            *    - case 1 : num channels for weight = 1, no issue here
            *    - case 2 : num channels for weight = 2, invalid case, neither group conv
            *    - case 3 : num channels for weight = 3, not group conv, NA
            * 4. num channels 4, no extension
            *    - case 1 : num channels for weight = 1, no issue here
            *    - case 2 : num channels for weight = 2, no issue here
            *    - case 3 : num channels for weight = 3, invalid case, neither group conv
            *    - case 4 : num channels for weight = 4, not group conv, NA
            */
        if ((dataInputDims.c % origWtDims.c) != 0)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Input channel size is not divisible by kernel channel size");
        }
        numGroups = dataInputDims.c / origWtDims.c;
    }

    groupedWtDims = WeightTrns::WeightDims(origWts.count * numGroups,
                                            origWtDims.n, dataInputDims.c, origWtDims.w, origWtDims.h,
                                            (int)params().stride().w, (int)params().stride().h);

    PRECISION_SWITCH(origWts.type.v(), computePrecision.v(), groupedWts, WeightTrns::padGroupWeights,
                                                                            origWtTrnsDims,
                                                                            groupedWtDims,
                                                                            origWts,
                                                                            params().numGroups());

    if (groupedWts.values == NULL)
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Splitting weights failed for weights of node '%s'", name().c_str());
    }

    groupedDims = Dims4(groupedWtDims.numKernels, groupedWtDims.numChannels,
                            groupedWtDims.height, groupedWtDims.width);
    auxEdge->tensorSurfaceDesc()->setDimensions(groupedDims);

    params().setRawWeights(groupedWts);
    params().setNumGroups(1);

fail:
    return e;
}

/*--------------------Pre-process Weight Data--------------------------*/
/* Weight data for IMG convolution needs mandatory channel pre-extension
 * and optional channel post-extension.
 */
NvDlaError engine_ast::ConvCoreNode::preProcessAuxData()
{
    NvDlaError e = NvDlaSuccess;

    bool isIMGConv      = false;
    Edge* dataInputEdge = NULL;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    dataInputEdge = inputEdges()[0];
    isIMGConv     = dataInputEdge->tensorSurfaceDesc()->surfaceFormat().category() == surface::SurfaceCategoryEnum::IMG;

    // dilation is not possible with IMG convolution
    if (isIMGConv && (params().dilation().h > 1 || params().dilation().w > 1))
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_NotSupported, "Dilation with image convolution is not supported\n");
    }

    // quantize weights if img Conv, since int8 weights have to undergo
    // pre and/or post Chnl Ext for IMG Conv
    if (isIMGConv)
    {
        PROPAGATE_ERROR_FAIL( quantizeAuxData() );
    }

    // pre-process weights for grouped DC/WG/Deconv convolutions
    if (params().numGroups() != 1)
    {
        PROPAGATE_ERROR_FAIL( squashWeightGroups() );

    }

    // pre-process weights for IMG convolution
    if (isIMGConv)
    {
           PROPAGATE_ERROR_FAIL( processWtsForIMG() );
    }


fail:
    return e;
}

/*----------------------Combine Similar Math Ops-----------------------*/
engine_ast::Node* engine_ast::ConvCoreNode::mergeWithSDPOp(SDPNode* nextSDP)
{
    Node* removableNode = NULL;

    if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_SCALE)
    {
        removableNode = tryToMergeWithScaleOp(nextSDP);
    }
    else if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_BATCH_NORM)
    {
        removableNode = tryToMergeWithBatchNormOp(nextSDP);
    }

    return removableNode;
}

engine_ast::Node* engine_ast::ConvCoreNode::tryToMergeWithScaleOp(SDPNode* SDPSclOp)
{
    Node* removableNode = NULL;
    SDPScaleOpNode* scaleOp = NodeFactory::nodeCast<SDPScaleOpNode*>(SDPSclOp);
    Weights rawKrnlWts = params().rawWeights();
    Weights rawSclData = scaleOp->params().rawScaleData();
    Dims4 krnlWtDims   = params().weightDims();
    Dims4 sclDims      = scaleOp->params().scaleDims();
    SDPMode sclMode    = scaleOp->params().x1Params().mode();
    Weights combinedWtsAndScaleData;
    WeightTrns wtTrns;

    nvdla::DataType modelPrec             = rawKrnlWts.type == rawSclData.type ?
                                            rawKrnlWts.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();

    NodeSequence scaleOpDownNodes;
    NodeSequenceIterator dni;
    NodeWithSameEngineType match_eng_type(EngineTypeEnum::SDP);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    // if there's no more SDP operations following the Scale Op, avoid the fusion since
    // the Conv Op will need an SDP write out proxy
    scaleOpDownNodes = graph()->downstreamDataNodes(scaleOp);
    dni = std::find_if(scaleOpDownNodes.begin(), scaleOpDownNodes.end(), match_eng_type);
    if (dni == scaleOpDownNodes.end())
    {
        goto fail;
    }
    // if there's a ReLU op after the scale, don't allow the conv+scale fusion since
    // the scale is a proxy node to perform the int8 rescaling before the relu op
    else if (graph()->profile()->computePrecision() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 &&
             scaleOpDownNodes.size() &&
             scaleOpDownNodes.at(0)->engineOpType() == engine_ast::EngineOpTypeEnum::SDP_ACTIVATION)
    {
        goto fail;
    }
    // if there's a ELTwise op after the scale, don't allow the conv+scale fusion since
    // the scale is a proxy node to perform the int8 rescaling before the EW op. since
    // the thread with which EW x2 fusion might happen is unknown, avoid the conv+scale fusion on all threads
    else if (graph()->profile()->computePrecision() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 &&
             scaleOpDownNodes.size() &&
             scaleOpDownNodes.at(0)->engineOpType() == engine_ast::EngineOpTypeEnum::SDP_ELEMENTWISE)
    {
        goto fail;
    }

    // xxx: skip if the dla weights are already arranged. plugging scale factors will be difficult now
    if ( params().DLAWeights().values != NULL )
    {
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedWtsAndScaleData,
                                                     wtTrns.combineKernelWeightsAndScaleData,
                                                     sclMode,
                                                     krnlWtDims,
                                                     sclDims,
                                                     rawKrnlWts,
                                                     rawSclData);
    if (combinedWtsAndScaleData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Kernel weights and Scale factors of "
                        << name() << " and " << scaleOp->name() << endl;
        }
        goto fail;
    }

    params().setRawWeights(combinedWtsAndScaleData);
    removableNode = scaleOp;

fail:
    return removableNode;
}

engine_ast::Node* engine_ast::ConvCoreNode::tryToMergeWithBatchNormOp(SDPNode* SDPBnOp)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode = NULL;
    SDPBatchNormOpNode* bnOp = NodeFactory::nodeCast<SDPBatchNormOpNode*>(SDPBnOp);
    Weights rawKrnlWts = params().rawWeights();
    Dims4 krnlWtDims   = params().weightDims();
    Weights rawMeanData = bnOp->params().rawMeanData();
    Weights rawVarData = bnOp->params().rawVarianceData();
    Dims4 bnDims      = bnOp->params().batchNormDims();
    SDPMode bnMode    = bnOp->params().x1Params().mode();
    SDPActType bnAct    = bnOp->params().x1Params().actType();
    Weights combinedWtsAndVarianceData;
    Weights combinedMeanAndVarData;
    WeightTrns wtTrns;
    NVDLA_UNUSED(e);

    nvdla::DataType modelPrec             = rawKrnlWts.type == rawVarData.type ?
                                            rawKrnlWts.type :
                                            nvdla::DataType::UNKNOWN;
    surface::SurfacePrecision computePrec = graph()->profile()->computePrecision();
    WeightTrns::WeightDims wtTrnsDims (rawVarData.count, bnDims.n, bnDims.c, bnDims.w, bnDims.h, 1, 1);

    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    // xxx: skip if the dla weights are already arranged. plugging BN factors will be difficult now
    if ( params().DLAWeights().values != NULL )
    {
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedWtsAndVarianceData,
                                                     wtTrns.combineKernelWeightsAndScaleData,
                                                     bnMode,
                                                     krnlWtDims,
                                                     bnDims,
                                                     rawKrnlWts,
                                                     rawVarData);
    if (combinedWtsAndVarianceData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Kernel weights and Scale factors of "
                        << name() << " and " << bnOp->name() << endl;
        }
        goto fail;
    }

    PRECISION_SWITCH(modelPrec.v(), computePrec.v(), combinedMeanAndVarData,
                                                     wtTrns.combineMultiplicationFactors,
                                                     bnMode,
                                                     wtTrnsDims,
                                                     rawMeanData,
                                                     rawVarData);

    if (combinedMeanAndVarData.values == NULL)
    {
        if ( debugFactorization() )
        {
            gLogWarning << "Can't successfully combine Variance and Scale factors of "
                        << name() << " and " << bnOp->name() << endl;
        }
        goto fail;
    }

    {
        NodeSequence substituteNodes;
        SDPBiasOpNode* newBiasReplaceNode = NULL;

        /* Step-1: Substitute BN with Bias node */
        newBiasReplaceNode = NodeFactory::newSDPBiasOpNode(NULL, graph());
        if ( !newBiasReplaceNode)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't create new Bias op for replacing %s",
                    bnOp->name().c_str());
        }
        newBiasReplaceNode->params().x1Params().setMode(bnMode);
        newBiasReplaceNode->params().x1Params().setActType(bnAct);
        newBiasReplaceNode->params().setBiasDims(bnDims);
        newBiasReplaceNode->params().setRawBiasData(combinedMeanAndVarData);
        newBiasReplaceNode->params().setDLABiasData(Weights(DataType::FLOAT, NULL, 0));
        PROPAGATE_ERROR_FAIL(newBiasReplaceNode->captureCanonicalBiasData());

        /* Step-2: Transfer the (w/v) factors to the existing Conv node */
        params().setRawWeights(combinedWtsAndVarianceData);

        /* Step-3: Substitute the BN node with Bias node */
        substituteNodes.push_back(newBiasReplaceNode);
        PROPAGATE_ERROR_FAIL(graph()->substituteNodeInAST(bnOp, substituteNodes));

        /* Step-4: Since Bias is replacing BN, inherit the op mode from the BN node before its removed*/
        newBiasReplaceNode->params().setConvMode(bnOp->params().convMode());
        newBiasReplaceNode->params().setWinogradParams(bnOp->params().winogradParams());

        /* Step-5: Finally remove the BN node */
        removableNode = bnOp;

        if ( graph()->debugMathOptz() )
        {
            gLogInfo << "Replace " << bnOp->name() << " with " << newBiasReplaceNode->name() << endl;
        }
    }

fail:
    return removableNode;
}

/*------------------Low Precision Conversions--------------------------*/
NvDlaError engine_ast::ConvCoreNode::handleLowPrecisionConversions()
{
    NvDlaError e = NvDlaSuccess;
    ConvCoreCVTParams convCvt;
    PrecisionCVTParams inCvt;

    if ( graph()->profile()->computePrecision().v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 )
    {
        // nop
        goto fail;
    }
    else if ( graph()->profile()->tensorScalingMode().v() != nvdla::TensorScalingMode::PER_TENSOR )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support tensor scaling mode: %s\n",
                                graph()->profile()->tensorScalingMode().c_str());
    }
    else if ( graph()->profile()->quantizationMode().v() != nvdla::QuantizationMode::PER_KERNEL &&
              graph()->profile()->quantizationMode().v() != nvdla::QuantizationMode::PER_FILTER )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support quantization mode: %s\n",
                                graph()->profile()->quantizationMode().c_str());
    }

    inCvt.setEnable(1);
    inCvt.setScale(1);
    inCvt.setOffset(0);
    inCvt.setTruncate(0);

    convCvt.setInputCVT(inCvt);
    convCvt.setOutTruncate(0);
    convCvt.setPraTruncate(0);

    params().setConvCoreCVT(convCvt);

fail:
    return e;
}

/*----------------------Weight Translation ----------------------------*/
NvDlaError engine_ast::ConvCoreNode::translateAuxData()
{
    NvDlaError e = NvDlaSuccess;

    bool isIMGConv      = false;
    Edge* dataInputEdge = NULL;
    Edge* auxEdge       = NULL;
    surface::SurfacePrecision computePrecision;
    NvU32 atomicCSize = 0;
    NvU32 atomicKSize = 0;
    NvU32 cbufWidth   = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    computePrecision  = graph()->profile()->computePrecision();
    atomicCSize = graph()->target_config()->atomicCSize();
    atomicKSize = graph()->target_config()->atomicKSize();
    cbufWidth   = graph()->target_config()->bufEntryWidth();

    dataInputEdge = inputEdges()[0];
    auxEdge       = auxEdges()[0];

    isIMGConv = dataInputEdge->tensorSurfaceDesc()->surfaceFormat().category() == surface::SurfaceCategoryEnum::IMG;
    if (isIMGConv && NodeFactory::nodeCast<ConvCoreConvolutionOpNode*>(this)->params().postExtension() > 0)
    {
        // when post chnl extension is done for IMG conv, weights are already remapped. No need to remap them again
        ASSERT(params().DLAWeights().values != NULL);
        goto fail;
    }

    {
        Weights trnsKrnlWts;
        Weights rawKrnlWts = params().rawWeights();

        if ( graph()->debugWeights() )
        {
            gLogInfo << "translating weights for " << id() << " kernel-dims kcrs = " <<
                                    auxEdge->tensorSurfaceDesc()->dimensions().n << "," <<
                                    auxEdge->tensorSurfaceDesc()->dimensions().c << "," <<
                                    auxEdge->tensorSurfaceDesc()->dimensions().h << "," <<
                                    auxEdge->tensorSurfaceDesc()->dimensions().w << "" <<
                                    " and size= " << rawKrnlWts.count << endl;
        }


        WeightTrns::WeightDims kernelDims (rawKrnlWts.count,
                                           auxEdge->tensorSurfaceDesc()->dimensions().n,
                                           auxEdge->tensorSurfaceDesc()->dimensions().c,
                                           auxEdge->tensorSurfaceDesc()->dimensions().w,
                                           auxEdge->tensorSurfaceDesc()->dimensions().h,
                                           (int)params().stride().w,
                                           (int)params().stride().h);
        if (rawKrnlWts.count != (auxEdge->tensorSurfaceDesc()->dimensions().n *
                                 auxEdge->tensorSurfaceDesc()->dimensions().c *
                                 auxEdge->tensorSurfaceDesc()->dimensions().h *
                                 auxEdge->tensorSurfaceDesc()->dimensions().w))
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "kernel dims dont match kernel size ");
        }

        if ( params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD )
        {
            PRECISION_SWITCH(rawKrnlWts.type.v(), computePrecision.v(), trnsKrnlWts, WeightTrns::translateWtsForWG,
                                                                                     kernelDims,
                                                                                     rawKrnlWts);
        }
        else if ( params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT )
        {
            PRECISION_SWITCH(rawKrnlWts.type.v(), computePrecision.v(), trnsKrnlWts, WeightTrns::translateWtsForDC,
                                                                                     kernelDims,
                                                                                     rawKrnlWts,
                                                                                     atomicKSize,
                                                                                     atomicCSize,
                                                                                     cbufWidth);
        }
        else
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown Conv mode : %s for %s",
                    params().convMode().c_str(), name().c_str());
        }

        if (trnsKrnlWts.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Kernel Wt trnaslation failed for node '%s'", name().c_str());
        }

        params().setDLAWeights(trnsKrnlWts);
    }


fail:
    return e;
}

/*--------------------------------Fuse Nodes---------------------------*/
/*
 *  Conv node doesn't have output port, it must have an sdp node downstream,
 *  else something went wrong in previous steps
 */
NvDlaError engine_ast::ConvCoreNode::fuseOnTheFlyNodes()
{
    NvDlaError e = NvDlaSuccess;

    engine_ast::Graph::NodeSequence consumer_nodes;
    EdgeSequence output_edges = graph()->downstreamEdges(this);
    for (EdgeSequenceIterator oei = output_edges.begin(); oei != output_edges.end(); ++oei)
    {
        consumer_nodes = graph()->downstreamNodes(*oei);
        for (NodeSequence::const_iterator cni = consumer_nodes.begin(); cni != consumer_nodes.end(); ++cni)
        {
            if ((*cni)->engineType().v() == EngineTypeEnum::SDP)
            {
                dependencyParams().setFusedNode(IODirectionEnum::OUTPUT, *cni);
                (*cni)->dependencyParams().setFusedNode(IODirectionEnum::INPUT, this);
            }
        }
    }
    if (dependencyParams().fusedNode(IODirectionEnum::OUTPUT) == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Conv node didn't find a fusable downstream SDP node");
    }
fail:
    return e;
}

/* FIXME: this API currently supports only partial-H split of data cube.
 * partial-W and partial-C split would follow in future
 */
NvDlaError engine_ast::ConvCoreNode::determineSplitDataRatios(NvU16& avlbDataBanks, std::vector<ConvCoreNode::SplitDataInfo>& splitChunks)
{
    NvDlaError e = NvDlaSuccess;
    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *weightTSD  = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    Dims4 weightDims = weightTSD->dimensions();
    Dims4 srcDims    = srcTSD->dimensions();
    Dims4 dstDims    = dstTSD->dimensions();

    bool isWinograd = false;
    bool isIMGConv  = false;
    const NvU8 FIRST_PARTIAL_H_SEG        = 0;
    const NvU8 INTERMEDIATE_PARTIAL_H_SEG = 1;
    const NvU8 LAST_PARTIAL_H_SEG         = 2;

    NvU32 entriesPerSlice  = 0;
    NvU32 srcSlicesInCBuff = 0;
    NvU32 totalCBuffEntriesAvlb = 0;
    NvS32 dilatedWeightH = 0;
    NvS32 dilatedWeightW = 0;
    NvU32 srcHeights[3] = {0};
    NvU32 avlbCBuffEntries[3] = {0};
    NvU32 outputHeightProcessed = 0;
    NvU32 wtBanksReserved = graph()->target_config()->bufBankAllotted() - avlbDataBanks;
    surface::SurfaceCategory srcSC = srcTSD->surfaceFormat().category();

    entriesPerSlice       = calculateEPS(srcTSD);

    totalCBuffEntriesAvlb = avlbDataBanks * (graph()->target_config()->bufEntriesPerBank());
    isIMGConv = srcSC.v() == surface::SurfaceCategoryEnum::IMG;

    if (srcSC.v() == surface::SurfaceCategoryEnum::IMG ||
        (srcSC.v() == surface::SurfaceCategoryEnum::FEATURE_DATA && params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT))
    {
        isWinograd = false;
    }
    else if (params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD)
    {
        isWinograd = true;
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Cant recognize conv mode %s", params().convMode().c_str());
    }

    if (isWinograd)
    {
        /* WG conv stores TP and BP padding values in CBUFF,
         * so avlb CBuff entries for data is lesser
         */
        srcSlicesInCBuff = srcDims.h + params().topLeftPadding().h + params().bottomRightPadding().h;
    }
    else
    {
        /* IMG and DC conv with feature data input dont store TP and BP padding values in CBUFF,
         * so all avlb CBuff entries are for data only
         */
        srcSlicesInCBuff = srcDims.h;
    }
    std::fill_n(avlbCBuffEntries, 3, totalCBuffEntriesAvlb);

    if (isWinograd)
    {
        bool isMinInPHSatisfied = false;
        do {
            totalCBuffEntriesAvlb = avlbDataBanks * (graph()->target_config()->bufEntriesPerBank());
            std::fill_n(avlbCBuffEntries, 3, totalCBuffEntriesAvlb);
            srcHeights[FIRST_PARTIAL_H_SEG]        = ROUNDDOWN_AND_ALIGN(avlbCBuffEntries[FIRST_PARTIAL_H_SEG] / entriesPerSlice, 4);
            srcHeights[INTERMEDIATE_PARTIAL_H_SEG] = ROUNDDOWN_AND_ALIGN(avlbCBuffEntries[INTERMEDIATE_PARTIAL_H_SEG] / entriesPerSlice, 4);
            srcHeights[LAST_PARTIAL_H_SEG]         = ROUNDDOWN_AND_ALIGN(avlbCBuffEntries[LAST_PARTIAL_H_SEG] / entriesPerSlice, 4);

            if (srcHeights[FIRST_PARTIAL_H_SEG] <= 4 ||
                srcHeights[INTERMEDIATE_PARTIAL_H_SEG] <= 4 ||
                srcHeights[LAST_PARTIAL_H_SEG] <= 4)
            {
                gLogWarning << "Input partial-H for WG should be atleast >4 for " <<  name() << endl;
                NvU32 minWtBanksNeeded = calculateMinBanksForWeight(weightTSD);
                if (2*minWtBanksNeeded < wtBanksReserved)
                {
                    gLogWarning << "Downsizing weight banks from " << wtBanksReserved << " to " << (2*minWtBanksNeeded)
                                << " such that wts use PING-PONG mode" << endl;
                    wtBanksReserved = 2*minWtBanksNeeded;
                    avlbDataBanks = graph()->target_config()->bufBankAllotted() - wtBanksReserved;
                }
                else if (minWtBanksNeeded < wtBanksReserved)
                {
                    gLogWarning << "Downsizing weight banks from " << wtBanksReserved << " to " << minWtBanksNeeded
                                << " such that wts use single-KG mode" << endl;
                    wtBanksReserved = minWtBanksNeeded;
                    avlbDataBanks = graph()->target_config()->bufBankAllotted() - wtBanksReserved;
                }
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Can't use Winograd mode for %s since insufficient banks for data. "
                            "Recompile with WG turned off", name().c_str());
                }
                isMinInPHSatisfied = false;
                params().setAllottedWeightBanks(wtBanksReserved);
            }
            else
            {
                isMinInPHSatisfied = true;
            }
        } while (!isMinInPHSatisfied);
    }
    else
    {
        srcHeights[FIRST_PARTIAL_H_SEG]        = (NvU32)floor(avlbCBuffEntries[FIRST_PARTIAL_H_SEG] / entriesPerSlice);
        srcHeights[INTERMEDIATE_PARTIAL_H_SEG] = (NvU32)floor(avlbCBuffEntries[INTERMEDIATE_PARTIAL_H_SEG] / entriesPerSlice);
        srcHeights[LAST_PARTIAL_H_SEG]         = (NvU32)floor(avlbCBuffEntries[LAST_PARTIAL_H_SEG] / entriesPerSlice);  // not used anywhere
    }

    if (isIMGConv)
    {
        NvS32 unswizzledWeightW = params().weightDims().w;
        NvS32 unswizzledWeightH = params().weightDims().h;

        dilatedWeightH = (unswizzledWeightH - 1)*params().dilation().h + 1;
        dilatedWeightW = (unswizzledWeightW - 1)*params().dilation().w + 1;
    }
    else
    {
        dilatedWeightH = (weightDims.h - 1)*params().dilation().h + 1;
        dilatedWeightW = (weightDims.w - 1)*params().dilation().w + 1;
    }

    if ( debugSplits() )
    {
        gLogInfo << "eps " << entriesPerSlice << endl;
        gLogInfo << "total srcSlices needed In CBuff " << srcSlicesInCBuff << endl;
        gLogInfo << "total CBuffEntries Avlb " << totalCBuffEntriesAvlb << endl;
        gLogInfo << "SRC dims: nchw " << srcDims.n << ", " << srcDims.c << ", "
                 << srcDims.h << ", " << srcDims.w << endl;
        gLogInfo << "orig DST dims: nchw " << dstDims.n << ", " << dstDims.c << ", "
                 << dstDims.h << ", " << dstDims.w << endl;
    }

    if (isWinograd)
    {
        /*
         * In WG, there are strict limitations on input and output dimensions x(4x4)
         * As a result, unlike DC, we cannot make maximum utilization of available CBUFF entries
         */
        NvU32 opSlider = 0;
        ConvCoreNode::SplitDataInfo firstPH;
        firstPH.topSliceID = 0;
        firstPH.bottomSliceID = srcHeights[FIRST_PARTIAL_H_SEG] - params().topLeftPadding().h - 1;
        firstPH.numOverlapSlices = 0;
        firstPH.numRetainSlices = 0;
        firstPH.topPadding   = params().topLeftPadding().h;
        firstPH.bottomPadding = 0;
        firstPH.leftPadding = params().topLeftPadding().w;
        firstPH.rightPadding = params().bottomRightPadding().w;
        opSlider = 4 - params().topLeftPadding().h - 1;

        if (firstPH.bottomSliceID >= (srcDims.h - 1))
        {
            firstPH.bottomPadding = firstPH.bottomSliceID - (srcDims.h - 1);
            firstPH.bottomSliceID = srcDims.h - 1;
            opSlider = firstPH.bottomSliceID;
        }
        splitChunks.push_back(firstPH);

        while (opSlider < srcSlicesInCBuff)
        {
            ConvCoreNode::SplitDataInfo newPH;
            newPH.topSliceID = (splitChunks.size() * (srcHeights[INTERMEDIATE_PARTIAL_H_SEG] - 4) * params().stride().h) - params().topLeftPadding().h;
            newPH.bottomSliceID = newPH.topSliceID + srcHeights[INTERMEDIATE_PARTIAL_H_SEG] - 1;
            newPH.topPadding = 0;
            newPH.leftPadding = params().topLeftPadding().w;
            newPH.rightPadding = params().bottomRightPadding().w;
            newPH.numOverlapSlices = 0;
            newPH.numRetainSlices = 0;

            // check if current pH is pHL
            if (newPH.bottomSliceID >= (srcDims.h + params().bottomRightPadding().h - 1))
            {
                newPH.bottomSliceID = srcDims.h - 1;
                newPH.bottomPadding = params().bottomRightPadding().h;
                ASSERT((newPH.bottomSliceID + newPH.bottomPadding - newPH.topSliceID + 1 ) % 4 == 0);
                splitChunks.push_back(newPH);
                break;
            }
            else if (newPH.bottomSliceID >= (srcDims.h - 1))
            {
                newPH.bottomPadding = newPH.bottomSliceID - (srcDims.h - 1);
                newPH.bottomSliceID = srcDims.h - 1;
                opSlider = newPH.bottomSliceID;
                splitChunks.push_back(newPH);
            }
            else
            {
                opSlider = newPH.bottomSliceID;
                newPH.bottomPadding = 0;
                splitChunks.push_back(newPH);
            }
        }
    }
    else
    {
        NvU32 opSlider = 0;
        ConvCoreNode::SplitDataInfo firstPH;
        firstPH.topSliceID       = 0;
        firstPH.bottomSliceID    = srcHeights[FIRST_PARTIAL_H_SEG] - 1;
        firstPH.numOverlapSlices = 0;
        firstPH.numRetainSlices  = 0;
        firstPH.topPadding       = params().topLeftPadding().h;
        firstPH.bottomPadding    = 0;
        firstPH.leftPadding      = params().topLeftPadding().w;
        firstPH.rightPadding     = params().bottomRightPadding().w;
        splitChunks.push_back(firstPH);
        //opSlider = isWinograd ? dilatedWeightH - params().topLeftPadding().h - 1 : dilatedWeightH - 1;
        opSlider = dilatedWeightH - params().topLeftPadding().h - 1;

        if ((dilatedWeightH - 1) > splitChunks.back().bottomSliceID)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Kernel height greater than single split possible");
        }

        while (opSlider < srcSlicesInCBuff)
        {
            while (opSlider <= (NvU32)splitChunks.back().bottomSliceID)
            {
                opSlider += params().stride().h;
            }
            // if opSlider overshoots bottomSliceID, roll it back to second-last valid op in lastPH chunk
            if (opSlider > (NvU32)splitChunks.back().bottomSliceID)
            {
                opSlider -= params().stride().h;
            }

            ConvCoreNode::SplitDataInfo newPH;
            newPH.topSliceID       = min(opSlider, (NvU32)splitChunks.back().bottomSliceID) - (dilatedWeightH - 1) + params().stride().h;
            newPH.bottomSliceID    = newPH.topSliceID + srcHeights[INTERMEDIATE_PARTIAL_H_SEG] - 1;
            newPH.topPadding       = 0;
            newPH.leftPadding      = params().topLeftPadding().w;
            newPH.rightPadding     = params().bottomRightPadding().w;
            newPH.numOverlapSlices = 0;
            newPH.numRetainSlices  = 0;

            // check if current pH is pHL
            if (newPH.bottomSliceID >= srcDims.h - 1)
            {
                newPH.bottomSliceID = srcDims.h - 1;
                newPH.bottomPadding = params().bottomRightPadding().h;
                splitChunks.push_back(newPH);
                break;
            }

            opSlider = newPH.topSliceID + dilatedWeightH - 1;
            newPH.bottomPadding = 0;
            splitChunks.push_back(newPH);
        }
    }

    // check if the last partial-H chunk overshoots bottom of the input, if so - trim it
    if (splitChunks.back().topSliceID >= srcDims.h ||
        splitChunks.back().bottomSliceID >= (srcDims.h + params().bottomRightPadding().h))
    {
        splitChunks.pop_back();
    }

    // determine overlap slices between 2 split chunks
    for (std::vector<ConvCoreNode::SplitDataInfo>::iterator itr = splitChunks.begin() + 1, pastItr = itr -1;
            itr != splitChunks.end(); ++itr, ++pastItr)
    {
        if ((*pastItr).bottomSliceID >= (*itr).topSliceID)
        {
            (*itr).numOverlapSlices = (*pastItr).bottomSliceID - (*itr).topSliceID + 1;
            (*pastItr).numRetainSlices = (*itr).numOverlapSlices;
        }
        else
        {
            (*itr).numOverlapSlices = 0;
            (*pastItr).numRetainSlices = 0;
        }
    }


    for (std::vector<ConvCoreNode::SplitDataInfo>::iterator itr = splitChunks.begin();
            itr != splitChunks.end(); ++itr)
    {
        NvU32 convSlider = 0;
        NvU32 convCounter = 0;
        if (isWinograd)
        {
            convSlider = (*itr).topSliceID + (4 - 1) - (*itr).topPadding;
        }
        else
        {
            convSlider = (*itr).topSliceID + dilatedWeightH -1 - (*itr).topPadding;
        }

        while(convSlider <= (NvU32)((*itr).bottomSliceID + (*itr).bottomPadding))
        {
            convSlider += params().stride().h;
            convCounter++;
        }
        (*itr).numConvs = convCounter;
        if ((*itr).bottomSliceID > srcDims.h - 1)
        {
            (*itr).bottomSliceID = srcDims.h - 1;
        }

        (*itr).inDims.n = srcDims.n;
        (*itr).inDims.c = srcDims.c;
        (*itr).inDims.h = (*itr).bottomSliceID - (*itr).topSliceID + 1;
        (*itr).inDims.w = srcDims.w;

        (*itr).outDims.n = dstDims.n;
        (*itr).outDims.c = dstDims.c;

        if (isWinograd)
        {
            (*itr).outDims.h = (*itr).inDims.h + (*itr).topPadding + (*itr).bottomPadding - 4;
            (*itr).outDims.w = (*itr).inDims.w + (*itr).leftPadding + (*itr).rightPadding - 4;
        }
        else
        {
            (*itr).outDims.h = (((*itr).inDims.h + (*itr).topPadding + (*itr).bottomPadding - dilatedWeightH) / params().stride().h) + 1;
            (*itr).outDims.w = (((*itr).inDims.w + (*itr).leftPadding + (*itr).rightPadding - dilatedWeightW) / params().stride().w) + 1;
        }

        (*itr).inputBufferOffset  = suggestLineStride(srcTSD) * (*itr).topSliceID;
        (*itr).outputBufferOffset = suggestLineStride(dstTSD) * outputHeightProcessed;

        outputHeightProcessed += (*itr).outDims.h;

        (*itr).wtBanks = wtBanksReserved;
        (*itr).dataBanks = avlbDataBanks;
    }

    PROPAGATE_ERROR_FAIL(verifyPartialHInfo(splitChunks, isWinograd));

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::splitData(NvU16 avlbDataBanks)
{
    NvDlaError e = NvDlaSuccess;
    int counter = 0;

    std::vector<ConvCoreNode::SplitDataInfo> splitChunks = std::vector<ConvCoreNode::SplitDataInfo>();
    std::vector<ConvCoreNode::SplitDataInfo>::iterator splitItr;
    engine_ast::ConvCoreNode* newSplitConvNode  = NULL;
    engine_ast::SDPNode* newSplitSDPNode        = NULL;
    engine_ast::SplitNode* swSplitNode          = NULL;
    engine_ast::ConcatenationNode* swConcatNode = NULL;
    surface::TensorSurfaceDesc* weightTSD       = NULL;

    Dims4 newSplitSrcDims     {0,0,0,0};    // unique dims for input to each splitOp combo
    Dims4 newSplitStreamDims  {0,0,0,0};    // unique dims for stream tensor between a split conv+sdp
    Dims4 newSplitDstDims     {0,0,0,0};    // unique dims for each splitOp combo's output


    engine_ast::Edge* origInputEdge     = NULL;
    engine_ast::Edge* origStreamEdge    = NULL;
    engine_ast::Edge* origOutputEdge    = NULL;
    engine_ast::Edge* origConvAuxEdge   = NULL;
    engine_ast::Edge* origSDPAuxEdge    = NULL;
    engine_ast::Edge* origConvComputeOutEdge = NULL;    // compute edge going out from existing conv node
    engine_ast::Edge* origSDPComputeOutEdge = NULL;     // compute edge going out from existing fused-sdp node


    engine_ast::Edge* newSplitSrcDataEdge    = NULL;    // new data edge carrying unique src tensor SD from sw-split to conv node
    engine_ast::Edge* newSplitStreamDataEdge = NULL;    // new data edge carrying unique dst tensor SD from a split conv node
    engine_ast::Edge* newSplitDstDataEdge    = NULL;    // new data edge carrying unique dst tensor SD from a split sdp to sw-concat node
    engine_ast::Edge* newSplitConvCompEdge  = NULL;    // new compute edge connecting a splitOp with its sibling
    engine_ast::Edge* newSplitSDPCompEdge   = NULL;    // new compute edge connecting a splitOp with its sibling
    engine_ast::Edge* newSplitConvRedundantAuxDataEdge = NULL;  // in partial-H, all split conv's share original wt edge
    engine_ast::Edge* newSplitSDPRedundantAuxDataEdge  = NULL;  // in partial-H, all split sdp's share original aux data edge

    Tensor* newSplitSrcTensor             = NULL;    // new tensor holding the src data chunk details to a splitOp combo
    Tensor* newSplitDstTensor             = NULL;    // new tensor detailing the chunk of output specific to a splitOp combo

    NVDLA_UNUSED(newSplitConvCompEdge);
    NVDLA_UNUSED(newSplitSDPCompEdge);
    NVDLA_UNUSED(newSplitSrcDataEdge);
    NVDLA_UNUSED(newSplitDstDataEdge);

    bool isWinograd = params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD;

    // list of split siblings
    vector<engine_ast::ConvCoreNode*> splitConvNodes = std::vector<engine_ast::ConvCoreNode*>();
    vector<engine_ast::SDPNode*>      splitSDPNodes  = std::vector<engine_ast::SDPNode*>();


    engine_ast::ConvCoreNode* origConvNode      = this;
    engine_ast::SDPNode*      origFusedSDPNode  = engine_ast::NodeFactory::nodeCast<SDPNode*>(dependencyParams().fusedNode(engine_ast::IODirectionEnum::OUTPUT));
    bool fusedSDPHasAuxData = false;

    if (!origFusedSDPNode ||
        (origFusedSDPNode->engineType().v() != engine_ast::EngineTypeEnum::SDP))
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "ConvCore op %s doesn't have any fused SDP partner."
                "Can't split the node without a memory-write capable fused partner", name().c_str());
    }

    origInputEdge   = origConvNode->inputEdges()[0];
    origConvAuxEdge = origConvNode->auxEdges()[0];
    origStreamEdge  = origConvNode->outputEdges()[0];
    origOutputEdge  = origFusedSDPNode->outputEdges()[0];
    weightTSD       = origConvAuxEdge->tensorSurfaceDesc();
    origConvComputeOutEdge = graph()->downstreamComputeEdges(this).size() ?
                             graph()->downstreamComputeEdges(this)[0] : NULL;
    origSDPComputeOutEdge  = graph()->downstreamComputeEdges(origFusedSDPNode).size() ?
                             graph()->downstreamComputeEdges(origFusedSDPNode)[0] : NULL;

    PROPAGATE_ERROR_FAIL(origFusedSDPNode->nodeAuxEdge(&origSDPAuxEdge));

    fusedSDPHasAuxData = origSDPAuxEdge ? true : false;

    PROPAGATE_ERROR_FAIL( determineSplitDataRatios(avlbDataBanks, splitChunks) );

    if ( debugSplits() )
    {
        gLogInfo << "orig input edge=" << origInputEdge->id()
                 << " orig aux edge="  << origConvAuxEdge->id()
                 << " orig output edge=" << origOutputEdge->id() << endl;

        for (splitItr = splitChunks.begin(); splitItr != splitChunks.end(); ++splitItr, ++counter)
        {
            gLogInfo << "ph " << counter << endl;
            gLogInfo << "\tstart "          << (*splitItr).topSliceID << endl;
            gLogInfo << "\tend "            << (*splitItr).bottomSliceID << endl;
            gLogInfo << "\ttoppadding "     << (*splitItr).topPadding << endl;
            gLogInfo << "\tbottompadding "  << (*splitItr).bottomPadding << endl;
            gLogInfo << "\tleftpadding "    << (*splitItr).leftPadding << endl;
            gLogInfo << "\trightpadding "   << (*splitItr).rightPadding << endl;
            gLogInfo << "\tnumConvs "       << (*splitItr).numConvs << endl;
            gLogInfo << "\tretainSlices "   << (*splitItr).numRetainSlices << endl;
            gLogInfo << "\tnumOverlapSlices " << (*splitItr).numOverlapSlices << endl;
            gLogInfo << "\tinputBufferOffset " << (*splitItr).inputBufferOffset << endl;
            gLogInfo << "\toutputBufferOffset " << (*splitItr).outputBufferOffset << endl;
            gLogInfo << "\tinDims: nchw "  << (*splitItr).inDims.n << ", " << (*splitItr).inDims.c << ", " << (*splitItr).inDims.h << ", " << (*splitItr).inDims.w << endl;
            gLogInfo << "\toutDims: nchw " << (*splitItr).outDims.n << ", " << (*splitItr).outDims.c << ", " << (*splitItr).outDims.h << ", " << (*splitItr).outDims.w << endl;
        }
        gLogInfo << "numPartialHs " << splitChunks.size() << endl;
    }

    if (splitChunks.size() <= 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Num of partial-H ops: %d. This function shouldn't have been invoked for %s", splitChunks.size(),
                name().c_str());
    }
    /*
     *                                (node-x)
     *                                   |
     *                   (weight)        | (orig input-surface: 1 edge : 1 TSD : 1 TBD)
     *                      |\  \        |
     *  (orig wt-surf:      | \  \    (split)___________
     *       n 2nd-nodes:   |  \  \    /   \            \
     *       1 edge:        |   \  \  /     \            \  (input-surfaces: n edges : n TSDs : 1TBD)
     *       1 TSD :        |    \  \/_______\__________  \
     *       1 TBD)         |     \_/_________\         \  \
     *                      |      /          \\         \  \
     *                       ----(C0)=========(C1)=-=-=-=-(Cn)
     *                             |            |           |  (stream-surfaces: n edges : n TSDs : no TBD)
     *                      -----(S0)=========(S1)=-=-=-=-(Sn)
     *                      |   ___\_________/ /         / /
     *                      |  /  __\_________/_________/ /   (output-surfaces: n edges : n TSDs : 1 TBD)
     *                      | /  /   \       /           /
     *  (orig bias-surf:    |/  /     \     /           /
     *    n 2nd-nodes:  (bias-data)   (concat)- - - - -
     *    1 edge:                        |
     *    1 TSD :                        | (orig output-surf: 1 edge : 1 TSD : 1 TBD)
     *    1 TBD)                         |
     *                              (node-x+1)
     */

    /* software split and concat nodes which appear in the graph but
     * do not get annotated in the firmware's action list
     */
    swSplitNode  = engine_ast::NodeFactory::newSplitNode(NULL, graph());
    swSplitNode->params().setSplitAxis(SplitAxisEnum::SPLIT_ALONG_H);
    swConcatNode = engine_ast::NodeFactory::newConcatNode(NULL, graph());
    swConcatNode->params().setConcatAxis(ConcatAxisEnum::CONCAT_ALONG_H);

    /* delegate orig input edge to conv node and reattach to swSplit node and
     * delegate orig output edge from SDP node and reattach to swConcat node
     */
    graph()->replaceEdgeNodes(origInputEdge, ast::EdgeSideEnum::SECOND, origConvNode, swSplitNode);
    graph()->replaceEdgeNodes(origOutputEdge, ast::EdgeSideEnum::FIRST, origFusedSDPNode, swConcatNode);

    splitConvNodes.push_back(origConvNode);
    splitSDPNodes.push_back(origFusedSDPNode);

    origConvNode->params().setTopLeftPadding(Dims2(splitChunks[0].topPadding, splitChunks[0].leftPadding));
    origConvNode->params().setBottomRightPadding(Dims2(splitChunks[0].bottomPadding, splitChunks[0].rightPadding));

    /********************* Handle 1st split op node ****************************/
    /* Step-1: Handle edges */
    // Handle split input edge
    newSplitSrcDims.n = 1;                          //FIXME: conv doesnt support HW multi-batch yet
    newSplitSrcDims.c = splitChunks[0].inDims.c;
    newSplitSrcDims.w = splitChunks[0].inDims.w;
    newSplitSrcDims.h = splitChunks[0].inDims.h;
    newSplitSrcTensor = origInputEdge->originalTensor()->clone();
    newSplitSrcTensor->setTensorType(TensorType::kIO);
    newSplitSrcTensor->setDimensions(newSplitSrcDims);

    // Handle split stream edge
    newSplitStreamDims.n = 1;                           //FIXME: conv doesnt support HW multi-batch yet
    newSplitStreamDims.c = splitChunks[0].outDims.c;
    newSplitStreamDims.h = splitChunks[0].outDims.h;
    newSplitStreamDims.w = splitChunks[0].outDims.w;
    origStreamEdge->originalTensor()->setDimensions(newSplitStreamDims);
    origStreamEdge->tensorSurfaceDesc()->setDimensions(newSplitStreamDims);

    // Handle split output edge
    newSplitDstDims = newSplitStreamDims;
    newSplitDstTensor = origOutputEdge->originalTensor()->clone();
    newSplitDstTensor->setTensorType(TensorType::kIO);
    newSplitDstTensor->setDimensions(newSplitDstDims);

    newSplitSrcDataEdge = graph()->addDataEdge(origInputEdge->canonicalEdge(), swSplitNode, origConvNode, newSplitSrcTensor);
    newSplitDstDataEdge = graph()->addDataEdge(origOutputEdge->canonicalEdge(), origFusedSDPNode, swConcatNode, newSplitDstTensor);

    if (isWinograd)
    {
        PROPAGATE_ERROR_FAIL(origConvNode->determineWinogradParams());
        PROPAGATE_ERROR_FAIL(origFusedSDPNode->determineWinogradParams(origConvNode));
    }

    origConvNode->setSplitDataInfo(splitChunks[0]);
    origConvNode->params().setRetainSlices(splitChunks[0].numRetainSlices);

    ASSERT(origConvNode->params().weightBanksAllotted() == splitChunks[0].wtBanks);
    ASSERT(splitChunks[0].dataBanks == avlbDataBanks);

    origConvNode->params().setAllottedDataBanks(splitChunks[0].dataBanks);

    /*******************************Handle remaining split ops ***************/
    for (splitItr = splitChunks.begin() + 1; splitItr != splitChunks.end(); ++splitItr)
    {
        /* Step-1: Add new splitOp node */
        switch(origConvNode->engineOpType().v())
        {
            case EngineOpTypeEnum::CONVOLUTION_CONV: {
                    canonical_ast::ConvolutionNode* origCanNode =
                        canonical_ast::NodeFactory::nodeCast<canonical_ast::ConvolutionNode*>(canonicalNode());
                    newSplitConvNode = engine_ast::NodeFactory::newConvCoreConvolutionOpNode(origCanNode, graph());
                    if (!isWinograd && (newSplitConvNode->params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD))
                    {
                        newSplitConvNode->params().setConvMode(ConvolutionModeEnum::CONV_DIRECT);
                        newSplitConvNode->setName("dc-conv-" + newSplitConvNode->name().substr(newSplitConvNode->name().find("wg-conv-") + 8));
                    }
                    newSplitSDPNode  = newSplitConvNode->addSDPJointOpNode(origFusedSDPNode);
                    NodeFactory::nodeCast<ConvCoreConvolutionOpNode*>(newSplitConvNode)->inheritParams(origConvNode);
                }; break;
            case EngineOpTypeEnum::CONVOLUTION_FC: {
                    canonical_ast::FullyConnectedNode* origCanNode =
                        canonical_ast::NodeFactory::nodeCast<canonical_ast::FullyConnectedNode*>(canonicalNode());
                    newSplitConvNode = engine_ast::NodeFactory::newConvCoreFullyConnectedOpNode(origCanNode, graph());
                    newSplitSDPNode  = newSplitConvNode->addSDPJointOpNode(origCanNode);
                }; break;
            case EngineOpTypeEnum::CONVOLUTION_DECONV: {
                    canonical_ast::DeconvolutionNode* origCanNode =
                        canonical_ast::NodeFactory::nodeCast<canonical_ast::DeconvolutionNode*>(canonicalNode());
                    newSplitConvNode = engine_ast::NodeFactory::newConvCoreDeconvolutionOpNode(origCanNode, graph());
                    newSplitSDPNode  = newSplitConvNode->addSDPJointOpNode(origFusedSDPNode);
                    NodeFactory::nodeCast<ConvCoreDeconvolutionOpNode*>(newSplitConvNode)->inheritParams(origConvNode);
                }; break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Invalid engine type for splitting weights: %s",
                        engineOpType().c_str());
        }

        newSplitConvNode->params().setTopLeftPadding(Dims2((*splitItr).topPadding, (*splitItr).leftPadding));
        newSplitConvNode->params().setBottomRightPadding(Dims2((*splitItr).bottomPadding, (*splitItr).rightPadding));

        /* Trim the extra aux edges attached to the nodes above. All the split conv's
         * are going to share weights and the split sdp's are going to share data (if any)
         */
        PROPAGATE_ERROR_FAIL(newSplitConvNode->nodeDataEdge(TensorType::kWEIGHT, ast::EdgeSideEnum::SECOND, &newSplitConvRedundantAuxDataEdge));
        newSplitConvNode->graph()->removeEdgeFromNode(newSplitConvRedundantAuxDataEdge, ast::EdgeSideEnum::SECOND, newSplitConvNode);
        newSplitConvNode->graph()->removeNodeFromEdge(newSplitConvRedundantAuxDataEdge, ast::EdgeSideEnum::SECOND, newSplitConvNode);
        newSplitConvNode->graph()->appendNodeToEdge(origConvAuxEdge, ast::EdgeSideEnum::SECOND, newSplitConvNode);
        newSplitConvNode->graph()->removeEdge(newSplitConvRedundantAuxDataEdge);

        if (fusedSDPHasAuxData)
        {
            PROPAGATE_ERROR_FAIL(newSplitSDPNode->nodeAuxEdge(&newSplitSDPRedundantAuxDataEdge));
            newSplitSDPNode->graph()->removeEdgeFromNode(newSplitSDPRedundantAuxDataEdge, ast::EdgeSideEnum::SECOND, newSplitSDPNode);
            newSplitSDPNode->graph()->removeNodeFromEdge(newSplitSDPRedundantAuxDataEdge, ast::EdgeSideEnum::SECOND, newSplitSDPNode);
            newSplitSDPNode->graph()->appendNodeToEdge(origSDPAuxEdge, ast::EdgeSideEnum::SECOND, newSplitSDPNode);
            newSplitSDPNode->graph()->removeEdge(newSplitSDPRedundantAuxDataEdge);
        }


        /* Step-1: Handle edges */
        // Handle split input edge
        newSplitSrcDims.n = 1;                      //FIXME: conv doesnt support HW multi-batch yet
        newSplitSrcDims.c = (*splitItr).inDims.c;
        newSplitSrcDims.h = (*splitItr).inDims.h;
        newSplitSrcDims.w = (*splitItr).inDims.w;
        newSplitSrcTensor = origInputEdge->originalTensor()->clone();
        newSplitSrcTensor->setTensorType(TensorType::kIO);
        newSplitSrcTensor->setDimensions(newSplitSrcDims);


        // Handle split stream edge
        PROPAGATE_ERROR_FAIL(newSplitConvNode->nodeDataEdge(TensorType::kSTREAM, ast::EdgeSideEnum::FIRST, &newSplitStreamDataEdge));
        newSplitStreamDims.n = 1;                       //FIXME: conv doesnt support HW multi-batch yet
        newSplitStreamDims.c = (*splitItr).outDims.c;
        newSplitStreamDims.h = (*splitItr).outDims.h;
        newSplitStreamDims.w = (*splitItr).outDims.w;
        newSplitStreamDataEdge->originalTensor()->setDimensions(newSplitStreamDims);

        // Handle split dst edge
        newSplitDstDims = newSplitStreamDims;
        newSplitDstTensor = origOutputEdge->originalTensor()->clone();
        newSplitDstTensor->setTensorType(TensorType::kIO);
        newSplitDstTensor->setDimensions(newSplitDstDims);

        newSplitSrcDataEdge = graph()->addDataEdge(origInputEdge->canonicalEdge(), swSplitNode, newSplitConvNode, newSplitSrcTensor);
        newSplitDstDataEdge = graph()->addDataEdge(origOutputEdge->canonicalEdge(), newSplitSDPNode, swConcatNode, newSplitDstTensor);

        if (isWinograd)
        {
            PROPAGATE_ERROR_FAIL(newSplitConvNode->determineWinogradParams());
            PROPAGATE_ERROR_FAIL(newSplitSDPNode->determineWinogradParams(newSplitConvNode));
        }

        /* Step-4: Connect the newly added splitOp to the last one using compute edge */
        newSplitConvCompEdge = graph()->addComputeEdge(splitConvNodes.back(), newSplitConvNode);
        newSplitSDPCompEdge = graph()->addComputeEdge(splitSDPNodes.back(), newSplitSDPNode);

        newSplitConvNode->setSplitDataInfo((*splitItr));
        splitConvNodes.push_back(newSplitConvNode);
        splitSDPNodes.push_back(newSplitSDPNode);
        PROPAGATE_ERROR_FAIL(newSplitConvNode->fuseOnTheFlyNodes());

        // Retain overlapping slices between adjacent conv ops
        newSplitConvNode->params().setRetainSlices((*splitItr).numRetainSlices);

        ASSERT(origConvNode->params().weightBanksAllotted() == (*splitItr).wtBanks);
        ASSERT((*splitItr).dataBanks == avlbDataBanks);

        newSplitConvNode->params().setAllottedWeightBanks((*splitItr).wtBanks);
        newSplitConvNode->params().setAllottedDataBanks((*splitItr).dataBanks);
    }

    // Delegate compute edge going out from existing conv and sdp to the last of the respective split siblings
    if (origConvComputeOutEdge)
    {
        graph()->replaceEdgeNodes(origConvComputeOutEdge, ast::EdgeSideEnum::FIRST, origConvNode, splitConvNodes.back());
    }
    if (origSDPComputeOutEdge)
    {
        graph()->replaceEdgeNodes(origSDPComputeOutEdge, ast::EdgeSideEnum::FIRST, origFusedSDPNode, splitSDPNodes.back());
    }

    // Reuse weights for each split conv iff all of the weight data can fit in the avlb wt banks
    if (params().weightBanksAllotted() == calculateTotalBanksForWeight(weightTSD))
    {
        ConvCoreNode* firstSplitConv = splitConvNodes[0];
        ConvCoreNode* lastSplitConv  = splitConvNodes.back();
        firstSplitConv->params().setReleaseWeights(false);
        for (std::vector<ConvCoreNode*>::iterator ni = splitConvNodes.begin() + 1; ni != splitConvNodes.end(); ++ni)
        {
            (*ni)->params().setReuseWeights(true);
            (*ni)->params().setReleaseWeights(false);
        }
        // Release weights after last split op
        lastSplitConv->params().setReleaseWeights(true);
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::splitWeightsAndData(NvU16 /*avlbWtBanks*/, NvU16 /*avlbDataBanks*/)
{
    NvDlaError e = NvDlaSuccess;
    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "%s is not yet supported", __func__);
fail:
    return e;
}

/*--------------------------------Split Nodes---------------------------*/
NvDlaError engine_ast::ConvCoreNode::splitNodes()
{
    NvDlaError e = NvDlaSuccess;

    if ( params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT )
    {
        PROPAGATE_ERROR_FAIL( splitNodesInternal() );
    }
    else if ( params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD )
    {
        ConvCoreConvolutionOpNode* convOp = NULL;
        ASSERT( engineOpType().v() == EngineOpTypeEnum::CONVOLUTION_CONV );
        convOp = NodeFactory::nodeCast<ConvCoreConvolutionOpNode*>(this);

        e = splitNodesInternal();
        // if compilation failed underneath due to any WG limitations, fallback to DC and retry
        // for any other failure flavors, actually fail
        if (e == NvDlaError_InsufficientMemory)
        {
            PROPAGATE_ERROR_FAIL( convOp->fallbackWGConvToDC() );
            PROPAGATE_ERROR_FAIL( splitNodesInternal() );
        }
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::splitNodesInternal()
{
    NvDlaError e = NvDlaSuccess;

    NvU32 totalDataBanksNeeded = 0;
    NvU32 totalWtBanksNeeded   = 0;
    NvU32 minWtBanksNeeded     = 0;
    NvU32 compWtReservedBank   = 0;
    NvU32 spareBanks           = 0;

    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *weightTSD  = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    bool weight_compression = graph()->profile()->canCompressWeights() && graph()->target_config()->isCompressWeightsCapable();
    NvU32 totalCbuffBanks = graph()->target_config()->bufBankAllotted();

    totalDataBanksNeeded = calculateTotalBanksForData(srcTSD);
    totalWtBanksNeeded   = calculateTotalBanksForWeight(weightTSD);
    minWtBanksNeeded     = calculateMinBanksForWeight(weightTSD);
    // If weights are compressed, WMB surface needs 1 bank (Bank-15)
    compWtReservedBank   = weight_compression ? 1 : 0;

    if ( debugSplits() )
    {
        gLogInfo << "(" << name() << ") " << params().convMode().c_str() << endl;
        gLogInfo << "\ttotal b4d needed " << totalDataBanksNeeded << endl;
        gLogInfo << "\ttotal b4w needed " << totalWtBanksNeeded << endl;
        gLogInfo << "\tmin b4w needed " << minWtBanksNeeded << endl;
        gLogInfo << "\treserved WMB bank " << compWtReservedBank << endl;
        gLogInfo << "(" << name() << ") ";
    }

    /* FIXME: For now, follow the following order of preference
     *  1.  Full input and Full weight        (no split needed)
     *  2.a Full input and Partial weight     (split-k mode)
     *  2.b Partial input and Full weight     (partial-H mode)
     *  3   Partial input and Partial weight  (split-k and partial-H)
     */
    //HW-powered: FI + FW
    if (totalWtBanksNeeded + totalDataBanksNeeded + compWtReservedBank <= totalCbuffBanks)
    {
        // nothing to split
        if ( debugSplits() )
        {
            gLogInfo << "FI + FW mode. Nothing to split" << endl;
        }
        params().setAllottedDataBanks(totalDataBanksNeeded);
        params().setAllottedWeightBanks(totalWtBanksNeeded);
        goto fail;
    }

    //HW-powered: FI + ping-pong Wts
    if ((2*minWtBanksNeeded) + totalDataBanksNeeded + compWtReservedBank <= totalCbuffBanks)
    {
        if ( debugSplits() )
        {
            gLogInfo << "FI + ping-pong mode on weights" << endl;
        }
        params().setAllottedDataBanks(totalDataBanksNeeded);
        params().setAllottedWeightBanks(2*minWtBanksNeeded);
        goto fail;
    }
    //HW-powered: FI + 1KG wts
    else if (minWtBanksNeeded + totalDataBanksNeeded + compWtReservedBank <= totalCbuffBanks)
    {
        if ( debugSplits() )
        {
            gLogInfo << "FI + 1KG of weights. Suboptimal but hw automated" << endl;
        }
        params().setAllottedDataBanks(totalDataBanksNeeded);
        params().setAllottedWeightBanks(minWtBanksNeeded);
        goto fail;
    }

    /* If control reaches here, then either the data is massive and/or single KG is massive,
     * software splits are unavoidable. Everything that follows is profile governed i.e.
     * we will split nodes if the profile forces us to. A better profile later will
     * eventually emerge a better KPI. However, we do attempt to determine if any scope of
     * hw automation is still left.
     */

    /* FI within (profile) alloted banks:-
     * FI + (hw-split-k(Ping-Pong) / hw-split-k(single KG))
     */
    if (totalDataBanksNeeded < graph()->profile()->dataBanksAlloted())
    {
        if ( debugSplits() )
        {
            gLogInfo << "FI + ";
        }
        params().setAllottedDataBanks(totalDataBanksNeeded);

        // hw-split-k (Ping-Pong)
        if ((2*minWtBanksNeeded) + compWtReservedBank < graph()->profile()->weightBanksAlloted())
        {
            if ( debugSplits() )
            {
                gLogInfo << "HW split-K with Ping-pong mode on weights" << endl;
            }
            params().setAllottedWeightBanks(2*minWtBanksNeeded);
            goto fail;
        }
        // hw-split-k (single KG)
        else if (minWtBanksNeeded + compWtReservedBank < graph()->profile()->weightBanksAlloted())
        {
            if ( debugSplits() )
            {
                gLogInfo << "HW split-K with single KG" << endl;
            }
            params().setAllottedWeightBanks(minWtBanksNeeded);
            goto fail;
        }
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Not enough banks to hold weight data. No software split supported");
    }

    /* Split-Input within (profile) alloted banks:-
     * SW-partial-h + (FW (all KGs) / hw-split-k(Ping-Pong) / hw-split-k(single KG) / sw-split-k)
     */
    else
    {
        if (totalWtBanksNeeded + compWtReservedBank <= graph()->profile()->weightBanksAlloted())
        {
            if ( debugSplits() )
            {
                gLogInfo << "sw-Partial-H + FW" << endl;
            }
            params().setAllottedWeightBanks(totalWtBanksNeeded);
            spareBanks = totalCbuffBanks - totalWtBanksNeeded - compWtReservedBank;
            PROPAGATE_ERROR_FAIL(splitData(spareBanks));
        }
        // hw-split-k(Ping-Pong)
        else if ((2*minWtBanksNeeded) + compWtReservedBank <= graph()->profile()->weightBanksAlloted())
        {
            if ( debugSplits() )
            {
                gLogInfo << "sw-Partial-H + hw-split-K (ping-pong mode)" << endl;
            }
            params().setAllottedWeightBanks(2*minWtBanksNeeded);
            spareBanks = totalCbuffBanks - (2*minWtBanksNeeded) - compWtReservedBank;
            PROPAGATE_ERROR_FAIL(splitData(spareBanks));
        }
        // hw-split-k(single KG)
        else if (minWtBanksNeeded + compWtReservedBank <= graph()->profile()->weightBanksAlloted())
        {
            if ( debugSplits() )
            {
                gLogInfo << "sw-Partial-H + hw-split-K (single KG mode)" << endl;
            }
            params().setAllottedWeightBanks(minWtBanksNeeded);
            spareBanks = totalCbuffBanks - minWtBanksNeeded - compWtReservedBank;
            PROPAGATE_ERROR_FAIL(splitData(spareBanks));
        }
        //SW-poweredL sw-partial-h + hw-split-k
        else if (minWtBanksNeeded < graph()->profile()->weightBanksAlloted())
        {
            // mostly remnant code block
        }
        // sw-split-k
        else
        {
            if ( debugSplits() )
            {
                gLogInfo << "sw-Partial-H + sw-split-K" << endl;
            }
            PROPAGATE_ERROR_FAIL(splitWeightsAndData(graph()->profile()->dataBanksAlloted(), graph()->profile()->weightBanksAlloted()));
        }
    }

fail:
    return e;
}

/*------------------------------Handle Multi-Batch---------------------*/
NvDlaError engine_ast::ConvCoreNode::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;

    NvU32 numBatches = graph()->profile()->multiBatchSize();
    NvU32 firstBatch = 0;
    NvU32 lastBatch  = numBatches - 1;
    surface::TensorSurfaceDesc *weightTSD = NULL;

    for (NvU32 nn = 1; nn < numBatches; ++nn)
    {
        params(nn) = params(0);
        switch(engineOpType().v()) {
            case EngineOpTypeEnum::CONVOLUTION_CONV:
                NodeFactory::nodeCast<ConvCoreConvolutionOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<ConvCoreConvolutionOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::CONVOLUTION_FC:
                NodeFactory::nodeCast<ConvCoreFullyConnectedOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<ConvCoreFullyConnectedOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::CONVOLUTION_DECONV:
                NodeFactory::nodeCast<ConvCoreDeconvolutionOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<ConvCoreDeconvolutionOpNode*>(this)->params(0);
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unsupported CONV Engine Op type: %s", engineOpType().c_str());
        }
    }

    /* Convolution ops can have weight reuse scenarios, handle them carefully among batches
     *
     * Whenever the entire weights for the conv - op can be fit in the conv_buff banks,
     * subsequent batches can reuse those weights.
     *
     * Whenever, there is hardware automated split-K selected (ping-pong mode(2KG) OR 1KG),
     * weight reuse among batches is not possible. So the weights have to be re-fetched
     * separately for each batch.
     */
    weightTSD = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());
    if (params(firstBatch).weightBanksAllotted() == calculateTotalBanksForWeight(weightTSD))
    {
        bool isSingleBatchWtRls = params(firstBatch).isReleaseWeights();
        for (NvU32 nn = firstBatch; nn < lastBatch; ++nn)
        {
            NvU32 currBatch = nn;
            NvU32 nextBatch = nn + 1;
            params(nextBatch).setReuseWeights(true);
            params(currBatch).setReleaseWeights(false);
        }
        params(lastBatch).setReleaseWeights(isSingleBatchWtRls);
    }

fail:
    return e;
}

/*---------------------------Resolve Data Dependencies-----------------*/
NvDlaError engine_ast::ConvCoreNode::resolveDataDependencies(Node* next)
{
    NvDlaError e = NvDlaSuccess;
    Node* downStreamSDP;
    // capture all consumers as normal
    PROPAGATE_ERROR_FAIL(engine_ast::Node::resolveDataDependencies(next));

    downStreamSDP = dependencyParams().consumer(EngineTypeEnum::SDP).node();
    if (downStreamSDP == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Conv node %s doesn't have a single SDP consumer!!"
                " Forgot to add even an SDP-NOP?", name().c_str());
    }

    // now retrofit the consumer array to specially handle the fused SDP op with conv
    if (downStreamSDP->dependencyParams().fusedNode(IODirectionEnum::INPUT) == this)
    {
        dependencyParams().consumer(EngineTypeEnum::SDP).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
        downStreamSDP->dependencyParams().producer(EngineTypeEnum::CONVOLUTION).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Conv node not fused with down stream SDP");
        //todo: probably fuse them here as a late on-fly fix??
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::determineWinogradParams()
{
    NvDlaError e = NvDlaSuccess;

    NvU16 bpe = 0;
    Dims2 origPadTL;
    NvU16 newWOut, newHOut;
    NvU16 WInExt, HInExt, CInExt;
    NvU16 newPadRight, newPadBottom;
    Dims4 origConvInDims, origConvOutDims, origConvAuxDims;
    engine_ast::ConvCoreEngineParams::WinogradParams convWGParams;

    if (params().convMode().v() != ConvolutionModeEnum::CONV_WINOGRAD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't determine WG params for %s which is"
                " not selected for WG mode", name().c_str());
    }

    PROPAGATE_ERROR_FAIL( repopulateEdgePorts() );

    bpe = graph()->profile()->computePrecision() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? 1 : 2;
    origConvInDims  = inputEdges()[0]->originalTensor()->getDimensions();
    origConvOutDims = outputEdges()[0]->originalTensor()->getDimensions();
    origConvAuxDims = auxEdges()[0]->originalTensor()->getDimensions();

    convWGParams.inDims   = origConvInDims;
    convWGParams.outDims  = origConvOutDims;
    convWGParams.auxDims  = origConvAuxDims;

    origPadTL = params().topLeftPadding();
    newWOut   = (NvU16)((origConvOutDims.w + 3)/4) * 4;
    newHOut   = (NvU16)((origConvOutDims.h + 3)/4) * 4;

    if ( debugWinograd() )
    {
        gLogInfo << "origconv in dims: " << origConvInDims.c << "x"
                 << origConvInDims.h << "x" << origConvInDims.w << endl;
        gLogInfo << "origconv aux dims: " << origConvAuxDims.n << "x"
                 << origConvAuxDims.c << "x" << origConvAuxDims.h << "x"
                 << origConvAuxDims.w << endl;
        gLogInfo << "origconv out dims: " << origConvOutDims.c << "x"
                 << origConvOutDims.h << "x" << origConvOutDims.w << endl;
    }

    WInExt       = newWOut + 4;
    HInExt       = newHOut + 4;
    CInExt       = origConvInDims.c;
    newPadRight  = WInExt - origConvInDims.w - origPadTL.w;
    newPadBottom = HInExt - origConvInDims.h - origPadTL.h;

    /* Step-1: determine input side wg details */
    if ( ((CInExt * bpe) % 32) != 0 )
    {
        CInExt = CInExt + (32 - ((CInExt * bpe) % 32))/bpe;
    }
    convWGParams.inDims.w   = WInExt;
    convWGParams.inDims.h   = HInExt;
    convWGParams.inDims.c   = CInExt;
    convWGParams.auxDims.w  = 4;
    convWGParams.auxDims.h  = 4;
    convWGParams.auxDims.c  = CInExt;
    params().setBottomRightPadding(Dims2(newPadBottom, newPadRight));

    /* Step-2: Determine output side wg details */
    convWGParams.outDims.w = newWOut;
    convWGParams.outDims.h = newHOut;

    params().setWinogradParams(convWGParams);

    if ( debugWinograd() )
    {
        gLogInfo << "conv wg " << name() << " in: "
                 << convWGParams.inDims.n << "x" << convWGParams.inDims.c << "x"
                 << convWGParams.inDims.h << "x" << convWGParams.inDims.w << endl;
        gLogInfo << "conv wg " << name() << " aux: "
                 << convWGParams.auxDims.n << "x" << convWGParams.auxDims.c << "x"
                 << convWGParams.auxDims.h << "x" << convWGParams.auxDims.w << endl;
        gLogInfo << "conv wg " << name() << " out: "
                 << convWGParams.outDims.n << "x" << convWGParams.outDims.c << "x"
                 << convWGParams.outDims.h << "x" << convWGParams.outDims.w << endl;
        gLogInfo << "conv wg " << name() << " TxL pad: "
                 << origPadTL.h << "x" << origPadTL.w << endl;
        gLogInfo << "conv wg " << name() << " BxR pad: "
                 << newPadBottom << "x" << newPadRight << endl;
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreNode::captureCanonicalWeights()
{
    NvDlaError e = NvDlaSuccess;

    Tensor* wt_tensor;

    wt_tensor = graph()->addAuxTensor(graph()->newAuxTensorName(), params().weightDims(), TensorType::kWEIGHT);
    Edge* aux = graph()->addDataEdge((canonical_ast::Edge*)0, 0, this, wt_tensor);
    NVDLA_UNUSED(aux);

    return e;
}

//----------------------------------------------------------------------
//                      Formula Book
//----------------------------------------------------------------------

NvS16 engine_ast::ConvCoreNode::calculateEPS(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvS16 eps = -1;
    NvU8  bpe = tsd->surfaceFormat().bytesPerElement();
    NvS8  cpa = tsd->surfaceFormat().channelsPerAtom();
    Dims4 input_dims = tsd->dimensions();
    NvU32 width      = input_dims.w;
    NvU32 width_ext  = params().topLeftPadding().w + params().bottomRightPadding().w + input_dims.w;
    NvU32 chnl_ext   = 0;
    NvU16 stride_x = params().stride().w;
    NvU16 stride_y = params().stride().h;
    NvF32 memory_atomic_size = graph()->target_config()->memoryAtomicSize();
    NvF32 buf_bank_width     = graph()->target_config()->bufEntryWidth();

    surface::SurfaceCategory sc = tsd->surfaceFormat().category();

    switch(sc.v())
    {
        case surface::SurfaceCategoryEnum::IMG: {
            switch(params().convMode().v())
            {
                case ConvolutionModeEnum::CONV_DIRECT:
                    eps = (NvS16)ceil(width_ext * cpa * bpe / (NvF32)(buf_bank_width));
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Conv Mode: %s for Pixel format", params().convMode().c_str());
            }
        }; break;
        case surface::SurfaceCategoryEnum::FEATURE_DATA: {
            switch(params().convMode().v())
            {
                case ConvolutionModeEnum::CONV_DIRECT: {
                    NvU16 total_c_atomics = (NvU16)ceil(input_dims.c * bpe / (NvF32)(memory_atomic_size));
                    NvU16 last_c_atomics  = total_c_atomics % (NvU16)(buf_bank_width / memory_atomic_size) ;
                    NvU16 int_c_entries   = (total_c_atomics / (NvU16)(buf_bank_width / memory_atomic_size)) * width;
                    NvU16 frac_c_entries  = (last_c_atomics == 3) ? width: (NvU16)ceil(last_c_atomics * width / ((NvF32)(buf_bank_width / memory_atomic_size)));
                    eps = int_c_entries + frac_c_entries;
                }; break;
                case ConvolutionModeEnum::CONV_WINOGRAD: {
                    width_ext = params().winogradParams().inDims.w;
                    chnl_ext  = params().winogradParams().inDims.c;
                    ASSERT(width_ext % (4 * params().stride().w) == 0);
                    NvU16 c_atomics = (NvU16)ceil(chnl_ext * bpe / 32.0);
                    NvU16 c_atomics_ext = c_atomics * stride_x * stride_y;
                    eps = ( (width_ext/(4*stride_x)) * c_atomics_ext * 4) / 4;
                }; break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Conv Mode: %s for Feature Data format", params().convMode().c_str());
            }
        }; break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "EPS cannot be calculated for: %s surface category", sc.c_str());
    }

fail:
    return eps;
}

NvU16 engine_ast::ConvCoreNode::calculateTotalBanksForData(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU16 b4d = 0;
    Dims4 input_dims = tsd->dimensions();
    NvU32 height_ext = 0;
    NvU32 buf_entry_per_bank = graph()->target_config()->bufEntriesPerBank();

    surface::SurfaceCategory sc = tsd->surfaceFormat().category();
    if ( sc.v() == surface::SurfaceCategoryEnum::IMG ||
        (sc.v() == surface::SurfaceCategoryEnum::FEATURE_DATA && params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT))
    {
        b4d = (NvU16)ceil(calculateEPS(tsd) * input_dims.h / (NvF32)(buf_entry_per_bank));
    }
    else if (params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD)
    {
        height_ext = params().winogradParams().inDims.h;
        b4d = (NvU16)ceil(calculateEPS(tsd) * height_ext / 256.0);
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown CONV mode for %s", name().c_str());
    }

fail:
    return b4d;
}

NvU16 engine_ast::ConvCoreNode::calculateMinBanksForWeight(surface::TensorSurfaceDesc* wt_tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU8 bpe = wt_tsd->surfaceFormat().bytesPerElement();
    NvU32 atom_k_size = graph()->target_config()->atomicKSize();
    NvU32 buf_ent_per_bank = graph()->target_config()->bufEntriesPerBank();
    NvU32 buf_entry_width = graph()->target_config()->bufEntryWidth();

    NvU16 min_b4w = 0;

    NvU16 kpg  = bpe == 1 ? atom_k_size : atom_k_size / 2;

    ASSERT(std::string(wt_tsd->surfaceFormat().c_str()).find("COMPRESSED") == std::string::npos);

    // compute how many banks are necessary to store minimum weight data (1 KG)
    switch(params().convMode().v())
    {
        case ConvolutionModeEnum::CONV_WINOGRAD:
        {
            NvU32 krnl_width_ext   = (NvU32)ceil((NvF32)wt_tsd->dimensions().w/params().stride().w);
            NvU32 krnl_height_ext  = (NvU32)ceil((NvF32)wt_tsd->dimensions().h/params().stride().h);
            NvU32 krnl_chnl        = ROUNDUP_AND_ALIGN(wt_tsd->dimensions().c * bpe, 32) / bpe;
            NvU32 krnl_chnl_ext    = krnl_chnl * params().stride().w * params().stride().h;

            ASSERT(krnl_width_ext  == 3);
            ASSERT(krnl_height_ext == 3);

            // 4x4 because the pre-transformed kernel is always 4x4
            min_b4w       = (NvU16)ceil(4 * 4 * krnl_chnl_ext * bpe * kpg / 128 / 256.0);
        }
        break;

        case ConvolutionModeEnum::CONV_DIRECT:
        {
            // weight_bank has the same formula for image input/DC
            NvU64 kpg_bytes           = ROUNDUP_AND_ALIGN(wt_tsd->dimensions().c *
                                                          wt_tsd->dimensions().h *
                                                          wt_tsd->dimensions().w *
                                                          kpg * bpe, buf_entry_width);
            min_b4w     = (NvU16)ceil(kpg_bytes / buf_entry_width / ((NvF32)buf_ent_per_bank));
        }
        break;

        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                 "Can't calculate Banks4Weight for unsupported Conv Mode: %s",
                                 params().convMode().c_str());
    }

fail:
    return min_b4w;
}

NvU16 engine_ast::ConvCoreNode::calculateTotalBanksForWeight(surface::TensorSurfaceDesc* wt_tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU8      bpe = wt_tsd->surfaceFormat().bytesPerElement();
    NvU16 needed_b4w = 0;
    NvU32 buf_ent_per_bank = graph()->target_config()->bufEntriesPerBank();
    NvU32 buf_entry_width = graph()->target_config()->bufEntryWidth();

    ASSERT(std::string(wt_tsd->surfaceFormat().c_str()).find("COMPRESSED") == std::string::npos);

    // compute how many banks are necessary to store entire weight data
    switch(params().convMode().v())
    {
        case ConvolutionModeEnum::CONV_WINOGRAD: {
            NvU32 krnl_width_ext   = (NvU32)ceil((NvF32)wt_tsd->dimensions().w/params().stride().w);
            NvU32 krnl_height_ext  = (NvU32)ceil((NvF32)wt_tsd->dimensions().h/params().stride().h);
            NvU32 krnl_chnl        = ROUNDUP_AND_ALIGN(wt_tsd->dimensions().c * bpe, 32) / bpe;
            NvU32 krnl_chnl_ext    = krnl_chnl * params().stride().w * params().stride().h;

            ASSERT(krnl_width_ext  == 3);
            ASSERT(krnl_height_ext == 3);

            // 4x4 because the pre-transformed kernel is always 4x4
            needed_b4w    = (NvU16)ceil(4 * 4 * krnl_chnl_ext * bpe * wt_tsd->dimensions().n / 128 / 256.0);
        }; break;
        case ConvolutionModeEnum::CONV_DIRECT: {
            // weight_bank has the same formula for image input/DC
            NvU64 total_krnl_bytes    = ROUNDUP_AND_ALIGN(wt_tsd->dimensions().n *
                                                          wt_tsd->dimensions().c *
                                                          wt_tsd->dimensions().h *
                                                          wt_tsd->dimensions().w *
                                                          bpe, buf_entry_width);
            needed_b4w  = (NvU16)ceil(total_krnl_bytes / buf_entry_width / (NvF32)(buf_ent_per_bank));
        }; break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Can't calculate Banks4Weight for unsupported Conv Mode: %s", params().convMode().c_str());
    }

fail:
    return needed_b4w;
}

NvDlaError engine_ast::ConvCoreNode::verifyPartialHInfo
(
    const std::vector<engine_ast::ConvCoreNode::SplitDataInfo>& splitChunks,
    bool isWinograd
)
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *weightTSD  = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    bool isIMGConv        = false;
    NvS32 dilatedWeightH  = 0;
    NvU32 unsplitNumConvs = 0;
    NvU32 splitNumConvs   = 0;
    std::vector<ConvCoreNode::SplitDataInfo>::const_iterator itr;

    isIMGConv = srcTSD->surfaceFormat().category().v() == surface::SurfaceCategoryEnum::IMG;

    if (isIMGConv)
    {
        NvS32 unswizzledWeightH = params().weightDims().h;
        dilatedWeightH = (unswizzledWeightH - 1)*params().dilation().h + 1;
    }
    else
    {
        dilatedWeightH = (weightTSD->dimensions().h - 1)*params().dilation().h + 1;
    }

    NvU32 unsplitConvSlider = dilatedWeightH - 1 - params().topLeftPadding().h;
    while(unsplitConvSlider <= (NvU32)(srcTSD->dimensions().h + params().bottomRightPadding().h - 1))
    {
        unsplitConvSlider += params().stride().h;
        unsplitNumConvs++;
    }

    if (isWinograd)
    {
        for(itr = splitChunks.begin(); itr != splitChunks.end(); ++itr)
        {
            NvS32 pHId        = std::distance(splitChunks.begin(), itr);
            NvS32 prevPHOutHt = !pHId ? 0 : (*(itr - 1)).outDims.h;
            NvS32 convYStr    = params().stride().h;
            NvS32 topPadding  = params().topLeftPadding().h;
            if ((*itr).topSliceID > (*itr).bottomSliceID)
            {
                gLogInfo << "pH-" << std::distance(splitChunks.begin(), itr) << " has "
                         << "topSlice Id (" << (*itr).topSliceID << ") > "
                         << "bottomSlice Id (" << (*itr).bottomSliceID << ")" << endl;
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Partial-H data split logic made some mistake for %s!!!", name().c_str());
            }

            bool topSliceIdOk = !pHId ? ((*itr).topSliceID == 0) : ((*itr).topSliceID == ((pHId * prevPHOutHt * convYStr) - topPadding));
            if ( !topSliceIdOk )
            {
                gLogInfo << "pH-" << pHId << " top slide-id: " << (*itr).topSliceID << " doesn't match "
                         "standard formula " << (!pHId ? 0 : (pHId * prevPHOutHt * convYStr - topPadding)) << endl;
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Partial-H data split logic made some mistake for %s!!!", name().c_str());
            }

            ASSERT((*itr).wtBanks  + (*itr).dataBanks <= graph()->target_config()->bufBankAllotted());
        }
    }
    else
    {
        for(itr = splitChunks.begin(); itr != splitChunks.end(); ++itr)
        {
            if ((*itr).topSliceID > (*itr).bottomSliceID)
            {
                gLogInfo << "pH-" << std::distance(splitChunks.begin(), itr) << " has "
                         << "topSlice Id (" << (*itr).topSliceID << ") > "
                         << "bottomSlice Id (" << (*itr).bottomSliceID << ")" << endl;
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Partial-H data split logic made some mistake!!!");
            }

            ASSERT((*itr).wtBanks  + (*itr).dataBanks <= graph()->target_config()->bufBankAllotted());

            splitNumConvs += (*itr).numConvs;
        }

        if ( splitNumConvs != unsplitNumConvs )
        {
            gLogInfo << "unsplit num convs(" << unsplitNumConvs << ") != "
                     << "split num convs(" << splitNumConvs << ")" << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Partial-H data split logic made some mistake!!!");
        }
    }

fail:
    return e;
}

}; // nvdla::priv::
}; // nvdla::
