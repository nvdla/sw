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
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{
/*--------------------------------Generic functions--------------------*/
NvDlaError engine_ast::SDPNode::nodeAuxEdge(engine_ast::Edge **ret_edge)
{
    NvDlaError e = NvDlaSuccess;

    switch(engineOpType().v()) {
        case EngineOpTypeEnum::SDP_BIAS:
            PROPAGATE_ERROR_FAIL(nodeDataEdge(TensorType::kBIAS,
                                              ast::EdgeSideEnum::SECOND, ret_edge));
            break;
        case EngineOpTypeEnum::SDP_BATCH_NORM:
            PROPAGATE_ERROR_FAIL(nodeDataEdge(TensorType::kBATCH_NORM,
                                              ast::EdgeSideEnum::SECOND, ret_edge));
            break;
        case EngineOpTypeEnum::SDP_SCALE:
            PROPAGATE_ERROR_FAIL(nodeDataEdge(TensorType::kSCALE,
                                              ast::EdgeSideEnum::SECOND, ret_edge));
            break;
        //FIXME SDP SUPER
        default:
            *ret_edge = NULL;
    }

fail:
    return e;
}

/*--------------Suggest Format/Dims/Strides/Size/Buffer-Offset----------------*/
/* DLA 1.0 can do int16 activation in int8 pipeline. So using int16 for both int8
 * and int16 computation to retain bits of precision in the aux data
 */
std::vector<surface::SurfaceFormat> engine_ast::SDPNode::suggestAuxSurfaceFormats(engine_ast::Edge* auxEdge)
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
        case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:{
            surface::IsSurfacePrecisionDifferent desiredSP(surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16);
            supportedAuxSFs.erase(std::remove_if(supportedAuxSFs.begin(), supportedAuxSFs.end(), desiredSP), supportedAuxSFs.end());
        } break;
        case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8: {
            surface::IsSurfacePrecisionDifferent desiredSP(surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8);
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

/*
 * DLA-CONV engine works on IMG and FEATURE_DATA input in either
 * DC or Winograd mode; whereas DC has no special stride/size alignment
 * requirements, Winograd mode needs perfect x(4x4) input and output
 * surfaces. This means tweaking input/output surface dims whenever
 * we want to chose WG.
 * But, code emission layer finally takes care of adjusting dims, the compiler
 * here should only accommodate for the larger stride/size requirements.
 */
Dims4 engine_ast::SDPNode::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    bool isSrcTSD = false;
    bool isAuxTSD = false;
    bool isDstTSD = false;
    Dims4 suggestedDims(-1,-1,-1,-1);
    EdgeSequence inDataEdges;
    EdgeSequence auxDataEdges;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    inDataEdges = inputEdges();
    for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            break;
        }
    }
    auxDataEdges = auxEdges();
    for (EdgeSequenceIterator iei = auxDataEdges.begin(); iei != auxDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isAuxTSD = true;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (params().convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD)
    {
        if (isSrcTSD || isDstTSD)
        {
            suggestedDims = params().winogradParams().ioDims;
        }
        else if (isAuxTSD)
        {
            suggestedDims = Node::suggestSurfaceDims(tsd);
        }
    }
    else if (params().convMode().v() == ConvolutionModeEnum::CONV_DIRECT)
    {
        if (isAuxTSD)
        {
            suggestedDims = tsd->dimensions();
        }
        else
        {
            // if no src, use the prevailing dims of the surface (no choice for more arbitration)
            // inherit output requirements of the source node. Required for winogradable src node
            Dims4 inSurfDims(-1,-1,-1,-1);
            Edge* inEdge = inputEdges()[0];
            Node* srcNode = graph()->upstreamNodes(inEdge).size() ? graph()->upstreamNodes(inEdge)[0] : NULL;
            // destination TSD
            // Even though unfused src is winogradable, this sdp can still remove
            // winograd padding requirement for output tensor
            if (srcNode && dependencyParams().fusedNode(IODirectionEnum::INPUT) != NULL)
            {
                inSurfDims = srcNode->suggestSurfaceDims(inEdge->tensorSurfaceDesc());
            }
            suggestedDims.n = std::max<NvS32>(tsd->dimensions().n, inSurfDims.n);
            suggestedDims.c = std::max<NvS32>(tsd->dimensions().c, inSurfDims.c);
            suggestedDims.h = std::max<NvS32>(tsd->dimensions().h, inSurfDims.h);
            suggestedDims.w = std::max<NvS32>(tsd->dimensions().w, inSurfDims.w);
        }
    }
    else
    {
        REPORT_ERROR(NvDlaError_BadValue, "Unknown conv mode for %s", name().c_str());
    }

fail:
    return suggestedDims;
}

NvU32 engine_ast::SDPNode::suggestLineStride(surface::TensorSurfaceDesc* tsd)
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

NvU32 engine_ast::SDPNode::suggestSurfaceStride(surface::TensorSurfaceDesc* tsd)
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

NvU64 engine_ast::SDPNode::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 size = 0;
    bool isAuxEdge = false;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

   for (EdgeSequence::const_iterator iei = auxEdges().begin(); iei != auxEdges().end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isAuxEdge = true;
            break;
        }
    }

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
        // if the op does int8 rescaling, it has both the
        // aux data for the op and the rescaling factors
        if (params().x1Params().isINT8Rescaling() && isAuxEdge)
        {
            size *= 2;
        }
    }

    m_nodeTSDSurfaceSize[tsd] = size;

fail:
    return size;
}

NvU64 engine_ast::SDPNode::suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 offset = 0;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    bool isAuxTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    EdgeSequence inDataEdges;
    EdgeSequence auxDataEdges;
    Node* sinkNode = NULL;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    if (m_nodeTSDSurfaceOffsetInBuffer.find(tsd) != m_nodeTSDSurfaceOffsetInBuffer.end())
    {
        offset = m_nodeTSDSurfaceOffsetInBuffer[tsd];
        goto fail;
    }

    inDataEdges = inputEdges();
    for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;
    auxDataEdges = auxEdges();
    for (EdgeSequenceIterator iei = auxDataEdges.begin(); iei != auxDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isAuxTSD = true;
            break;
        }
    }
    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    // pick any one of the src edges for consideration here
    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();

    if (isAuxTSD)
    {
        offset = 0;
    }
    else if (isSrcTSD)
    {
        /* if input is a stream tensor, it acts as a proxy for the output tensor
         * so, inherit the surface offset in buffer from the stream tensor into the output tensor;
         * else, input and output tensors both are independent
         */
        if (srcTSD->tensorCategory() == memory::TensorCategoryEnum::STREAM_TENSOR)
        {
            offset = srcTSD->bufferOffset();
        }
        else
        {
            offset = 0;
        }
    }
    else if (isDstTSD)
    {
        sinkNode = graph()->downstreamDataNodes(this).size() ? graph()->downstreamDataNodes(this)[0] : NULL;
        if (sinkNode && sinkNode->engineType().v() == EngineTypeEnum::CONCATENATION)
        {
            ConcatenationNode* concatNode = NodeFactory::nodeCast<ConcatenationNode*>(sinkNode);
            // fixme: this kind of diving catches can be avoided if slice Ids are maintained inside tsd
            switch(concatNode->params().concatAxis().v())
            {
                case ConcatAxisEnum::CONCAT_ALONG_W:
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support split-w mode for %s",
                            name().c_str());
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_H:
                    // in case of concat along Chnl direction, let the concat node decide suitable surface offset
                    offset = concatNode->suggestSurfaceOffsetInBuffer(tsd);
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_C:
                    // in case of concat along Chnl direction, let the concat node decide suitable surface offset
                    offset = concatNode->suggestSurfaceOffsetInBuffer(tsd);
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown concat type for %s",
                            concatNode->name().c_str());
            }
        }
        else
        {
            offset = 0;
        }
    }

    m_nodeTSDSurfaceOffsetInBuffer[tsd] = offset;

fail:
    return offset;
}

NvDlaError engine_ast::SDPNode::verifySurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    Dims4 auxDims;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    bool isAuxTSD = false;
    bool isPerLayer = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    surface::TensorSurfaceDesc* auxTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;
    EdgeSequence inDataEdges;
    EdgeSequence auxDataEdges;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    inDataEdges = inputEdges();
    for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;
    auxDataEdges = auxEdges();
    for (EdgeSequenceIterator iei = auxDataEdges.begin(); iei != auxDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isAuxTSD = true;
            auxTSD = tsd;
            break;
        }
    }
    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    // pick any 1 of the src edges for consideration here
    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();
    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    if (!srcTSD || !dstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Not all TSDs are registered yet for %s", name().c_str());
    }
    else if (inDataEdges.size() > 1)
    {
        // sdp supports max 2 src edges (for EW)
        ASSERT(inDataEdges.size() == 2)
        surface::TensorSurfaceDesc* src1TSD = inDataEdges[0]->tensorSurfaceDesc();
        surface::TensorSurfaceDesc* src2TSD = inDataEdges[1]->tensorSurfaceDesc();
        if ( !src1TSD->isSurfaceSymmetricTo(src2TSD) )
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) All input tensors to SDP should have the same dimensions",
                                                    name().c_str());
        }
    }

    if ( isAuxTSD )
    {
        isPerLayer = auxTSD->dimensions().w == 1 &&
                    auxTSD->dimensions().h == 1 &&
                    auxTSD->dimensions().c == 1;

        if(auxTSD->dimensions().c != srcTSD->dimensions().c && !isPerLayer)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Aux and Input tensors should have the same number of channels: %d != %d",
                                                    name().c_str(),
                                                    auxTSD->dimensions().c,
                                                    srcTSD->dimensions().c);
        }
    }
    else
    {
        Dims4 srcDims = srcTSD->dimensions();
        Dims4 dstDims = dstTSD->dimensions();
        if ((srcDims.c != dstDims.c) || (srcDims.h != dstDims.h) || (srcDims.w != dstDims.w))
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Input and Output tensors should have the same dimensions",
                                                    name().c_str());
        }
    }

fail:
    return e;
}

/*--------------------------------Fuse Nodes---------------------------*/
NvDlaError engine_ast::SDPNode::fuseOnTheFlyNodes()
{
    NvDlaError e = NvDlaError_Success;

    if ( graph()->profile()->canSDPFuseVerticalOps() )
    {
        NodeSequence consumer_nodes;
        EdgeSequence output_edges = graph()->downstreamEdges(this);
        for (EdgeSequenceIterator oei = output_edges.begin(); oei != output_edges.end(); ++oei)
        {
            consumer_nodes = graph()->downstreamNodes(*oei);
            for (NodeSequence::const_iterator cni = consumer_nodes.begin(); cni != consumer_nodes.end(); ++cni)
            {
                // Find scope of fusing adjacent sdp nodes.
                // FIXME: add logic to determine sub-engine requirements and then fuse
                if ((*cni)->engineType().v() == EngineTypeEnum::SDP)
                {
                    dependencyParams().setFusedNode(IODirectionEnum::OUTPUT, (*cni));
                    (*cni)->dependencyParams().setFusedNode(IODirectionEnum::INPUT, this);
                }
            }
        }
    }

    // We will determine fusing SDP + PDP in the pdp api

    return e;
}

/*---------------------------Merge SDP Operations----------------------------*/
engine_ast::Node* engine_ast::SDPNode::mergeWithSDPOp(SDPNode* nextSDP)
{
    Node* removableNode = NULL;

    /*
     * In DLA architecture, CONV engine doesn't have a write out pipe. So any canonical
     * convolution operation when translated to the engine land needs 2 translation ops:
     * (conv + sdp-nop): where conv does the math op and sdp-nop provides the SDP engine's
     * write out pipe.
     * However such a sdp-nop can be pruned if there's another SDP-op following it.
     */
    if (engineOpType().v() == EngineOpTypeEnum::SDP_NOP)
    {
        /* Irrespective of whatever other type of SDP the nextSDP op is (including
         * a NOP itself), always remove the current node. The next NOP will be trimmed
         * in its own pass.
         */
        removableNode = this;
        goto fail;
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

engine_ast::Node* engine_ast::SDPNode::mergeUnitScaleOp(SDPNode* SDPScaleOp)
{
    Node* removableNode     = NULL;
    SDPScaleOpNode* scaleOp = NodeFactory::nodeCast<SDPScaleOpNode*>(SDPScaleOp);

    if (scaleOp->params(/*batch_id*/0).x1Params().actType().v() == SDPActTypeEnum::SDP_ACT_TYPE_RELU)
    {
        switch(engineOpType().v()) {
            case EngineOpTypeEnum::SDP_BIAS:
                NodeFactory::nodeCast<SDPBiasOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                    SDPActTypeEnum::SDP_ACT_TYPE_RELU);
                break;
            case EngineOpTypeEnum::SDP_ELEMENTWISE:
                NodeFactory::nodeCast<SDPElementWiseOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                    SDPActTypeEnum::SDP_ACT_TYPE_RELU);
                break;
            case EngineOpTypeEnum::SDP_SCALE:
                NodeFactory::nodeCast<SDPScaleOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                    SDPActTypeEnum::SDP_ACT_TYPE_RELU);
                break;
            case EngineOpTypeEnum::SDP_BATCH_NORM:
                NodeFactory::nodeCast<SDPBatchNormOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                    SDPActTypeEnum::SDP_ACT_TYPE_RELU);
                break;
            case EngineOpTypeEnum::SDP_SUPER:
                NodeFactory::nodeCast<SDPSuperOpNode*>(this)->params(/*batch_id*/0).x2Params().setActType(
                    SDPActTypeEnum::SDP_ACT_TYPE_RELU);
                break;
            default:
                goto fail;
        }
    }
    removableNode = SDPScaleOp;

fail:
    return removableNode;
}

/*
 * Often many of the non-linear SDP operations are followed by a ReLU unit.
 *      ReLU: y = max(x, 0)
 * Although, it's as simple as trimming all the negative elements in the tensor
 * and can be achieved by a simple bit flip in the hardware, it is executed
 * independently of the preceeding SDP op and incurs a whole memory pass.
 * Try to consume the following ReLU operation into the existing sdp op-node and
 * prune the redundant ReLU node from AST.
 */
engine_ast::Node* engine_ast::SDPNode::tryToMergeWithActOp(SDPNode* SDPActOp)
{
    Node* removableNode     = NULL;
    SDPActivationOpNode* actOp = NodeFactory::nodeCast<SDPActivationOpNode*>(SDPActOp);

    // fixme: when vertical fusion is turned on, this logic to remove ReLU will move there.
    // at that point, this API will cease to exist.
    if (!graph()->profile()->canSDPMergeMathOps())
    {
        goto fail;
    }

    if (engineOpType().e() == SDPActOp->engineOpType().e())
    {
        // cannot combine 2 adjacent Activation operations
        goto fail;
    }
    else if (actOp->params(/*batch_id*/0).x1Params().actType().v() != SDPActTypeEnum::SDP_ACT_TYPE_RELU)
    {
        // cannot combine any other activation op than ReLU
        goto fail;
    }

    switch(engineOpType().v()) {
        case EngineOpTypeEnum::SDP_BIAS:
            NodeFactory::nodeCast<SDPBiasOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                SDPActTypeEnum::SDP_ACT_TYPE_RELU);
            break;
        case EngineOpTypeEnum::SDP_ELEMENTWISE:
            NodeFactory::nodeCast<SDPElementWiseOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                SDPActTypeEnum::SDP_ACT_TYPE_RELU);
            break;
        case EngineOpTypeEnum::SDP_NOP:
            NodeFactory::nodeCast<SDPNOPNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                SDPActTypeEnum::SDP_ACT_TYPE_RELU);
            break;
        case EngineOpTypeEnum::SDP_SCALE:
            NodeFactory::nodeCast<SDPScaleOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                SDPActTypeEnum::SDP_ACT_TYPE_RELU);
            break;
        case EngineOpTypeEnum::SDP_BATCH_NORM:
            NodeFactory::nodeCast<SDPBatchNormOpNode*>(this)->params(/*batch_id*/0).x1Params().setActType(
                SDPActTypeEnum::SDP_ACT_TYPE_RELU);
            break;
        case EngineOpTypeEnum::SDP_SUPER:
            NodeFactory::nodeCast<SDPSuperOpNode*>(this)->params(/*batch_id*/0).x2Params().setActType(
                SDPActTypeEnum::SDP_ACT_TYPE_RELU);
            break;
        default:
            goto fail;
    }

    removableNode = SDPActOp;

fail:
    return removableNode;
}

/* Check if nextSDP op fusion feasible in X2 */
bool engine_ast::SDPNode::isFeasibleToFuseSDPSubEngineOp(SDPNode* nextSDP)
{
    bool retval = false;

    // check for open x2 slot
    if ( params().x2Params().enabled()
        || params().yParams().enabled()
        || ( params().outCVT().isEnable() && params().outCVT().scale() != 1 )
        || ( params().outCVT().isEnable() && params().outCVT().truncate() != 0 )
        || nextSDP->params().x2Params().enabled()
       )
    {
        if ( graph()->debugFuseSubEngineOps() )
        {
            gLogInfo << "SDP X2 fusion NOT feasible for " << this->name() << " & " << nextSDP->name();
            gLogInfo << ". No x2 slot available" << std::endl;
        }
        goto fail;
    }

    // Fusing of Eltwise Op in X2, enforces additional restrictions
    if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_ELEMENTWISE)
    {
        retval = isFeasibleToFuseSDPEltwiseOp(nextSDP);
    }
    else
    {
        retval = true;
    }
fail:
    if ( retval & graph()->debugFuseSubEngineOps() )
    {
        gLogInfo << "SDP fusion feasible for " << this->name() << " & " << nextSDP->name() << std::endl;
    }
    return retval;
}

/* Check if eltwise op can be fused in current SDP node */
bool engine_ast::SDPNode::isFeasibleToFuseSDPEltwiseOp(SDPNode* nextSDP)
{
    bool retval = false;
    NodeSequence sourceNodes;
    engine_ast::Node* currentSource = this;
    engine_ast::Node* peerSource = NULL;
    bool current2peerDep = false;
    bool peer2currentDep = false;
    NodeSequence allNodes;
    NodeSequenceIterator startNodeIter;
    NodeSequenceIterator ni;
    int eltNodeDistance = -1;

    SDPElementWiseOpNode* eltwiseOp = NodeFactory::nodeCast<SDPElementWiseOpNode*>(nextSDP);

    // Allow eltwise fuse, only if it is betn two source nodes.
    sourceNodes = graph()->upstreamDataNodes(eltwiseOp);
    if (sourceNodes.size() != 2)
    {
        if ( graph()->debugFuseSubEngineOps() )
        {
            gLogInfo << "Eltwise merge NOT feasible for " << this->name() << " & " << eltwiseOp->name();
            gLogInfo << ". Eltwise op has " << sourceNodes.size() << " (more than 2) source layers" << std::endl;
        }
        goto fail;
    }

    // Determine eltwise source node dependencies on each other.
    currentSource = this;
    if (currentSource == sourceNodes[0])
    {
        peerSource = sourceNodes[1];
    }
    else
    {
        peerSource = sourceNodes[0];
    }

    allNodes = graph()->orderedNodes();
    startNodeIter = std::find(allNodes.begin(), allNodes.end(), this);
    for (NodeSequenceIterator ni = startNodeIter; ni != allNodes.end(); ++ni) {
        if (*ni == this || *ni == peerSource)
        {
            eltNodeDistance++;
        }
        else if (*ni == eltwiseOp)
        {
            eltNodeDistance++;
            break;
        }
    }
    if (eltNodeDistance !=1 && eltNodeDistance !=2)
    {
        if ( graph()->debugFuseSubEngineOps() )
        {
            gLogInfo << "Eltwise merge NOT feasible for " << this->name() << " & " << eltwiseOp->name();
            gLogInfo << ". Invalid state. Eltwise node distance incorrect: " << eltNodeDistance << std::endl;
        }
        goto fail;
    }

    // current node depends on peer sibling
    current2peerDep = currentSource->dependsOn(peerSource, engine_ast::viaComputeData, engine_ast::allowAll);
    // peer sibling depends on current node
    peer2currentDep = peerSource->dependsOn(currentSource, engine_ast::viaComputeData, engine_ast::allowAll);

    if ( graph()->debugFuseSubEngineOps() )
    {
        gLogInfo << eltwiseOp->name() << " sources: " << currentSource->name() << " & " <<  peerSource->name();
        gLogInfo << " eltNodeDistance: " << eltNodeDistance;
        gLogInfo << " c2p: " << current2peerDep << " p2c: " << peer2currentDep << std::endl;
    }

    /*
    current node depends on peer -> fuse elt in current node
    peer node depends on curent node -> Do not fuse elt in current node
    current and peer nodes are independent of each other, then we have option of fusing it at any node.
    Based on traversal, check which source node visited later i.e. elt node distance.
        - if elt node distance = 1 fuse in current source node
        - if elt node distance = 2, fuse in peer source.
    */
    if ( current2peerDep
        || ((!current2peerDep && !peer2currentDep) && (eltNodeDistance == 1)) )
    {
        // Eltwise fuse in current node feasible
        retval = true;
    }
    else
    {
        if ( graph()->debugFuseSubEngineOps() )
        {
            gLogInfo << "Eltwise merge NOT feasible for " << currentSource->name() << " & " << eltwiseOp->name();
            gLogInfo << ". Eltwise op source nodes dependency requirements. eltNodeDistance: " << eltNodeDistance << std::endl;
        }
        goto fail;
    }
fail:
    return retval;
}

/* Merge currentSDP and nextSDP Operation in SDP Super Op.
   Creates new SDP super op with both X1 and X2 engines enabled.
   currentSDP Op is configured in X1.
   nextSDP Op is configured in X2.
   Removes nextSDP op node.
   Mark currentSDP op node for removal.
*/
engine_ast::Node* engine_ast::SDPNode::fuseSDPSubEngineOp(SDPNode* nextSDP)
{
    NvDlaError e = NvDlaSuccess;
    Node* removableNode = NULL;

    if (isFeasibleToFuseSDPSubEngineOp(nextSDP))
    {
        engine_ast::Edge* edgeToMoveX1 = NULL;
        engine_ast::Edge* edgeToMoveX2 = NULL;
        SDPSuperOpNode* sdpSuperOp = NULL;
        NodeSequence substituteNodes;
        NodeSequence sourceNodes;
        engine_ast::Node* eltPeerSource = NULL;

        // Create new SDP super node
        sdpSuperOp = NodeFactory::newSDPSuperOpNode(NULL, graph());
        if (!sdpSuperOp)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't create new SDP super op for replacing %s and %s",
                    this->name().c_str(), nextSDP->name().c_str());
        }

        // configure this SDP op in X1 engine
        switch(engineOpType().v())
        {
            case EngineOpTypeEnum::SDP_ELEMENTWISE:
                // Do not support elt op in x1.
                // Limitation is finding the aux edge of elt in x1 i.e. differentiating betn 2 inputs edges
                goto fail;
            default:
                PROPAGATE_ERROR_FAIL(this->configureSDPSuperOpSubEngine(sdpSuperOp, SDP_ENGINE_X1));
        }

        // Find X1 Aux Edge
        PROPAGATE_ERROR_FAIL(this->nodeAuxEdge(&edgeToMoveX1));

        // configure next SDP OP in X2 engine
        PROPAGATE_ERROR_FAIL(nextSDP->configureSDPSuperOpSubEngine(sdpSuperOp, SDP_ENGINE_X2));

        // Find X2 Aux Edge
        if (nextSDP->engineOpType().v() == EngineOpTypeEnum::SDP_ELEMENTWISE)
        {
            // Aux edge of Elementwise is connecting edge betn elt and peer source
            eltPeerSource = NodeFactory::nodeCast<SDPElementWiseOpNode*>(nextSDP)->getPeerSource(this);
            edgeToMoveX2 = graph()->connectingDataEdge(nextSDP, eltPeerSource, ast::EdgeSideEnum::SECOND);
        }
        else
        {
            PROPAGATE_ERROR_FAIL(nextSDP->nodeAuxEdge(&edgeToMoveX2));
        }

        // Graph Updates
        sourceNodes = graph()->upstreamDataNodes(this);

        // 1. Replace this SDP node with Super Op node in the graph
        substituteNodes.push_back(sdpSuperOp);
        PROPAGATE_ERROR_FAIL(graph()->substituteNodeInAST(this, substituteNodes));

        // 2. Move X1 Aux Edge
        sdpSuperOp->markSdpAuxEdge(SDP_ENGINE_X1, edgeToMoveX1);
        graph()->replaceEdgeNodes(edgeToMoveX1, ast::EdgeSideEnum::SECOND, this, sdpSuperOp);

        // 3. Move X2 Aux Edge
        sdpSuperOp->markSdpAuxEdge(SDP_ENGINE_X2, edgeToMoveX2);
        graph()->replaceEdgeNodes(edgeToMoveX2, ast::EdgeSideEnum::SECOND, nextSDP, sdpSuperOp);

        // 4. nextSDP node should be removed manually, by disconnecting it from the input side
        PROPAGATE_ERROR_FAIL(graph()->removeNodeFromAST(nextSDP, IODirectionEnum::INPUT));

        // 5. Framework will remove this SDP node
        removableNode = this;

        if (sourceNodes.size() != 0)
        {
            if ( graph()->debugFuseSubEngineOps() )
            {
                gLogInfo << "SDP x2 fusion parent: " << sourceNodes[0]->name() << std::endl;
            }
            PROPAGATE_ERROR_FAIL(sourceNodes[0]->fuseOnTheFlyNodes());

            if (eltPeerSource && dependencyParams().fusedNode(IODirectionEnum::INPUT) == sourceNodes[0])
            {
                // for elt x2 fusion, add compute edge betn elt sources.
                // This gurantees that SDPSuperOp fused parent will always be
                // traversed last.
                Edge* compEdge = graph()->addComputeEdge(eltPeerSource, sourceNodes[0]);
                if ( graph()->debugFuseSubEngineOps() )
                {
                    gLogInfo << "elt x2 fusion compute edge " << compEdge->id();
                    gLogInfo << " betn " << eltPeerSource->name();
                    gLogInfo << " & " << sourceNodes[0]->name() << std::endl;
                }
            }
        }
    }
fail:
    return removableNode;
}

/*------------------------------Handle Multi-Batch---------------------*/
NvDlaError engine_ast::SDPNode::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;

    //Handle operation parameters for the multi-batch operations
    NvU32 numBatches = graph()->profile()->multiBatchSize();
    for (NvU32 nn = 1; nn < numBatches; ++nn)
    {
        switch(engineOpType().v()) {
            case EngineOpTypeEnum::SDP_ACTIVATION:
                NodeFactory::nodeCast<SDPActivationOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPActivationOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::SDP_BIAS:
                NodeFactory::nodeCast<SDPBiasOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPBiasOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::SDP_ELEMENTWISE:
                NodeFactory::nodeCast<SDPElementWiseOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPElementWiseOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::SDP_NOP:
                NodeFactory::nodeCast<SDPNOPNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPNOPNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::SDP_SCALE:
                NodeFactory::nodeCast<SDPScaleOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPScaleOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::SDP_BATCH_NORM:
                NodeFactory::nodeCast<SDPBatchNormOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPBatchNormOpNode*>(this)->params(0);
                break;
            case EngineOpTypeEnum::SDP_SUPER:
                NodeFactory::nodeCast<SDPSuperOpNode*>(this)->params(nn) =
                        NodeFactory::nodeCast<SDPSuperOpNode*>(this)->params(0);
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unsupported SDP Engine Op type: %s", engineOpType().c_str());
        }
    }

fail:
    return e;
}

/*---------------------------Resolve Data Dependencies-----------------*/
NvDlaError engine_ast::SDPNode::resolveDataDependencies(engine_ast::Node* next)
{
    NvDlaError e = NvDlaSuccess;
    Node* downStreamPDP;

    // capture all consumers as normal
    PROPAGATE_ERROR_FAIL(engine_ast::Node::resolveDataDependencies(next));

    downStreamPDP = dependencyParams().consumer(EngineTypeEnum::PDP).node();
    if (downStreamPDP == NULL) {
        // no pdp consumer at all. do nothing
        goto fail;
    }

    // now retrofit the consumer array to specially handle the fused PDP op with SDP
    if (downStreamPDP->dependencyParams().fusedNode(IODirectionEnum::INPUT) == this) {
        dependencyParams().consumer(EngineTypeEnum::PDP).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
        downStreamPDP->dependencyParams().producer(EngineTypeEnum::SDP).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
    }

fail:
    return e;
}

NvDlaError engine_ast::SDPNode::determineWinogradParams(ConvCoreNode* wgConvNode)
{
    NvDlaError e = NvDlaSuccess;

    Dims4 origSdpAuxDims;
    engine_ast::Edge* sdpAuxEdge = NULL;
    engine_ast::SDPEngineParams::WinogradParams sdpWGParams;

    if (params().convMode().v() != ConvolutionModeEnum::CONV_WINOGRAD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't determine WG params for %s which is"
                " not selected for WG mode", name().c_str());
    }

    PROPAGATE_ERROR_FAIL( repopulateEdgePorts() );
    PROPAGATE_ERROR_FAIL( nodeAuxEdge(&sdpAuxEdge) );

    /* Use the input and output side wg details of associated Conv-WG node for this SDP as well*/
    sdpWGParams.ioDims    = wgConvNode->params().winogradParams().outDims;

    params().setWinogradParams(sdpWGParams);

    if ( debugWinograd() )
    {
        gLogInfo << "sdp wg " << name() << " io: "
                 << sdpWGParams.ioDims.n << "x" << sdpWGParams.ioDims.c << "x"
                 << sdpWGParams.ioDims.h << "x" << sdpWGParams.ioDims.w << endl;
    }

fail:
    return e;
}

};  // nvdla::priv::
};  // nvdla::
