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
#include <algorithm>

#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

void engine_ast::ConcatenationNode::captureCanonicalParams()
{
    // default to concat along Channel direction.
    // TODO: For other modes, add support in network/layer and canonical ast too
    params().setConcatAxis(ConcatAxisEnum::CONCAT_ALONG_C);
}

// not idempotent since relies on some details canonical details
NvDlaError engine_ast::ConcatenationNode::populateEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    typedef engine_ast::Graph::EdgeSequence EngineEdges;
    typedef engine_ast::Graph::EdgeSequenceIterator EngineEdgeIterator;

    typedef canonical_ast::Graph::EdgeSequence CanonicalEdges;
    typedef canonical_ast::Graph::EdgeSequenceIterator CanonicalEdgeIterator;

    EngineEdges engInEdges = graph()->upstreamDataEdges(this);
    EngineEdges engOutEdges = graph()->downstreamDataEdges(this);

    // if canonical equivalent exists, use order of i/o edge insertions in canonical land to establish strict order
    if (canonicalNode())
    {
        CanonicalEdges canInEdges = canonicalNode()->inputEdges();
        for (CanonicalEdgeIterator cei = canInEdges.begin(); cei != canInEdges.end(); ++cei)
        {
            IsSameCanonicalEdge match_can_edge(*cei);
            EngineEdgeIterator eei = std::find_if(engInEdges.begin(), engInEdges.end(), match_can_edge);
            if (eei == engInEdges.end())
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s is not an input edge to %s", (*eei)->id().c_str(), name().c_str());
            }
            markInputEdge(*eei);
        }
    }
    // else simply mark input edges in order of appearance
    else
    {
        for (EngineEdgeIterator eei = engInEdges.begin(); eei != engInEdges.end(); ++eei)
        {
            markInputEdge(*eei);
        }
    }

    if (engOutEdges.size() > 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s has > 1 output edges", name().c_str());
    }
    else
    {
        markOutputEdge(engOutEdges[0]);
    }

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

fail:
    return e;
}

Dims4 engine_ast::ConcatenationNode::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    EdgeSequence inDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    Edge* inputEdge = NULL;
    Dims4 suggestedDims(-1,-1,-1,-1);

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    inDataEdges = inputEdges();

    for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            inputEdge = *iei;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    // concat node doesn't affect any tensor dimensions,
    // except just aggregating them from the upstream nodes.
    // as a result, respect the suggested dims from upstream node if any
    if (isSrcTSD)
    {
        suggestedDims = graph()->upstreamNodes(inputEdge)[0]->suggestSurfaceDims(tsd);
    }
    else
    {
        NvS32 sumConcatInHeight = 0;
        NvS32 sumConcatInChannel = 0;
        Dims4 inSurfDims;
        Dims4 outSurfDims;
        for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
        {
            Node* srcNode = graph()->upstreamNodes(*iei).size() ? graph()->upstreamNodes(*iei)[0] : NULL;
            if (srcNode)
            {
                inSurfDims = srcNode->suggestSurfaceDims((*iei)->tensorSurfaceDesc());
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No upstream node for input edge %s to node %s",
                        (*iei)->id().c_str(), name().c_str());
            }

            switch(params().concatAxis().v())
            {
                case ConcatAxisEnum::CONCAT_ALONG_W:
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support concat along Width direction for %s",
                            name().c_str());
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_H:
                    sumConcatInHeight += inSurfDims.h;
                    outSurfDims.w = std::max<NvS32>(outSurfDims.w, inSurfDims.w);
                    outSurfDims.h = std::max<NvS32>(outSurfDims.h, sumConcatInHeight);
                    outSurfDims.c = std::max<NvS32>(outSurfDims.c, inSurfDims.c);
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_C:
                    sumConcatInChannel += inSurfDims.c;
                    outSurfDims.w = std::max<NvS32>(outSurfDims.w, inSurfDims.w);
                    outSurfDims.h = std::max<NvS32>(outSurfDims.h, inSurfDims.h);
                    outSurfDims.c = std::max<NvS32>(outSurfDims.c, sumConcatInChannel);
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown concat mode %s", params().concatAxis().c_str());
            }
        }
        suggestedDims = outSurfDims;
    }

fail:
    return suggestedDims;
}

NvU32 engine_ast::ConcatenationNode::suggestLineStride(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 lineStride = 0;
    EdgeSequence inDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    surface::TensorSurfaceDesc* dstTSD = NULL;
    ConcatenationNode* sinkConcat = NULL;
    surface::TensorSurfaceDesc* currConcatOutTSD = NULL;
    bool isConcatChain = false;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDLineStride.find(tsd) != m_nodeTSDLineStride.end())
    {
        lineStride = m_nodeTSDLineStride[tsd];
        goto fail;
    }

    isConcatChain = graph()->downstreamDataNodes(this).size() ?
                    graph()->downstreamDataNodes(this)[0]->engineType().v() == EngineTypeEnum::CONCATENATION :
                    false;
    if (isConcatChain)
    {
        sinkConcat = NodeFactory::nodeCast<ConcatenationNode*>(graph()->downstreamDataNodes(this)[0]);
        currConcatOutTSD = outputEdges()[0]->tensorSurfaceDesc();
        lineStride = sinkConcat->suggestLineStride(currConcatOutTSD);
        m_nodeTSDLineStride[tsd] = lineStride;
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

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    /* In DLA, a small cube can be read from/written to a larger cube
     * provided the strides are programmed correctly. Concat is one such
     * operation, where multiple smaller cubes can written into single
     * larger output cube provided the strides of the larger cube are
     * programmed into each of them.
     */
    {
        NvS32 sumConcatInWidth = 0;
        NvS32 sumConcatInHeight = 0;
        NvS32 sumConcatInChannel = 0;
        Dims4 largeSurfDims;
        surface::TensorSurfaceDesc tempConcatLargeSurface = *dstTSD;
        for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
        {
            Dims4 inSurfDims;
            Node* srcNode = graph()->upstreamNodes(*iei).size() ? graph()->upstreamNodes(*iei)[0] : NULL;
            if (srcNode)
            {
                inSurfDims = srcNode->suggestSurfaceDims((*iei)->tensorSurfaceDesc());
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No upstream node for input edge %s to node %s",
                        (*iei)->id().c_str(), name().c_str());
            }

            switch(params().concatAxis().v())
            {
                case ConcatAxisEnum::CONCAT_ALONG_W:
                    sumConcatInWidth += inSurfDims.w;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, sumConcatInWidth);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, inSurfDims.h);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, inSurfDims.c);
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_H:
                    sumConcatInHeight += inSurfDims.h;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, inSurfDims.w);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, sumConcatInHeight);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, inSurfDims.c);
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_C:
                    sumConcatInChannel += inSurfDims.c;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, inSurfDims.w);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, inSurfDims.h);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, sumConcatInChannel);
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown concat mode %s", params().concatAxis().c_str());
            }
        }

        tempConcatLargeSurface.setDimensions(largeSurfDims);
        tempConcatLargeSurface.resetLineStride();
        lineStride = tempConcatLargeSurface.lineStride();
    }

    m_nodeTSDLineStride[tsd] = lineStride;

fail:
    return lineStride;
}

NvU32 engine_ast::ConcatenationNode::suggestSurfaceStride(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 surfaceStride = 0;
    EdgeSequence inDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    bool isConcatChain = false;
    ConcatenationNode* sinkConcat = NULL;
    surface::TensorSurfaceDesc* currConcatOutTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    if (m_nodeTSDSurfaceStride.find(tsd) != m_nodeTSDSurfaceStride.end())
    {
        surfaceStride = m_nodeTSDSurfaceStride[tsd];
        goto fail;
    }

    isConcatChain = graph()->downstreamDataNodes(this).size() ?
                    graph()->downstreamDataNodes(this)[0]->engineType().v() == EngineTypeEnum::CONCATENATION :
                    false;
    if (isConcatChain)
    {
        sinkConcat = NodeFactory::nodeCast<ConcatenationNode*>(graph()->downstreamDataNodes(this)[0]);
        currConcatOutTSD = outputEdges()[0]->tensorSurfaceDesc();
        surfaceStride = sinkConcat->suggestSurfaceStride(currConcatOutTSD);
        m_nodeTSDSurfaceStride[tsd] = surfaceStride;
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

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    /* In DLA, a small cube can be read from/written to a larger cube
     * provided the strides are programmed correctly. Concat is one such
     * operation, where multiple smaller cubes can written into single
     * larger output cube provided the strides of the larger cube are
     * programmed into each of them.
     */
    {
        NvS32 sumConcatInWidth = 0;
        NvS32 sumConcatInHeight = 0;
        NvS32 sumConcatInChannel = 0;
        Dims4 largeSurfDims;
        surface::TensorSurfaceDesc tempConcatLargeSurface = *dstTSD;
        for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
        {
            Dims4 inSurfDims;
            Node* srcNode = graph()->upstreamNodes(*iei).size() ? graph()->upstreamNodes(*iei)[0] : NULL;
            if (srcNode)
            {
                inSurfDims = srcNode->suggestSurfaceDims((*iei)->tensorSurfaceDesc());
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No upstream node for input edge %s to node %s",
                        (*iei)->id().c_str(), name().c_str());
            }

            switch(params().concatAxis().v())
            {
                case ConcatAxisEnum::CONCAT_ALONG_W:
                    sumConcatInWidth += inSurfDims.w;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, sumConcatInWidth);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, inSurfDims.h);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, inSurfDims.c);
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_H:
                    sumConcatInHeight += inSurfDims.h;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, inSurfDims.w);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, sumConcatInHeight);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, inSurfDims.c);
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_C:
                    sumConcatInChannel += inSurfDims.c;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, inSurfDims.w);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, inSurfDims.h);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, sumConcatInChannel);
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown concat mode %s", params().concatAxis().c_str());
            }
        }

        tempConcatLargeSurface.setDimensions(largeSurfDims);
        tempConcatLargeSurface.resetSurfaceStride();
        surfaceStride = tempConcatLargeSurface.surfaceStride();
    }

    m_nodeTSDSurfaceStride[tsd] = surfaceStride;

fail:
    return surfaceStride;
}

NvU64 engine_ast::ConcatenationNode::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU64 size = 0;
    EdgeSequence inDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    surface::TensorSurfaceDesc* dstTSD = NULL;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    if (m_nodeTSDSurfaceSize.find(tsd) != m_nodeTSDSurfaceSize.end())
    {
        size = m_nodeTSDSurfaceSize[tsd];
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

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    {
        NvS32 sumConcatInWidth = 0;
        NvS32 sumConcatInHeight = 0;
        NvS32 sumConcatInChannel = 0;
        Dims4 largeSurfDims;
        NvS32 requestedEdgeWidth = 0;
        NvS32 requestedEdgeHeight = 0;
        NvS32 requestedEdgeChnls = 0;
        surface::TensorSurfaceDesc tempConcatLargeSurface = *dstTSD;
        for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
        {
            Dims4 inSurfDims;
            Node* srcNode = graph()->upstreamNodes(*iei).size() ? graph()->upstreamNodes(*iei)[0] : NULL;
            if (srcNode)
            {
                inSurfDims = srcNode->suggestSurfaceDims((*iei)->tensorSurfaceDesc());
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No upstream node for input edge %s to node %s",
                        (*iei)->id().c_str(), name().c_str());
            }

            switch(params().concatAxis().v())
            {
                case ConcatAxisEnum::CONCAT_ALONG_W:
                    sumConcatInWidth += inSurfDims.w;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, sumConcatInWidth);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, inSurfDims.h);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, inSurfDims.c);
                    if ((*iei)->tensorSurfaceDesc() == tsd)
                    {
                        requestedEdgeWidth = inSurfDims.w;
                    }
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_H:
                    sumConcatInHeight += inSurfDims.h;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, inSurfDims.w);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, sumConcatInHeight);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, inSurfDims.c);
                    if ((*iei)->tensorSurfaceDesc() == tsd)
                    {
                        requestedEdgeHeight = inSurfDims.h;
                    }
                    break;
                case ConcatAxisEnum::CONCAT_ALONG_C:
                    sumConcatInChannel += inSurfDims.c;
                    largeSurfDims.w = std::max<NvS32>(largeSurfDims.w, inSurfDims.w);
                    largeSurfDims.h = std::max<NvS32>(largeSurfDims.h, inSurfDims.h);
                    largeSurfDims.c = std::max<NvS32>(largeSurfDims.c, sumConcatInChannel);
                    if ((*iei)->tensorSurfaceDesc() == tsd)
                    {
                        requestedEdgeChnls = inSurfDims.c;
                    }
                    break;
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown concat mode %s", params().concatAxis().c_str());
            }
        }

        // if size for an individual smaller input edge is requested, then ensure to use
        // the #chnls/width/height represented by just that edge
        if (isSrcTSD)
        {
            switch(params().concatAxis().v())
            {
                case ConcatAxisEnum::CONCAT_ALONG_W: largeSurfDims.w = requestedEdgeWidth; break;
                case ConcatAxisEnum::CONCAT_ALONG_H: largeSurfDims.h = requestedEdgeHeight; break;
                case ConcatAxisEnum::CONCAT_ALONG_C: largeSurfDims.c = requestedEdgeChnls; break;
            }
        }

        tempConcatLargeSurface.setDimensions(largeSurfDims);
        tempConcatLargeSurface.resetSize();
        size = tempConcatLargeSurface.size();
    }

    m_nodeTSDSurfaceSize[tsd] = size;

fail:
    return size;
}

NvU64 engine_ast::ConcatenationNode::suggestOffsetInConcatChain(surface::TensorSurfaceDesc* inTSD)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU64 offsetInCommonBuff = 0;
    bool isConcatChain = graph()->downstreamDataNodes(this).size() ?
                         graph()->downstreamDataNodes(this)[0]->engineType().v() == EngineTypeEnum::CONCATENATION :
                         false;
    if (isConcatChain)
    {
        ConcatenationNode* sinkConcat = NodeFactory::nodeCast<ConcatenationNode*>(graph()->downstreamDataNodes(this)[0]);
        surface::TensorSurfaceDesc* currConcatOutTSD = outputEdges()[0]->tensorSurfaceDesc();
        offsetInCommonBuff += sinkConcat->suggestOffsetInConcatChain(currConcatOutTSD);
    }
    else
    {
        EdgeSequence inEdges = inputEdges();
        for (EdgeSequenceIterator iei = inEdges.begin(); iei != inEdges.end(); ++iei)
        {
            if ((*iei)->tensorSurfaceDesc() == inTSD)
            {
                break;
            }
            else
            {
                Dims4 inSurfDims;
                Node* srcNode = graph()->upstreamNodes(*iei).size() ? graph()->upstreamNodes(*iei)[0] : NULL;
                if (srcNode)
                {
                    inSurfDims = srcNode->suggestSurfaceDims((*iei)->tensorSurfaceDesc());
                }
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No upstream node for input edge %s to node %s",
                            (*iei)->id().c_str(), name().c_str());
                }

                switch (params().concatAxis().v())
                {
                    case ConcatAxisEnum::CONCAT_ALONG_W:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support concat along W dimension yet for %s", name().c_str());
                        break;
                    case ConcatAxisEnum::CONCAT_ALONG_H:
                        offsetInCommonBuff += (*iei)->tensorSurfaceDesc()->lineStride() * inSurfDims.h;
                        break;
                    case ConcatAxisEnum::CONCAT_ALONG_C:
                        offsetInCommonBuff += (*iei)->tensorSurfaceDesc()->size();
                        break;
                    default:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unknown concat mode for %s", name().c_str());
                }
            }
        }
    }

fail:
    return offsetInCommonBuff;
}

NvU64 engine_ast::ConcatenationNode::suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU64 offsetInCommonBuff = 0;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    EdgeSequence outEdges;
    EdgeSequence inEdges;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    if (m_nodeTSDSurfaceOffsetInBuffer.find(tsd) != m_nodeTSDSurfaceOffsetInBuffer.end())
    {
        offsetInCommonBuff = m_nodeTSDSurfaceOffsetInBuffer[tsd];
        goto fail;
    }

    outEdges = outputEdges();
    inEdges  = inputEdges();

    for (EdgeSequenceIterator iei = inEdges.begin(); iei != inEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (isSrcTSD)
    {
        EdgeSequenceIterator iei;
        for (iei = inEdges.begin(); iei != inEdges.end(); ++iei)
        {
            if ((*iei)->tensorSurfaceDesc() == tsd)
            {
                break;
            }
            else
            {
                Dims4 inSurfDims;
                Node* srcNode = graph()->upstreamNodes(*iei).size() ? graph()->upstreamNodes(*iei)[0] : NULL;
                if (srcNode)
                {
                    inSurfDims = srcNode->suggestSurfaceDims((*iei)->tensorSurfaceDesc());
                }
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No upstream node for input edge %s to node %s",
                            (*iei)->id().c_str(), name().c_str());
                }

                switch (params().concatAxis().v())
                {
                    case ConcatAxisEnum::CONCAT_ALONG_W:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support concat along W dimension yet for %s", name().c_str());
                        break;
                    case ConcatAxisEnum::CONCAT_ALONG_H:
                        offsetInCommonBuff += (suggestLineStride((*iei)->tensorSurfaceDesc()) * inSurfDims.h);
                        break;
                    case ConcatAxisEnum::CONCAT_ALONG_C:
                        offsetInCommonBuff += suggestSurfaceSize((*iei)->tensorSurfaceDesc());
                        break;
                    default:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unknown concat mode for %s", name().c_str());
                }
            }
        }
        // add to the calculated offset, the offset of the output edge since it could be part of a larger buffer
        offsetInCommonBuff += suggestSurfaceOffsetInBuffer(outputEdges()[0]->tensorSurfaceDesc());

        if ( debugConcat() )
        {
            gLogInfo << "(" << name() << ") input edge " << (*iei)->id()
                     << "(" << tsd->dimensions().c << "x" << tsd->dimensions().h << "x" << tsd->dimensions().w << ")"
                     << " is at offset " << offsetInCommonBuff
                     << " in its buffer: " << (tsd->tensorBufferDesc() ? tsd->tensorBufferDesc()->id() : "TBR") << endl;
        }
    }
    else if (isDstTSD)
    {
        offsetInCommonBuff = 0;
        // if output of concat is part of a downstream concat, it may be at a non-0 offset in the
        // revalent buffer; return presiding buffer offset in that case
        bool isConcatChain = graph()->downstreamDataNodes(this).size() ?
                             graph()->downstreamDataNodes(this)[0]->engineType().v() == EngineTypeEnum::CONCATENATION :
                             false;
        if (isConcatChain)
        {
            ConcatenationNode* sinkConcat = NodeFactory::nodeCast<ConcatenationNode*>(graph()->downstreamDataNodes(this)[0]);
            surface::TensorSurfaceDesc* currConcatDstTSD = tsd;
            offsetInCommonBuff = sinkConcat->suggestOffsetInConcatChain(currConcatDstTSD);
        }
        else
        {
            offsetInCommonBuff = 0;
        }

        if ( debugConcat() )
        {
            gLogInfo << "(" << name() << ") output edge " << outputEdges().at(0)->id()
                     << "(" << tsd->dimensions().c << "x" << tsd->dimensions().h << "x" << tsd->dimensions().w << ")"
                     << " is at offset " << offsetInCommonBuff
                     << " in its buffer: " << (tsd->tensorBufferDesc() ? tsd->tensorBufferDesc()->id() : "TBR") << endl;
        }
    }

    m_nodeTSDSurfaceOffsetInBuffer[tsd] = offsetInCommonBuff;

fail:
    return offsetInCommonBuff;
}

memory::TensorBufferDesc* engine_ast::ConcatenationNode::suggestBuffer(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    memory::TensorBufferDesc* commonTBD = NULL;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    EdgeSequence outEdges;
    EdgeSequence inEdges;
    NvU16 numBatches = graph()->profile()->multiBatchSize();

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    outEdges = outputEdges();
    inEdges  = inputEdges();

    for (EdgeSequenceIterator iei = inEdges.begin(); iei != inEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }
    else if ( tsd->tensorCategory().v() == memory::TensorCategoryEnum::UNKNOWN_TENSOR )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Edge %s has 'unknown' tensor category",
                             tsd->id().c_str());
    }

    // hunt across all edges to find a common TBD if any was registered before
    commonTBD = isDstTSD ? tsd->tensorBufferDesc() : outputEdges()[0]->tensorBufferDesc();
    if ( !commonTBD )
    {
        for (EdgeSequenceIterator iei = inEdges.begin(); iei != inEdges.end(); ++iei)
        {
            if ((*iei)->tensorBufferDesc())
            {
                commonTBD = (*iei)->tensorBufferDesc();
                break;
            }
        }
    }

    if ( !commonTBD )
    {
        commonTBD = graph()->resourceMgr()->regTensorBufferDesc(numBatches);
    }

fail:
    return commonTBD;
}

/* A concat-node could be receiving inputs from operations following different
 * limitations (For eg: some of the upstream ops could be winograd and some not).
 * Hence accomodate the worst case scenarios to perform this check
 */
NvDlaError engine_ast::ConcatenationNode::verifySurfaceIsPartOfConcat
(
    surface::TensorSurfaceDesc* srcTSD,
    surface::TensorSurfaceDesc* dstTSD,
    engine_ast::ConcatAxis axis
)
{
    NvDlaError e = NvDlaSuccess;
    Dims4 srcDims = suggestSurfaceDims(srcTSD);
    Dims4 dstDims = suggestSurfaceDims(dstTSD);

    if (axis.v() == engine_ast::ConcatAxisEnum::CONCAT_ALONG_C)
    {
        /* some of the src tensors could be smaller in WxH than the dst
         * if at least one of the concat participants is Winogradable
         */
        if (srcDims.w > dstDims.w || srcDims.h > dstDims.h)
        {
            gLogError << "In-" << srcTSD->id() << " WxH : "
                      << srcDims.w << "x" << srcDims.h << endl;
            gLogError << "Out-" << dstTSD->id() << " WxH : "
                      << dstDims.w << "x" << dstDims.h << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Concat along C should have "
                    "WxH of all input edges <= that of the output tensor", name().c_str());
        }
    }
    else if (axis.v() == engine_ast::ConcatAxisEnum::CONCAT_ALONG_H)
    {
        if (srcDims.w != dstDims.w || srcDims.c != dstDims.c)
        {
            gLogError << "In-" << srcTSD->id() << " WxC : "
                      << srcDims.w << "x" << srcDims.c << endl;
            gLogError << "Out-" << dstTSD->id() << " WxC : "
                      << dstDims.w << "x" << dstDims.c << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Concat along H should have "
                    "all edges of the same WxC", name().c_str());
        }
    }
    else if (axis.v() == engine_ast::ConcatAxisEnum::CONCAT_ALONG_W)
    {
        if (srcDims.h != dstDims.h || srcDims.c != dstDims.c)
        {
            gLogError << "In-" << srcTSD->id() << " HxC : "
                      << srcDims.h << "x" << srcDims.c << endl;
            gLogError << "Out-" << dstTSD->id() << " HxC : "
                      << dstDims.h << "x" << dstDims.c << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Concat along W should have "
                    "all edges of the same HXC", name().c_str());
        }
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown concat axis: %s", axis.c_str());
    }

fail:
     return e;
}

NvDlaError engine_ast::ConcatenationNode::verifySurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    EdgeSequence inDataEdges;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;

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

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : NULL;
    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    if (isSrcTSD)
    {
        PROPAGATE_ERROR_FAIL( verifySurfaceIsPartOfConcat(srcTSD, dstTSD, params().concatAxis()) );
    }
    else
    {
        for (EdgeSequenceIterator iei = inDataEdges.begin(); iei != inDataEdges.end(); ++iei)
        {
            surface::TensorSurfaceDesc* srcTSD = (*iei)->tensorSurfaceDesc();
            PROPAGATE_ERROR_FAIL( verifySurfaceIsPartOfConcat(srcTSD, dstTSD, params().concatAxis()) );
        }
    }

fail:
    return e;
}

};  // nvdla::priv::
};  // nvdla::
