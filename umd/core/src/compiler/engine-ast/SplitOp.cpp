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
#include "priv/Surface.h"

using std::endl;

namespace nvdla
{
namespace priv
{

void engine_ast::SplitNode::captureCanonicalParams()
{
    // default to split along Channel direction.
    // TODO: For other modes, add support in network/layer and canonical ast too
    params().setSplitAxis(SplitAxisEnum::SPLIT_ALONG_C);
}

// not idempotent since relies on some canonical details
NvDlaError engine_ast::SplitNode::populateEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    typedef engine_ast::Graph::EdgeSequence EngineEdges;
    typedef engine_ast::Graph::EdgeSequenceIterator EngineEdgeIterator;

    typedef canonical_ast::Graph::EdgeSequence CanonicalEdges;
    typedef canonical_ast::Graph::EdgeSequenceIterator CanonicalEdgeIterator;

    EngineEdges engInEdges = graph()->upstreamDataEdges(this);
    EngineEdges engOutEdges = graph()->downstreamDataEdges(this);


    if (engInEdges.size() > 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s has > 1 input edges", name().c_str());
    }
    else
    {
        markInputEdge(engInEdges[0]);
    }

    // if canonical equivalent exists, use order of output edge insertions in canonical land to establish strict order
    if (canonicalNode())
    {
        CanonicalEdges canOutEdges = canonicalNode()->outputEdges();
        for (CanonicalEdgeIterator cei = canOutEdges.begin(); cei != canOutEdges.end(); ++cei)
        {
            IsSameCanonicalEdge match_can_edge(*cei);
            EngineEdgeIterator eei = std::find_if(engOutEdges.begin(), engOutEdges.end(), match_can_edge);
            if (eei == engOutEdges.end())
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s is not an output edge of %s", (*eei)->id().c_str(), name().c_str());
            }
            markOutputEdge(*eei);
        }
    }
    // else simply mark output edges in order of appearance
    else
    {
        for (EngineEdgeIterator eei = engOutEdges.begin(); eei != engOutEdges.end(); ++eei)
        {
            markOutputEdge(*eei);
        }
    }

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());
fail:
    return e;
}

Dims4 engine_ast::SplitNode::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    EdgeSequence outDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    Node* srcNode = NULL;
    Dims4 srcSuggestedDims(-1,-1,-1,-1);
    Dims4 returnSuggestedDims(-1,-1,-1,-1);
    Dims4 tsdActualDims = tsd->dimensions();

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    outDataEdges = outputEdges();

    for (EdgeSequenceIterator oei = outDataEdges.begin(); oei != outDataEdges.end(); ++oei)
    {
        if ((*oei)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    // split node doesn't affect any tensor dimensions,
    // except just distributing them to the downstream nodes.
    // as a result, respect the suggested dims from upstream node if any
    srcNode = graph()->upstreamNodes(inputEdges()[0]).size() ?
              graph()->upstreamNodes(inputEdges()[0])[0] : NULL;
    if (srcNode)
    {
        srcSuggestedDims = srcNode->suggestSurfaceDims(inputEdges()[0]->tensorSurfaceDesc());
    }

    switch(params().splitAxis().v())
    {
        case SplitAxisEnum::SPLIT_ALONG_W:
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support split along Width direction for %s",
                    name().c_str());
            break;
        case SplitAxisEnum::SPLIT_ALONG_H:
            returnSuggestedDims.w = std::max<NvS32>(tsdActualDims.w, srcSuggestedDims.w);
            returnSuggestedDims.h = tsdActualDims.h;
            returnSuggestedDims.c = std::max<NvS32>(tsdActualDims.c, srcSuggestedDims.c);
            break;
        case SplitAxisEnum::SPLIT_ALONG_C:
            returnSuggestedDims.w = std::max<NvS32>(tsdActualDims.w, srcSuggestedDims.w);
            returnSuggestedDims.h = std::max<NvS32>(tsdActualDims.h, srcSuggestedDims.h);
            returnSuggestedDims.c = tsdActualDims.c;
            break;
        case SplitAxisEnum::SPLIT_ALONG_NONE:
            // split-along-none is a pass through
            returnSuggestedDims.w = std::max<NvS32>(tsdActualDims.w, srcSuggestedDims.w);
            returnSuggestedDims.h = std::max<NvS32>(tsdActualDims.h, srcSuggestedDims.h);
            returnSuggestedDims.c = std::max<NvS32>(tsdActualDims.c, srcSuggestedDims.c);
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue,  "Unknown split mode %s for %s",
                    params().splitAxis().c_str(), name().c_str());
    }

fail:
    return returnSuggestedDims;
}

NvU32 engine_ast::SplitNode::suggestLineStride(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 lineStride  = 0;
    EdgeSequence outDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDLineStride.find(tsd) != m_nodeTSDLineStride.end())
    {
        lineStride = m_nodeTSDLineStride[tsd];
        goto fail;
    }

    outDataEdges = outputEdges();

    for (EdgeSequenceIterator oei = outDataEdges.begin(); oei != outDataEdges.end(); ++oei)
    {
        if ((*oei)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();

    /* In DLA, a small cube can be read from/written to a larger cube
     * provided the sizes are programmed correctly. Split is one such
     * operation, where multiple smaller cubes can be read from a single
     * larger input cube provided the strides of the larger cube are
     * programmed into each of them.
     * Linestride is same for all input and output tensors of split node;
     * request the LS suggestion from the upstream node if any
     */
    lineStride = graph()->upstreamNodes(inputEdges()[0]).size() ?
                 graph()->upstreamNodes(inputEdges()[0])[0]->suggestLineStride(srcTSD) :
                 tsd->lineStride();

    m_nodeTSDLineStride[tsd] = lineStride;

fail:
    return lineStride;
}

NvU32 engine_ast::SplitNode::suggestSurfaceStride(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 surfaceStride  = 0;
    EdgeSequence outDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDSurfaceStride.find(tsd) != m_nodeTSDSurfaceStride.end())
    {
        surfaceStride = m_nodeTSDSurfaceStride[tsd];
        goto fail;
    }

    outDataEdges = outputEdges();

    for (EdgeSequenceIterator oei = outDataEdges.begin(); oei != outDataEdges.end(); ++oei)
    {
        if ((*oei)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();

    /* In DLA, a small cube can be read from/written to a larger cube
     * provided the sizes are programmed correctly. Split is one such
     * operation, where multiple smaller cubes can be read from a single
     * larger input cube provided the strides of the larger cube are
     * programmed into each of them.
     * Surfacestride is same for all input and output tensors of split node;
     * request the SS suggestion from the upstream node if any
     */
    surfaceStride = graph()->upstreamNodes(inputEdges()[0]).size() ?
                    graph()->upstreamNodes(inputEdges()[0])[0]->suggestSurfaceStride(srcTSD) :
                    srcTSD->surfaceStride();

    m_nodeTSDSurfaceStride[tsd] = surfaceStride;

fail:
    return surfaceStride;
}

NvU64 engine_ast::SplitNode::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    NvU64 size = 0;
    EdgeSequence outDataEdges;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    EdgeSequenceIterator outEdgeItr;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    if (m_nodeTSDSurfaceSize.find(tsd) != m_nodeTSDSurfaceSize.end())
    {
        size = m_nodeTSDSurfaceSize[tsd];
        goto fail;
    }

    outDataEdges = outputEdges();
    for (outEdgeItr = outDataEdges.begin(); outEdgeItr != outDataEdges.end(); ++outEdgeItr)
    {
        if ((*outEdgeItr)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();

    if (isDstTSD)
    {
        Dims4 largeSurfDims;
        Dims4 inSurfDims;
        surface::TensorSurfaceDesc tempSplitLargeSurface = *srcTSD;
        Node* srcNode = graph()->upstreamNodes(inputEdges()[0]).size() ?
                        graph()->upstreamNodes(inputEdges()[0])[0] : NULL;
        if (srcNode)
        {
            inSurfDims = srcNode->suggestSurfaceDims(inputEdges()[0]->tensorSurfaceDesc());
        }
        else
        {
            inSurfDims = inputEdges()[0]->tensorSurfaceDesc()->dimensions();
        }

        switch(params().splitAxis().v())
        {
            case SplitAxisEnum::SPLIT_ALONG_W:
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support split along Width direction for %s",
                        name().c_str());
                break;
            case SplitAxisEnum::SPLIT_ALONG_H:
                largeSurfDims.w = std::max<NvS32>(tsd->dimensions().w, inSurfDims.w);
                largeSurfDims.h = tsd->dimensions().h;
                largeSurfDims.c = std::max<NvS32>(tsd->dimensions().c, inSurfDims.c);
                break;
            case SplitAxisEnum::SPLIT_ALONG_C:
                largeSurfDims.w = std::max<NvS32>(tsd->dimensions().w, inSurfDims.w);
                largeSurfDims.h = std::max<NvS32>(tsd->dimensions().h, inSurfDims.h);
                largeSurfDims.c = tsd->dimensions().c;
                break;
            case SplitAxisEnum::SPLIT_ALONG_NONE:
                // split-along-none is a pass through
                largeSurfDims.w = inSurfDims.w;
                largeSurfDims.h = inSurfDims.h;
                largeSurfDims.c = inSurfDims.c;
                break;
            default:
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown split mode %s for %s",
                        params().splitAxis().c_str(), name().c_str());
        }

        tempSplitLargeSurface.setDimensions(largeSurfDims);
        tempSplitLargeSurface.resetSize();
        size = tempSplitLargeSurface.size();
    }
    else if (isSrcTSD)
    {
        size = graph()->upstreamNodes(inputEdges()[0]).size() ?
               graph()->upstreamNodes(inputEdges()[0])[0]->suggestSurfaceSize(tsd) :
               tsd->size();
    }

    m_nodeTSDSurfaceSize[tsd] = size;

fail:
    return size;
}

NvU64 engine_ast::SplitNode::suggestOffsetInSplitChain(surface::TensorSurfaceDesc* outTSD)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU64 offsetInCommonBuff = 0;
    bool isSplitChain = graph()->upstreamDataNodes(this).size() ?
                        graph()->upstreamDataNodes(this)[0]->engineType().v() == EngineTypeEnum::SPLIT :
                        false;
    if (isSplitChain)
    {
        SplitNode* srcSplit = NodeFactory::nodeCast<SplitNode*>(graph()->upstreamDataNodes(this)[0]);
        surface::TensorSurfaceDesc* currSplitSrcTSD = inputEdges()[0]->tensorSurfaceDesc();
        offsetInCommonBuff += srcSplit->suggestOffsetInSplitChain(currSplitSrcTSD);
    }
    else
    {
        EdgeSequence outEdges = outputEdges();
        for (EdgeSequenceIterator oei = outEdges.begin(); oei != outEdges.end(); ++oei)
        {
            if ((*oei)->tensorSurfaceDesc() == outTSD)
            {
                break;
            }
            else
            {
                Dims4 outSurfDims;
                Node* sinkNode = graph()->downstreamNodes(*oei).size() ? graph()->downstreamNodes(*oei)[0] : NULL;
                if (sinkNode)
                {
                    outSurfDims = sinkNode->suggestSurfaceDims((*oei)->tensorSurfaceDesc());
                }
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No downstream node for output edge %s to node %s",
                            (*oei)->id().c_str(), name().c_str());
                }

                switch (params().splitAxis().v())
                {
                    case SplitAxisEnum::SPLIT_ALONG_W:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't yet support split along W dimension for %s", name().c_str());
                        break;
                    case SplitAxisEnum::SPLIT_ALONG_H:
                        offsetInCommonBuff += ((*oei)->tensorSurfaceDesc()->lineStride() * outSurfDims.h);
                        break;
                    case SplitAxisEnum::SPLIT_ALONG_C:
                        offsetInCommonBuff += (*oei)->tensorSurfaceDesc()->size();
                        break;
                    case SplitAxisEnum::SPLIT_ALONG_NONE:
                        // split-along-none is a pass through
                        offsetInCommonBuff += 0;
                        break;
                    default:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unknown split mode for %s", name().c_str());
                }
            }
        }
    }

fail:
    return offsetInCommonBuff;
}

NvU64 engine_ast::SplitNode::suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd)
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

    for (EdgeSequenceIterator oei = outEdges.begin(); oei != outEdges.end(); ++oei)
    {
        if ((*oei)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inEdges[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (isDstTSD)
    {
        Dims4 inSurfDims;
        EdgeSequenceIterator outEdgeItr;
        Node* srcNode = graph()->upstreamNodes(inputEdges()[0]).size() ?
                        graph()->upstreamNodes(inputEdges()[0])[0] : NULL;
        if (srcNode)
        {
            inSurfDims = srcNode->suggestSurfaceDims(inputEdges()[0]->tensorSurfaceDesc());
        }
        else
        {
            inSurfDims = inputEdges()[0]->tensorSurfaceDesc()->dimensions();
        }

        for (outEdgeItr = outEdges.begin(); outEdgeItr != outEdges.end(); ++outEdgeItr)
        {
            if ((*outEdgeItr)->tensorSurfaceDesc() == tsd)
            {
                break;
            }
            else
            {
                Dims4 outSurfDims;
                Node* sinkNode = graph()->downstreamNodes(*outEdgeItr).size() ?
                                 graph()->downstreamNodes(*outEdgeItr)[0] : NULL;
                if (sinkNode)
                {
                    outSurfDims = sinkNode->suggestSurfaceDims((*outEdgeItr)->tensorSurfaceDesc());
                }
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No downstream node for output edge %s of %s",
                            (*outEdgeItr)->id().c_str(), name().c_str());
                }

                switch(params().splitAxis().v())
                {
                    case SplitAxisEnum::SPLIT_ALONG_W:
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Dont yet support split along Widht direction for %s",
                                name().c_str());
                        break;
                    case SplitAxisEnum::SPLIT_ALONG_H:
                        offsetInCommonBuff += (suggestLineStride((*outEdgeItr)->tensorSurfaceDesc()) * outSurfDims.h);
                        break;
                    case SplitAxisEnum::SPLIT_ALONG_C:
                        offsetInCommonBuff += suggestSurfaceSize((*outEdgeItr)->tensorSurfaceDesc());
                        break;
                    case SplitAxisEnum::SPLIT_ALONG_NONE:
                        // split-along-none is a pass through
                        offsetInCommonBuff = 0;
                        break;
                    default:
                        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown split mode %s for %s",
                                params().splitAxis().c_str(), name().c_str());
                }
            }
        }

        if (params().splitAxis().v() == SplitAxisEnum::SPLIT_ALONG_H)
        {
            NvS32 sumSplitOutHeight = 0;
            for (EdgeSequenceIterator oei = outEdges.begin(); oei != outEdges.end(); ++oei)
            {
                Dims4 surfDims;
                Node* sink = graph()->downstreamNodes(*oei).size() ?
                             graph()->downstreamNodes(*oei)[0] : NULL;
                if (sink)
                {
                    surfDims = sink->suggestSurfaceDims((*oei)->tensorSurfaceDesc());
                }
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No downstream node for output edge %s of %s",
                            (*oei)->id().c_str(), name().c_str());
                }
                sumSplitOutHeight += surfDims.h;
            }

            if ((sumSplitOutHeight == inSurfDims.h) &&
                (sumSplitOutHeight == inputEdges()[0]->tensorSurfaceDesc()->dimensions().h))
            {
                // this means there are no overlapping slices in H-direction
                // among the split out edges; in this case offset of a split tensor
                // in common buffer is deterministic
            }
            else
            {
                // otherwise there are overlapping slices;
                // in this case, offset of a split tensor in common buffer is undeterministic
                // so, consider the offset calculated by the client on the other end of the tsd as gospel
                Node* sinkNode = graph()->downstreamNodes((*outEdgeItr)).size() ?
                                 graph()->downstreamNodes((*outEdgeItr))[0] : NULL;
                if (!sinkNode)
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No sink node for output edge %s of %s",
                            (*outEdgeItr)->id().c_str(), name().c_str());
                }
                offsetInCommonBuff = sinkNode->suggestSurfaceOffsetInBuffer(tsd);
            }
        }
        // add to the calculated offset, the offset of the input edge since it could be part of a larger buffer
        offsetInCommonBuff += suggestSurfaceOffsetInBuffer(inputEdges()[0]->tensorSurfaceDesc());

        if ( debugSplit() )
        {
            gLogInfo << "(" << name() << ") output edge " << (*outEdgeItr)->id()
                     << "(" << tsd->dimensions().c << "x" << tsd->dimensions().h << "x" << tsd->dimensions().w << ")"
                     << " is at offset " << offsetInCommonBuff
                     << " in its buffer: " << (tsd->tensorBufferDesc() ? tsd->tensorBufferDesc()->id() : "TBR") << endl;
        }
    }
    else if (isSrcTSD)
    {
        offsetInCommonBuff = 0;
        // if input of split is part of an upstream split, it may be at a non-0 offset in the
        // revalent buffer; return presiding buffer offset in that case
        bool isSplitChain = graph()->upstreamDataNodes(this).size() ?
                            graph()->upstreamDataNodes(this)[0]->engineType().v() == EngineTypeEnum::SPLIT :
                            false;
        if (isSplitChain)
        {
            SplitNode* srcSplit = NodeFactory::nodeCast<SplitNode*>(graph()->upstreamDataNodes(this)[0]);
            surface::TensorSurfaceDesc* currSplitSrcTSD = tsd;
            offsetInCommonBuff = srcSplit->suggestOffsetInSplitChain(currSplitSrcTSD);
        }
        else
        {
            offsetInCommonBuff = 0;
        }

        if ( debugSplit() )
        {
            gLogInfo << "(" << name() << ") input edge " << inputEdges().at(0)->id()
                     << "(" << tsd->dimensions().c << "x" << tsd->dimensions().h << "x" << tsd->dimensions().w << ")"
                     << " is at offset " << offsetInCommonBuff
                     << " in its buffer: " << (tsd->tensorBufferDesc() ? tsd->tensorBufferDesc()->id() : "TBR") << endl;
        }
    }

    m_nodeTSDSurfaceOffsetInBuffer[tsd] = offsetInCommonBuff;

fail:
    return offsetInCommonBuff;
}

memory::TensorBufferDesc* engine_ast::SplitNode::suggestBuffer(surface::TensorSurfaceDesc* tsd)
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

    for (EdgeSequenceIterator oei = outEdges.begin(); oei != outEdges.end(); ++oei)
    {
        if ((*oei)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;

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
    commonTBD = isSrcTSD ? tsd->tensorBufferDesc() : inputEdges()[0]->tensorBufferDesc();
    if ( !commonTBD )
    {
        for (EdgeSequenceIterator oei = outEdges.begin(); oei != outEdges.end(); ++oei)
        {
            if ((*oei)->tensorBufferDesc())
            {
                commonTBD = (*oei)->tensorBufferDesc();
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

NvDlaError engine_ast::SplitNode::verifySurfaceIsPartOfSplit
(
    surface::TensorSurfaceDesc* srcTSD,
    surface::TensorSurfaceDesc* dstTSD,
    engine_ast::SplitAxis axis
)
{
    NvDlaError e = NvDlaSuccess;
    Dims4 srcDims = suggestSurfaceDims(srcTSD);
    Dims4 dstDims = suggestSurfaceDims(dstTSD);

    if (axis.v() == engine_ast::SplitAxisEnum::SPLIT_ALONG_C)
    {
        if (srcDims.w != dstDims.w || srcDims.h != dstDims.h)
        {
            gLogError << "In-" << srcTSD->id() << " WxH : "
                      << srcDims.w << "x" << srcDims.h << endl;
            gLogError << "Out-" << dstTSD->id() << " WxH : "
                      << dstDims.w << "x" << dstDims.h << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Split along C should have "
                    "all edges of the same WxH");
        }
    }
    else if (axis.v() == engine_ast::SplitAxisEnum::SPLIT_ALONG_H)
    {
        if (srcDims.w != dstDims.w || srcDims.c != dstDims.c)
        {
            gLogError << "In-" << srcTSD->id() << " WxC : "
                      << srcDims.w << "x" << srcDims.c << endl;
            gLogError << "Out-" << dstTSD->id() << " WxC : "
                      << dstDims.w << "x" << dstDims.c << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Split along H should have "
                    "all edges of the same WxC");
        }
    }
    else if (axis.v() == engine_ast::SplitAxisEnum::SPLIT_ALONG_W)
    {
        if (srcDims.h != dstDims.h || srcDims.c != dstDims.c)
        {
            gLogError << "In-" << srcTSD->id() << " HxC : "
                      << srcDims.h << "x" << srcDims.c << endl;
            gLogError << "Out-" << dstTSD->id() << " HxC : "
                      << dstDims.h << "x" << dstDims.c << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Split along W should have "
                    "all edges of the same HXC");
        }
    }
    else if (axis.v() == engine_ast::SplitAxisEnum::SPLIT_ALONG_NONE)
    {
        if (srcDims.w != dstDims.w || srcDims.h != dstDims.h || srcDims.c != dstDims.c)
        {
            gLogError << "In-" << srcTSD->id() << " WxHxC : "
                      << srcDims.w << "x" << srcDims.h << "x" << srcDims.c << endl;
            gLogError << "Out-" << dstTSD->id() << " WxHxC : "
                      << dstDims.w << "x" << dstDims.h << "x" << dstDims.c << endl;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Split along none should have "
                    "all edges of the same WxHXC");
        }
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown split axis: %s", axis.c_str());
    }

fail:
    return e;
}

NvDlaError engine_ast::SplitNode::verifySurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    EdgeSequence outDataEdges;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    outDataEdges = outputEdges();
    for (EdgeSequenceIterator oei = outDataEdges.begin(); oei != outDataEdges.end(); ++oei)
    {
        if ((*oei)->tensorSurfaceDesc() == tsd)
        {
            isDstTSD = true;
            break;
        }
    }
    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();
    dstTSD = isDstTSD ? tsd : NULL;

    if (isSrcTSD)
    {
        for (EdgeSequenceIterator oei = outDataEdges.begin(); oei != outDataEdges.end(); ++oei)
        {
            surface::TensorSurfaceDesc* dstTSD = (*oei)->tensorSurfaceDesc();
            PROPAGATE_ERROR_FAIL( verifySurfaceIsPartOfSplit(srcTSD, dstTSD, params().splitAxis()) );
        }
    }
    else
    {
        PROPAGATE_ERROR_FAIL( verifySurfaceIsPartOfSplit(srcTSD, dstTSD, params().splitAxis()) );
    }

fail:
    return e;
}
};  // nvdla::priv::
};  // nvdla::
