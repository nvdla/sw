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
#include <math.h>

#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

using std::endl;
using std::max;


namespace nvdla
{
namespace priv
{

/* TODO: If in future PDP engine could run >1 operation,
 * strip them into separate files and just keep the common stuff here
 */

void engine_ast::PDPNode::captureCanonicalParams()
{
    params().setPoolingType(canonicalNode()->params().poolType());
    params().setTopLeftPadding(canonicalNode()->params().topLeftPadding());
    params().setBottomRightPadding(canonicalNode()->params().bottomRightPadding());
    params().setPaddingValue(0);
    params().setStride(canonicalNode()->params().stride());
    params().setPoolingWindow(canonicalNode()->params().kernelDims());
}

NvDlaError engine_ast::PDPNode::verifySurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    bool isSrcTSD, isDstTSD;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;
    Dims4 poolOutDimsFromNw;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();
    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    poolOutDimsFromNw = outputEdges()[0]->originalTensor()->getDimensions();

    if (!srcTSD->bindable() && !dstTSD->bindable() && (srcTSD->dimensions().n != dstTSD->dimensions().n))
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Non-bindable Input and Output tensors should have the same "
                "#batches (N)", name().c_str());
    }
    else if (srcTSD->dimensions().c != dstTSD->dimensions().c)
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Input and Output tensors should have the same "
                "#channels (C)", name().c_str());
    }
    else if(dstTSD->dimensions().w != poolOutDimsFromNw.w ||
            dstTSD->dimensions().h != poolOutDimsFromNw.h)
    {
        gLogError << "Nw pool out dims " << poolOutDimsFromNw.w << "x" << poolOutDimsFromNw.h << endl;
        gLogError << "PDP out dims " << dstTSD->dimensions().w << "x" << dstTSD->dimensions().h << endl;
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Out dims determined from compiler have diverged"
                    " from that expected by original network", name().c_str());
    }

fail:
    return e;
}

//----------------------------------------------------------------------
//                      Formula Book
//----------------------------------------------------------------------
void engine_ast::PDPNode::adjustBRPadding()
{
    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    Dims2 brPadding         = params().bottomRightPadding();
    Dims2 tlPadding         = params().topLeftPadding();
    NvS32 kernelW           = params().poolingWindow().w;
    NvS32 kernelH           = params().poolingWindow().h;
    NvS32 strideX           = params().stride().w;
    NvS32 strideY           = params().stride().h;
    NvS32 leftPadding       = tlPadding.w;
    NvS32 topPadding        = tlPadding.h;
    NvS32 pdpInputWidth     = srcTSD->dimensions().w;
    NvS32 pdpInputHeight    = srcTSD->dimensions().h;
    NvS32 pdpOutputWidth    = dstTSD->dimensions().w;
    NvS32 pdpOutputHeight   = dstTSD->dimensions().h;
    NvS32 newRightPadding   = 0;
    NvS32 newBottomPadding  = 0;

    newRightPadding = ((pdpOutputWidth - 1)*strideX + kernelW) - (pdpInputWidth + leftPadding);
    newBottomPadding = ((pdpOutputHeight - 1)*strideY + kernelH) - (pdpInputHeight + topPadding);

    brPadding.w = newRightPadding;
    brPadding.h = newBottomPadding;

    params().setBottomRightPadding(brPadding);
}

NvDlaError engine_ast::PDPNode::verifySplitWInfo
(
    engine_ast::PDPEngineParams::hwSplitWidthInfo splitWInfo
)
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    Dims2 tlPadding         = params().topLeftPadding();
    NvS32 kernelW           = params().poolingWindow().w;
    NvS32 kernelH           = params().poolingWindow().h;
    NvS32 strideX           = params().stride().w;
    NvS32 leftPadding       = tlPadding.w;
    NvS32 pdpInputWidth     = srcTSD->dimensions().w;
    NvS32 pdpOutputWidth    = dstTSD->dimensions().w;
    NvS32 maxFlyingWidth    = calculateMaxWidth(srcTSD->surfaceFormat().precision(),
                                             params().poolingWindow().h,
                                             params().stride().h);

    NvS32 totalWidth = pdpInputWidth + leftPadding + splitWInfo.rightPadding;

    ASSERT(splitWInfo.firstOutWidth <= maxFlyingWidth);
    ASSERT(splitWInfo.midOutWidth <= maxFlyingWidth);
    ASSERT(splitWInfo.lastOutWidth <= maxFlyingWidth);

    if (splitWInfo.numSplits == 1)
    {
        ASSERT(splitWInfo.firstOutWidth == pdpOutputWidth);
        ASSERT((splitWInfo.firstInWidth + leftPadding) == ((splitWInfo.firstOutWidth-1)*strideX + kernelW));
    }
    else if (splitWInfo.numSplits == 2)
    {
        ASSERT((splitWInfo.firstOutWidth + splitWInfo.lastOutWidth) == pdpOutputWidth);
        ASSERT((splitWInfo.firstInWidth + leftPadding) == ((splitWInfo.firstOutWidth-1)*strideX + kernelW));
        ASSERT((splitWInfo.lastInWidth  + splitWInfo.rightPadding + splitWInfo.numOverlapStripes) == ((splitWInfo.lastOutWidth-1)*strideX + kernelW));
        if (kernelW >= strideX)
        {
            ASSERT((kernelW - strideX) <= splitWInfo.firstInWidth);
        }
        else
        {
            ASSERT((strideX - kernelW) < splitWInfo.lastInWidth);
        }
    }
    else
    {
        ASSERT((splitWInfo.firstOutWidth + splitWInfo.lastOutWidth + splitWInfo.midOutWidth*(splitWInfo.numSplits-2)) == pdpOutputWidth);
        ASSERT((splitWInfo.firstInWidth + leftPadding) == ((splitWInfo.firstOutWidth-1)*strideX + kernelW));
        ASSERT((splitWInfo.midInWidth + splitWInfo.numOverlapStripes) == ((splitWInfo.midOutWidth-1)*strideX + kernelW));
        ASSERT((splitWInfo.lastInWidth  + splitWInfo.rightPadding + splitWInfo.numOverlapStripes) == ((splitWInfo.lastOutWidth-1)*strideX + kernelW));

        if (kernelW >= strideX)
        {
            ASSERT((kernelW - strideX) <= splitWInfo.firstInWidth);
            ASSERT((kernelW - strideX) <= splitWInfo.midInWidth);
        }
        else
        {
            ASSERT((strideX - kernelW) < splitWInfo.lastInWidth);
            ASSERT((strideX - kernelW) < splitWInfo.midInWidth);
        }
    }

    ASSERT((totalWidth >= kernelW));
    ASSERT((totalWidth - kernelW + strideX) % strideX == 0);
    ASSERT(leftPadding < kernelW);
    ASSERT(splitWInfo.rightPadding < kernelW);
    ASSERT(splitWInfo.bottomPadding < kernelH);

fail:
    return e;
}

NvU16 engine_ast::PDPNode::calculateMaxWidth
(
    surface::SurfacePrecision pdpPrecision,
    NvU16 poolingKernelHeight,
    NvU16 poolingStrideY
)
{
    NvU16 pdpMaxFlyingWidth   = 0;

    const NvU32 PDP_BUF_SIZE  = (7 * 1024);
    const NvU16 rtlOverlapLines[9] = {1, 1, 2, 4, 4, 8, 8, 8, 8};

    NvU16 logicalOverlapLines = (NvU16) ceil(float(poolingKernelHeight)/float(poolingStrideY));
    NvU32 atom_k_size         = graph()->target_config()->atomicKSize();
    NvU16 kernelPerGroup      = pdpPrecision.v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? atom_k_size : atom_k_size / 2;
    NvU16 bitsPerElement      = pdpPrecision.v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? 14 : 28;

    pdpMaxFlyingWidth = (PDP_BUF_SIZE * 8) / (rtlOverlapLines[logicalOverlapLines] * kernelPerGroup * bitsPerElement);

    return pdpMaxFlyingWidth;
}

/*--------------------------------Fuse Nodes---------------------------*/
NvDlaError engine_ast::PDPNode::fuseOnTheFlyNodes()
{
    NvDlaError e = NvDlaSuccess;
    NvU32 pdpMaxFlyingWidth = 0;

    if ( graph()->profile()->canSDPPDPOnFly() )
    {
        std::vector<Node *> upstreamNodes = graph()->upstreamNodes(this);

        /* Bail out if pdp has multiple upstream nodes - albeit this should NEVER happen
         * we did something wrong if it did
         */
        if (upstreamNodes.size() > 1)
        {
            if ( debugFusion() )
            {
                gLogInfo << "(" << name() << ") PDP node has multiple upstream nodes. Can't do on-fly " << endl;
            }
            params().setPDPFlyingMode(false);
            goto fail;
        }

        {
            engine_ast::Node* upstreamNode = upstreamNodes.at(0);
            surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
            surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

            /* Bail out if pdp is attached to a non-sdp node above.
             * If upstream is a concat node, we also effectively weed out split conv+sdp's
             * above. Because pdp on-fly is not possible with split conv(+sdp)
             */
            if (upstreamNode->engineType().v() != EngineTypeEnum::SDP)
            {
                if ( debugFusion() )
                {
                    gLogInfo << "(" << name() << ") Upstream node to this PDP (" << name() << ") is of type: "
                             << upstreamNode->engineType().c_str() << ". No scope of PDP on-fly" << endl;
                }
                params().setPDPFlyingMode(false);
                goto fail;
            }

            /* Bail out if this pdp is attached to upstream-sdp with multiple fan outs */
            if (graph()->downstreamEdges(upstreamNode).size() > 1)
            {
                if ( debugFusion() )
                {
                    gLogInfo << "(" << name() << ") Upstream SDP (" << upstreamNode->name() << ") to this PDP node ("
                             << name() << ") has multiple output edges. Can't fuse with such an SDP" << endl;
                }
                params().setPDPFlyingMode(false);
                goto fail;
            }

            /* Bail out if this pdp is attached to upstream-sdp with groups > 1 */
            // fixme: allow such fusion in future; by splitting pdp along C in boundGraph()
            if (NodeFactory::nodeCast<SDPNode*>(upstreamNode)->params().numGroups() > 1)
            {
                if ( debugFusion() )
                {
                    gLogInfo << "(" << name() << ") Upstream SDP (" << upstreamNode->name() << ") to this PDP node ("
                             << name() << ") numGroups > 1. Can't fuse with such an SDP" << endl;
                }
                params().setPDPFlyingMode(false);
                goto fail;
            }

            /* Bail out if pdp line buffers can't hold overlapping lines between adjacent vertical kernels*/
            pdpMaxFlyingWidth = calculateMaxWidth(srcTSD->surfaceFormat().precision(),
                                                  params().poolingWindow().h,
                                                  params().stride().h);
            if ( pdpMaxFlyingWidth < (NvU32)dstTSD->dimensions().w )
            {
                if ( debugFusion() )
                {
                    gLogInfo << "(" << name() << ") Max flying width " << pdpMaxFlyingWidth
                             << " < output_width " << dstTSD->dimensions().w
                             << " . Can't do On-Fly mode" << endl;
                }
                params().setPDPFlyingMode(false);
                goto fail;
            }

            // After escaping all the booby-traps, finally fuse.
            params().setPDPFlyingMode(true);
            dependencyParams().setFusedNode(IODirectionEnum::INPUT, upstreamNode);
            upstreamNode->dependencyParams().setFusedNode(IODirectionEnum::OUTPUT, this);

            if ( debugFusion() )
            {
                gLogInfo << "(" << name() << ") Fusing with " << upstreamNode->name() << endl;
            }
        }
    }

fail:
    return e;
}

/*--------------------------------Split Nodes--------------------------*/
NvDlaError engine_ast::PDPNode::pdpHWSplitWidth()
{
    NvDlaError e = NvDlaSuccess;

    engine_ast::PDPEngineParams::hwSplitWidthInfo splitWInfo;

    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    NvS32 remainingOutWidth = 0;
    Dims2 brPadding         = params().bottomRightPadding();
    Dims2 tlPadding         = params().topLeftPadding();
    NvS32 kernelW           = params().poolingWindow().w;
    NvS32 strideX           = params().stride().w;
    NvS32 leftPadding       = tlPadding.w;
    NvS32 pdpOutputWidth    = dstTSD->dimensions().w;
    NvS32 maxFlyingWidth    = calculateMaxWidth(srcTSD->surfaceFormat().precision(),
                                             params().poolingWindow().h,
                                             params().stride().h);

    remainingOutWidth = pdpOutputWidth;

    while (remainingOutWidth > 0)
    {
        splitWInfo.numSplits++;
        remainingOutWidth -= maxFlyingWidth;
    }

    splitWInfo.firstOutWidth = std::min(maxFlyingWidth, pdpOutputWidth);
    if (splitWInfo.numSplits == 2)
    {
        splitWInfo.lastOutWidth = pdpOutputWidth - splitWInfo.firstOutWidth;
    }
    else
    {
        splitWInfo.midOutWidth = maxFlyingWidth;
        splitWInfo.lastOutWidth = pdpOutputWidth - (maxFlyingWidth * (splitWInfo.numSplits - 1));
        ASSERT(splitWInfo.lastOutWidth > 0);
    }

    // map to input details
    splitWInfo.numOverlapStripes = std::max<NvU32>(0, kernelW - strideX);
    splitWInfo.firstInWidth      = std::max<NvU32>(0, (splitWInfo.firstOutWidth - 1)*strideX + kernelW - leftPadding);

    // extra right, bottom padding (it might be less than left_pad/top_pad; depends on start point of the last pooling)
    adjustBRPadding();
    brPadding = params().bottomRightPadding();
    splitWInfo.rightPadding  = brPadding.w;
    splitWInfo.bottomPadding = brPadding.h;

    if (splitWInfo.numSplits > 1)
    {
        // lastInWidth has to be >0, otherwise we have to adjust firstInWidth to meet this constraint
        splitWInfo.lastInWidth = (splitWInfo.lastOutWidth - 1)*strideX + kernelW -
                                  splitWInfo.rightPadding - splitWInfo.numOverlapStripes;
        if ( splitWInfo.lastInWidth > 0)
        {
            ASSERT(std::min(splitWInfo.firstInWidth, splitWInfo.lastInWidth) > 0);
        }
        else
        {
            NvU32 firstLastTotalOut = splitWInfo.firstOutWidth + splitWInfo.lastOutWidth;
            NvU32 lastInWidth = 1;
            while ((lastInWidth + splitWInfo.numOverlapStripes + splitWInfo.rightPadding - kernelW)%strideX != 0)
            {
                lastInWidth++;
            }
            splitWInfo.lastInWidth = lastInWidth;
            splitWInfo.lastOutWidth = (lastInWidth + splitWInfo.numOverlapStripes + splitWInfo.rightPadding - kernelW)/strideX + 1;
            ASSERT(splitWInfo.lastOutWidth > 0);

            splitWInfo.firstOutWidth = firstLastTotalOut - splitWInfo.lastOutWidth;
            ASSERT(splitWInfo.firstOutWidth > 0);

            splitWInfo.firstInWidth = (splitWInfo.firstOutWidth - 1)*strideX + kernelW - leftPadding;
            ASSERT(splitWInfo.firstInWidth > 0);
        }
    }

    if (splitWInfo.numSplits > 2)
    {
        splitWInfo.midInWidth = std::max<NvU32>(0, (splitWInfo.midOutWidth-1)*strideX + kernelW - splitWInfo.numOverlapStripes);
        ASSERT(std::min(splitWInfo.firstInWidth, std::min(splitWInfo.midInWidth, splitWInfo.lastInWidth)) > 0);
    }

    if ( debugSplits() ) {
        gLogInfo << "(" << name() << ") splitInfo: " << endl;
        gLogInfo << "\tnum splits: " << splitWInfo.numSplits << endl;
        gLogInfo << "\trightPadding: " << splitWInfo.rightPadding << endl;
        gLogInfo << "\tbottomPadding: " << splitWInfo.bottomPadding << endl;
        gLogInfo << "\tfirstInWidth: " << splitWInfo.firstInWidth << endl;
        gLogInfo << "\tmidInWidth: " << splitWInfo.midInWidth << endl;
        gLogInfo << "\tlastInWidth: " << splitWInfo.lastInWidth << endl;
        gLogInfo << "\tnumOverlapStripes: " << splitWInfo.numOverlapStripes << endl;
        gLogInfo << "\tfirstOutWidth: " << splitWInfo.firstOutWidth << endl;
        gLogInfo << "\tmidOutWidth: " << splitWInfo.midOutWidth << endl;
        gLogInfo << "\tlastOutWidth: " << splitWInfo.lastOutWidth << endl;
    }

    PROPAGATE_ERROR_FAIL(verifySplitWInfo(splitWInfo));

    params().setHwSplitWidthInfo(splitWInfo);

fail:
    return e;
}

NvDlaError engine_ast::PDPNode::pdpSWSplit()
{
    NvDlaError e = NvDlaSuccess;
    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "(%s) PDP software split data is not supported yet",
                                                name().c_str());

fail:
    return e;
}

/*--------------------------------Split Nodes---------------------------*/
NvDlaError engine_ast::PDPNode::splitNodes()
{
    NvDlaError e = NvDlaSuccess;
    NvU32 pdpMaxFlyingWidth = 0;
    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    // Make sure pdp-on-fly mode is not turned ON
    if ( params().isPDPOnFlying() )
    {
        if ( debugFusion() ) {
            gLogInfo << "(" << name() << ") PDP on-flying mode is ON. No need to split" << endl;
        }
        goto fail;
    }

    /* PDP can end up in off-flying mode and has to read from memory:
     *  - when it can't be fused to split conv+sdp combo above (is_full_conv = False) OR
     *  - when upstream node is not SDP
     *
     * However, that doesn't necessarily mean that hw/sw-powered split is needed.
     */

    // Soft exit if maxFlyingWidth >= output_width
    pdpMaxFlyingWidth = calculateMaxWidth(srcTSD->surfaceFormat().precision(),
                                          params().poolingWindow().h,
                                          params().stride().h);

    if ( debugFusion() )
    {
        gLogInfo << "(" << name() << ") maxflywidth " << pdpMaxFlyingWidth << " out width " << dstTSD->dimensions().w << endl;
    }

    if ( pdpMaxFlyingWidth >= (NvU32)dstTSD->dimensions().w )
    {
        if ( debugFusion() ) {
            gLogInfo << "(" << name() << ") maxFlyingWidth >= output_width. No need to do hw/sw PDP splits" << endl;
        }

        /* Before quitting, adjust right/bottom padding if the pooling window
         * is going to overshoot the input tensor boundaries
         */
        adjustBRPadding();
        goto fail;
    }
    else
    {
        // HW-powered split-Width can automatically stream overlapping vertical stripes
        // in a single HWL. So try that first
        PROPAGATE_ERROR_FAIL(pdpHWSplitWidth());

        /* Since PDP line buffer size is (64*112bits*8)bits, it could accommodate 2048/4096 elements
         * in int16/int8 format, the maximum line buffer limited output width is 2048/16=128 or 4096/32=128.
         * Since the width configuration could be 8192 max, so the max split_num could be 8192/128=64.
         *
         * Anything greater than that, needs software split of PDP node
         */
        if (params().getHwSplitWidthInfo().numSplits > 64)
        {
           gLogInfo << "Can't support splits > 64 yet." << endl;
           PROPAGATE_ERROR_FAIL(pdpSWSplit());
        }
    }

fail:
    return e;
}

/*------------------------------Handle Multi-Batch---------------------*/
NvDlaError engine_ast::PDPNode::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;

    //Handle operation parameters for the multi-batch operations
    NvU32 numBatches = graph()->profile()->multiBatchSize();
    for (NvU32 nn = 1; nn < numBatches; ++nn)
    {
        params(nn) = params(0);
    }

    return e;
}

//----------------------------------------------------------------------
//                      Code Emission
//----------------------------------------------------------------------
NvDlaError engine_ast::PDPNode::emitOp(Graph *g,
                                    DLAInterface *target_dla,
                                    NvU32 op_slot, NvU32 batch_id,
                                    DLACommonOpDescAccessor       dep,
                                    DLAOperationContainerAccessor op,
                                    DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;

    DLAPDPOpDescAccessor  pdp_op = op.pdpOpDescAccessor(0);
    DLAPDPSurfaceDescAccessor surf_acc = surf.pdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(fused_acc);

    surface::TensorSurfaceDesc *srcTSD     = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    *pdp_op.precision() = ASTToDLAInterface::getPDPPrecision(target_dla, srcTSD->surfaceFormat().precision());
    *pdp_op.poolMode()  = ASTToDLAInterface::getPDPMode(target_dla, params(batch_id).poolingType());
    *pdp_op.paddingValue(0)     = params(batch_id).paddingValue();
    *pdp_op.splitNum()            = max(params(batch_id).getHwSplitWidthInfo().numSplits, 1);
    *pdp_op.partialInWidthFirst() = max(params(batch_id).getHwSplitWidthInfo().firstInWidth, 0);
    *pdp_op.partialInWidthLast()  = max(params(batch_id).getHwSplitWidthInfo().lastInWidth, 0);
    *pdp_op.partialInWidthMid()   = max(params(batch_id).getHwSplitWidthInfo().midInWidth, 0);
    *pdp_op.partialWidthFirst()   = max(params(batch_id).getHwSplitWidthInfo().firstOutWidth, 0);
    *pdp_op.partialWidthLast()    = max(params(batch_id).getHwSplitWidthInfo().lastOutWidth, 0);
    *pdp_op.partialWidthMid()     = max(params(batch_id).getHwSplitWidthInfo().midOutWidth, 0);
    *pdp_op.poolHeight()          = params(batch_id).poolingWindow().h - 1;   // h/w friendly value
    *pdp_op.poolWidth()           = params(batch_id).poolingWindow().w - 1;
    *pdp_op.strideX()             = params(batch_id).stride().w;
    *pdp_op.strideY()             = params(batch_id).stride().h;
    *pdp_op.padLeft()             = params(batch_id).topLeftPadding().w;
    *pdp_op.padRight()            = params(batch_id).bottomRightPadding().w;
    *pdp_op.padTop()              = params(batch_id).topLeftPadding().h;
    *pdp_op.padBottom()           = params(batch_id).bottomRightPadding().h;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, srcTSD, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dst_data_acc, dstTSD, IODirectionEnum::UNKNOWN, batch_id);

    if ( params(batch_id).bottomRightPadding().w < 0 ) {
        *src_data_acc.width() += params(batch_id).bottomRightPadding().w;
        *pdp_op.padRight() = 0;
    }

    if ( params(batch_id).bottomRightPadding().h < 0 ) {
        *src_data_acc.height() += params(batch_id).bottomRightPadding().h;
        *pdp_op.padBottom() = 0;
    }

    if ( g->debugOps() )
    {
        gLogInfo << "PDP node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;

        gLogInfo << "\tpdp precision" << (int)*pdp_op.precision() << endl;
        gLogInfo << "\tpdp pool mode" << (int)*pdp_op.poolMode() << endl;
        gLogInfo << "\tsrc tsd:" << srcTSD->id() << endl;
        gLogInfo << "\tdst tsd:" << dstTSD->id() << endl;
        gLogInfo << "\tsrc addr=" << (int) *src_data_acc.address() << endl;
        gLogInfo << "\tsrc type=" << (int) *src_data_acc.type() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tsplitNum " << (int)*pdp_op.splitNum() << endl;
        gLogInfo << "\tpadLeft " << (int)*pdp_op.padLeft() << endl;
        gLogInfo << "\tpadTop " << (int)*pdp_op.padTop() << endl;
        gLogInfo << "\tpadRight " << (int)*pdp_op.padRight() << endl;
        gLogInfo << "\tpadBottom " << (int)*pdp_op.padBottom() << endl;
        gLogInfo << "\tpool height " << (int)*pdp_op.poolHeight() << endl;
        gLogInfo << "\tpool width " << (int)*pdp_op.poolWidth() << endl;
        gLogInfo << "\tstride x " << (int)*pdp_op.strideX() << endl;
        gLogInfo << "\tstride y " << (int)*pdp_op.strideY() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << (int) *dst_data_acc.address() << endl;
        gLogInfo << "\tdst type=" << (int) *dst_data_acc.type() << endl;
        gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
        gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
        gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
        gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
    }

    return e;
}

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
NvDlaError engine_ast::PDPNode::emitOp(NvU32 op_slot, NvU32 batch_id,
                                    DLAInterface *target_dla,
                                    DLACommonOpDescAccessor&       dep,
                                    DLAOperationContainerAccessor& op,
                                    DLASurfaceContainerAccessor&   surf,
                                    nvdla_prototest_interface::Layer* protoLayer)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 numConsumers = 0;

    DLAPDPOpDescAccessor  pdp_op = op.pdpOpDescAccessor(0);
    DLAPDPSurfaceDescAccessor surf_acc = surf.pdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(batch_id);

    surface::TensorSurfaceDesc *srcTSD     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    nvdla_prototest_interface::PDPOpDesc* protoPDPOpDesc        = protoLayer->mutable_op_config()->mutable_pdp_op();
    nvdla_prototest_interface::PDPSurfaceDesc* protoPDPSurfDesc = protoLayer->mutable_surface()->mutable_pdp_surface();
    nvdla_prototest_interface::DataCube* protoSrcDataCube       = protoPDPSurfDesc->mutable_src_data();
    nvdla_prototest_interface::DataCube* protoDstDataCube       = protoPDPSurfDesc->mutable_dst_data();

    nvdla_prototest_interface::DataPrecision pdpPrec;

    protoLayer->set_index(op_slot);
    protoLayer->set_roi_index(0);
    protoLayer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_PDP);
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
            ++numConsumers;
            nvdla_prototest_interface::Consumer* protoConsumer = protoLayer->add_bottom();
            protoConsumer->set_index(*cons_acc.index());
            switch(c) {
                case EngineTypeEnum::BDMA : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_BDMA); break;
                case EngineTypeEnum::CONVOLUTION : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
                case EngineTypeEnum::SDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP); break;
                case EngineTypeEnum::PDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_PDP); break;
                case EngineTypeEnum::CDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CDP); break;
                case EngineTypeEnum::RUBIK: protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_RUBIK); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer %d", c);
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
            case EngineTypeEnum::SDP: protoFusedConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP); break;
            default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "PDP can have only SDP op as its fused partner on input side if at all possible");
        }
    }

    switch(srcTSD->surfaceFormat().precision().v()) {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : pdpPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : pdpPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : pdpPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized input precision: %s", srcTSD->surfaceFormat().precision().c_str());
    }

    protoPDPOpDesc->set_precision(pdpPrec);

    switch(params(batch_id).poolingType().v()) {
        case nvdla::PoolingType::kAVERAGE: protoPDPOpDesc->set_pool_mode(nvdla_prototest_interface::PDPOpDesc_PoolingMode_MODE_AVG); break;
        case nvdla::PoolingType::kMAX:     protoPDPOpDesc->set_pool_mode(nvdla_prototest_interface::PDPOpDesc_PoolingMode_MODE_MAX); break;
        case nvdla::PoolingType::kMIN:     protoPDPOpDesc->set_pool_mode(nvdla_prototest_interface::PDPOpDesc_PoolingMode_MODE_MIN); break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized pool mode: %s", params(batch_id).poolingType().c_str());
    }

    protoPDPOpDesc->add_padding_value(*pdp_op.paddingValue(0));
    protoPDPOpDesc->set_split_num(*pdp_op.splitNum());
    protoPDPOpDesc->set_partial_in_width_first(*pdp_op.partialInWidthFirst());
    protoPDPOpDesc->set_partial_in_width_last(*pdp_op.partialInWidthLast());
    protoPDPOpDesc->set_partial_in_width_mid(*pdp_op.partialInWidthMid());
    protoPDPOpDesc->set_partial_width_first(*pdp_op.partialWidthFirst());
    protoPDPOpDesc->set_partial_width_last(*pdp_op.partialWidthLast());
    protoPDPOpDesc->set_partial_width_mid(*pdp_op.partialWidthMid());

    switch(*pdp_op.poolHeight()) {
        case 0: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_1); break;
        case 1: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_2); break;
        case 2: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_3); break;
        case 3: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_4); break;
        case 4: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_5); break;
        case 5: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_6); break;
        case 6: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_7); break;
        case 7: protoPDPOpDesc->set_pool_height(nvdla_prototest_interface::PoolSize::SIZE_8); break;
    }

    switch(*pdp_op.poolWidth()) {
        case 0: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_1); break;
        case 1: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_2); break;
        case 2: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_3); break;
        case 3: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_4); break;
        case 4: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_5); break;
        case 5: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_6); break;
        case 6: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_7); break;
        case 7: protoPDPOpDesc->set_pool_width(nvdla_prototest_interface::PoolSize::SIZE_8); break;
    }

    protoPDPOpDesc->set_stride_x(*pdp_op.strideX());
    protoPDPOpDesc->set_stride_y(*pdp_op.strideY());
    protoPDPOpDesc->set_pad_left(*pdp_op.padLeft());
    protoPDPOpDesc->set_pad_right(*pdp_op.padRight());
    protoPDPOpDesc->set_pad_top(*pdp_op.padTop());
    protoPDPOpDesc->set_pad_bottom(*pdp_op.padBottom());

    if (dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT)) {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_HW);
    } else if (srcTSD->tensorBufferDesc()->memoryLoc(batch_id).v() == memory::LocationEnum::lCVSRAM) {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_CV);
    } else {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    }
    protoSrcDataCube->set_address(*src_data_acc.address());
    protoSrcDataCube->set_size(srcTSD->tensorBufferDesc()->size() - srcTSD->bufferOffset());
    protoSrcDataCube->set_width(*src_data_acc.width());
    protoSrcDataCube->set_height(*src_data_acc.height());
    protoSrcDataCube->set_channel(*src_data_acc.channel());
    protoSrcDataCube->set_line_stride(*src_data_acc.lineStride());
    protoSrcDataCube->set_surf_stride(*src_data_acc.surfStride());
    protoSrcDataCube->set_plane_stride(*src_data_acc.planeStride());
    protoSrcDataCube->mutable_mem_info()->set_mem_id(srcTSD->tensorBufferDesc()->memoryId(batch_id));
    protoSrcDataCube->mutable_mem_info()->set_mem_size(srcTSD->tensorBufferDesc()->size());
    protoSrcDataCube->mutable_mem_info()->set_offset(srcTSD->bufferOffset());
    protoSrcDataCube->mutable_mem_info()->set_fill_type(nvdla_prototest_interface::FillerType::FILL_RANDOM);
    protoSrcDataCube->mutable_mem_info()->set_flag(nvdla_prototest_interface::MemFlag::DLA_MEM_SET);
    protoSrcDataCube->mutable_mem_info()->set_precision(pdpPrec);

    protoDstDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoDstDataCube->set_address(*dst_data_acc.address());
    protoDstDataCube->set_size(dstTSD->tensorBufferDesc()->size() - dstTSD->bufferOffset());
    protoDstDataCube->set_width(*dst_data_acc.width());
    protoDstDataCube->set_height(*dst_data_acc.height());
    protoDstDataCube->set_channel(*dst_data_acc.channel());
    protoDstDataCube->set_line_stride(*dst_data_acc.lineStride());
    protoDstDataCube->set_surf_stride(*dst_data_acc.surfStride());
    protoDstDataCube->set_plane_stride(*dst_data_acc.planeStride());
    protoDstDataCube->mutable_mem_info()->set_mem_id(dstTSD->tensorBufferDesc()->memoryId(batch_id));
    protoDstDataCube->mutable_mem_info()->set_mem_size(dstTSD->tensorBufferDesc()->size());
    protoDstDataCube->mutable_mem_info()->set_offset(dstTSD->bufferOffset());
    if (numConsumers == 0) {
        protoDstDataCube->mutable_mem_info()->set_fill_type(nvdla_prototest_interface::FillerType::FILL_NONE);
        protoDstDataCube->mutable_mem_info()->set_flag(nvdla_prototest_interface::MemFlag::DLA_MEM_OUTPUT);
        protoDstDataCube->mutable_mem_info()->set_precision(pdpPrec);
    }

fail:
    return e;
}
#endif

};  // nvdla::priv::
};  // nvdla::
