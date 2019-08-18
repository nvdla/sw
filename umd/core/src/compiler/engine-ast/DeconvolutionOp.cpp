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
#include "priv/WeightTranslationUnit.h"

#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

void engine_ast::ConvCoreDeconvolutionOpNode::captureCanonicalParams()
{
    params().setHasBiasTerm(canonicalNode()->params().hasBiasTerm() == true ? 1 : 0);
    params().setWeightDims(canonicalNode()->params().weightDims());
    params().setTopLeftPadding(canonicalNode()->params().topLeftPadding());
    params().setBottomRightPadding(canonicalNode()->params().bottomRightPadding());
    params().setPaddingValue(canonicalNode()->params().paddingValue());
    params().setStride(canonicalNode()->params().stride());
    params().setDilation(canonicalNode()->params().dilation());
    params().setRawWeights(canonicalNode()->params().weights());
    params().setDLAWeights(Weights(DataType::FLOAT, NULL, 0));
    params().setNumGroups(canonicalNode()->params().numGroups());
    captureCanonicalWeights();
}

void engine_ast::ConvCoreDeconvolutionOpNode::inheritParams(Node* otherNode)
{
    // inherit the parameters that make sense (keep adding later)
    ConvCoreDeconvolutionOpNode* otherDeconv = NodeFactory::nodeCast<ConvCoreDeconvolutionOpNode*>(otherNode);
    params().setStride(otherDeconv->params().stride());
    params().setConvMode(otherDeconv->params().convMode());
    // WG doesn't work with deconv - following is redundant
    params().setWinogradParams(otherDeconv->params().winogradParams());
    params().setNumGroups(otherDeconv->params().numGroups());
}

/*
 * DLA-CONV engine is used to implement software based deconvolution followed
 * but Rubik engine in contract mode. Rubik likes to move 32B atoms in contract mode;
 * as a result conv output should be laid out in a manner such that it
 * accommodates the channel padding required by Rubik. As a result conv output
 * should be dumped into a chnl aligned buffer (16 for fp16/int16 or 32 for int8)
 */
Dims4 engine_ast::ConvCoreDeconvolutionOpNode::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    bool isSrcTSD = false;
    bool isAuxTSD = false;
    bool isDstTSD = false;
    Dims4 suggestedDims(-1,-1,-1,-1);
    NvU32 atom_k_size = graph()->target_config()->atomicKSize();
    NvU32 chnlAlign = graph()->profile()->computePrecision() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? atom_k_size : atom_k_size / 2;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isAuxTSD = auxEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isAuxTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (isSrcTSD || isAuxTSD)
    {
        suggestedDims = tsd->dimensions();
    }
    else if (isDstTSD)
    {
        Dims4 chnlAlignedSurf = tsd->dimensions();
        chnlAlignedSurf.c = ROUNDUP_AND_ALIGN(chnlAlignedSurf.c, chnlAlign);
        suggestedDims = chnlAlignedSurf;
    }

fail:
    return suggestedDims;
}

/*
 * Deconvolution on DLA is performed as a series of convolution ops
 * whose outputs are concatenated in C direction and later transformed by
 * Rubik engine to shuffle bytes such that the final output is as desired.
 */
NvDlaError engine_ast::ConvCoreDeconvolutionOpNode::preProcessAuxData()
{
    NvDlaError e = NvDlaSuccess;
    Edge*   deconvAuxEdge = NULL;
    NvU16   numSplits = 0;
    Weights rawKCRSWts;
    Weights rawCKRSWts;
    Weights splitWts;
    Dims4   origWtDims;
    Dims4   splitSetWtDims;
    std::vector< Weights > splitSetWts;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    /* Step-1: Caffe weights for deconv are laid out in CKRS format;
     *         Interchange the C and K dimensions to get to KCRS format (which matches
     *         caffe's Conv weight layout) since DLA performs Deconv as series of Convs
     *         such that:
     *         Strns = Sorig
     *         Rtrns = Rorig
     *         Ctrns = Korig
     *         Ktrns = Corig
     */
    deconvAuxEdge = auxEdges()[0];
    rawCKRSWts    = params().rawWeights();

    {
        WeightTrns::WeightDims kernelDims (rawCKRSWts.count,
                                       params().weightDims().n,
                                       params().weightDims().c,
                                       params().weightDims().w,
                                       params().weightDims().h,
                                       (int)params().stride().w,
                                       (int)params().stride().h);

        PRECISION_SWITCH(rawCKRSWts.type.v(), nvdla::DataType::FLOAT, rawKCRSWts, WeightTrns::convertWtsToKCRS,
                                                                                 kernelDims,
                                                                                 rawCKRSWts);
        if (rawKCRSWts.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Kernel Wt trnaslation failed for node '%s'", name().c_str());
        }

        params().setRawWeights(rawKCRSWts);
    }

    /*
     * Quantization of weights.
     * Safer to call in preprocess aux data since no sdp math fusion possible.
     */
    if (computePrecision.v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        SDPNode* origFusedSDPNode    = NodeFactory::nodeCast<SDPNode*>(graph()->downstreamDataNodes(this)[0]);

        if (origFusedSDPNode->engineOpType().v() == EngineOpTypeEnum::SDP_NOP)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue,
                                "Expecting node other than SDP_NOP (to handle rescaling)");
        }
        ConvCoreNode::quantizeAuxData();
        rawKCRSWts = params().rawWeights();
    }

    /* Step-2: The Deconv operation in DLA is performed as a series of Conv ops where
     *
     *         #numSplits = deconv_x_stride * deconv_y_stride
     *
     *         Distribute the weights into #numSplits sets such that:
     *         Ssplit = ceil(Strns/deconv_x_stride)
     *         Rsplit = ceil(Rtrns/deconv_y_stride)
     *         Csplit = Ctrns
     *         Ksplit = Ktrns
     */

    numSplits = params().stride().w * params().stride().h;
    origWtDims = params().weightDims();
    {
        NvS32 splitSetW = (NvS32)ceilf(origWtDims.w/float(params().stride().w));
        NvS32 splitSetH = (NvS32)ceilf(origWtDims.h/float(params().stride().h));
        splitSetWtDims = Dims4(origWtDims.n, origWtDims.c, splitSetH, splitSetW);

        /* Split weights based on deconvolution strides */
        PRECISION_SWITCH(rawKCRSWts.type.v(), computePrecision.v(), splitSetWts, WeightTrns::splitWeightsForDeconv,
                                                                                    rawKCRSWts,
                                                                                    origWtDims,
                                                                                    params().stride(),
                                                                                    splitSetWtDims);

        if (splitSetWts.size() == 0)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                "Problem in splitting the deconvolution weight data!");
        }
    }

    /* Step-3: Create conv+sdp sets and distribute the weight sets to each of
     *         these. Enforce strict order of execution by chaining them with
     *         compute edges
     */
    {
        canonical_ast::DeconvolutionNode* canDeconvNode = canonicalNode();
        ConvCoreNode* origDeconvNode = this;
        SDPNode* origFusedSDPNode    = NodeFactory::nodeCast<SDPNode*>(graph()->downstreamDataNodes(this)[0]);
        bool fusedSDPHasAuxData      = origFusedSDPNode->auxEdges().size() > 0;
        Edge* sdpAuxEdge        = fusedSDPHasAuxData ? origFusedSDPNode->auxEdges()[0] : NULL;
        Edge* origInputEdge     = origDeconvNode->inputEdges()[0];
        Edge* origStreamEdge    = origDeconvNode->outputEdges()[0];
        Edge* origOutputEdge    = origFusedSDPNode->outputEdges()[0];
        Edge* splitDeconvEdge   = NULL;
        Dims2 origDeconvStride  = origDeconvNode->params().stride();
        Dims2 splitDeconvStride = Dims2(1,1);
        Dims4 origSrcDims       = origInputEdge->tensorSurfaceDesc()->dimensions();
        Dims4 origDstDims       = origOutputEdge->tensorSurfaceDesc()->dimensions();
        Dims4 splitSrcDims;
        Dims4 splitStreamDims;
        Dims4 splitDstDims;

        Tensor* newSplitSrcTensor;
        Tensor* newSplitDstTensor;
        Tensor* concatDstTensor;
        Dims4 concatDstDims;
        Dims4 rubikDstDims;

        std::vector< ConvCoreNode* > splitDeconvNodes;
        std::vector< SDPNode* > splitSDPNodes;

        SplitNode* swSplitNode          = engine_ast::NodeFactory::newSplitNode(NULL, graph());
        ConcatenationNode* swConcatNode = engine_ast::NodeFactory::newConcatNode(NULL, graph());
        RubikNode* rubikNode            = engine_ast::NodeFactory::newRubikNode(canDeconvNode, graph());
        rubikNode->params().setMode(RubikModeEnum::RUBIK_MODE_CONTRACT);
        swSplitNode->params().setSplitAxis(SplitAxisEnum::SPLIT_ALONG_NONE);
        swConcatNode->params().setConcatAxis(ConcatAxisEnum::CONCAT_ALONG_C);

        origDeconvNode->params().setStride(splitDeconvStride);
        rubikNode->params().setDeconvStride(Dims2(origDeconvStride.h, origDeconvStride.w));

        origDeconvNode->params().setRawWeights(splitSetWts[0]);
        origDeconvNode->params().setWeightDims(splitSetWtDims);
        deconvAuxEdge->originalTensor()->setDimensions(splitSetWtDims);
        deconvAuxEdge->tensorSurfaceDesc()->setDimensions(splitSetWtDims);

        concatDstDims = Dims4(origDstDims.n,
                              origDstDims.c * numSplits,
                              origDstDims.h / origDeconvStride.h,
                              origDstDims.w / origDeconvStride.w);
        rubikDstDims = Dims4(concatDstDims.n,
                             concatDstDims.c / numSplits,
                             concatDstDims.h * origDeconvStride.h,
                             concatDstDims.w * origDeconvStride.w);

        ASSERT(origDstDims == rubikDstDims);

        splitSrcDims = origSrcDims;

        /**
         * Currently, we don't support if padding is involved in input or output data cube.
         *
         * split_output_height = ((input_height + input_pad_top + input_pad_bottom - split_kernel_height) /
         *                           stride) + 1
         *
         * rubik_output_height = deconv_output_height + output_pad_top + output_pad_bottom
         *
         * With stride = 1 and with current assumption, input_padding and output padding are zero.
         *
         * split_output_height = input_height - split_kernel_height + 1         --------- [1]
         * rubik_output_height = deconv_output_height                           --------- [2]
         *
         * Both ([1] * deconv_y_stride) and [2] need to be equal to continue with our assumptions.
         *
         * if (split_output_height * deconv_y_stride) != rubik_output_height, our assumption on padding is
         * wrong and we error out as unsupported.
         *
         * Similar justification holds true for width computation.
         **/
        splitStreamDims = Dims4(origDstDims.n,
                                origDstDims.c,
                                origSrcDims.h - splitSetWtDims.h + 1,
                                origSrcDims.w - splitSetWtDims.w + 1);

        if (splitStreamDims.h * origDeconvStride.h != origDstDims.h ||
            splitStreamDims.w * origDeconvStride.w != origDstDims.w)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported,
                                "Padding issue, dims difference: (orig - computed) (H x W) = (%d x %d)",
                                origDstDims.h - splitStreamDims.h * origDeconvStride.h,
                                origDstDims.w - splitStreamDims.w * origDeconvStride.w);
        }
        splitDstDims = splitStreamDims;

        newSplitSrcTensor = origInputEdge->originalTensor()->clone();
        newSplitSrcTensor->setTensorType(TensorType::kIO);
        newSplitSrcTensor->setDimensions(splitSrcDims);

        splitStreamDims.n = 1;
        origStreamEdge->originalTensor()->setDimensions(splitStreamDims);
        origStreamEdge->tensorSurfaceDesc()->setDimensions(splitStreamDims);

        splitDstDims.n = 1;
        newSplitDstTensor = origOutputEdge->originalTensor()->clone();
        newSplitDstTensor->setTensorType(TensorType::kIO);
        newSplitDstTensor->setDimensions(splitDstDims);

        splitDeconvEdge = graph()->addDataEdge(origInputEdge->canonicalEdge(), swSplitNode, origDeconvNode, newSplitSrcTensor);
        splitDeconvEdge->setBindId(origInputEdge->bindId(), origInputEdge->bindDomain());
        graph()->addDataEdge(origOutputEdge->canonicalEdge(), origFusedSDPNode, swConcatNode, newSplitDstTensor);

        graph()->replaceEdgeNodes(origInputEdge, ast::EdgeSideEnum::SECOND, origDeconvNode, swSplitNode);
        graph()->replaceEdgeNodes(origOutputEdge, ast::EdgeSideEnum::FIRST, origFusedSDPNode, rubikNode);

        concatDstDims.n = 1;
        concatDstTensor = origOutputEdge->originalTensor()->clone();
        concatDstTensor->setTensorType(TensorType::kIO);
        concatDstTensor->setDimensions(concatDstDims);

        // since concat out dims is different from orig conv's output dims,
        // expand the tensor scales in concat's outputs to populate the same value in the extra channels
        if (graph()->profile()->computePrecision().v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
        {
            std::vector<NvF32> deconvOutScales = concatDstTensor->getChannelScales();
            deconvOutScales.resize(concatDstDims.c, deconvOutScales.at(0));
            concatDstTensor->setChannelScales(deconvOutScales);
        }

        Edge* concatRubikEdge = graph()->addDataEdge((Edge*)0, swConcatNode, rubikNode, concatDstTensor);

        if ( debugSplits() )
        {
            gLogInfo << "swsplit node=" << swSplitNode->id() << " now takes orig input edge="
                     << origInputEdge->id() << " in " << graph()->name() << endl;
            gLogInfo << "swconcat node=" << swConcatNode->id() << " connects to rubik via ="
                     << concatRubikEdge->id() << " in " << graph()->name() << endl;
            gLogInfo << "rubik node=" << rubikNode->id() << " now takes orig output edge="
                     << origOutputEdge->id() << " in " << graph()->name() << endl;
        }

        splitDeconvNodes.push_back(origDeconvNode);
        splitSDPNodes.push_back(origFusedSDPNode);

        for (NvU16 n = 1; n < numSplits; ++n)
        {
            ConvCoreDeconvolutionOpNode* newSplitDeconvNode = NodeFactory::newConvCoreDeconvolutionOpNode(canDeconvNode, graph());
            SDPNode* newSplitSDPNode = newSplitDeconvNode->addSDPJointOpNode(origFusedSDPNode);
            Edge* splitStreamEdge = NULL;
            Edge* splitDeconvAuxEdge = NULL;

            newSplitDeconvNode->params().setStride(splitDeconvStride);
            newSplitDeconvNode->params().setRawWeights(splitSetWts[n]);
            newSplitDeconvNode->params().setWeightDims(splitSetWtDims);
            // fixme: non-functional deconv: copy filter scales in all deconv split nodes; in future assign apt filter scales to each
            newSplitDeconvNode->params().setFilterScales(origDeconvNode->params().filterScales());

            PROPAGATE_ERROR_FAIL(newSplitDeconvNode->nodeDataEdge(TensorType::kWEIGHT, ast::EdgeSideEnum::SECOND, &splitDeconvAuxEdge));
            splitDeconvAuxEdge->originalTensor()->setDimensions(splitSetWtDims);

            if (fusedSDPHasAuxData)
            {
                Edge* newSplitSDPRedundantAuxEdge;
                PROPAGATE_ERROR_FAIL(newSplitSDPNode->nodeAuxEdge(&newSplitSDPRedundantAuxEdge));
                newSplitSDPNode->graph()->removeEdgeFromNode(newSplitSDPRedundantAuxEdge, ast::EdgeSideEnum::SECOND, newSplitSDPNode);
                newSplitSDPNode->graph()->removeNodeFromEdge(newSplitSDPRedundantAuxEdge, ast::EdgeSideEnum::SECOND, newSplitSDPNode);
                newSplitSDPNode->graph()->appendNodeToEdge(sdpAuxEdge, ast::EdgeSideEnum::SECOND, newSplitSDPNode);
                newSplitSDPNode->graph()->removeEdge(newSplitSDPRedundantAuxEdge);
            }

            newSplitSrcTensor = origInputEdge->originalTensor()->clone();
            newSplitSrcTensor->setTensorType(TensorType::kIO);
            newSplitSrcTensor->setDimensions(splitSrcDims);

            PROPAGATE_ERROR_FAIL(newSplitDeconvNode->nodeDataEdge(TensorType::kSTREAM, ast::EdgeSideEnum::FIRST, &splitStreamEdge));
            splitStreamEdge->originalTensor()->setDimensions(splitStreamDims);

            newSplitDstTensor = origOutputEdge->originalTensor()->clone();
            newSplitDstTensor->setTensorType(TensorType::kIO);
            newSplitDstTensor->setDimensions(splitDstDims);

            splitDeconvEdge = graph()->addDataEdge(origInputEdge->canonicalEdge(), swSplitNode, newSplitDeconvNode, newSplitSrcTensor);
            splitDeconvEdge->setBindId(origInputEdge->bindId(), origInputEdge->bindDomain());

            graph()->addDataEdge(origOutputEdge->canonicalEdge(), newSplitSDPNode, swConcatNode, newSplitDstTensor);

            graph()->addComputeEdge(splitDeconvNodes.back(), newSplitDeconvNode);
            graph()->addComputeEdge(splitSDPNodes.back(), newSplitSDPNode);

            splitDeconvNodes.push_back(newSplitDeconvNode);
            splitSDPNodes.push_back(newSplitSDPNode);
        }

        // finally determine contract op params that satisfy hardware requirements for Rubik
        PROPAGATE_ERROR_FAIL( rubikNode->determineContractOpParams() );
    }

fail:
    return e;
}

NvDlaError engine_ast::ConvCoreDeconvolutionOpNode::quantizeAuxData()
{
    NvDlaError e = NvDlaSuccess;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();

    if (computePrecision.v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        Weights rawWeights = params().rawWeights();
        if (rawWeights.type != nvdla::DataType::INT8)
        {
            // Expected quantization to have happened during preprocess aux pass itself.
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                                "Unquantized weights found."
                                "Quantization expected to have happened during preprocess aux pass.");
        }
    }

fail:
    return e;
}

/*----------------------Weight Translation ----------------------------*/
NvDlaError engine_ast::ConvCoreDeconvolutionOpNode::translateAuxData()
{
    NvDlaError e = NvDlaSuccess;

    Edge* auxEdge       = NULL;
    surface::SurfacePrecision computePrecision;
    NvU32 atomicCSize = 0;
    NvU32 atomicKSize = 0;
    NvU32 cbufWidth = 0;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    computePrecision  = graph()->profile()->computePrecision();
    atomicCSize = graph()->target_config()->atomicCSize();
    atomicKSize = graph()->target_config()->atomicKSize();
    cbufWidth   = graph()->target_config()->bufEntryWidth();

    auxEdge       = auxEdges()[0];

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

        PRECISION_SWITCH(rawKrnlWts.type.v(), computePrecision.v(), trnsKrnlWts, WeightTrns::translateWtsForDeconv,
                                                                                 kernelDims,
                                                                                 rawKrnlWts,
                                                                                 atomicKSize,
                                                                                 atomicCSize,
                                                                                 cbufWidth);

        if (trnsKrnlWts.values == NULL)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Kernel Wt trnaslation failed for node '%s'", name().c_str());
        }

        params().setDLAWeights(trnsKrnlWts);
    }


fail:
    return e;
}

NvDlaError engine_ast::ConvCoreDeconvolutionOpNode::emitOp(Graph *g,
                                                        DLAInterface *target_dla,
                                                        NvU32 op_slot, NvU32 batch_id,
                                                        DLACommonOpDescAccessor       dep,
                                                        DLAOperationContainerAccessor op,
                                                        DLASurfaceContainerAccessor   surf)
{
    NvDlaError e  = NvDlaSuccess;

    DLAConvOpDescAccessor       deconvOp   = op.convOpDescAccessor(0);
    DLACVTParamAccessor         outCVTAcc  = deconvOp.outCVTAccessor();
    DLACVTParamAccessor         inCVTAcc   = deconvOp.inCVTAccessor();
    DLAConvSurfaceDescAccessor  surfAcc    = surf.convSurfaceDescAccessor(0);
    DLADataCubeAccessor         srcDataAcc = surfAcc.srcDataAccessor();
    DLADataCubeAccessor         dstDataAcc = surfAcc.dstDataAccessor();
    DLADataCubeAccessor         wtDataAcc  = surfAcc.weightDataAccessor();

    surface::TensorSurfaceDesc *srcTSD = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *wtTSD  = g->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    *deconvOp.padVal()       = 0; // FIXME: assuming same padding on both dimensions
    *deconvOp.dataReuse()    = 0;
    *deconvOp.weightReuse()  = 0;
    *deconvOp.skipDataRls()  = 0;
    *deconvOp.skipWeightRls()= 0;
    *deconvOp.batch()        = 1;
    *deconvOp.batchStride()  = 0;
    *deconvOp.release()      = srcTSD->dimensions().h;
    *deconvOp.meanFormat()   = deconvOp.meanFormat_Disable();
    *deconvOp.meanRY()       = 0;
    *deconvOp.meanGU()       = 0;
    *deconvOp.meanBV()       = 0;
    *deconvOp.padXLeft()     = 0;
    *deconvOp.padXRight()    = 0;
    *deconvOp.padYTop()      = 0;
    *deconvOp.padYBottom()   = 0;
    *deconvOp.dilationX()    = 1;
    *deconvOp.dilationY()    = 1;
    *deconvOp.pixelMapping() = deconvOp.pixelMapping_PitchLinear();   //default

    *inCVTAcc.scale()     = 0;
    *inCVTAcc.truncate()  = 0;
    *inCVTAcc.enable()    = 0;
    *inCVTAcc.offset()    = 0;

    *outCVTAcc.scale()    = 1;
    *outCVTAcc.truncate() = 0;
    *outCVTAcc.enable()   = 1;
    *outCVTAcc.offset()   = 0;

    /* Common parameters */
    *deconvOp.inPrecision()  = ASTToDLAInterface::getConvCorePrecision(target_dla, srcTSD->surfaceFormat().precision());
    *deconvOp.outPrecision() = ASTToDLAInterface::getConvCorePrecision(target_dla, dstTSD->surfaceFormat().precision());
    *deconvOp.fetchGrain()   = 1;                                           //FIXME: right now its max of requirements of all conv nodes in mnist
    *deconvOp.dataFormat()   = ASTToDLAInterface::getDataFormat(target_dla, srcTSD->surfaceFormat());
    *deconvOp.weightFormat() = deconvOp.weightFormat_Uncompressed();
    *deconvOp.convStrideX()  = params(batch_id).stride().w;
    *deconvOp.convStrideY()  = params(batch_id).stride().h;
    *deconvOp.inputWidthCMAC()   = dstTSD->dimensions().w;
    *deconvOp.inputHeightCMAC()  = dstTSD->dimensions().h;
    *deconvOp.bytesPerKernel()   = surface::WeightDesc::bytesPerKernel(wtTSD);

    *deconvOp.convMode()         = deconvOp.convMode_Direct();
    *deconvOp.inputWidthCSC()    = srcTSD->dimensions().w;
    *deconvOp.inputHeightCSC()   = srcTSD->dimensions().h;
    *deconvOp.inputChannelCSC()  = srcTSD->dimensions().c;
    *deconvOp.kernelHeightCSC()  = wtTSD->dimensions().h;
    *deconvOp.kernelWidthCSC()   = wtTSD->dimensions().w;
    *deconvOp.kernelChannelCSC() = wtTSD->dimensions().c;

    /* entry-per-slice & banks should be calculated after conv-mode is determined */
    params(batch_id).setConvMode(ConvolutionModeEnum::CONV_DIRECT);
    *deconvOp.entryPerSlice()= calculateEPS(srcTSD);
    *deconvOp.dataBank()     = params(batch_id).dataBanksAllotted();
    *deconvOp.weightBank()   = params(batch_id).weightBanksAllotted();

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(srcDataAcc, srcTSD, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(wtDataAcc, wtTSD, IODirectionEnum::UNKNOWN, batch_id);
    setDataCubeAccessor(dstDataAcc, dstTSD, IODirectionEnum::OUTPUT, batch_id);

    if ( g->debugOps() )
    {
        gLogInfo << "Deconv node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tin precision " << (int)*deconvOp.inPrecision() << endl;
        gLogInfo << "\tout precision " << (int)*deconvOp.outPrecision() << endl;
        gLogInfo << "\tsrc data loc: " << (int) *srcDataAcc.type() << endl;
        gLogInfo << "\tdst data loc: " << (int) *dstDataAcc.type() << endl;
        gLogInfo << "\tpost y extension: " << (int)*deconvOp.postExtension() << endl;
        gLogInfo << "\tin_precision " << (int)*deconvOp.inPrecision() << endl;
        gLogInfo << "\tout_precision " << (int)*deconvOp.outPrecision() << endl;
        gLogInfo << "\tpad_val " << (int)*deconvOp.padVal() << endl;
        gLogInfo << "\tconv mode " << (int)*deconvOp.convMode() << endl;
        gLogInfo << "\tdata_reuse " << (int)*deconvOp.dataReuse() << endl;
        gLogInfo << "\tweight_reuse " << (int)*deconvOp.weightReuse() << endl;
        gLogInfo << "\tskip_data_rls " << (int)*deconvOp.skipDataRls() << endl;
        gLogInfo << "\tskip_wt_rls " << (int)*deconvOp.skipWeightRls() << endl;
        gLogInfo << "\teps " << *deconvOp.entryPerSlice() << endl;
        gLogInfo << "\tfetch_grain " << (int)*deconvOp.fetchGrain() << endl;
        gLogInfo << "\tdata_format " << (int)*deconvOp.dataFormat() << endl;
        gLogInfo << "\tpixel_mapping " << (int)*deconvOp.pixelMapping() << endl;
        gLogInfo << "\tbatch " << (int)*deconvOp.batch()  << endl;
        gLogInfo << "\tweight_format " << (int)*deconvOp.weightFormat()  << endl;
        gLogInfo << "\tb4d " << (int)*deconvOp.dataBank() << endl;
        gLogInfo << "\tb4w " << (int)*deconvOp.weightBank() << endl;
        gLogInfo << "\tbatch_stride " << (int)*deconvOp.batchStride()  << endl;
        gLogInfo << "\trelease " << (int)*deconvOp.release()  << endl;
        gLogInfo << "\tpost_extension " << (int)*deconvOp.postExtension()  << endl;
        gLogInfo << "\tpixel_override " << (int)*deconvOp.pixelOverride() << endl;
        gLogInfo << "\tmean_format " << (int)*deconvOp.meanFormat() << endl;
        gLogInfo << "\tstride-x " << (int)*deconvOp.convStrideX() << endl;
        gLogInfo << "\tstride-y " << (int)*deconvOp.convStrideY() << endl;
        gLogInfo << "\tpad-left " << (int)*deconvOp.padXLeft() << endl;
        gLogInfo << "\tpad-top " << (int)*deconvOp.padYTop() << endl;
        gLogInfo << "\tpad-right " << (int)*deconvOp.padXRight() << endl;
        gLogInfo << "\tpad-bottom " << (int)*deconvOp.padYBottom() << endl;
        gLogInfo << "\tdilationx-x " << (int)*deconvOp.dilationX() << endl;
        gLogInfo << "\tdilation-y " << (int)*deconvOp.dilationY() << endl;
        gLogInfo << "\tpra_truncate " << (int)*deconvOp.praTruncate() << endl;
        gLogInfo << "\tinputwidthcsc " << *deconvOp.inputWidthCSC() << endl;
        gLogInfo << "\tinputheightcsc " << *deconvOp.inputHeightCSC() << endl;
        gLogInfo << "\tinputchannelcsc " << *deconvOp.inputChannelCSC() << endl;
        gLogInfo << "\tkernelwidthcsc " << *deconvOp.kernelWidthCSC() << endl;
        gLogInfo << "\tkernelheightcsc " << *deconvOp.kernelHeightCSC() << endl;
        gLogInfo << "\tkernelchannelcsc " << *deconvOp.kernelChannelCSC() << endl;
        gLogInfo << "\tinputwidthcmac " << *deconvOp.inputWidthCMAC() << endl;
        gLogInfo << "\tinputheightcmac " << *deconvOp.inputHeightCMAC() << endl;
        gLogInfo << "\tbytesperkernel " << *deconvOp.bytesPerKernel() << endl;
        gLogInfo << "\toffsetU " << (int)*surfAcc.offsetU() << endl;
        gLogInfo << "\tdependencyCount " << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tsrc tsd:" << srcTSD->id() << "/" << srcTSD->tensorBufferDesc()->id()
                                                        << ":off= " << srcTSD->bufferOffset() << endl;
        gLogInfo << "\tsrc addr=" << *srcDataAcc.address() << endl;
        gLogInfo << "\tsrc size " << *srcDataAcc.size()    << endl;
        gLogInfo << "\tsrc width " << *srcDataAcc.width()   << endl;
        gLogInfo << "\tsrc height " << *srcDataAcc.height()   << endl;
        gLogInfo << "\tsrc channel " << *srcDataAcc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *srcDataAcc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *srcDataAcc.surfStride()  << endl;
        gLogInfo << "\tdst tsd:" << dstTSD->id() << endl;
        gLogInfo << "\tdst addr=" << *dstDataAcc.address() << endl;
        gLogInfo << "\tdst size " << *dstDataAcc.size()    << endl;
        gLogInfo << "\tdst width " << *dstDataAcc.width()   << endl;
        gLogInfo << "\tdst height " << *dstDataAcc.height()   << endl;
        gLogInfo << "\tdst channel " << *dstDataAcc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dstDataAcc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dstDataAcc.surfStride()  << endl;
        gLogInfo << "\twt  tsd:" << wtTSD->id() << endl;
        gLogInfo << "\tweight addr=" << *wtDataAcc.address() << endl;
        gLogInfo << "\twt size " << *wtDataAcc.size()    << endl;
        gLogInfo << "\twt width " << *wtDataAcc.width()   << endl;
        gLogInfo << "\twt height " << *wtDataAcc.height()   << endl;
        gLogInfo << "\twt channel " << *wtDataAcc.channel()  << endl;
    }

    return e;
}

}; // nvdla::priv::
}; // nvdla::
