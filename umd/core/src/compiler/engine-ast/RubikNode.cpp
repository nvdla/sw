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
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{


void engine_ast::RubikNode::captureCanonicalParams()
{

}

/*
 * Rubik engine has special requirement that its input and output channels
 * should be aligned to: 16 (for fp16/int16) and 32 (for int8)
 */
Dims4 engine_ast::RubikNode::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    bool isSrcTSD = false;
    bool isDstTSD = false;
    Dims4 suggestedDims(-1,-1,-1,-1);

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    if (isSrcTSD)
    {
        Dims4 inSurfDims(-1,-1,-1,-1);
        Edge* inEdge = inputEdges()[0];
        Node* srcNode = graph()->upstreamNodes(inEdge).size() ? graph()->upstreamNodes(inEdge)[0] : NULL;
        if (srcNode)
        {
            inSurfDims = srcNode->suggestSurfaceDims(inEdge->tensorSurfaceDesc());
        }
        suggestedDims.n = std::max<NvS32>(params().contractOpParams().inDims.n, inSurfDims.n);
        suggestedDims.c = std::max<NvS32>(params().contractOpParams().inDims.c, inSurfDims.c);
        suggestedDims.h = std::max<NvS32>(params().contractOpParams().inDims.h, inSurfDims.h);
        suggestedDims.w = std::max<NvS32>(params().contractOpParams().inDims.w, inSurfDims.w);

        // use this opportunity to update the contract op params if they seem outdated
        if (suggestedDims != params().contractOpParams().inDims)
        {
            RubikEngineParams::ContractOpParams updatedContractOps = params().contractOpParams();
            updatedContractOps.inDims = suggestedDims;
            params().setContractOpParams(updatedContractOps);
        }
    }
    else
    {
        suggestedDims.n = std::max<NvS32>(params().contractOpParams().outDims.n, tsd->dimensions().n);
        suggestedDims.c = std::max<NvS32>(params().contractOpParams().outDims.c, tsd->dimensions().c);
        suggestedDims.h = std::max<NvS32>(params().contractOpParams().outDims.h, tsd->dimensions().h);
        suggestedDims.w = std::max<NvS32>(params().contractOpParams().outDims.w, tsd->dimensions().w);

        // use this opportunity to update the contract op params if they seem outdated
        if (suggestedDims != params().contractOpParams().outDims)
        {
            RubikEngineParams::ContractOpParams updatedContractOps = params().contractOpParams();
            updatedContractOps.outDims = suggestedDims;
            params().setContractOpParams(updatedContractOps);
        }
    }

fail:
    return suggestedDims;
}

NvU32 engine_ast::RubikNode::suggestLineStride(surface::TensorSurfaceDesc* tsd)
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

NvU32 engine_ast::RubikNode::suggestSurfaceStride(surface::TensorSurfaceDesc* tsd)
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

NvU64 engine_ast::RubikNode::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
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

/*------------------------------Handle Multi-Batch---------------------*/
NvDlaError engine_ast::RubikNode::handleMultiBatch()
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

NvDlaError engine_ast::RubikNode::determineContractOpParams()
{
    NvDlaError e = NvDlaSuccess;

    NvU16 CInExt, COutExt;
    Dims4 origRubikInDims, origRubikOutDims;
    engine_ast::RubikEngineParams::ContractOpParams contractOpParams;
    NvU32 atom_k_size = graph()->target_config()->atomicKSize();
    NvU32 chnlAlign = graph()->profile()->computePrecision() ==
            surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? atom_k_size : atom_k_size / 2;

    if (params().mode().v() != RubikModeEnum::RUBIK_MODE_CONTRACT)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't determine Contract op params for %s which is"
                " not selected for Contract mode", name().c_str());
    }

    PROPAGATE_ERROR_FAIL( repopulateEdgePorts() );

    origRubikInDims  = inputEdges()[0]->originalTensor()->getDimensions();
    origRubikOutDims = outputEdges()[0]->originalTensor()->getDimensions();

    contractOpParams.inDims   = origRubikInDims;
    contractOpParams.outDims  = origRubikOutDims;

    if ( debugRubik() )
    {
        gLogInfo << "orig rubik in dims: " << origRubikInDims.c << "x"
                 << origRubikInDims.h << "x" << origRubikInDims.w << endl;
        gLogInfo << "orig rubik out dims: " << origRubikOutDims.c << "x"
                 << origRubikOutDims.h << "x" << origRubikOutDims.w << endl;
    }

    /* Step-1: determine input side contract op details */
    CInExt   = ROUNDUP_AND_ALIGN(origRubikInDims.c, chnlAlign);
    contractOpParams.inDims.c  = CInExt;

    /* Step-2: Determine output side contract op details */
    COutExt  = ROUNDUP_AND_ALIGN(origRubikOutDims.c, chnlAlign);
    contractOpParams.outDims.c = COutExt;

    params().setContractOpParams(contractOpParams);

    if ( debugRubik() )
    {
        gLogInfo << "rubik contract op " << name() << " in: "
                 << contractOpParams.inDims.n << "x" << contractOpParams.inDims.c << "x"
                 << contractOpParams.inDims.h << "x" << contractOpParams.inDims.w << endl;
        gLogInfo << "rubik contract op " << name() << " out: "
                 << contractOpParams.outDims.n << "x" << contractOpParams.outDims.c << "x"
                 << contractOpParams.outDims.h << "x" << contractOpParams.outDims.w << endl;
    }

fail:
    return e;
}

/*----------------------Code Emission-----------------------------------*/
NvDlaError engine_ast::RubikNode::emitOp(Graph *g,
                                      DLAInterface *target_dla,
                                      NvU32 op_slot, NvU32 batch_id,
                                      DLACommonOpDescAccessor       dep,
                                      DLAOperationContainerAccessor op,
                                      DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;

    DLARubikOpDescAccessor      rubikOp    = op.rubikOpDescAccessor(0);
    DLARubikSurfaceDescAccessor surfAcc    = surf.rubikSurfaceDescAccessor(0);
    DLADataCubeAccessor         srcDataAcc = surfAcc.srcDataAccessor();
    DLADataCubeAccessor         dstDataAcc = surfAcc.dstDataAccessor();

    surface::TensorSurfaceDesc *srcTSD     = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dstTSD     = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    *rubikOp.precision() = ASTToDLAInterface::getRubikPrecision(target_dla, srcTSD->surfaceFormat().precision());
    *rubikOp.mode()      = ASTToDLAInterface::getRubikMode(target_dla, params(batch_id).mode());
    *rubikOp.strideX()   = (int)params(batch_id).deconvStride().w;
    *rubikOp.strideY()   = (int)params(batch_id).deconvStride().h;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(srcDataAcc, srcTSD, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dstDataAcc, dstTSD, IODirectionEnum::UNKNOWN, batch_id);

    if (params(batch_id).mode() == RubikModeEnum::RUBIK_MODE_CONTRACT)
    {
        *srcDataAcc.channel() = params(batch_id).contractOpParams().inDims.c;
        *dstDataAcc.channel() = params(batch_id).contractOpParams().outDims.c;
    }

    if ( g->debugOps() )
    {
        gLogInfo << "Rubik node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\trubik precision " << (int)*rubikOp.precision() << endl;
        gLogInfo << "\trubik mode " << (int)*rubikOp.mode() << endl;
        gLogInfo << "\tdeconv-stride-x " << (int)*rubikOp.strideX() << endl;
        gLogInfo << "\tdeconv-stride-Y " << (int)*rubikOp.strideY() << endl;
        gLogInfo << "\tsrc tsd:" << srcTSD->id() << endl;
        gLogInfo << "\tdst tsd:" << dstTSD->id() << endl;
        gLogInfo << "\tsrc addr=" << (int) *srcDataAcc.address() << endl;
        gLogInfo << "\tsrc type=" << (int) *srcDataAcc.type() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tsrc size " << *srcDataAcc.size()    << endl;
        gLogInfo << "\tsrc width " << *srcDataAcc.width()   << endl;
        gLogInfo << "\tsrc height " << *srcDataAcc.height()   << endl;
        gLogInfo << "\tsrc channel " << *srcDataAcc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *srcDataAcc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *srcDataAcc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << (int) *dstDataAcc.address() << endl;
        gLogInfo << "\tdst type=" << (int) *dstDataAcc.type() << endl;
        gLogInfo << "\tdst size " << *dstDataAcc.size()    << endl;
        gLogInfo << "\tdst width " << *dstDataAcc.width()   << endl;
        gLogInfo << "\tdst height " << *dstDataAcc.height()   << endl;
        gLogInfo << "\tdst channel " << *dstDataAcc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dstDataAcc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dstDataAcc.surfStride()  << endl;
    }

    return e;
}

};  // nvdla::priv::
};  // nvdla::
