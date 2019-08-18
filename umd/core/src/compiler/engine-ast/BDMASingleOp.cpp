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
#include "priv/Tensor.h"
#include "priv/Compiler.h"

#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

//----------------------------------------------------------------------
//                           Code Emission Utils
//----------------------------------------------------------------------


NvDlaError engine_ast::BDMASingleDMAOpNode::emitOp(engine_ast::Graph *g,
                                             DLAInterface *target_dla,
                                             NvU32 op_slot, NvU32 batch_id,
                                             DLACommonOpDescAccessor       dep,
                                             DLAOperationContainerAccessor op,
                                             DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;
    DLABDMAOpDescAccessor bdma_op = op.bdmaOpDescAccessor(0);
    surface::TensorSurfaceDesc *src_tsd = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    DLABDMASurfaceDescAccessor surf_acc = surf.bdmaSurfaceDescAccessor(0);
    DLABDMATransferDescAccessor trns_acc   = surf_acc.transferAccessor(0);    // always the 1st one for single-dma-op
    NVDLA_UNUSED(fused_acc);

    *bdma_op.numTransfers() = 1;

    emitDependencyParams(target_dla, dep, batch_id);

    *surf_acc.numTransfers() = getTransferParams().numTransfers();

    switch (src_tsd->tensorBufferDesc()->memoryLoc(batch_id).e())
    {
        case memory::LocationEnum::lDRAM:
            *surf_acc.srcType() = surf_acc.type_MC();
            break;

        case memory::LocationEnum::lCVSRAM:
            *surf_acc.srcType() = surf_acc.type_CV();
            break;

        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "goofy surf acc src type=%d\n",
                                 (int)src_tsd->tensorBufferDesc()->memoryLoc(batch_id).v());
            break;
    }

    switch ( dst_tsd->tensorBufferDesc()->memoryLoc(batch_id).e() )
    {
        case memory::LocationEnum::lDRAM:
            *surf_acc.dstType() = surf_acc.type_MC();
            break;
        case memory::LocationEnum::lCVSRAM:
            *surf_acc.dstType() = surf_acc.type_CV();
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "goofy surf acc dst type=%d\n",
                                 (int)dst_tsd->tensorBufferDesc()->memoryLoc(batch_id).v());
            break;
    }

    *trns_acc.srcAddress() = src_tsd->addressId(batch_id);
    *trns_acc.dstAddress() = dst_tsd->addressId(batch_id);

    *trns_acc.lineSize()   = getTransferParams().lineSize();
    *trns_acc.lineRepeat() = getTransferParams().lineRepeat();
    *trns_acc.srcLine()    = getTransferParams().srcLine();
    *trns_acc.dstLine()    = getTransferParams().destLine();
    *trns_acc.surfaceRepeat() = getTransferParams().surfaceRepeat();
    *trns_acc.srcSurface() = getTransferParams().srcSurface();
    *trns_acc.dstSurface() = getTransferParams().destSurface();

    if ( g->debugOps() )
    {
        gLogInfo << "Single BDMA node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc address: " << *trns_acc.srcAddress() << " type:" << (int) *surf_acc.srcType() << endl;
        gLogInfo << "\tdst address: " << *trns_acc.dstAddress() << " type:" << (int) *surf_acc.dstType() << endl;
        gLogInfo << "\tnum transfers: " << *surf_acc.numTransfers() << endl;
        gLogInfo << "\tline size: " << *trns_acc.lineSize() << " repeat: " << *trns_acc.lineRepeat() << endl;
        gLogInfo << "\tsrc line: " << *trns_acc.srcLine() << " dst line: " << *trns_acc.dstLine() << endl;
        gLogInfo << "\tsrc surface: " << *trns_acc.srcSurface() << " dst surface: " << *trns_acc.dstSurface() << endl;
        gLogInfo << "\tsurface repeat: " << *trns_acc.surfaceRepeat() << endl;
    }

 fail:
    return e;
}

};  // nvdla::priv::
};  // nvdla::
