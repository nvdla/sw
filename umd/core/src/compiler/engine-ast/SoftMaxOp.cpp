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
void engine_ast::CPUSoftMaxOpNode::captureCanonicalParams() { }

NvDlaError engine_ast::CPUSoftMaxOpNode::emitOp(Graph *g,
                                             EMUInterface *emu_if,
                                             NvU32 op_slot, NvU32 batch_id,
                                             EMUOperationContainerAccessor op,
                                             EMUOperationBufferContainerAccessor buf)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 mem_atomic_size = graph()->target_config()->memoryAtomicSize();
    EMUSoftmaxOpDescAccessor softmax_op = op.softmaxOpDescAccessor(0);
    EMUCommonOpDescAccessor softmax_op_common = softmax_op.commonOpDescAccessor();

    surface::TensorSurfaceDesc *src_tsd     = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    *softmax_op_common.op_type() = 1; //<-- EMU_OP_SOFTMAX
    if (graph()->profile()->computePrecision().v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        *softmax_op_common.input_scale_factor()  = inputEdges().at(0)->originalTensor()->getChannelScales().at(0);
        *softmax_op_common.output_scale_factor() = outputEdges().at(0)->originalTensor()->getChannelScales().at(0);
    }
    else
    {
        *softmax_op_common.input_scale_factor()  = 1.0f;
        *softmax_op_common.output_scale_factor() = 1.0f;
    }

    *softmax_op.axis() = 1;           //fixme: this->getParams().axis()

    EMUPowerBufferDescsAccessor power_buffer = buf.powerBufferDescsAccessor(0);
    EMUBufferDescAccessor src_data_acc = power_buffer.srcDataAccessor();
    EMUBufferDescAccessor dst_data_acc = power_buffer.dstDataAccessor();

    NvS16 src_id, dst_id;

    src_id = src_tsd->addressId(batch_id);
    dst_id = dst_tsd->addressId(batch_id);

    *src_data_acc.addressIndex() = src_id;
    *src_data_acc.addressIndexOffset() = src_tsd->addressIdOffset(batch_id);
    *src_data_acc.size()       = (NvU32)src_tsd->size();    //fixme: 64b -> 32b
    *src_data_acc.format()     = ASTToEMUInterface::getDataFormat(emu_if, src_tsd->surfaceFormat(), mem_atomic_size);
    *src_data_acc.width()      = (NvU16)src_tsd->dimensions().w; //fixme: 32b -> 16b
    *src_data_acc.height()     = (NvU16)src_tsd->dimensions().h; //fixme: 32b -> 16b
    *src_data_acc.channel()    = (NvU16)src_tsd->dimensions().c; //fixme: 32b -> 16b
    if ( src_tsd->bindable() ) {
        NvS16 addrId = src_tsd->addressId(batch_id);
        uintptr_t lineOffs = uintptr_t(src_data_acc.lineStride());// - uintptr_t(power_buffer.struct_base());
        uintptr_t surfOffs = uintptr_t(src_data_acc.surfStride());// - uintptr_t(power_buffer.struct_base());
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, lineOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Line));
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, surfOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Surf));
    }
    *src_data_acc.lineStride() = src_tsd->lineStride();
    *src_data_acc.surfStride() = src_tsd->surfaceStride();

    *dst_data_acc.addressIndex() = dst_id;
    *dst_data_acc.addressIndexOffset() = dst_tsd->addressIdOffset(batch_id);
    *dst_data_acc.size()       = (NvU32)dst_tsd->size();    //fixme: 64b -> 32b
    *dst_data_acc.format()     = ASTToEMUInterface::getDataFormat(emu_if, dst_tsd->surfaceFormat(), mem_atomic_size);
    *dst_data_acc.width()      = (NvU16)dst_tsd->dimensions().w; //fixme: 32b -> 16b
    *dst_data_acc.height()     = (NvU16)dst_tsd->dimensions().h; //fixme: 32b -> 16b
    *dst_data_acc.channel()    = (NvU16)dst_tsd->dimensions().c; //fixme: 32b -> 16b
    if ( dst_tsd->bindable() ) {
        NvS16 addrId = dst_tsd->addressId(batch_id);
        uintptr_t lineOffs = uintptr_t(dst_data_acc.lineStride());// - uintptr_t(power_buffer.struct_base());
        uintptr_t surfOffs = uintptr_t(dst_data_acc.surfStride());// - uintptr_t(power_buffer.struct_base());
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, lineOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Line));
        g->insertRelocEntry(ILoadable::RelocEntry(addrId, surfOffs,
                                                  NVDLA_LOADABLE_INTERFACE_EMU1,
                                                  NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS,
                                                  ELST_Surf));
    }
    *dst_data_acc.lineStride() = dst_tsd->lineStride();
    *dst_data_acc.surfStride() = dst_tsd->surfaceStride();

    if ( g->debugOps() )
    {
        gLogInfo << "Softmax node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.addressIndex() << "[" << *src_data_acc.addressIndexOffset() << "]" << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.addressIndex() << "[" << *dst_data_acc.addressIndexOffset() << "]" << endl;
        gLogInfo << "\tinput scale factor " << *softmax_op_common.input_scale_factor() << endl;
        gLogInfo << "\toutput scale factor " << *softmax_op_common.output_scale_factor() << endl;

        gLogInfo << "\tsrc size=" << *src_data_acc.size() << endl;
        gLogInfo << "\tsrc format=" << *src_data_acc.format() << endl;
        gLogInfo << "\tsrc width=" << *src_data_acc.width() << endl;
        gLogInfo << "\tsrc height=" << *src_data_acc.height() << endl;
        gLogInfo << "\tsrc channel=" << *src_data_acc.channel() << endl;

        gLogInfo << "\tdst size=" << *dst_data_acc.size() << endl;
        gLogInfo << "\tdst format=" << *dst_data_acc.format() << endl;
        gLogInfo << "\tdst width=" << *dst_data_acc.width() << endl;
        gLogInfo << "\tdst height=" << *dst_data_acc.height() << endl;
        gLogInfo << "\tdst channel=" << *dst_data_acc.channel() << endl;
    }

    return e;
}

};  // nvdla::priv::
};  // nvdla::
