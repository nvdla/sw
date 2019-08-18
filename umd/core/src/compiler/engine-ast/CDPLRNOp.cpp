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
#include "priv/Compiler.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{
namespace engine_ast
{

void CDPLRNOpNode::captureCanonicalParams()
{
    NvDlaError e = NvDlaSuccess;

    params().setLocalSize(canonicalNode()->params().localSize());
    params().setAlpha(canonicalNode()->params().alpha());
    params().setBeta(canonicalNode()->params().beta());
    params().setK(canonicalNode()->params().k());

    // Register the LRN lut with the LutManager
    PROPAGATE_ERROR_FAIL(graph()->lutManager()->registerLRN(graph()->profile()->computePrecision().e(),
                                                            params().localSize(), params().alpha(), params().beta(), params().k(), &m_hLut));


fail:
    return;
}

NvDlaError engine_ast::CDPLRNOpNode::emitOp(Graph *g,
                                         DLAInterface *target_dla,
                                         NvU32 op_slot, NvU32 batch_id,
                                         DLACommonOpDescAccessor       dep,
                                         DLAOperationContainerAccessor op,
                                         DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;


    DLACDPOpDescAccessor cdp_op = op.cdpOpDescAccessor(0);
    DLACVTParamAccessor in_cvt_acc = cdp_op.inCVTAccessor();
    DLACVTParamAccessor out_cvt_acc = cdp_op.outCVTAccessor();

    DLACDPSurfaceDescAccessor cdp_surf = surf.cdpSurfaceDescAccessor(0);

    DLADataCubeAccessor src_data_acc = cdp_surf.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc = cdp_surf.dstDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(fused_acc);

    surface::TensorSurfaceDesc *src_tsd     = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    *cdp_op.inPrecision()     = ASTToDLAInterface::getCDPPrecision(target_dla, src_tsd->surfaceFormat().precision());
    *cdp_op.outPrecision()    = ASTToDLAInterface::getCDPPrecision(target_dla, dst_tsd->surfaceFormat().precision());
    *cdp_op.LUTIndex()        = graph()->lutManager()->getIndex(m_hLut);

    *in_cvt_acc.scale()       = 1;
    *in_cvt_acc.truncate()    = 0;
    *in_cvt_acc.offset()      = 0;
    *in_cvt_acc.enable()      = 1;

    *out_cvt_acc.scale()      = 1;
    *out_cvt_acc.truncate()   = 0;
    *out_cvt_acc.offset()     = 0;
    *out_cvt_acc.enable()     = 1;

    *cdp_op.localSize()       = params(batch_id).localSize();
    *cdp_op.bypassSquareSum() = 0;
    *cdp_op.bypassOutMul()    = 0;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::UNKNOWN, batch_id);

    if ( g->debugOps() )
    {
        gLogInfo << "CDP activation node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tin precision " << (int)*cdp_op.inPrecision() << endl;
        gLogInfo << "\tout precision " << (int)*cdp_op.outPrecision() << endl;
        gLogInfo << "\tsrc tsd:" << src_tsd->id() << endl;
        gLogInfo << "\tdst tsd:" << dst_tsd->id() << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.address() << endl;
        gLogInfo << "\tsrc type=" << (int)*src_data_acc.type() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
        gLogInfo << "\tdst type=" << (int)*dst_data_acc.type() << endl;
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
NvDlaError engine_ast::CDPLRNOpNode::emitOp(NvU32 op_slot, NvU32 batch_id,
                                         DLAInterface* target_dla,
                                         DLACommonOpDescAccessor&        dep,
                                         DLAOperationContainerAccessor&  op,
                                         DLASurfaceContainerAccessor&    surf,
                                         nvdla_prototest_interface::Layer* protoLayer)
{
    NvDlaError e = NvDlaSuccess;
    NvU8 numConsumers = 0;

    DLACDPOpDescAccessor cdp_op = op.cdpOpDescAccessor(0);
    DLACVTParamAccessor in_cvt_acc = cdp_op.inCVTAccessor();
    DLACVTParamAccessor out_cvt_acc = cdp_op.outCVTAccessor();

    DLACDPSurfaceDescAccessor cdp_surf = surf.cdpSurfaceDescAccessor(0);

    DLADataCubeAccessor src_data_acc = cdp_surf.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc = cdp_surf.dstDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(fused_acc);

    surface::TensorSurfaceDesc *src_tsd     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    nvdla_prototest_interface::CDPOpDesc* protoCDPOpDesc        = protoLayer->mutable_op_config()->mutable_cdp_op();
    nvdla_prototest_interface::CDPSurfaceDesc* protoCDPSurfDesc = protoLayer->mutable_surface()->mutable_cdp_surface();
    nvdla_prototest_interface::DataCube* protoSrcDataCube       = protoCDPSurfDesc->mutable_src_data();
    nvdla_prototest_interface::DataCube* protoDstDataCube       = protoCDPSurfDesc->mutable_dst_data();
    nvdla_prototest_interface::DataPrecision protoSrcPrec, protoDstPrec;

    protoLayer->set_index(op_slot);
    protoLayer->set_roi_index(0);
    protoLayer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CDP);
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

    protoCDPOpDesc->set_in_precision(protoSrcPrec);
    protoCDPOpDesc->set_out_precision(protoDstPrec);
    protoCDPOpDesc->set_lut_index(*cdp_op.LUTIndex());
    protoCDPOpDesc->set_local_size(*cdp_op.localSize());
    protoCDPOpDesc->set_bypass_sqsum(*cdp_op.bypassSquareSum());
    protoCDPOpDesc->set_bypass_out_mul(*cdp_op.bypassOutMul());

    protoCDPOpDesc->mutable_in_cvt()->set_enable(*in_cvt_acc.enable());
    protoCDPOpDesc->mutable_in_cvt()->set_offset(*in_cvt_acc.offset());
    protoCDPOpDesc->mutable_in_cvt()->set_scale(*in_cvt_acc.scale());
    protoCDPOpDesc->mutable_in_cvt()->set_truncate(*in_cvt_acc.truncate());

    protoCDPOpDesc->mutable_out_cvt()->set_enable(*out_cvt_acc.enable());
    protoCDPOpDesc->mutable_out_cvt()->set_offset(*out_cvt_acc.offset());
    protoCDPOpDesc->mutable_out_cvt()->set_scale(*out_cvt_acc.scale());
    protoCDPOpDesc->mutable_out_cvt()->set_truncate(*out_cvt_acc.truncate());

    protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
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
fail:
    return e;
}
#endif

};  // nvdla::priv::engine_ast::
};  // nvdla::priv::
};  // nvdla::
