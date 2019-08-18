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

#include <fstream>

#include "priv/EngineAST.h"
#include "priv/Profile.h"


using std::endl;

namespace nvdla
{
namespace priv
{

void engine_ast::SDPNOPNode::captureCanonicalParams()
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    switch(canonicalNode()->canonicalOpType().v())
    {
        case canonical_ast::CanonicalOpTypeEnum::CONVOLUTION: {
             canonical_ast::ConvolutionNode* canConvNode = canonical_ast::NodeFactory::nodeCast<canonical_ast::ConvolutionNode*>(canonicalNode());
             params().setNumGroups(canConvNode->params().numGroups());
        } break;
        case canonical_ast::CanonicalOpTypeEnum::DECONVOLUTION: {
             canonical_ast::DeconvolutionNode* canDeconvNode = canonical_ast::NodeFactory::nodeCast<canonical_ast::DeconvolutionNode*>(canonicalNode());
             params().setNumGroups(canDeconvNode->params().numGroups());
        } break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized canonical op type: %s", canonicalNode()->canonicalOpType().c_str());
    }

fail:
    return;
}

void engine_ast::SDPNOPNode::inheritParams(Node* otherNode)
{
    // inherit the parameters that make sense (keep adding later)
    SDPNOPNode* otherNOP = NodeFactory::nodeCast<SDPNOPNode*>(otherNode);
    params().setX1Params(otherNOP->params().x1Params());
    params().setConvMode(otherNOP->params().convMode());
    params().setWinogradParams(otherNOP->params().winogradParams());
    params().setNumGroups(otherNOP->params().numGroups());
    params().setOutCVT(otherNOP->params().outCVT());
}

NvDlaError engine_ast::SDPNOPNode::emitOp(Graph *g,
                                       DLAInterface *target_dla,
                                       NvU32 op_slot, NvU32 batch_id,
                                       DLACommonOpDescAccessor       dep,
                                       DLAOperationContainerAccessor op,
                                       DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;

    DLASDPOpDescAccessor  sdp_op    = op.sdpOpDescAccessor(0);
    DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
    DLASDPOpAccessor x1_op_acc      = sdp_op.x1OpAccessor();
    DLASDPOpAccessor x2_op_acc      = sdp_op.x2OpAccessor();
    DLASDPOpAccessor y_op_acc       = sdp_op.yOpAccessor();
    DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
    DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
    DLADataCubeAccessor y_data_acc      = surf_acc.yDataAccessor();
    NVDLA_UNUSED(x1_data_acc);
    NVDLA_UNUSED(x2_data_acc);
    NVDLA_UNUSED(y_data_acc);

    surface::TensorSurfaceDesc *src_tsd     = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    *sdp_op.srcPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, src_tsd->surfaceFormat().precision());
    *sdp_op.dstPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, dst_tsd->surfaceFormat().precision());
    *sdp_op.LUTIndex()       = -1;
    *sdp_op.batchNum()       = 1;
    *sdp_op.batchStride()    = 0;

    *out_cvt_acc.scale()     = params().outCVT().scale();
    *out_cvt_acc.truncate()  = params().outCVT().truncate();
    *out_cvt_acc.offset()    = params().outCVT().offset();
    *out_cvt_acc.enable()    = static_cast<NvU8>(params().outCVT().isEnable());

    *x1_op_acc.enable() = 0;
    *x2_op_acc.enable() = 0;
    *y_op_acc.enable()  = 0;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);

    if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD )
    {
        *sdp_op.convMode() = sdp_op.convMode_Winograd();

        *src_data_acc.width() = params().winogradParams().ioDims.w;
        *src_data_acc.height() = params().winogradParams().ioDims.h;

        *dst_data_acc.width() = params().winogradParams().ioDims.w;
        *dst_data_acc.height() = params().winogradParams().ioDims.h;
    }
    else if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_DIRECT )
    {
        *sdp_op.convMode() = sdp_op.convMode_Direct();
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Conv mode for %s", name().c_str());
    }

    if ( g->debugOps() )
    {
        gLogInfo << "SDP NOP node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc precision " << (int)*sdp_op.srcPrecision() << endl;
        gLogInfo << "\tdst precision " << (int)*sdp_op.dstPrecision() << endl;
        gLogInfo << "\tx1 enable " << (int)*x1_op_acc.enable() << endl;
        if (*x1_op_acc.enable())
        {
            gLogInfo << "\tx1 precision " << (int)*x1_op_acc.precision() << endl;
            gLogInfo << "\tx1 aluType " << (int)*x1_op_acc.ALUType() << endl;
            gLogInfo << "\tx1 type " << (int)*x1_op_acc.type() << endl;
            gLogInfo << "\tx1 mode " << (int)*x1_op_acc.mode() << endl;
            gLogInfo << "\tx1 act " << (int)*x1_op_acc.act() << endl;
            gLogInfo << "\tx1 shiftValue " << (int)*x1_op_acc.shiftValue() << endl;
            gLogInfo << "\tx1 aluOperand " << (int)*x1_op_acc.ALUOperand() << endl;
            gLogInfo << "\tx1 mulOperand " << (int)*x1_op_acc.MulOperand() << endl;
            gLogInfo << "\tx1 truncate " << (int)*x1_op_acc.truncate() << endl;
        }
        gLogInfo << "\tx2 enable " << (int)*x2_op_acc.enable() << endl;
        if (*x2_op_acc.enable())
        {
            gLogInfo << "\tx2 precision " << (int)*x2_op_acc.precision() << endl;
            gLogInfo << "\tx2 aluType " << (int)*x2_op_acc.ALUType() << endl;
            gLogInfo << "\tx2 type " << (int)*x2_op_acc.type() << endl;
            gLogInfo << "\tx2 mode " << (int)*x2_op_acc.mode() << endl;
            gLogInfo << "\tx2 act " << (int)*x2_op_acc.act() << endl;
            gLogInfo << "\tx2 shiftValue " << (int)*x2_op_acc.shiftValue() << endl;
            gLogInfo << "\tx2 aluOperand " << (int)*x2_op_acc.ALUOperand() << endl;
            gLogInfo << "\tx2 mulOperand " << (int)*x2_op_acc.MulOperand() << endl;
            gLogInfo << "\tx2 truncate " << (int)*x2_op_acc.truncate() << endl;
        }
        gLogInfo << "\ty enable " << (int)*y_op_acc.enable() << endl;
        if (*y_op_acc.enable())
        {
            gLogInfo << "\ty precision " << (int)*y_op_acc.precision() << endl;
            gLogInfo << "\ty aluType " << (int)*y_op_acc.ALUType() << endl;
            gLogInfo << "\ty type " << (int)*y_op_acc.type() << endl;
            gLogInfo << "\ty mode " << (int)*y_op_acc.mode() << endl;
            gLogInfo << "\ty act " << (int)*y_op_acc.act() << endl;
            gLogInfo << "\ty shiftValue " << (int)*y_op_acc.shiftValue() << endl;
            gLogInfo << "\ty aluOperand " << (int)*y_op_acc.ALUOperand() << endl;
            gLogInfo << "\ty mulOperand " << (int)*y_op_acc.MulOperand() << endl;
            gLogInfo << "\ty truncate " << (int)*y_op_acc.truncate() << endl;
        }
        gLogInfo << "\tsrc tsd:" << src_tsd->id() << endl;
        gLogInfo << "\tdst tsd:" << dst_tsd->id() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.address() << endl;
        gLogInfo << "\tsrc type=" << (int)*src_data_acc.type() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tscale addr=" << *x1_data_acc.address() <<  endl;
        gLogInfo << "\tscale type=" << (int)*x1_data_acc.type() << endl;
        gLogInfo << "\tscale size " << *x1_data_acc.size()    << endl;
        gLogInfo << "\tscale width " << *x1_data_acc.width()   << endl;
        gLogInfo << "\tscale height " << *x1_data_acc.height()   << endl;
        gLogInfo << "\tscale channel " << *x1_data_acc.channel()  << endl;
        gLogInfo << "\tscale linestride " << *x1_data_acc.lineStride() << endl;
        gLogInfo << "\tscale surfstride " << *x1_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
        gLogInfo << "\tdst type=" << (int)*dst_data_acc.type() << endl;
        gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
        gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
        gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
        gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
        gLogInfo << "\tout_cvt enable " << (int)*out_cvt_acc.enable() << endl;
        gLogInfo << "\tout_cvt scale " << (int)*out_cvt_acc.scale() << endl;
        gLogInfo << "\tout_cvt offset " << (int)*out_cvt_acc.offset() << endl;
        gLogInfo << "\tout_cvt truncate " << (int)*out_cvt_acc.truncate() << endl;
    }

fail:
    return e;
}

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
NvDlaError engine_ast::SDPNOPNode::emitOp(NvU32 op_slot, NvU32 batch_id,
                                       DLAInterface* target_dla,
                                       DLACommonOpDescAccessor&        dep,
                                       DLAOperationContainerAccessor&  op,
                                       DLASurfaceContainerAccessor&    surf,
                                       nvdla_prototest_interface::Layer* protoLayer)
{
    NvDlaError e = NvDlaSuccess;
    NvU8 numConsumers = 0;

    DLASDPOpDescAccessor  sdp_op    = op.sdpOpDescAccessor(0);
    DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
    DLASDPOpAccessor x1_op_acc      = sdp_op.x1OpAccessor();
    DLASDPOpAccessor x2_op_acc      = sdp_op.x2OpAccessor();
    DLASDPOpAccessor y_op_acc       = sdp_op.yOpAccessor();
    DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
    DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
    DLADataCubeAccessor y_data_acc      = surf_acc.yDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(x1_data_acc);
    NVDLA_UNUSED(x2_data_acc);
    NVDLA_UNUSED(y_data_acc);
    NVDLA_UNUSED(batch_id);

    surface::TensorSurfaceDesc *src_tsd     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    nvdla_prototest_interface::SDPOpDesc* protoSDPOpDesc        = protoLayer->mutable_op_config()->mutable_sdp_op();
    nvdla_prototest_interface::SDPSurfaceDesc* protoSDPSurfDesc = protoLayer->mutable_surface()->mutable_sdp_surface();
    nvdla_prototest_interface::SDPOp*          protoSDPX1OpDesc = protoSDPOpDesc->mutable_x1_op();
    nvdla_prototest_interface::SDPOp*          protoSDPX2OpDesc = protoSDPOpDesc->mutable_x2_op();
    nvdla_prototest_interface::SDPOp*          protoSDPYOpDesc  = protoSDPOpDesc->mutable_y_op();
    nvdla_prototest_interface::DataCube* protoSrcDataCube       = protoSDPSurfDesc->mutable_src_data();
    nvdla_prototest_interface::DataCube* protoDstDataCube       = protoSDPSurfDesc->mutable_dst_data();
    nvdla_prototest_interface::DataPrecision protoSrcPrec, protoDstPrec;

    protoLayer->set_index(op_slot);
    protoLayer->set_roi_index(0);
    protoLayer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP);
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

    /* fused node */
    if (dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT))
    {
        nvdla_prototest_interface::Consumer* protoFusedConsumer = protoLayer->mutable_fused();
        protoFusedConsumer->set_index(*fused_acc.index());
        protoFusedConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_ENABLED);
        switch(dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT)->engineType().v()) {
            case EngineTypeEnum::CONVOLUTION: protoFusedConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
            default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "SDP can have only Conv op as its fused partner on input side");
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

    protoSDPOpDesc->set_src_precision(protoSrcPrec);
    protoSDPOpDesc->set_dst_precision(protoDstPrec);
    protoSDPOpDesc->set_lut_index(-1);

    protoSDPOpDesc->mutable_out_cvt()->set_enable(*out_cvt_acc.enable());
    protoSDPOpDesc->mutable_out_cvt()->set_offset(*out_cvt_acc.offset());
    protoSDPOpDesc->mutable_out_cvt()->set_scale(*out_cvt_acc.scale());
    protoSDPOpDesc->mutable_out_cvt()->set_truncate(*out_cvt_acc.truncate());

    protoSDPOpDesc->set_conv_mode(nvdla_prototest_interface::ConvMode::DIRECT);
    protoSDPOpDesc->set_batch_num(1);
    protoSDPOpDesc->set_batch_stride(0);

    protoSDPX1OpDesc->set_enable(*x1_op_acc.enable());
    protoSDPX1OpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPX1OpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
    protoSDPX1OpDesc->set_mode(nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL);
    protoSDPX1OpDesc->set_act(nvdla_prototest_interface::SDPActivation::ACT_NONE);
    protoSDPX1OpDesc->set_shift_value(*x1_op_acc.shiftValue());
    protoSDPX1OpDesc->set_alu_operand(*x1_op_acc.ALUOperand());
    protoSDPX1OpDesc->set_mul_operand(*x1_op_acc.MulOperand());
    protoSDPX1OpDesc->set_truncate(*x1_op_acc.truncate());
    protoSDPX1OpDesc->set_precision(protoSrcPrec);
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*x1_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*x1_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*x1_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*x1_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*x1_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*x1_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*x1_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*x1_op_acc.cvt().mulCVTAccessor().offset());

    protoSDPX2OpDesc->set_enable(*x2_op_acc.enable());
    protoSDPX2OpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPX2OpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
    protoSDPX2OpDesc->set_mode(nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL);
    protoSDPX2OpDesc->set_act(nvdla_prototest_interface::SDPActivation::ACT_NONE);
    protoSDPX2OpDesc->set_shift_value(*x2_op_acc.shiftValue());
    protoSDPX2OpDesc->set_alu_operand(*x2_op_acc.ALUOperand());
    protoSDPX2OpDesc->set_mul_operand(*x2_op_acc.MulOperand());
    protoSDPX2OpDesc->set_truncate(*x2_op_acc.truncate());
    protoSDPX2OpDesc->set_precision(protoSrcPrec);
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*x2_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*x2_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*x2_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*x2_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*x2_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*x2_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*x2_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*x2_op_acc.cvt().mulCVTAccessor().offset());


    protoSDPYOpDesc->set_enable(*y_op_acc.enable());
    protoSDPYOpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPYOpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
    protoSDPYOpDesc->set_mode(nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL);
    protoSDPYOpDesc->set_act(nvdla_prototest_interface::SDPActivation::ACT_NONE);
    protoSDPYOpDesc->set_shift_value(*y_op_acc.shiftValue());
    protoSDPYOpDesc->set_alu_operand(*y_op_acc.ALUOperand());
    protoSDPYOpDesc->set_mul_operand(*y_op_acc.MulOperand());
    protoSDPYOpDesc->set_truncate(*y_op_acc.truncate());
    protoSDPYOpDesc->set_precision(protoSrcPrec);
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*y_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*y_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*y_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*y_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*y_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*y_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*y_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*y_op_acc.cvt().mulCVTAccessor().offset());


    if (*fused_acc.index() != -1) {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_HW);
    } else {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    }
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

};  // nvdla::priv::
};  // nvdla::

