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
#include "priv/TargetConfig.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

void engine_ast::ConvCoreFullyConnectedOpNode::captureCanonicalParams()
{
    params().setHasBiasTerm(canonicalNode()->params().hasBiasTerm() == true ? 1 : 0);
    params().setWeightDims(canonicalNode()->params().weightDims());
    // FIXME: assuming stride for FC in both directions is 1
    params().setStride(Dims2(1, 1));
    params().setRawWeights(canonicalNode()->params().weights());
    params().setDLAWeights(Weights(DataType::FLOAT, NULL, 0));

    captureCanonicalWeights();
}


NvDlaError engine_ast::ConvCoreFullyConnectedOpNode::emitOp(Graph *g,
                                                         DLAInterface *target_dla,
                                                         NvU32 op_slot, NvU32 batch_id,
                                                         DLACommonOpDescAccessor       dep,
                                                         DLAOperationContainerAccessor op,
                                                         DLASurfaceContainerAccessor   surf)
{
    NvDlaError e  = NvDlaSuccess;

    DLAConvOpDescAccessor       conv_op         = op.convOpDescAccessor(0);
    DLACVTParamAccessor         out_cvt_acc     = conv_op.outCVTAccessor();
    DLACVTParamAccessor         in_cvt_acc      = conv_op.inCVTAccessor();
    DLAConvSurfaceDescAccessor  surf_acc        = surf.convSurfaceDescAccessor(0);
    DLADataCubeAccessor         src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor         dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor         weight_data_acc = surf_acc.weightDataAccessor();
    DLADataCubeAccessor         wmb_data_acc    = surf_acc.wmbDataAccessor();
    DLADataCubeAccessor         wgs_data_acc    = surf_acc.wgsDataAccessor();
    DLAConsumerAccessor         fused_acc       = dep.fusedParentAccessor();
    NVDLA_UNUSED(wmb_data_acc);
    NVDLA_UNUSED(wgs_data_acc);
    NVDLA_UNUSED(fused_acc);

    surface::TensorSurfaceDesc *src_tsd     = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *weight_tsd  = g->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    /* Parameters we don't need/care for MNIST except their safe default values*/
    *conv_op.padVal()       = 0; // FIXME: assuming same padding on both dimensions
    *conv_op.dataReuse()    = 0;
    *conv_op.weightReuse()  = 0;
    *conv_op.skipDataRls()  = 0;
    *conv_op.skipWeightRls()= 0;
    *conv_op.batch()        = 1;
    *conv_op.batchStride()  = 0;
    *conv_op.release()      = src_tsd->dimensions().h;
    *conv_op.meanFormat()   = conv_op.meanFormat_Disable();
    *conv_op.meanRY()       = 0;
    *conv_op.meanGU()       = 0;
    *conv_op.meanBV()       = 0;
    *conv_op.padXLeft()     = 0;
    *conv_op.padXRight()    = 0;
    *conv_op.padYTop()      = 0;
    *conv_op.padYBottom()   = 0;
    *conv_op.dilationX()    = 1;
    *conv_op.dilationY()    = 1;
    *conv_op.pixelMapping() = conv_op.pixelMapping_PitchLinear();   //default

    *in_cvt_acc.scale()     = params().convCoreCVT().inputCVT().scale();
    *in_cvt_acc.truncate()  = params().convCoreCVT().inputCVT().truncate();
    *in_cvt_acc.enable()    = (int)params().convCoreCVT().inputCVT().isEnable();
    *in_cvt_acc.offset()    = params().convCoreCVT().inputCVT().offset();

    *out_cvt_acc.scale()    = 1;
    *out_cvt_acc.truncate() = params().convCoreCVT().outTruncate();
    *out_cvt_acc.enable()   = 1;
    *out_cvt_acc.offset()   = 0;

    /* Common parameters */
    *conv_op.inPrecision()  = ASTToDLAInterface::getConvCorePrecision(target_dla, src_tsd->surfaceFormat().precision());
    *conv_op.outPrecision() = ASTToDLAInterface::getConvCorePrecision(target_dla, dst_tsd->surfaceFormat().precision());
    *conv_op.fetchGrain()   = 1;                                           //FIXME: right now its max of requirements of all conv nodes in mnist
    *conv_op.dataFormat()   = ASTToDLAInterface::getDataFormat(target_dla, src_tsd->surfaceFormat());
    *conv_op.weightFormat() = conv_op.weightFormat_Uncompressed();
    *conv_op.convStrideX()  = params(batch_id).stride().w;
    *conv_op.convStrideY()  = params(batch_id).stride().h;
    *conv_op.inputWidthCMAC()   = dst_tsd->dimensions().w;
    *conv_op.inputHeightCMAC()  = dst_tsd->dimensions().h;
    *conv_op.bytesPerKernel()   = surface::WeightDesc::bytesPerKernel(weight_tsd);

    *conv_op.convMode()         = conv_op.convMode_Direct();
    *conv_op.inputWidthCSC()    = src_tsd->dimensions().w;
    *conv_op.inputHeightCSC()   = src_tsd->dimensions().h;
    *conv_op.inputChannelCSC()  = src_tsd->dimensions().c;
    *conv_op.kernelHeightCSC()  = weight_tsd->dimensions().h;
    *conv_op.kernelWidthCSC()   = weight_tsd->dimensions().w;
    *conv_op.kernelChannelCSC() = weight_tsd->dimensions().c;

    /* entry-per-slice & banks should be calculated after conv-mode is determined */
    params(batch_id).setConvMode(ConvolutionModeEnum::CONV_DIRECT);
    *conv_op.entryPerSlice()= calculateEPS(src_tsd);
    *conv_op.dataBank()     = params(batch_id).dataBanksAllotted();
    *conv_op.weightBank()   = params(batch_id).weightBanksAllotted();

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(weight_data_acc, weight_tsd, IODirectionEnum::UNKNOWN, batch_id);
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);

    if ( g->debugOps() )
    {
        gLogInfo << "FullyConnected node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc data loc: " << (int) *src_data_acc.type() << endl;
        gLogInfo << "\tdst data loc: " << (int) *dst_data_acc.type() << endl;
        gLogInfo << "\tpost y extension: " << (int)*conv_op.postExtension() << endl;
        gLogInfo << "\tin_precision " << (int)*conv_op.inPrecision() << endl;
        gLogInfo << "\tout_precision " << (int)*conv_op.outPrecision() << endl;
        gLogInfo << "\tpad_val " << (int)*conv_op.padVal() << endl;
        gLogInfo << "\tconv mode " << (int)*conv_op.convMode() << endl;
        gLogInfo << "\tdata_reuse " << (int)*conv_op.dataReuse() << endl;
        gLogInfo << "\tweight_reuse " << (int)*conv_op.weightReuse() << endl;
        gLogInfo << "\tskip_data_rls " << (int)*conv_op.skipDataRls() << endl;
        gLogInfo << "\tskip_wt_rls " << (int)*conv_op.skipWeightRls() << endl;
        gLogInfo << "\teps " << *conv_op.entryPerSlice() << endl;
        gLogInfo << "\tfetch_grain " << (int)*conv_op.fetchGrain() << endl;
        gLogInfo << "\tdata_format " << (int)*conv_op.dataFormat() << endl;
        gLogInfo << "\tpixel_mapping " << (int)*conv_op.pixelMapping() << endl;
        gLogInfo << "\tbatch " << (int)*conv_op.batch()  << endl;
        gLogInfo << "\tweight_format " << (int)*conv_op.weightFormat()  << endl;
        gLogInfo << "\tb4d " << (int)*conv_op.dataBank() << endl;
        gLogInfo << "\tb4w " << (int)*conv_op.weightBank() << endl;
        gLogInfo << "\tbatch_stride " << (int)*conv_op.batchStride()  << endl;
        gLogInfo << "\trelease " << (int)*conv_op.release()  << endl;
        gLogInfo << "\tpost_extension " << (int)*conv_op.postExtension()  << endl;
        gLogInfo << "\tpixel_override " << (int)*conv_op.pixelOverride() << endl;
        gLogInfo << "\tmean_format " << (int)*conv_op.meanFormat() << endl;
        gLogInfo << "\tstride-x " << (int)*conv_op.convStrideX() << endl;
        gLogInfo << "\tstride-y " << (int)*conv_op.convStrideY() << endl;
        gLogInfo << "\tpad-left " << (int)*conv_op.padXLeft() << endl;
        gLogInfo << "\tpad-top " << (int)*conv_op.padYTop() << endl;
        gLogInfo << "\tpad-right " << (int)*conv_op.padXRight() << endl;
        gLogInfo << "\tpad-bottom " << (int)*conv_op.padYBottom() << endl;
        gLogInfo << "\tdilationx-x " << (int)*conv_op.dilationX() << endl;
        gLogInfo << "\tdilation-y " << (int)*conv_op.dilationY() << endl;
        gLogInfo << "\tpra_truncate " << (int)*conv_op.praTruncate() << endl;
        gLogInfo << "\tinputwidthcsc " << *conv_op.inputWidthCSC() << endl;
        gLogInfo << "\tinputheightcsc " << *conv_op.inputHeightCSC() << endl;
        gLogInfo << "\tinputchannelcsc " << *conv_op.inputChannelCSC() << endl;
        gLogInfo << "\tkernelwidthcsc " << *conv_op.kernelWidthCSC() << endl;
        gLogInfo << "\tkernelheightcsc " << *conv_op.kernelHeightCSC() << endl;
        gLogInfo << "\tkernelchannelcsc " << *conv_op.kernelChannelCSC() << endl;
        gLogInfo << "\tinputwidthcmac " << *conv_op.inputWidthCMAC() << endl;
        gLogInfo << "\tinputheightcmac " << *conv_op.inputHeightCMAC() << endl;
        gLogInfo << "\tbytesperkernel " << *conv_op.bytesPerKernel() << endl;
        gLogInfo << "\toffsetU " << (int)*surf_acc.offsetU() << endl;
        gLogInfo << "\tdependencyCount " << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tsrc tsd:" << src_tsd->id() << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.address() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst tsd:" << dst_tsd->id() << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
        gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
        gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
        gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
        gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
        gLogInfo << "\twt  tsd:" << weight_tsd->id() << endl;
        gLogInfo << "\tweight addr=" << *weight_data_acc.address() <<endl;
        gLogInfo << "\twt size " << *weight_data_acc.size()    << endl;
        gLogInfo << "\twt width " << *weight_data_acc.width()   << endl;
        gLogInfo << "\twt height " << *weight_data_acc.height()   << endl;
        gLogInfo << "\twt channel " << *weight_data_acc.channel()  << endl;
    }

    return e;
}

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
NvDlaError engine_ast::ConvCoreFullyConnectedOpNode::emitOp(NvU32 op_slot, NvU32 batch_id,
                                                         DLAInterface* target_dla,
                                                         DLACommonOpDescAccessor&       dep,
                                                         DLAOperationContainerAccessor& op,
                                                         DLASurfaceContainerAccessor&   surf,
                                                         nvdla_prototest_interface::Layer* protoLayer)
{
    NvDlaError e = NvDlaSuccess;

    DLAConvOpDescAccessor       conv_op         = op.convOpDescAccessor(0);
    DLACVTParamAccessor         out_cvt_acc     = conv_op.outCVTAccessor();
    DLACVTParamAccessor         in_cvt_acc      = conv_op.inCVTAccessor();
    DLAConvSurfaceDescAccessor  surf_acc        = surf.convSurfaceDescAccessor(0);
    DLADataCubeAccessor         src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor         dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor         weight_data_acc = surf_acc.weightDataAccessor();
    DLADataCubeAccessor         wmb_data_acc    = surf_acc.wmbDataAccessor();
    DLADataCubeAccessor         wgs_data_acc    = surf_acc.wgsDataAccessor();
    DLAConsumerAccessor         fused_acc       = dep.fusedParentAccessor();
    NVDLA_UNUSED(wmb_data_acc);
    NVDLA_UNUSED(wgs_data_acc);
    NVDLA_UNUSED(fused_acc);
    NVDLA_UNUSED(batch_id);

    surface::TensorSurfaceDesc *src_tsd     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *weight_tsd  = graph()->nodeInputTensorSurface(this, 0, supportedAuxSurfCategories());

    nvdla_prototest_interface::CONVOpDesc* protoConvOpDesc        = protoLayer->mutable_op_config()->mutable_conv_op();
    nvdla_prototest_interface::CONVSurfaceDesc* protoConvSurfDesc = protoLayer->mutable_surface()->mutable_conv_surface();
    nvdla_prototest_interface::DataCube* protoWtDataCube          = protoConvSurfDesc->mutable_weight_data();
    nvdla_prototest_interface::DataCube* protoSrcDataCube         = protoConvSurfDesc->mutable_src_data();
    nvdla_prototest_interface::DataCube* protoDstDataCube         = protoConvSurfDesc->mutable_dst_data();

    nvdla_prototest_interface::DataPrecision protoInPrec, protoOutPrec;

    protoLayer->set_index(op_slot);
    protoLayer->set_roi_index(0);
    protoLayer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV);
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
        if (*cons_acc.index() != -1)
        {
            nvdla_prototest_interface::Consumer* protoConsumer = protoLayer->add_bottom();
            protoConsumer->set_index(*cons_acc.index());
            switch(c)
            {
                case EngineTypeEnum::BDMA : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_BDMA); break;
                case EngineTypeEnum::CONVOLUTION : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
                case EngineTypeEnum::SDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP); break;
                case EngineTypeEnum::PDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_PDP); break;
                case EngineTypeEnum::CDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CDP); break;
                case EngineTypeEnum::RUBIK: protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_RUBIK); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer");
            }
            switch(dependencyParams().consumer(c).opEvent().v())
            {
                case OperationEventTypeEnum::OP_CDMA_WEIGHT_DONE : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_CDMA_WT_DONE); break;
                case OperationEventTypeEnum::OP_CDMA_DATA_DONE   : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_CDMA_DT_DONE); break;
                case OperationEventTypeEnum::OP_COMPLETED        : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_COMPLETED); break;
                case OperationEventTypeEnum::OP_ENABLED          : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_ENABLED); break;
                case OperationEventTypeEnum::OP_PROGRAMMED       : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_PROGRAMMED); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer event");
            }
        }
    }

    /* Fused operation NOP */

    switch(src_tsd->surfaceFormat().precision().v())
    {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : protoInPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : protoInPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : protoInPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized input precision");
    }

    switch(dst_tsd->surfaceFormat().precision().v())
    {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : protoOutPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : protoOutPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : protoOutPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized output precision");
    }

    protoConvOpDesc->set_in_precision(protoInPrec);
    protoConvOpDesc->set_out_precision(protoOutPrec);
    protoConvOpDesc->set_pad_val(*conv_op.padVal());

    switch(params(batch_id).convMode().v())
    {
        case ConvolutionModeEnum::CONV_DIRECT:   protoConvOpDesc->set_conv_mode(nvdla_prototest_interface::ConvMode::DIRECT); break;
        case ConvolutionModeEnum::CONV_WINOGRAD: protoConvOpDesc->set_conv_mode(nvdla_prototest_interface::ConvMode::WINOGRAD); break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized conv mode");
    }

    protoConvOpDesc->set_data_reuse(*conv_op.dataReuse());
    protoConvOpDesc->set_weight_reuse(*conv_op.weightReuse());
    protoConvOpDesc->set_skip_data_rls(*conv_op.skipDataRls());
    protoConvOpDesc->set_skip_weight_rls(*conv_op.skipWeightRls());
    protoConvOpDesc->set_entry_per_slice(*conv_op.entryPerSlice());
    protoConvOpDesc->set_fetch_grain(*conv_op.fetchGrain());

    switch(src_tsd->surfaceFormat().v())
    {
        case surface::SurfaceFormatEnum::NVDLA_IMG_R8:                  protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R10:                 protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R10); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R12:                 protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R12); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R16:                 protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R16); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R16_I:               protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R16_I); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R16_F:               protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R16_F); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A16B16G16R16:        protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A16B16G16R16); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_X16B16G16R16:        protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_X16B16G16R16); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A16B16G16R16_F:      protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A16B16G16R16_F); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A16Y16U16V16:        protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A16Y16U16V16); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_V16U16Y16A16:        protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_V16U16Y16A16); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A16Y16U16V16_F:      protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A16Y16U16V16_F); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A8B8G8R8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A8B8G8R8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A8R8G8B8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A8R8G8B8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_B8G8R8A8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_B8G8R8A8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R8G8B8A8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R8G8B8A8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_X8B8G8R8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_X8B8G8R8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_X8R8G8B8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_X8R8G8B8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_B8G8R8X8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_B8G8R8X8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R8G8B8X8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R8G8B8X8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A2B10G10R10:         protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A2B10G10R10); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A2R10G10B10:         protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A2R10G10B10); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_B10G10R10A2:         protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_B10G10R10A2); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_R10G10B10A2:         protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_R10G10B10A2); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A2Y10U10V10:         protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A2Y10U10V10); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_V10U10Y10A2:         protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_V10U10Y10A2); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_A8Y8U8V8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_A8Y8U8V8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_V8U8Y8A8:            protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_V8U8Y8A8); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y8___U8V8_N444:      protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y8___U8V8_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y8___V8U8_N444:      protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y8___V8U8_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y10___U10V10_N444:   protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y10___U10V10_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y10___V10U10_N444:   protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y10___V10U10_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y12___U12V12_N444:   protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y12___U12V12_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y12___V12U12_N444:   protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y12___V12U12_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y16___U16V16_N444:   protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y16___U16V16_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_IMG_Y16___V16U16_N444:   protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_T_Y16___V16U16_N444); break;
        case surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8:
        case surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT16:
        case surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16:       protoConvOpDesc->set_data_format(nvdla_prototest_interface::DataFormat::FORMAT_FEATURE); break;
        default: REPORT_ERROR(NvDlaError_BadParameter, "Unsupported surface format provided %s", src_tsd->surfaceFormat().c_str());
    }

    protoConvOpDesc->mutable_in_cvt()->set_enable(*in_cvt_acc.enable());
    protoConvOpDesc->mutable_in_cvt()->set_truncate(*in_cvt_acc.truncate());
    protoConvOpDesc->mutable_in_cvt()->set_scale(*in_cvt_acc.scale());
    protoConvOpDesc->mutable_in_cvt()->set_offset(*in_cvt_acc.offset());

    protoConvOpDesc->mutable_out_cvt()->set_enable(*out_cvt_acc.enable());
    protoConvOpDesc->mutable_out_cvt()->set_truncate(*out_cvt_acc.truncate());
    protoConvOpDesc->mutable_out_cvt()->set_scale(*out_cvt_acc.scale());
    protoConvOpDesc->mutable_out_cvt()->set_offset(*out_cvt_acc.offset());

    /* FIXME: dynamically figure out pitch-linear */
    protoConvOpDesc->set_pixel_mapping(nvdla_prototest_interface::ConvPixelMAP::PITCH_LINEAR);
    protoConvOpDesc->set_pixel_offset_x(graph()->profile()->networkInputPixelOffX());
    protoConvOpDesc->set_pixel_offset_y(graph()->profile()->networkInputPixelOffY());
    protoConvOpDesc->set_batch(1);
    protoConvOpDesc->set_weight_format(nvdla_prototest_interface::WeightFormat::UNCOMPRESSED);
    protoConvOpDesc->set_weight_bank(*conv_op.weightBank());
    protoConvOpDesc->set_data_bank(*conv_op.dataBank());
    protoConvOpDesc->set_batch_stride(0);
    protoConvOpDesc->set_release(*conv_op.release());
    protoConvOpDesc->set_post_extension(*conv_op.postExtension());
    protoConvOpDesc->set_pixel_override(nvdla_prototest_interface::PixelOverride::OVERRIDE_UINT);
    protoConvOpDesc->set_mean_format(nvdla_prototest_interface::MeanFormat::MEAN_DISABLE);
    protoConvOpDesc->set_mean_ax(0);
    protoConvOpDesc->set_mean_bv(0);
    protoConvOpDesc->set_mean_gu(0);
    protoConvOpDesc->set_mean_ry(0);
    protoConvOpDesc->set_conv_stride_x(*conv_op.convStrideX());
    protoConvOpDesc->set_conv_stride_y(*conv_op.convStrideY());
    protoConvOpDesc->set_pad_x_left(*conv_op.padXLeft());
    protoConvOpDesc->set_pad_x_right(*conv_op.padXRight());
    protoConvOpDesc->set_pad_y_bottom(*conv_op.padYBottom());
    protoConvOpDesc->set_pad_y_top(*conv_op.padYTop());
    protoConvOpDesc->set_dilation_x(*conv_op.dilationX());
    protoConvOpDesc->set_dilation_y(*conv_op.dilationY());
    protoConvOpDesc->set_pra_truncate(*conv_op.praTruncate());
    protoConvOpDesc->set_input_width_csc(*conv_op.inputWidthCSC());
    protoConvOpDesc->set_input_height_csc(*conv_op.inputHeightCSC());
    protoConvOpDesc->set_input_channel_csc(*conv_op.inputChannelCSC());
    protoConvOpDesc->set_kernel_width_csc(*conv_op.kernelWidthCSC());
    protoConvOpDesc->set_kernel_height_csc(*conv_op.kernelHeightCSC());
    protoConvOpDesc->set_kernel_channel_csc(*conv_op.kernelChannelCSC());
    protoConvOpDesc->set_input_width_cmac(*conv_op.inputWidthCMAC());
    protoConvOpDesc->set_input_height_cmac(*conv_op.inputHeightCMAC());
    protoConvOpDesc->set_bytes_per_kernel(*conv_op.bytesPerKernel());

    protoConvSurfDesc->set_offset_u(*surf_acc.offsetU());

    protoWtDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoWtDataCube->set_address(*weight_data_acc.address());
    protoWtDataCube->set_size(*weight_data_acc.size());
    protoWtDataCube->set_width(*weight_data_acc.width());
    protoWtDataCube->set_height(*weight_data_acc.height());
    protoWtDataCube->set_channel(*weight_data_acc.channel());
    protoWtDataCube->set_line_stride(*weight_data_acc.lineStride());
    protoWtDataCube->set_surf_stride(*weight_data_acc.surfStride());
    protoWtDataCube->set_plane_stride(*weight_data_acc.planeStride());
    protoWtDataCube->mutable_mem_info()->set_mem_id(weight_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoWtDataCube->mutable_mem_info()->set_mem_size(weight_tsd->tensorBufferDesc()->size());
    protoWtDataCube->mutable_mem_info()->set_offset(0);
    protoWtDataCube->mutable_mem_info()->set_fill_type(nvdla_prototest_interface::FillerType::FILL_RANDOM);
    protoWtDataCube->mutable_mem_info()->set_flag(nvdla_prototest_interface::MemFlag::DLA_MEM_SET);
    protoWtDataCube->mutable_mem_info()->set_precision(protoInPrec);

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
    protoSrcDataCube->mutable_mem_info()->set_fill_type(nvdla_prototest_interface::FillerType::FILL_RANDOM);
    protoSrcDataCube->mutable_mem_info()->set_flag(nvdla_prototest_interface::MemFlag::DLA_MEM_SET);
    protoSrcDataCube->mutable_mem_info()->set_precision(protoInPrec);

    protoDstDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_HW);
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
fail:
    return e;
}
#endif


}; // nvdla::priv::
}; // nvdla::
