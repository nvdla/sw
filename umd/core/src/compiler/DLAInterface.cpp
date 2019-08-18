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

#include "priv/DLAInterface.h"

namespace nvdla
{

namespace priv
{

//
// dla_network_desc
//
DLANetworkDescAccessor::DLANetworkDescAccessor(NvU8 *base, const DLANetworkDesc &n) : _base(base), _n(n) { }

NvU8 * DLANetworkDescAccessor::struct_base()  const { return _base;      }
size_t DLANetworkDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t DLANetworkDescAccessor::struct_align() const { return _n.struct_align(); }

size_t DLANetworkDescAccessor::op_BDMA()  const { return _n.op_BDMA(); }
size_t DLANetworkDescAccessor::op_CONV()  const { return _n.op_CONV(); }
size_t DLANetworkDescAccessor::op_SDP()   const { return _n.op_SDP(); }
size_t DLANetworkDescAccessor::op_PDP()   const { return _n.op_PDP(); }
size_t DLANetworkDescAccessor::op_CDP()   const { return _n.op_CDP(); }
size_t DLANetworkDescAccessor::op_RUBIK() const { return _n.op_RUBIK(); }
size_t DLANetworkDescAccessor::numOpHeads() const { return _n.numOpHeads(); }

int16_t  * DLANetworkDescAccessor::operationDescIndex()   const { return _n.operationDescIndex(_base); }
int16_t  * DLANetworkDescAccessor::surfaceDescIndex()     const { return _n.surfaceDescIndex(_base); }
int16_t  * DLANetworkDescAccessor::dependencyGraphIndex() const { return _n.dependencyGraphIndex(_base); }
int16_t  * DLANetworkDescAccessor::LUTDataIndex()         const { return _n.LUTDataIndex(_base); }
int16_t  * DLANetworkDescAccessor::ROIArrayIndex()        const { return _n.ROIArrayIndex(_base); }
int16_t  * DLANetworkDescAccessor::surfaceIndex()         const { return _n.surfaceIndex(_base); }
int16_t  * DLANetworkDescAccessor::statListIndex()        const { return _n.statListIndex(_base); }
int16_t  * DLANetworkDescAccessor::opHead(size_t h)       const { return _n.opHead(_base, h); }
uint16_t * DLANetworkDescAccessor::numROIs()              const { return _n.numROIs(_base); }
uint16_t * DLANetworkDescAccessor::numOperations()        const { return _n.numOperations(_base); }
uint16_t * DLANetworkDescAccessor::numLUTs()              const { return _n.numLUTs(_base); }
uint16_t * DLANetworkDescAccessor::numAddresses()         const { return _n.numAddresses(_base); }
uint8_t  * DLANetworkDescAccessor::dynamicROI()           const { return _n.dynamicROI(_base); }
int16_t  * DLANetworkDescAccessor::inputLayer()           const { return _n.inputLayer(_base); }

//
// dla_consumer
//
DLAConsumerAccessor::DLAConsumerAccessor(NvU8 *base, const DLAConsumer &c) : _base(base), _c(c) { }

NvU8 *    DLAConsumerAccessor::struct_base()  const { return _base;      }
size_t    DLAConsumerAccessor::struct_size()  const { return _c.struct_size();  }
size_t    DLAConsumerAccessor::struct_align() const { return _c.struct_align(); }

int16_t * DLAConsumerAccessor::index() const { return _c.index(_base); }
uint8_t * DLAConsumerAccessor::event() const { return _c.event(_base); }
uint8_t   DLAConsumerAccessor::event_OpCompleted()      const { return _c.event_OpCompleted();      }
uint8_t   DLAConsumerAccessor::event_OpProgrammed()     const { return _c.event_OpProgrammed();     }
uint8_t   DLAConsumerAccessor::event_OpEnabled()        const { return _c.event_OpEnabled();        }
uint8_t   DLAConsumerAccessor::event_OpCDMAWeightDone() const { return _c.event_OpCDMAWeightDone(); }
uint8_t   DLAConsumerAccessor::event_OpCDMADataDone()   const { return _c.event_OpCDMADataDone();   }
uint8_t * DLAConsumerAccessor::res()   const { return _c.res(_base);   }


//
// dla_common_op_desc
//
DLACommonOpDescAccessor::DLACommonOpDescAccessor(NvU8 *base, const DLACommonOpDesc &c) : _base(base), _c(c) { }

NvU8 *    DLACommonOpDescAccessor::struct_base()  const { return _base;      }
size_t    DLACommonOpDescAccessor::struct_size()  const { return _c.struct_size();  }
size_t    DLACommonOpDescAccessor::struct_align() const { return _c.struct_align(); }

int16_t * DLACommonOpDescAccessor::index()           const { return _c.index(_base); }
int8_t * DLACommonOpDescAccessor::roiIndex()        const { return _c.roiIndex(_base); }
uint8_t * DLACommonOpDescAccessor::opType()          const { return _c.opType(_base);     }
uint8_t   DLACommonOpDescAccessor::opType_BDMA()     const {     return _c.opType_BDMA(); }
uint8_t   DLACommonOpDescAccessor::opType_CONV()     const {     return _c.opType_CONV(); }
uint8_t   DLACommonOpDescAccessor::opType_SDP()      const {     return _c.opType_SDP();  }
uint8_t   DLACommonOpDescAccessor::opType_PDP()      const {     return _c.opType_PDP();  }
uint8_t   DLACommonOpDescAccessor::opType_CDP()      const {     return _c.opType_CDP();  }
uint8_t   DLACommonOpDescAccessor::opType_RUBIK()    const {     return _c.opType_RUBIK();  }
uint8_t * DLACommonOpDescAccessor::dependencyCount() const { return _c.dependencyCount(_base);   }
uint8_t * DLACommonOpDescAccessor::reserved_xxx()    const { return _c.reserved_xxx(_base); }
uint8_t * DLACommonOpDescAccessor::reserved0(size_t i) const { return _c.reserved0(_base, i); }
size_t    DLACommonOpDescAccessor::numConsumers()    const { return _c.numConsumers(); }

DLAConsumerAccessor DLACommonOpDescAccessor::consumerAccessor(size_t c) const { return _c.consumerAccessor(_base, c); }
DLAConsumerAccessor DLACommonOpDescAccessor::fusedParentAccessor()      const { return _c.fusedParentAccessor(_base); }


//
// dla_bdma_transfer_desc
//
DLABDMATransferDescAccessor::DLABDMATransferDescAccessor(NvU8 *base, const DLABDMATransferDesc &t) : _base(base), _t(t) { }

NvU8 * DLABDMATransferDescAccessor::struct_base()  const { return _base;      }
size_t DLABDMATransferDescAccessor::struct_size()  const { return _t.struct_size();  }
size_t DLABDMATransferDescAccessor::struct_align() const { return _t.struct_align(); }

int16_t   * DLABDMATransferDescAccessor::srcAddress()    const { return _t.srcAddress(_base); }
int16_t   * DLABDMATransferDescAccessor::dstAddress()    const { return _t.dstAddress(_base); }
uint32_t  * DLABDMATransferDescAccessor::lineSize()      const { return _t.lineSize(_base); }
uint32_t  * DLABDMATransferDescAccessor::lineRepeat()    const { return _t.lineRepeat(_base); }
uint32_t  * DLABDMATransferDescAccessor::srcLine()       const { return _t.srcLine(_base); }
uint32_t  * DLABDMATransferDescAccessor::dstLine()       const { return _t.dstLine(_base); }
uint32_t  * DLABDMATransferDescAccessor::surfaceRepeat() const { return _t.surfaceRepeat(_base); }
uint32_t  * DLABDMATransferDescAccessor::srcSurface()    const { return _t.srcSurface(_base); }
uint32_t  * DLABDMATransferDescAccessor::dstSurface()    const { return _t.dstSurface(_base); }


//
// dla_bdma_surface_desc
//
DLABDMASurfaceDescAccessor::DLABDMASurfaceDescAccessor(NvU8 *base, const DLABDMASurfaceDesc &s) : _base(base), _s(s) { }

NvU8 * DLABDMASurfaceDescAccessor::struct_base()  const { return _base;      }
size_t DLABDMASurfaceDescAccessor::struct_size()  const { return _s.struct_size();  }
size_t DLABDMASurfaceDescAccessor::struct_align() const { return _s.struct_align(); }

uint8_t   * DLABDMASurfaceDescAccessor::srcType()      const { return _s.srcType(_base); }
uint8_t   * DLABDMASurfaceDescAccessor::dstType()      const { return _s.dstType(_base); }
uint16_t  * DLABDMASurfaceDescAccessor::numTransfers() const { return _s.numTransfers(_base); }

uint8_t    DLABDMASurfaceDescAccessor::type_MC()        const { return _s.type_MC(); }
uint8_t    DLABDMASurfaceDescAccessor::type_CV()        const { return _s.type_CV(); }
uint8_t    DLABDMASurfaceDescAccessor::type_HW()        const { return _s.type_HW(); }
uint16_t   DLABDMASurfaceDescAccessor::maxNumTransfers() const { return _s.maxNumTransfers(); }


DLABDMATransferDescAccessor DLABDMASurfaceDescAccessor::transferAccessor(size_t c) const { return _s.transferAccessor(_base, c); }

//
// dla_bdma_op_desc
//
DLABDMAOpDescAccessor::DLABDMAOpDescAccessor(NvU8 *base, const DLABDMAOpDesc &s) : _base(base), _s(s) { }

NvU8 * DLABDMAOpDescAccessor::struct_base()  const { return _base;      }
size_t DLABDMAOpDescAccessor::struct_size()  const { return _s.struct_size();  }
size_t DLABDMAOpDescAccessor::struct_align() const { return _s.struct_align(); }

uint16_t  * DLABDMAOpDescAccessor::numTransfers() const { return _s.numTransfers(_base); }
uint16_t  * DLABDMAOpDescAccessor::reserved0()    const { return _s.reserved0(_base); }


//
// dla_cvt_param
//
DLACVTParamAccessor::DLACVTParamAccessor(NvU8 *base, const DLACVTParam &l) : _base(base), _l(l) { }

NvU8 * DLACVTParamAccessor::struct_base()  const { return _base;      }
size_t DLACVTParamAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLACVTParamAccessor::struct_align() const { return _l.struct_align(); }

int16_t  * DLACVTParamAccessor::scale()    const { return _l.scale(_base); }
uint8_t  * DLACVTParamAccessor::truncate() const { return _l.truncate(_base); }
int32_t  * DLACVTParamAccessor::offset()   const { return _l.offset(_base); }
uint8_t  * DLACVTParamAccessor::enable()   const { return _l.enable(_base); }
uint16_t * DLACVTParamAccessor::reserved_xxx() const { return _l.reserved_xxx(_base); }

//
// dla_data_cube
//
DLADataCubeAccessor::DLADataCubeAccessor(NvU8 *base, const DLADataCube &l) : _base(base), _l(l) { }

NvU8 * DLADataCubeAccessor::struct_base()  const { return _base;      }
size_t DLADataCubeAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLADataCubeAccessor::struct_align() const { return _l.struct_align(); }

uint8_t  * DLADataCubeAccessor::type_xxx()       const { return _l.type_xxx(_base); }
uint8_t    DLADataCubeAccessor::type_MC_xxx()    const  { return _l.type_MC_xxx(); }
uint8_t    DLADataCubeAccessor::type_CV_xxx()    const  { return _l.type_CV_xxx(); }
uint8_t    DLADataCubeAccessor::type_HW_xxx()    const  { return _l.type_HW_xxx(); }
uint16_t * DLADataCubeAccessor::type()       const { return _l.type(_base); }
uint16_t   DLADataCubeAccessor::type_MC()    const  { return _l.type_MC(); }
uint16_t   DLADataCubeAccessor::type_CV()    const  { return _l.type_CV(); }
uint16_t   DLADataCubeAccessor::type_HW()    const  { return _l.type_HW(); }
int16_t  * DLADataCubeAccessor::address()    const  { return _l.address(_base); }
uint32_t * DLADataCubeAccessor::offset()     const  { return _l.offset(_base); }
uint32_t * DLADataCubeAccessor::size()       const  { return _l.size(_base); }
uint16_t * DLADataCubeAccessor::width()      const  { return _l.width(_base); }
uint16_t * DLADataCubeAccessor::height()     const  { return _l.height(_base); }
uint16_t * DLADataCubeAccessor::channel()    const  { return _l.channel(_base); }
uint16_t * DLADataCubeAccessor::reserved0()  const  { return _l.reserved0(_base); }
uint32_t * DLADataCubeAccessor::lineStride() const  { return _l.lineStride(_base); }
uint32_t * DLADataCubeAccessor::surfStride() const  { return _l.surfStride(_base); }
uint32_t * DLADataCubeAccessor::planeStride()const  { return _l.planeStride(_base); }

//
// dla_conv_surface_desc
//
DLAConvSurfaceDescAccessor::DLAConvSurfaceDescAccessor(NvU8 *base, const DLAConvSurfaceDesc &l) : _base(base), _l(l) { }

NvU8 * DLAConvSurfaceDescAccessor::struct_base()  const { return _base;      }
size_t DLAConvSurfaceDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLAConvSurfaceDescAccessor::struct_align() const { return _l.struct_align(); }

DLADataCubeAccessor DLAConvSurfaceDescAccessor::weightDataAccessor() const { return _l.weightDataAccessor(_base); }
DLADataCubeAccessor DLAConvSurfaceDescAccessor::meanDataAccessor()   const { return _l.meanDataAccessor(_base); }
DLADataCubeAccessor DLAConvSurfaceDescAccessor::wmbDataAccessor()    const { return _l.wmbDataAccessor(_base); }
DLADataCubeAccessor DLAConvSurfaceDescAccessor::wgsDataAccessor()    const { return _l.wgsDataAccessor(_base); }
DLADataCubeAccessor DLAConvSurfaceDescAccessor::srcDataAccessor()    const { return _l.srcDataAccessor(_base); }
DLADataCubeAccessor DLAConvSurfaceDescAccessor::dstDataAccessor()    const { return _l.dstDataAccessor(_base); }
uint64_t * DLAConvSurfaceDescAccessor::offsetU_xxx()    const { return _l.offsetU_xxx(_base); }
int64_t  * DLAConvSurfaceDescAccessor::offsetU()        const { return _l.offsetU(_base); }
uint32_t * DLAConvSurfaceDescAccessor::offsetV()        const { return _l.offsetV(_base); }
uint32_t * DLAConvSurfaceDescAccessor::inLineUVStride() const { return _l.inLineUVStride(_base); }

//
// dla_conv_op_desc
//
DLAConvOpDescAccessor::DLAConvOpDescAccessor(NvU8 *base, const DLAConvOpDesc &l) : _base(base), _l(l) { }

NvU8 * DLAConvOpDescAccessor::struct_base()  const { return _base;      }
size_t DLAConvOpDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLAConvOpDescAccessor::struct_align() const { return _l.struct_align(); }

//a_type  * DLAConvOpDescAccessor::a()         const { return _l.a(_base);    }
//b_type  * DLAConvOpDescAccessor::b(size_t i) const { return _l.b(_base, i); }
//size_t    DLAConvOpDescAccessor::numB()      const { return _l.numB();      }
// DLAConsumerAccessor DLACommonOpDescAccessor::fusedParentAccessor()      const { return _c.fusedParentAccessor(_base); }

uint8_t * DLAConvOpDescAccessor::inPrecision()  const { return _l.inPrecision(_base); }
uint8_t   DLAConvOpDescAccessor::inPrecision_Int8()  const { return _l.inPrecision_Int8(); }
uint8_t   DLAConvOpDescAccessor::inPrecision_Int16()  const { return _l.inPrecision_Int16(); }
uint8_t   DLAConvOpDescAccessor::inPrecision_FP16()  const { return _l.inPrecision_FP16(); }

uint8_t * DLAConvOpDescAccessor::outPrecision() const { return _l.outPrecision(_base); }
uint8_t   DLAConvOpDescAccessor::outPrecision_Int8()  const { return _l.outPrecision_Int8(); }
uint8_t   DLAConvOpDescAccessor::outPrecision_Int16()  const { return _l.outPrecision_Int16(); }
uint8_t   DLAConvOpDescAccessor::outPrecision_FP16()  const { return _l.outPrecision_FP16(); }

DLACVTParamAccessor DLAConvOpDescAccessor::inCVTAccessor()  const { return _l.inCVTAccessor(_base); }
DLACVTParamAccessor DLAConvOpDescAccessor::outCVTAccessor() const { return _l.outCVTAccessor(_base); }
int16_t  * DLAConvOpDescAccessor::padVal()                  const { return _l.padVal(_base); }
uint8_t  * DLAConvOpDescAccessor::convMode() const { return _l.convMode(_base); }
uint8_t    DLAConvOpDescAccessor::convMode_Direct()    const { return _l.convMode_Direct(); }
uint8_t    DLAConvOpDescAccessor::convMode_Winograd()  const { return _l.convMode_Winograd(); }
uint8_t  * DLAConvOpDescAccessor::dataReuse()     const { return _l.dataReuse(_base); }
uint8_t  * DLAConvOpDescAccessor::weightReuse()   const { return _l.weightReuse(_base); }
uint8_t  * DLAConvOpDescAccessor::skipDataRls()   const { return _l.skipDataRls(_base); }
uint8_t  * DLAConvOpDescAccessor::skipWeightRls() const { return _l.skipWeightRls(_base); }
uint8_t  * DLAConvOpDescAccessor::reserved0()     const { return _l.reserved0(_base); }
uint16_t * DLAConvOpDescAccessor::entryPerSlice() const { return _l.entryPerSlice(_base); }
uint16_t * DLAConvOpDescAccessor::fetchGrain()  const { return _l.fetchGrain(_base); }
uint8_t  * DLAConvOpDescAccessor::dataFormat()  const { return _l.dataFormat(_base); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R8()    const { return _l.dataFormat_T_R8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R10()   const { return _l.dataFormat_T_R10(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R12()   const { return _l.dataFormat_T_R12(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R16()   const { return _l.dataFormat_T_R16(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R16_I() const { return _l.dataFormat_T_R16_I(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R16_F() const { return _l.dataFormat_T_R16_F(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A16B16G16R16()   const { return _l.dataFormat_T_A16B16G16R16(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_X16B16G16R16()   const { return _l.dataFormat_T_X16B16G16R16(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A16B16G16R16_F() const { return _l.dataFormat_T_A16B16G16R16_F(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A16Y16U16V16()   const { return _l.dataFormat_T_A16Y16U16V16(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_V16U16Y16A16()   const { return _l.dataFormat_T_V16U16Y16A16(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A16Y16U16V16_F() const { return _l.dataFormat_T_A16Y16U16V16_F(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A8B8G8R8() const { return _l.dataFormat_T_A8B8G8R8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A8R8G8B8() const { return _l.dataFormat_T_A8R8G8B8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_B8G8R8A8() const { return _l.dataFormat_T_B8G8R8A8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R8G8B8A8() const { return _l.dataFormat_T_R8G8B8A8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_X8B8G8R8() const { return _l.dataFormat_T_X8B8G8R8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_X8R8G8B8() const { return _l.dataFormat_T_X8R8G8B8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_B8G8R8X8() const { return _l.dataFormat_T_B8G8R8X8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R8G8B8X8() const { return _l.dataFormat_T_R8G8B8X8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A2B10G10R10() const { return _l.dataFormat_T_A2B10G10R10(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A2R10G10B10() const { return _l.dataFormat_T_A2R10G10B10(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_B10G10R10A2() const { return _l.dataFormat_T_B10G10R10A2(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_R10G10B10A2() const { return _l.dataFormat_T_R10G10B10A2(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A2Y10U10V10()          const { return _l.dataFormat_T_A2Y10U10V10(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_V10U10Y10A2()          const { return _l.dataFormat_T_V10U10Y10A2(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_A8Y8U8V8()             const { return _l.dataFormat_T_A8Y8U8V8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_V8U8Y8A8()             const { return _l.dataFormat_T_V8U8Y8A8(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y8___U8V8_N444()       const { return _l.dataFormat_T_Y8___U8V8_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y8___V8U8_N444()       const { return _l.dataFormat_T_Y8___V8U8_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y10___U10V10_N444()    const { return _l.dataFormat_T_Y10___U10V10_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y10___V10U10_N444()    const { return _l.dataFormat_T_Y10___V10U10_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y12___U12V12_N444()    const { return _l.dataFormat_T_Y12___U12V12_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y12___V12U12_N444()    const { return _l.dataFormat_T_Y12___V12U12_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y16___U16V16_N444()    const { return _l.dataFormat_T_Y16___U16V16_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y16___V16U16_N444()    const { return _l.dataFormat_T_Y16___V16U16_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y16___U8V8_N444()      const { return _l.dataFormat_T_Y16___U8V8_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y16___V8U8_N444()      const { return _l.dataFormat_T_Y16___V8U8_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y8___U8___V8_N444()    const { return _l.dataFormat_T_Y8___U8___V8_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y10___U10___V10_N444() const { return _l.dataFormat_T_Y10___U10___V10_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y12___U12___V12_N444() const { return _l.dataFormat_T_Y12___U12___V12_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y16___U16___V16_N444() const { return _l.dataFormat_T_Y16___U16___V16_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_T_Y16___U8___V8_N444()   const { return _l.dataFormat_T_Y16___U8___V8_N444(); }
uint8_t    DLAConvOpDescAccessor::dataFormat_FEATURE() const { return _l.dataFormat_FEATURE(); }
uint8_t  * DLAConvOpDescAccessor::pixelMapping()   const { return _l.pixelMapping(_base); }
uint8_t    DLAConvOpDescAccessor::pixelMapping_PitchLinear() const { return _l.pixelMapping_PitchLinear(); }
uint8_t  * DLAConvOpDescAccessor::batch()         const { return _l.batch(_base); }
uint8_t  * DLAConvOpDescAccessor::weightFormat()    const { return _l.weightFormat(_base); }
uint8_t    DLAConvOpDescAccessor::weightFormat_Uncompressed() const { return _l.weightFormat_Uncompressed(); }
uint8_t    DLAConvOpDescAccessor::weightFormat_Compressed()   const { return _l.weightFormat_Compressed(); }
uint8_t  * DLAConvOpDescAccessor::dataBank()    const { return _l.dataBank(_base); }
uint8_t  * DLAConvOpDescAccessor::weightBank()  const { return _l.weightBank(_base); }
uint32_t * DLAConvOpDescAccessor::batchStride() const { return _l.batchStride(_base); }
uint16_t * DLAConvOpDescAccessor::release()     const { return _l.release(_base); }
uint8_t  * DLAConvOpDescAccessor::postExtension() const { return _l.postExtension(_base); }
uint8_t  * DLAConvOpDescAccessor::reserved1_xxx()     const { return _l.reserved1_xxx(_base); }
uint8_t  * DLAConvOpDescAccessor::pixelOverride()    const { return _l.pixelOverride(_base); }
uint8_t    DLAConvOpDescAccessor::pixelOverride_UINT()    const { return _l.pixelOverride_UINT(); }
uint8_t    DLAConvOpDescAccessor::pixelOverride_INT()    const { return _l.pixelOverride_INT(); }
uint8_t  * DLAConvOpDescAccessor::meanFormat() const { return _l.meanFormat(_base); }
uint8_t    DLAConvOpDescAccessor::meanFormat_None()      const { return _l.meanFormat_None(); }
uint8_t    DLAConvOpDescAccessor::meanFormat_Global()    const { return _l.meanFormat_Global(); }
uint8_t    DLAConvOpDescAccessor::meanFormat_PerPixel()  const { return _l.meanFormat_PerPixel(); }
uint8_t    DLAConvOpDescAccessor::meanFormat_Disable()  const { return _l.meanFormat_Disable(); }
uint8_t    DLAConvOpDescAccessor::meanFormat_Enable()  const { return _l.meanFormat_Enable(); }
int16_t  * DLAConvOpDescAccessor::meanRY()  const { return _l.meanRY(_base); }
int16_t  * DLAConvOpDescAccessor::meanGU()  const { return _l.meanGU(_base); }
int16_t  * DLAConvOpDescAccessor::meanBV()  const { return _l.meanBV(_base); }
int16_t  * DLAConvOpDescAccessor::meanAX()  const { return _l.meanAX(_base); }
uint8_t  * DLAConvOpDescAccessor::convStrideX() const { return _l.convStrideX(_base); }
uint8_t  * DLAConvOpDescAccessor::convStrideY() const { return _l.convStrideY(_base); }
uint8_t  * DLAConvOpDescAccessor::padXLeft()   const { return _l.padXLeft(_base); }
uint8_t  * DLAConvOpDescAccessor::padXRight()  const { return _l.padXRight(_base); }
uint8_t  * DLAConvOpDescAccessor::padYTop()    const { return _l.padYTop(_base); }
uint8_t  * DLAConvOpDescAccessor::padYBottom() const { return _l.padYBottom(_base); }
uint8_t  * DLAConvOpDescAccessor::dilationX() const { return _l.dilationX(_base); }
uint8_t  * DLAConvOpDescAccessor::dilationY() const { return _l.dilationY(_base); }
uint8_t  * DLAConvOpDescAccessor::reserved2(size_t i)   const { return _l.reserved2(_base, i); }
uint8_t  * DLAConvOpDescAccessor::praTruncate()      const { return _l.praTruncate(_base); }
uint16_t * DLAConvOpDescAccessor::inputWidthCSC()    const { return _l.inputWidthCSC(_base); }
uint16_t * DLAConvOpDescAccessor::inputHeightCSC()   const { return _l.inputHeightCSC(_base); }
uint16_t * DLAConvOpDescAccessor::inputChannelCSC()  const { return _l.inputChannelCSC(_base); }
uint16_t * DLAConvOpDescAccessor::kernelWidthCSC()   const { return _l.kernelWidthCSC(_base); }
uint16_t * DLAConvOpDescAccessor::kernelHeightCSC()  const { return _l.kernelHeightCSC(_base); }
uint16_t * DLAConvOpDescAccessor::kernelChannelCSC() const { return _l.kernelChannelCSC(_base); }
uint16_t * DLAConvOpDescAccessor::inputWidthCMAC()   const { return _l.inputWidthCMAC(_base); }
uint16_t * DLAConvOpDescAccessor::inputHeightCMAC()  const { return _l.inputHeightCMAC(_base); }
uint32_t * DLAConvOpDescAccessor::bytesPerKernel()   const { return _l.bytesPerKernel(_base); }


//
// dla_lut_offset
//
DLALUTOffsetAccessor::DLALUTOffsetAccessor(NvU8 *base, const DLALUTOffset &l) : _base(base), _l(l) { }

NvU8 * DLALUTOffsetAccessor::struct_base()  const { return _base;      }
size_t DLALUTOffsetAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLALUTOffsetAccessor::struct_align() const { return _l.struct_align(); }

uint8_t  * DLALUTOffsetAccessor::expOffset_xxx() const { return _l.expOffset_xxx(_base); }
int8_t  * DLALUTOffsetAccessor::expOffset() const { return _l.expOffset(_base); }
uint8_t  * DLALUTOffsetAccessor::fracBits_xxx()  const { return _l.fracBits_xxx(_base); }
int8_t  * DLALUTOffsetAccessor::fracBits()  const { return _l.fracBits(_base); }
uint16_t * DLALUTOffsetAccessor::reserved0() const { return _l.reserved0(_base); }


//
// dla_float_data
//
DLAFloatDataAccessor::DLAFloatDataAccessor(NvU8 *base, const DLAFloatData &l) : _base(base), _l(l) { }

NvU8 * DLAFloatDataAccessor::struct_base()  const { return _base;      }
size_t DLAFloatDataAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLAFloatDataAccessor::struct_align() const { return _l.struct_align(); }

int16_t  * DLAFloatDataAccessor::scale()        const { return _l.scale(_base); }
uint8_t  * DLAFloatDataAccessor::shifter_xxx()  const { return _l.shifter_xxx(_base); }
int8_t   * DLAFloatDataAccessor::shifter()      const { return _l.shifter(_base); }
uint8_t  * DLAFloatDataAccessor::reserved0()    const { return _l.reserved0(_base); }

//
// dla_slope
//
DLASlopeAccessor::DLASlopeAccessor(NvU8 *base, const DLASlope &l) : _base(base), _l(l) { }

NvU8 * DLASlopeAccessor::struct_base()  const { return _base;      }
size_t DLASlopeAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLASlopeAccessor::struct_align() const { return _l.struct_align(); }

DLAFloatDataAccessor DLASlopeAccessor::dataIAccessor() const { return _l.dataIAccessor(_base); }
uint16_t * DLASlopeAccessor::dataF() const { return _l.dataF(_base); }

//a_type  * DLASlopeAccessor::a()         const { return _l.a(_base);    }
//b_type  * DLASlopeAccessor::b(size_t i) const { return _l.b(_base, i); }
//size_t    DLASlopeAccessor::numB()      const { return _l.numB();      }
// DLAConsumerAccessor DLACommonOpDescAccessor::fusedParentAccessor()      const { return _c.fusedParentAccessor(_base); }


//
// dla_lut_param
//
DLALUTParamAccessor::DLALUTParamAccessor(NvU8 *base, const DLALUTParam &l) : _base(base), _l(l) { }

NvU8 * DLALUTParamAccessor::struct_base()  const { return _base;      }
size_t DLALUTParamAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLALUTParamAccessor::struct_align() const { return _l.struct_align(); }

int16_t * DLALUTParamAccessor::linearExpTable(size_t i)  const { return _l.linearExpTable(_base, i); }
size_t DLALUTParamAccessor::numLinearExpTable()          const { return _l.numLinearExpTable(); }
int16_t * DLALUTParamAccessor::linearOnlyTable(size_t i) const { return _l.linearOnlyTable(_base, i); }
size_t DLALUTParamAccessor::numLinearOnlyTable()         const { return _l.numLinearOnlyTable(); }
uint8_t * DLALUTParamAccessor::method()                  const { return _l.method(_base); }
uint8_t   DLALUTParamAccessor::method_Exponential()      const { return _l.method_Exponential();  }
uint8_t   DLALUTParamAccessor::method_Linear()           const { return _l.method_Linear();  }
DLALUTOffsetAccessor DLALUTParamAccessor::linearExpOffsetAccessor()  const { return _l.linearExpOffsetAccessor(_base); }
DLALUTOffsetAccessor DLALUTParamAccessor::linearOnlyOffsetAccessor()  const { return _l.linearOnlyOffsetAccessor(_base); }
uint64_t * DLALUTParamAccessor::linearExpStart()         const { return _l.linearExpStart(_base); }
uint64_t * DLALUTParamAccessor::linearExpEnd()           const { return _l.linearExpEnd(_base); }
uint64_t * DLALUTParamAccessor::linearOnlyStart()        const { return _l.linearOnlyStart(_base); }
uint64_t * DLALUTParamAccessor::linearOnlyEnd()          const { return _l.linearOnlyEnd(_base); }
DLASlopeAccessor DLALUTParamAccessor::linearExpUnderflowSlopeAccessor()  const { return _l.linearExpUnderflowSlopeAccessor(_base); }
DLASlopeAccessor DLALUTParamAccessor::linearExpOverflowSlopeAccessor()   const { return _l.linearExpOverflowSlopeAccessor(_base); }
DLASlopeAccessor DLALUTParamAccessor::linearOnlyUnderflowSlopeAccessor() const { return _l.linearOnlyUnderflowSlopeAccessor(_base); }
DLASlopeAccessor DLALUTParamAccessor::linearOnlyOverflowSlopeAccessor()  const { return _l.linearOnlyOverflowSlopeAccessor(_base); }
uint8_t * DLALUTParamAccessor::hybridPriority()    const { return _l.hybridPriority(_base); }
uint8_t * DLALUTParamAccessor::underflowPriority() const { return _l.underflowPriority(_base); }
uint8_t * DLALUTParamAccessor::overflowPriority()  const { return _l.overflowPriority(_base); }
uint8_t   DLALUTParamAccessor::priority_LinearExp()      const { return _l.priority_LinearExp();  }
uint8_t   DLALUTParamAccessor::priority_LinearOnly()     const { return _l.priority_LinearOnly();  }
int8_t  * DLALUTParamAccessor::inputScaleLog2()    const { return _l.inputScaleLog2(_base); }

//
// dla_sdp_surface_desc
//
DLASDPSurfaceDescAccessor::DLASDPSurfaceDescAccessor(NvU8 *base, const DLASDPSurfaceDesc &l) : _base(base), _l(l) { }

NvU8 * DLASDPSurfaceDescAccessor::struct_base()  const { return _base;      }
size_t DLASDPSurfaceDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLASDPSurfaceDescAccessor::struct_align() const { return _l.struct_align(); }

DLADataCubeAccessor DLASDPSurfaceDescAccessor::srcDataAccessor() const { return _l.srcDataAccessor(_base); }
DLADataCubeAccessor DLASDPSurfaceDescAccessor::x1DataAccessor()  const  { return _l.x1DataAccessor(_base); }
DLADataCubeAccessor DLASDPSurfaceDescAccessor::x2DataAccessor()  const  { return _l.x2DataAccessor(_base); }
DLADataCubeAccessor DLASDPSurfaceDescAccessor::yDataAccessor()   const  { return _l.yDataAccessor(_base); }
DLADataCubeAccessor DLASDPSurfaceDescAccessor::dstDataAccessor() const  { return _l.dstDataAccessor(_base); }


//
// dla_sdp_op
//
DLASDPOpAccessor::DLASDPOpAccessor(NvU8 *base, const DLASDPOp &l) : _base(base), _l(l) { }

NvU8 * DLASDPOpAccessor::struct_base()  const { return _base;      }
size_t DLASDPOpAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLASDPOpAccessor::struct_align() const { return _l.struct_align(); }

uint8_t * DLASDPOpAccessor::enable() const { return _l.enable(_base); }
uint8_t * DLASDPOpAccessor::ALUType() const { return _l.ALUType(_base); }
uint8_t   DLASDPOpAccessor::ALUType_Max() const { return _l.ALUType_Max(); }
uint8_t   DLASDPOpAccessor::ALUType_Min() const { return _l.ALUType_Min(); }
uint8_t   DLASDPOpAccessor::ALUType_Sum() const { return _l.ALUType_Sum(); }
uint8_t   DLASDPOpAccessor::ALUType_Eql() const { return _l.ALUType_Eql(); }
uint8_t * DLASDPOpAccessor::type() const { return _l.type(_base); }
uint8_t   DLASDPOpAccessor::type_None() const { return _l.type_None(); }
uint8_t   DLASDPOpAccessor::type_Mul()  const { return _l.type_Mul();  }
uint8_t   DLASDPOpAccessor::type_Add()  const { return _l.type_Add();  }
uint8_t   DLASDPOpAccessor::type_Both() const { return _l.type_Both(); }
uint8_t * DLASDPOpAccessor::mode() const { return _l.mode(_base); }
uint8_t   DLASDPOpAccessor::mode_PerLayer()  const { return _l.mode_PerLayer();  }
uint8_t   DLASDPOpAccessor::mode_PerKernel() const { return _l.mode_PerKernel(); }
uint8_t   DLASDPOpAccessor::mode_PerPoint()  const { return _l.mode_PerPoint();  }
uint8_t * DLASDPOpAccessor::act() const { return _l.act(_base); }
uint8_t   DLASDPOpAccessor::act_None() const { return _l.act_None(); }
uint8_t   DLASDPOpAccessor::act_RelU() const { return _l.act_RelU(); }
uint8_t   DLASDPOpAccessor::act_LUT()  const { return _l.act_LUT();  }
uint8_t * DLASDPOpAccessor::shiftValue() const { return _l.shiftValue(_base); }
int16_t * DLASDPOpAccessor::ALUOperand_xxx() const { return _l.ALUOperand_xxx(_base); }
int16_t * DLASDPOpAccessor::MulOperand_xxx() const { return _l.MulOperand_xxx(_base); }
int32_t * DLASDPOpAccessor::ALUOperand() const { return _l.ALUOperand(_base); }
int32_t * DLASDPOpAccessor::MulOperand() const { return _l.MulOperand(_base); }
uint8_t * DLASDPOpAccessor::truncate() const { return _l.truncate(_base);    }
uint8_t * DLASDPOpAccessor::precision()  const { return _l.precision(_base);   }
DLASDPCVTAccessor DLASDPOpAccessor::cvt() const { return _l.cvt(_base); }

//
// dla_sdp_op_desc
//
DLASDPOpDescAccessor::DLASDPOpDescAccessor(NvU8 *base, const DLASDPOpDesc &l) : _base(base), _l(l) { }

NvU8 * DLASDPOpDescAccessor::struct_base()  const { return _base;      }
size_t DLASDPOpDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLASDPOpDescAccessor::struct_align() const { return _l.struct_align(); }

uint8_t * DLASDPOpDescAccessor::srcPrecision()       const { return _l.srcPrecision(_base); }
uint8_t   DLASDPOpDescAccessor::srcPrecision_Int8()  const { return _l.srcPrecision_Int8(); }
uint8_t   DLASDPOpDescAccessor::srcPrecision_Int16() const { return _l.srcPrecision_Int16(); }
uint8_t   DLASDPOpDescAccessor::srcPrecision_FP16()  const { return _l.srcPrecision_FP16(); }
uint8_t * DLASDPOpDescAccessor::dstPrecision()       const { return _l.dstPrecision(_base); }
uint8_t   DLASDPOpDescAccessor::dstPrecision_Int8()  const { return _l.dstPrecision_Int8(); }
uint8_t   DLASDPOpDescAccessor::dstPrecision_Int16() const { return _l.dstPrecision_Int16(); }
uint8_t   DLASDPOpDescAccessor::dstPrecision_FP16()  const { return _l.dstPrecision_FP16(); }
int16_t * DLASDPOpDescAccessor::LUTIndex()     const { return _l.LUTIndex(_base); }
DLACVTParamAccessor DLASDPOpDescAccessor::outCVTAccessor() const { return _l.outCVTAccessor(_base); }
uint8_t  * DLASDPOpDescAccessor::convMode()    const { return _l.convMode(_base); }
uint8_t    DLASDPOpDescAccessor::convMode_Direct()      const { return _l.convMode_Direct(); }
uint8_t    DLASDPOpDescAccessor::convMode_Winograd()    const { return _l.convMode_Winograd(); }
uint8_t  * DLASDPOpDescAccessor::batchNum()    const { return _l.batchNum(_base); }
uint16_t * DLASDPOpDescAccessor::reserved0()   const { return _l.reserved0(_base); }
uint32_t * DLASDPOpDescAccessor::batchStride() const { return _l.batchStride(_base); }
DLASDPOpAccessor DLASDPOpDescAccessor::x1OpAccessor() const { return _l.x1OpAccessor(_base); }
DLASDPOpAccessor DLASDPOpDescAccessor::x2OpAccessor() const { return _l.x2OpAccessor(_base); }
DLASDPOpAccessor DLASDPOpDescAccessor::yOpAccessor()  const { return _l.yOpAccessor(_base); }

DLASDPCVTAccessor::DLASDPCVTAccessor(NvU8 *base, const DLASDPCVT &l) : _base(base), _l(l) { }

NvU8 * DLASDPCVTAccessor::struct_base()  const { return _base; }
size_t DLASDPCVTAccessor::struct_size()  const { return _l.struct_size(); }
size_t DLASDPCVTAccessor::struct_align() const { return _l.struct_align(); }

DLACVTParamAccessor DLASDPCVTAccessor::aluCVTAccessor() const { return _l.aluCVTAccessor(_base); }
DLACVTParamAccessor DLASDPCVTAccessor::mulCVTAccessor() const { return _l.mulCVTAccessor(_base); }

//
// dla_pdp_surface_desc
//
DLAPDPSurfaceDescAccessor::DLAPDPSurfaceDescAccessor(NvU8 *base, const DLAPDPSurfaceDesc &l) : _base(base), _l(l) { }

NvU8 * DLAPDPSurfaceDescAccessor::struct_base()  const { return _base;      }
size_t DLAPDPSurfaceDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLAPDPSurfaceDescAccessor::struct_align() const { return _l.struct_align(); }

DLADataCubeAccessor DLAPDPSurfaceDescAccessor::srcDataAccessor() const { return _l.srcDataAccessor(_base); }
DLADataCubeAccessor DLAPDPSurfaceDescAccessor::dstDataAccessor() const { return _l.dstDataAccessor(_base); }

//
// dla_pdp_op_desc
//
DLAPDPOpDescAccessor::DLAPDPOpDescAccessor(NvU8 *base, const DLAPDPOpDesc &l) : _base(base), _l(l) { }

NvU8 * DLAPDPOpDescAccessor::struct_base()  const { return _base;      }
size_t DLAPDPOpDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLAPDPOpDescAccessor::struct_align() const { return _l.struct_align(); }

uint8_t * DLAPDPOpDescAccessor::precision() const { return _l.precision(_base); }
uint8_t      DLAPDPOpDescAccessor::precision_Int8() const { return _l.precision_Int8(); }
uint8_t      DLAPDPOpDescAccessor::precision_Int16() const { return _l.precision_Int16(); }
uint8_t      DLAPDPOpDescAccessor::precision_FP16() const { return _l.precision_FP16(); }
uint8_t * DLAPDPOpDescAccessor::reserved0_xxx(size_t i) const { return _l.reserved0_xxx(_base, i); }
uint8_t * DLAPDPOpDescAccessor::reserved0() const { return _l.reserved0(_base); }

int16_t * DLAPDPOpDescAccessor::paddingValue_xxx(size_t i) const { return _l.paddingValue_xxx(_base, i); }
int32_t * DLAPDPOpDescAccessor::paddingValue(size_t i) const { return _l.paddingValue(_base, i); }
uint8_t * DLAPDPOpDescAccessor::splitNum() const { return _l.splitNum(_base); }
uint8_t * DLAPDPOpDescAccessor::reserved1_xxx(size_t i) const { return _l.reserved1_xxx(_base, i); }
uint16_t * DLAPDPOpDescAccessor::partialInWidthFirst() const { return _l.partialInWidthFirst(_base); }
uint16_t * DLAPDPOpDescAccessor::partialInWidthMid() const { return _l.partialInWidthMid(_base); }
uint16_t * DLAPDPOpDescAccessor::partialInWidthLast() const { return _l.partialInWidthLast(_base); }

uint16_t * DLAPDPOpDescAccessor::partialWidthFirst() const { return _l.partialWidthFirst(_base); }
uint16_t * DLAPDPOpDescAccessor::partialWidthMid() const { return _l.partialWidthMid(_base); }
uint16_t * DLAPDPOpDescAccessor::partialWidthLast() const { return _l.partialWidthLast(_base); }

uint8_t *DLAPDPOpDescAccessor::poolMode() const { return _l.poolMode(_base); }
uint8_t      DLAPDPOpDescAccessor::poolMode_AVG() const { return _l.poolMode_AVG(); }
uint8_t      DLAPDPOpDescAccessor::poolMode_MAX() const { return _l.poolMode_MAX(); }
uint8_t      DLAPDPOpDescAccessor::poolMode_MIN() const { return _l.poolMode_MIN(); }
uint8_t *DLAPDPOpDescAccessor::poolWidth()  const { return _l.poolWidth(_base); }
uint8_t *DLAPDPOpDescAccessor::poolHeight() const { return _l.poolHeight(_base); }
uint8_t *DLAPDPOpDescAccessor::reserved2_xxx() const { return _l.reserved2_xxx(_base); }

uint8_t *DLAPDPOpDescAccessor::strideX() const { return _l.strideX(_base); }
uint8_t *DLAPDPOpDescAccessor::strideY() const { return _l.strideY(_base); }
uint16_t *DLAPDPOpDescAccessor::strideX_xxx() const { return _l.strideX_xxx(_base); }
uint16_t *DLAPDPOpDescAccessor::strideY_xxx() const { return _l.strideY_xxx(_base); }
uint16_t *DLAPDPOpDescAccessor::reserved3_xxx() const { return _l.reserved3_xxx(_base); }

uint8_t *DLAPDPOpDescAccessor::padLeft()   const { return _l.padLeft(_base); }
uint8_t *DLAPDPOpDescAccessor::padRight()  const { return _l.padRight(_base); }
uint8_t *DLAPDPOpDescAccessor::padTop()    const { return _l.padTop(_base); }
uint8_t *DLAPDPOpDescAccessor::padBottom() const { return _l.padBottom(_base); }

//
// dla_cdp_surface_desc
//
DLACDPSurfaceDescAccessor::DLACDPSurfaceDescAccessor(NvU8 *base, const DLACDPSurfaceDesc &l) : _base(base), _l(l) { }

NvU8 * DLACDPSurfaceDescAccessor::struct_base()  const { return _base;      }
size_t DLACDPSurfaceDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLACDPSurfaceDescAccessor::struct_align() const { return _l.struct_align(); }

DLADataCubeAccessor DLACDPSurfaceDescAccessor::srcDataAccessor() const { return _l.srcDataAccessor(_base); }
DLADataCubeAccessor DLACDPSurfaceDescAccessor::dstDataAccessor() const { return _l.dstDataAccessor(_base); }


//
// dla_cdp_op_desc
//
DLACDPOpDescAccessor::DLACDPOpDescAccessor(NvU8 *base, const DLACDPOpDesc &l) : _base(base), _l(l) { }

NvU8 * DLACDPOpDescAccessor::struct_base()  const { return _base;      }
size_t DLACDPOpDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLACDPOpDescAccessor::struct_align() const { return _l.struct_align(); }

uint8_t * DLACDPOpDescAccessor::inPrecision() const { return _l.inPrecision(_base); }
uint8_t      DLACDPOpDescAccessor::inPrecision_Int8() const { return _l.inPrecision_Int8(); }
uint8_t      DLACDPOpDescAccessor::inPrecision_Int16() const { return _l.inPrecision_Int16(); }
uint8_t      DLACDPOpDescAccessor::inPrecision_FP16() const { return _l.inPrecision_FP16(); }

uint8_t * DLACDPOpDescAccessor::outPrecision() const { return _l.outPrecision(_base); }
uint8_t      DLACDPOpDescAccessor::outPrecision_Int8() const { return _l.outPrecision_Int8(); }
uint8_t      DLACDPOpDescAccessor::outPrecision_Int16() const { return _l.outPrecision_Int16(); }
uint8_t      DLACDPOpDescAccessor::outPrecision_FP16() const { return _l.outPrecision_FP16(); }

int16_t * DLACDPOpDescAccessor::LUTIndex() const { return _l.LUTIndex(_base); }
DLACVTParamAccessor DLACDPOpDescAccessor::inCVTAccessor()  const { return _l.inCVTAccessor(_base); }
DLACVTParamAccessor DLACDPOpDescAccessor::outCVTAccessor() const { return _l.outCVTAccessor(_base); }

uint8_t * DLACDPOpDescAccessor::localSize() const { return _l.localSize(_base); }
uint8_t * DLACDPOpDescAccessor::bypassSquareSum() const { return _l.bypassSquareSum(_base); }
uint8_t * DLACDPOpDescAccessor::bypassOutMul() const { return _l.bypassOutMul(_base); }
uint8_t * DLACDPOpDescAccessor::reserved0() const { return _l.reserved0(_base); }

//
// dla_rubik_surface_desc
//
DLARubikSurfaceDescAccessor::DLARubikSurfaceDescAccessor(NvU8 *base, const DLARubikSurfaceDesc &l) : _base(base), _l(l) { }

NvU8 * DLARubikSurfaceDescAccessor::struct_base()  const { return _base;      }
size_t DLARubikSurfaceDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLARubikSurfaceDescAccessor::struct_align() const { return _l.struct_align(); }

DLADataCubeAccessor DLARubikSurfaceDescAccessor::srcDataAccessor() const { return _l.srcDataAccessor(_base); }
DLADataCubeAccessor DLARubikSurfaceDescAccessor::dstDataAccessor() const { return _l.dstDataAccessor(_base); }


//
// dla_rubik_op_desc
//
DLARubikOpDescAccessor::DLARubikOpDescAccessor(NvU8 *base, const DLARubikOpDesc &l) : _base(base), _l(l) { }

NvU8 * DLARubikOpDescAccessor::struct_base()  const { return _base;      }
size_t DLARubikOpDescAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLARubikOpDescAccessor::struct_align() const { return _l.struct_align(); }

uint8_t * DLARubikOpDescAccessor::mode() const { return _l.mode(_base); }
uint8_t      DLARubikOpDescAccessor::mode_Contract() const { return _l.mode_Contract(); }
uint8_t      DLARubikOpDescAccessor::mode_Split() const { return _l.mode_Split(); }
uint8_t      DLARubikOpDescAccessor::mode_Merge() const { return _l.mode_Merge(); }

uint8_t * DLARubikOpDescAccessor::precision() const { return _l.precision(_base); }
uint8_t      DLARubikOpDescAccessor::precision_Int8() const { return _l.precision_Int8(); }
uint8_t      DLARubikOpDescAccessor::precision_Int16() const { return _l.precision_Int16(); }
uint8_t      DLARubikOpDescAccessor::precision_FP16() const { return _l.precision_FP16(); }

uint8_t * DLARubikOpDescAccessor::strideX() const { return _l.strideX(_base); }
uint8_t * DLARubikOpDescAccessor::strideY() const { return _l.strideY(_base); }


//
// dla_surface_container
//
DLASurfaceContainerAccessor::DLASurfaceContainerAccessor(NvU8 *base, const DLASurfaceContainer &l) : _base(base), _l(l) { }

NvU8 * DLASurfaceContainerAccessor::struct_base()  const { return _base;      }
size_t DLASurfaceContainerAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLASurfaceContainerAccessor::struct_align() const { return _l.struct_align(); }

DLABDMASurfaceDescAccessor  DLASurfaceContainerAccessor::bdmaSurfaceDescAccessor(size_t c) const { return _l.bdmaSurfaceDescAccessor(_base, c); }
DLAConvSurfaceDescAccessor  DLASurfaceContainerAccessor::convSurfaceDescAccessor(size_t c) const { return _l.convSurfaceDescAccessor(_base, c); }
DLASDPSurfaceDescAccessor   DLASurfaceContainerAccessor::sdpSurfaceDescAccessor(size_t c)  const { return _l.sdpSurfaceDescAccessor(_base, c); }
DLAPDPSurfaceDescAccessor   DLASurfaceContainerAccessor::pdpSurfaceDescAccessor(size_t c)  const { return _l.pdpSurfaceDescAccessor(_base, c); }
DLACDPSurfaceDescAccessor   DLASurfaceContainerAccessor::cdpSurfaceDescAccessor(size_t c)  const { return _l.cdpSurfaceDescAccessor(_base, c); }
DLARubikSurfaceDescAccessor DLASurfaceContainerAccessor::rubikSurfaceDescAccessor(size_t c)  const { return _l.rubikSurfaceDescAccessor(_base, c); }


//
// dla_operation_container
//
DLAOperationContainerAccessor::DLAOperationContainerAccessor(NvU8 *base, const DLAOperationContainer &l) : _base(base), _l(l) { }

NvU8 * DLAOperationContainerAccessor::struct_base()  const { return _base;      }
size_t DLAOperationContainerAccessor::struct_size()  const { return _l.struct_size();  }
size_t DLAOperationContainerAccessor::struct_align() const { return _l.struct_align(); }
DLABDMAOpDescAccessor  DLAOperationContainerAccessor::bdmaOpDescAccessor(size_t c) const { return _l.bdmaOpDescAccessor(_base, c); }
DLAConvOpDescAccessor  DLAOperationContainerAccessor::convOpDescAccessor(size_t c) const { return _l.convOpDescAccessor(_base, c); }
DLASDPOpDescAccessor   DLAOperationContainerAccessor::sdpOpDescAccessor (size_t c) const { return _l.sdpOpDescAccessor(_base, c);  }
DLAPDPOpDescAccessor   DLAOperationContainerAccessor::pdpOpDescAccessor (size_t c) const { return _l.pdpOpDescAccessor(_base, c);  }
DLACDPOpDescAccessor   DLAOperationContainerAccessor::cdpOpDescAccessor (size_t c) const { return _l.cdpOpDescAccessor(_base, c);  }
DLARubikOpDescAccessor DLAOperationContainerAccessor::rubikOpDescAccessor (size_t c) const { return _l.rubikOpDescAccessor(_base, c);  }


//
// DLAInterface::
//
DLANetworkDescAccessor  DLAInterface::networkDescAccessor(NvU8 *b)  const { return DLANetworkDescAccessor (b, networkDesc());  }
DLAConsumerAccessor     DLAInterface::consumerAccessor(NvU8 *b)     const { return DLAConsumerAccessor    (b, consumer());     }
DLACommonOpDescAccessor DLAInterface::commonOpDescAccessor(NvU8 *b) const { return DLACommonOpDescAccessor(b, commonOpDesc()); }
DLACVTParamAccessor     DLAInterface::cvtParamAccessor(NvU8 *b)   const   { return DLACVTParamAccessor(b, cvtParam()); }
DLADataCubeAccessor        DLAInterface::dataCubeAccessor(NvU8 *b)     const { return DLADataCubeAccessor(b, dataCube()); }
DLAConvSurfaceDescAccessor DLAInterface::convSurfaceDescAccessor(NvU8 *b) const { return DLAConvSurfaceDescAccessor(b, convSurfaceDesc()); }
DLAConvOpDescAccessor      DLAInterface::convOpDescAccessor(NvU8 *b)      const { return DLAConvOpDescAccessor(b, convOpDesc()); }
DLALUTOffsetAccessor       DLAInterface::lutOffsetAccessor(NvU8 *b)       const { return DLALUTOffsetAccessor(b, lutOffset()); }
DLAFloatDataAccessor       DLAInterface::floatDataAccessor(NvU8 *b)       const { return DLAFloatDataAccessor(b, floatData()); }
DLASlopeAccessor           DLAInterface::slopeAccessor(NvU8 *b)           const { return DLASlopeAccessor(b, slope()); }
DLALUTParamAccessor        DLAInterface::lutParamAccessor(NvU8 *b)        const { return DLALUTParamAccessor(b, lutParam()); }
DLASDPSurfaceDescAccessor  DLAInterface::sdpSurfaceDescAccessor(NvU8 *b)  const { return DLASDPSurfaceDescAccessor(b, sdpSurfaceDesc()); }
DLASDPOpAccessor           DLAInterface::sdpOpAccessor(NvU8 *b)           const { return DLASDPOpAccessor(b, sdpOp()); }
DLASDPOpDescAccessor       DLAInterface::sdpOpDescAccessor(NvU8 *b)       const { return DLASDPOpDescAccessor(b, sdpOpDesc()); }
DLASDPCVTAccessor          DLAInterface::sdpCVTAccessor(NvU8 *b)          const { return DLASDPCVTAccessor(b, sdpCVT()); }
DLAPDPOpDescAccessor       DLAInterface::pdpOpDescAccessor(NvU8* b)       const { return DLAPDPOpDescAccessor(b, pdpOpDesc()); }
DLAPDPSurfaceDescAccessor  DLAInterface::pdpSurfaceDescAccessor(NvU8* b)  const { return DLAPDPSurfaceDescAccessor(b, pdpSurfaceDesc()); }
DLACDPOpDescAccessor       DLAInterface::cdpOpDescAccessor(NvU8* b)       const { return DLACDPOpDescAccessor(b, cdpOpDesc()); }
DLACDPSurfaceDescAccessor  DLAInterface::cdpSurfaceDescAccessor(NvU8* b)  const { return DLACDPSurfaceDescAccessor(b, cdpSurfaceDesc()); }
DLARubikOpDescAccessor       DLAInterface::rubikOpDescAccessor(NvU8* b)       const { return DLARubikOpDescAccessor(b, rubikOpDesc()); }
DLARubikSurfaceDescAccessor  DLAInterface::rubikSurfaceDescAccessor(NvU8* b)  const { return DLARubikSurfaceDescAccessor(b, rubikSurfaceDesc()); }
DLASurfaceContainerAccessor   DLAInterface::surfaceContainerAccessor(NvU8 *b)   const { return DLASurfaceContainerAccessor(b, surfaceContainer()); }
DLAOperationContainerAccessor DLAInterface::operationContainerAccessor(NvU8 *b) const { return DLAOperationContainerAccessor(b, operationContainer()); }


} // nvdla::priv

} // nvdla
