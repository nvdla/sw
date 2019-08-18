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

#include <cstddef> // for offsetof

#include "priv/DLAInterface.h"

#include "priv/dla/dla1/A/dla_fw_version.h"
#include "priv/dla/dla1/A/dla_interface.h"

namespace nvdla
{

namespace priv
{

//
// struct dla_network_descriptor
//
class DLANetworkDescA : public DLANetworkDesc
{
public:
    virtual ~DLANetworkDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_network_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual size_t op_BDMA() const { return DLA_OP_BDMA; }
    virtual size_t op_CONV() const { return DLA_OP_CONV; }
    virtual size_t op_SDP() const { return DLA_OP_SDP; }
    virtual size_t op_PDP() const { return DLA_OP_PDP; }
    virtual size_t op_CDP() const { return DLA_OP_CDP; }
    virtual size_t op_RUBIK() const { return DLA_OP_RUBIK; }
    virtual size_t numOpHeads() const { return DLA_OP_NUM; }

    virtual int16_t  * operationDescIndex(NvU8 *b)   const { return &ric(b)->operation_desc_index;   }
    virtual int16_t  * surfaceDescIndex(NvU8 *b)     const { return &ric(b)->surface_desc_index;     }
    virtual int16_t  * dependencyGraphIndex(NvU8 *b) const { return &ric(b)->dependency_graph_index; }
    virtual int16_t  * LUTDataIndex(NvU8 *b)         const { return &ric(b)->lut_data_index;         }
    virtual int16_t  * ROIArrayIndex(NvU8 *b)        const { return &ric(b)->roi_array_index;        }
    virtual int16_t  * surfaceIndex(NvU8 *b)         const { return &ric(b)->surface_index;          }
    virtual int16_t  * statListIndex(NvU8 *b)        const { return &ric(b)->stat_list_index;        }
    virtual int16_t  * reserved1(NvU8 *b)            const { return &ric(b)->reserved1;              }
    virtual int16_t  * opHead(NvU8 *b, size_t h)     const { return &ric(b)->op_head[h];             }
    virtual uint16_t * numROIs(NvU8 *b)              const { return &ric(b)->num_rois;               }
    virtual uint16_t * numOperations(NvU8 *b)        const { return &ric(b)->num_operations;         }
    virtual uint16_t * numLUTs(NvU8 *b)              const { return &ric(b)->num_luts;               }
    virtual uint16_t * numAddresses(NvU8 *b)         const { return &ric(b)->num_addresses;          }
    virtual int16_t  * inputLayer(NvU8 *b)           const { return &ric(b)->input_layer;            }
    virtual uint8_t  * dynamicROI(NvU8 *b)           const { return &ric(b)->dynamic_roi;            }
    virtual uint8_t  * reserved0(NvU8 *b)            const { return &ric(b)->reserved0;              }
protected:
    static inline dla_network_desc *ric(NvU8 *base) { return reinterpret_cast<dla_network_desc *>(base); }
};
static DLANetworkDescA g_dla_network_desc;
const DLANetworkDesc & DLAInterfaceA::networkDesc() const { return g_dla_network_desc; }


//
// struct dla_consumer
//
class DLAConsumerA : public DLAConsumer
{
public:
    virtual ~DLAConsumerA() { }

    virtual size_t struct_size()  const { return sizeof(dla_consumer);      }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual int16_t * index(NvU8 *base) const { return & ric(base)->index; }
    virtual uint8_t * event(NvU8 *base) const { return & ric(base)->event; }
    virtual uint8_t   event_OpCompleted()      const { return DLA_EVENT_OP_COMPLETED; }
    virtual uint8_t   event_OpProgrammed()     const { return DLA_EVENT_OP_PROGRAMMED; }
    virtual uint8_t   event_OpEnabled()        const { return DLA_EVENT_OP_ENABLED; }
    virtual uint8_t   event_OpCDMAWeightDone() const { return DLA_EVENT_CDMA_WT_DONE; }
    virtual uint8_t   event_OpCDMADataDone()   const { return DLA_EVENT_CDMA_DT_DONE; }

    virtual uint8_t * res  (NvU8 *base) const { return & ric(base)->res;   }
protected:
    static inline dla_consumer *ric(NvU8 *base) { return reinterpret_cast<dla_consumer *>(base); }
};

static DLAConsumerA g_dla_consumer;
const DLAConsumer & DLAInterfaceA::consumer() const { return g_dla_consumer; }


//
// struct dla_common_op_desc
//
class DLACommonOpDescA : public DLACommonOpDesc
{
public:
    virtual ~DLACommonOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_common_op_desc);   }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */    }

    virtual int16_t * index(NvU8 *b)  const { return &ric(b)->index;            }
    virtual int8_t  * roiIndex(NvU8 *b) const { return &ric(b)->roi_index;      }
    virtual uint8_t * opType(NvU8 *b) const { return &ric(b)->op_type; }
    virtual uint8_t   opType_BDMA()   const {     return DLA_OP_BDMA;  }
    virtual uint8_t   opType_CONV()   const {     return DLA_OP_CONV;  }
    virtual uint8_t   opType_SDP()    const {     return DLA_OP_SDP;   }
    virtual uint8_t   opType_PDP()    const {     return DLA_OP_PDP;   }
    virtual uint8_t   opType_CDP()    const {     return DLA_OP_CDP;   }
    virtual uint8_t   opType_RUBIK()  const {     return DLA_OP_RUBIK; }
    virtual uint8_t * dependencyCount(NvU8 *b) const { return &ric(b)->dependency_count; }
    virtual uint8_t * reserved_xxx(NvU8 *)      const { return 0; }   // XXX not supported for this version
    virtual uint8_t * reserved0(NvU8 *b, size_t i) const { return &ric(b)->reserved0[i]; }

    virtual size_t numConsumers()           const { return DLA_OP_NUM; }

    virtual DLAConsumerAccessor consumerAccessor(NvU8 *b, size_t c) const { return DLAConsumerAccessor(cir(&ric(b)->consumers[c]), g_dla_consumer); }
    virtual DLAConsumerAccessor fusedParentAccessor(NvU8 *b)        const { return DLAConsumerAccessor(cir(&ric(b)->fused_parent), g_dla_consumer); }

protected:

    virtual DLAConsumer &consumer()    const { return g_dla_consumer; }
    virtual DLAConsumer &fusedParent() const { return g_dla_consumer; }

    static inline NvU8               *cir(dla_consumer *c) { return reinterpret_cast<NvU8 *>(c);                  }
    static inline dla_common_op_desc *ric(NvU8 *base)      { return reinterpret_cast<dla_common_op_desc *>(base); }
};

static DLACommonOpDescA g_dla_common_op_desc;
const DLACommonOpDesc & DLAInterfaceA::commonOpDesc() const { return g_dla_common_op_desc; }


//
// dla_bdma_transfer_desc
//
class DLABDMATransferDescA : public DLABDMATransferDesc
{
public:
    virtual ~DLABDMATransferDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_bdma_transfer_desc);       }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual int16_t  * srcAddress(NvU8 *b)    const { return &ric(b)->source_address; }
    virtual int16_t  * dstAddress(NvU8 *b)    const { return &ric(b)->destination_address; }
    virtual uint32_t * lineSize(NvU8 *b)      const { return &ric(b)->line_size; }
    virtual uint32_t * lineRepeat(NvU8 *b)    const { return &ric(b)->line_repeat; }
    virtual uint32_t * srcLine(NvU8 *b)       const { return &ric(b)->source_line; }
    virtual uint32_t * dstLine(NvU8 *b)       const { return &ric(b)->destination_line; }
    virtual uint32_t * surfaceRepeat(NvU8 *b) const { return &ric(b)->surface_repeat; }
    virtual uint32_t * srcSurface(NvU8 *b)    const { return &ric(b)->source_surface; }
    virtual uint32_t * dstSurface(NvU8 *b)    const { return &ric(b)->destination_surface; }

protected:
    static inline dla_bdma_transfer_desc *ric(NvU8 *base) { return reinterpret_cast<dla_bdma_transfer_desc *>(base); }
};
static DLABDMATransferDescA g_dla_bdma_transfer_desc;


//
// dla_bdma_surface_desc
//
class DLABDMASurfaceDescA : public DLABDMASurfaceDesc
{
public:
    virtual ~DLABDMASurfaceDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_bdma_surface_desc);       }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t  * srcType(NvU8 *b)    const { return &ric(b)->source_type; }
    virtual uint8_t  * dstType(NvU8 *b)    const { return &ric(b)->destination_type; }
    virtual uint16_t * numTransfers(NvU8 *b)    const { return &ric(b)->num_transfers; }
    virtual DLABDMATransferDescAccessor transferAccessor(NvU8 *b, size_t i) const { return DLABDMATransferDescAccessor(dir(&ric(b)->transfers[i]), g_dla_bdma_transfer_desc); }

    virtual uint8_t type_MC() const { return DLA_MEM_MC; }
    virtual uint8_t type_CV() const { return DLA_MEM_CV; }
    virtual uint8_t type_HW() const { return DLA_MEM_HW; }
    virtual uint16_t maxNumTransfers() const { return NUM_MAX_BDMA_OPS; }

protected:
    static inline NvU8 *dir(dla_bdma_transfer_desc *d)   { return reinterpret_cast<NvU8 *>(d); }
    static inline dla_bdma_surface_desc *ric(NvU8 *base) { return reinterpret_cast<dla_bdma_surface_desc *>(base); }
};
static DLABDMASurfaceDescA g_dla_bdma_surface_desc;


//
// dla_bdma_op_desc
//
class DLABDMAOpDescA : public DLABDMAOpDesc
{
public:
    virtual ~DLABDMAOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_bdma_op_desc);       }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint16_t * numTransfers(NvU8 *b)    const { return &ric(b)->num_transfers; }
    virtual uint16_t * reserved0(NvU8 *b)       const { return &ric(b)->reserved0; }

protected:
    static inline dla_bdma_op_desc *ric(NvU8 *base) { return reinterpret_cast<dla_bdma_op_desc *>(base); }
};
static DLABDMAOpDescA g_dla_bdma_op_desc;

//
// dla_cvt_param
//
class DLACVTParamA : public DLACVTParam
{
public:
    virtual ~DLACVTParamA() { }

    virtual size_t struct_size()  const { return sizeof(dla_cvt_param);       }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }


    virtual int16_t  * scale(NvU8 *b)    const { return &ric(b)->scale;    }
    virtual uint8_t  * truncate(NvU8 *b) const { return &ric(b)->truncate; }
    virtual int32_t  * offset(NvU8 *b)   const { return &ric(b)->offset;   }
    virtual uint8_t  * enable(NvU8 *b)   const { return &ric(b)->enable;   }
    virtual uint16_t * reserved_xxx(NvU8 *) const { return 0; }    // XXX: Not supported for this version


    //virtual int16_t * index(NvU8 *b)           const { return &ric(b)->index; }
    //    virtual size_t numOpHeads() const { return DLA_OP_NUM; }
    // virtual AnotherAccessor anotherAccessor(NvU8 *b, size_t c) const { return AnotherAccessor(cir(&ric(b)->another[c]), g_another); }

protected:
    //    static inline NvU8          *cir(dla_cvt_param *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_cvt_param *ric(NvU8 *base)       { return reinterpret_cast<dla_cvt_param *>(base); }
};
static DLACVTParamA g_dla_cvt_param;
const DLACVTParam & DLAInterfaceA::cvtParam() const { return g_dla_cvt_param; }

//
// dla_data_cube
//
class DLADataCubeA : public DLADataCube
{
public:
    virtual ~DLADataCubeA() { }

    virtual size_t struct_size()  const { return sizeof(dla_data_cube);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t  * type_xxx(NvU8 *)    const { return 0; }      // XXX: Not supported for this version
    virtual uint8_t    type_MC_xxx()       const { return 0xFFU; }  // XXX: Not supported for this version
    virtual uint8_t    type_CV_xxx()       const { return 0xFFU; }  // XXX: Not supported for this version
    virtual uint8_t    type_HW_xxx()       const { return 0xFFU; }  // XXX: Not supported for this version
    virtual uint16_t * type(NvU8 *b)       const { return &ric(b)->type; }
    virtual uint16_t   type_MC()           const { return     DLA_MEM_MC; }
    virtual uint16_t   type_CV()           const { return     DLA_MEM_CV; }
    virtual uint16_t   type_HW()           const { return     DLA_MEM_HW; }
    virtual int16_t  * address(NvU8 *b)    const { return &ric(b)->address; }
    virtual uint32_t * offset(NvU8 *b)     const { return &ric(b)->offset;    }
    virtual uint32_t * size(NvU8 *b)       const { return &ric(b)->size;    }
    virtual uint16_t * width(NvU8 *b)      const { return &ric(b)->width;   }
    virtual uint16_t * height(NvU8 *b)     const { return &ric(b)->height;  }
    virtual uint16_t * channel(NvU8 *b)    const { return &ric(b)->channel; }
    virtual uint16_t * reserved0(NvU8 *b)  const { return &ric(b)->reserved0; }
    virtual uint32_t * lineStride(NvU8 *b) const { return &ric(b)->line_stride; }
    virtual uint32_t * surfStride(NvU8 *b) const { return &ric(b)->surf_stride; }
    virtual uint32_t * planeStride(NvU8 *b) const { return &ric(b)->plane_stride; }

protected:
    //static inline NvU8          *cir(dla_data_cube *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_data_cube *ric(NvU8 *base)       { return reinterpret_cast<dla_data_cube *>(base); }
};
static DLADataCubeA g_dla_data_cube;
const DLADataCube & DLAInterfaceA::dataCube() const { return g_dla_data_cube; }

//
// struct dla_conv_surface_desc
//

class DLAConvSurfaceDescA : public DLAConvSurfaceDesc
{
public:
    virtual ~DLAConvSurfaceDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_conv_surface_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual DLADataCubeAccessor weightDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->weight_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor meanDataAccessor(NvU8*)     const { return DLADataCubeAccessor(dir(NULL), g_dla_data_cube); } // not supported in this version
    virtual DLADataCubeAccessor wmbDataAccessor(NvU8 *b)    const { return DLADataCubeAccessor(dir(&ric(b)->wmb_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor wgsDataAccessor(NvU8 *b)    const { return DLADataCubeAccessor(dir(&ric(b)->wgs_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor srcDataAccessor(NvU8 *b)    const { return DLADataCubeAccessor(dir(&ric(b)->src_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor dstDataAccessor(NvU8 *b)    const { return DLADataCubeAccessor(dir(&ric(b)->dst_data), g_dla_data_cube); }
    virtual uint64_t * offsetU_xxx(NvU8 *)     const { return 0; } // XXX not supported in this version
    virtual int64_t  * offsetU(NvU8 *b)        const { return &ric(b)->offset_u;          }
    virtual uint32_t * offsetV(NvU8*)          const { return 0; } // XXX not supported in this version
    virtual uint32_t * inLineUVStride(NvU8 *b) const { return &ric(b)->in_line_uv_stride; }

protected:
    static inline NvU8          *dir(dla_data_cube *d)         { return reinterpret_cast<NvU8 *>(d);             }
    static inline NvU8          *cir(dla_conv_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_conv_surface_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_conv_surface_desc *>(base); }
};
static DLAConvSurfaceDescA g_dla_conv_surface_desc;
const DLAConvSurfaceDesc & DLAInterfaceA::convSurfaceDesc() const { return g_dla_conv_surface_desc; }

//
// struct dla_conv_op_desc
//
class DLAConvOpDescA : public DLAConvOpDesc
{
public:
    virtual ~DLAConvOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_conv_op_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t * inPrecision(NvU8 *b)              const  { return &ric(b)->in_precision;  }
    virtual uint8_t   inPrecision_Int8()         const { return PRECISION_INT8;  }
    virtual uint8_t   inPrecision_Int16()        const { return PRECISION_INT16; }
    virtual uint8_t   inPrecision_FP16()         const { return PRECISION_FP16;  }
    virtual uint8_t * outPrecision(NvU8 *b)             const  { return &ric(b)->out_precision; }
    virtual uint8_t   outPrecision_Int8()         const { return PRECISION_INT8;  }
    virtual uint8_t   outPrecision_Int16()        const { return PRECISION_INT16; }
    virtual uint8_t   outPrecision_FP16()         const { return PRECISION_FP16;  }
    virtual DLACVTParamAccessor inCVTAccessor(NvU8 *b)  const  { return DLACVTParamAccessor(cir(&ric(b)->in_cvt), g_dla_cvt_param); }
    virtual DLACVTParamAccessor outCVTAccessor(NvU8 *b) const  { return DLACVTParamAccessor(cir(&ric(b)->out_cvt), g_dla_cvt_param); }
    virtual int16_t  * padVal(NvU8 *b)                  const  { return &ric(b)->pad_val; }
    virtual uint8_t  * convMode(NvU8 *b)    const  { return &ric(b)->conv_mode;     }
    virtual uint8_t    convMode_Direct()    const  {     return CONV_MODE_DIRECT;   }
    virtual uint8_t    convMode_Winograd()  const  {     return CONV_MODE_WINOGRAD; }
    virtual uint8_t  * dataReuse(NvU8 *b)     const  { return &ric(b)->data_reuse;      }
    virtual uint8_t  * weightReuse(NvU8 *b)   const  { return &ric(b)->weight_reuse;    }
    virtual uint8_t  * skipDataRls(NvU8 *b)   const  { return &ric(b)->skip_data_rls;   }
    virtual uint8_t  * skipWeightRls(NvU8 *b) const  { return &ric(b)->skip_weight_rls; }
    virtual uint8_t  * reserved0(NvU8 *b)     const  { return &ric(b)->reserved0; }
    virtual uint16_t * entryPerSlice(NvU8 *b) const  { return &ric(b)->entry_per_slice; }
    virtual uint16_t * fetchGrain(NvU8 *b)    const  { return &ric(b)->fetch_grain;     }
    virtual uint8_t  * dataFormat(NvU8 *b)  const  { return &ric(b)->data_format; }
    virtual uint8_t    dataFormat_T_R8()    const  {     return FORMAT_T_R8; }
    virtual uint8_t    dataFormat_T_R10()   const  {     return FORMAT_T_R10; }
    virtual uint8_t    dataFormat_T_R12()   const  {     return FORMAT_T_R12; }
    virtual uint8_t    dataFormat_T_R16()   const  {     return FORMAT_T_R16; }
    virtual uint8_t    dataFormat_T_R16_I() const  {     return FORMAT_T_R16_I; }
    virtual uint8_t    dataFormat_T_R16_F() const  {     return FORMAT_T_R16_F; }
    virtual uint8_t    dataFormat_T_A16B16G16R16()   const  {     return FORMAT_T_A16B16G16R16; }
    virtual uint8_t    dataFormat_T_X16B16G16R16()   const  {     return FORMAT_T_X16B16G16R16; }
    virtual uint8_t    dataFormat_T_A16B16G16R16_F() const  {     return FORMAT_T_A16B16G16R16_F; }
    virtual uint8_t    dataFormat_T_A16Y16U16V16()   const  {     return FORMAT_T_A16Y16U16V16; }
    virtual uint8_t    dataFormat_T_V16U16Y16A16()   const  {     return FORMAT_T_V16U16Y16A16; }
    virtual uint8_t    dataFormat_T_A16Y16U16V16_F() const  {     return FORMAT_T_A16Y16U16V16_F; }
    virtual uint8_t    dataFormat_T_A8B8G8R8() const  {     return FORMAT_T_A8B8G8R8; }
    virtual uint8_t    dataFormat_T_A8R8G8B8() const  {     return FORMAT_T_A8R8G8B8; }
    virtual uint8_t    dataFormat_T_B8G8R8A8() const  {     return FORMAT_T_B8G8R8A8; }
    virtual uint8_t    dataFormat_T_R8G8B8A8() const  {     return FORMAT_T_R8G8B8A8; }
    virtual uint8_t    dataFormat_T_X8B8G8R8() const  {     return FORMAT_T_X8B8G8R8; }
    virtual uint8_t    dataFormat_T_X8R8G8B8() const  {     return FORMAT_T_X8R8G8B8; }
    virtual uint8_t    dataFormat_T_B8G8R8X8() const  {     return FORMAT_T_B8G8R8X8; }
    virtual uint8_t    dataFormat_T_R8G8B8X8() const  {     return FORMAT_T_R8G8B8X8; }
    virtual uint8_t    dataFormat_T_A2B10G10R10() const  {     return FORMAT_T_A2B10G10R10; }
    virtual uint8_t    dataFormat_T_A2R10G10B10() const  {     return FORMAT_T_A2R10G10B10; }
    virtual uint8_t    dataFormat_T_B10G10R10A2() const  {     return FORMAT_T_B10G10R10A2; }
    virtual uint8_t    dataFormat_T_R10G10B10A2() const  {     return FORMAT_T_R10G10B10A2; }
    virtual uint8_t    dataFormat_T_A2Y10U10V10()          const  {     return FORMAT_T_A2Y10U10V10;    }
    virtual uint8_t    dataFormat_T_V10U10Y10A2()          const  {     return FORMAT_T_V10U10Y10A2;    }
    virtual uint8_t    dataFormat_T_A8Y8U8V8()             const  {     return FORMAT_T_A8Y8U8V8;    }
    virtual uint8_t    dataFormat_T_V8U8Y8A8()             const  {     return FORMAT_T_V8U8Y8A8;    }
    virtual uint8_t    dataFormat_T_Y8___U8V8_N444()       const  {     return FORMAT_T_Y8___U8V8_N444; }
    virtual uint8_t    dataFormat_T_Y8___V8U8_N444()       const  {     return FORMAT_T_Y8___V8U8_N444; }
    virtual uint8_t    dataFormat_T_Y10___U10V10_N444()    const  {     return FORMAT_T_Y10___U10V10_N444; }
    virtual uint8_t    dataFormat_T_Y10___V10U10_N444()    const  {     return FORMAT_T_Y10___V10U10_N444; }
    virtual uint8_t    dataFormat_T_Y12___U12V12_N444()    const  {     return FORMAT_T_Y12___U12V12_N444; }
    virtual uint8_t    dataFormat_T_Y12___V12U12_N444()    const  {     return FORMAT_T_Y12___V12U12_N444; }
    virtual uint8_t    dataFormat_T_Y16___U16V16_N444()    const  {     return FORMAT_T_Y16___U16V16_N444; }
    virtual uint8_t    dataFormat_T_Y16___V16U16_N444()    const  {     return FORMAT_T_Y16___V16U16_N444; }
    virtual uint8_t    dataFormat_T_Y16___U8V8_N444()      const  {     return 0xFF; }
    virtual uint8_t    dataFormat_T_Y16___V8U8_N444()      const  {     return 0xFF; }
    virtual uint8_t    dataFormat_T_Y8___U8___V8_N444()    const  {     return 0xFF; }
    virtual uint8_t    dataFormat_T_Y10___U10___V10_N444() const  {     return 0xFF; }
    virtual uint8_t    dataFormat_T_Y12___U12___V12_N444() const  {     return 0xFF; }
    virtual uint8_t    dataFormat_T_Y16___U16___V16_N444() const  {     return 0xFF; }
    virtual uint8_t    dataFormat_T_Y16___U8___V8_N444()   const  {     return 0xFF; }
    virtual uint8_t    dataFormat_FEATURE() const  {     return FORMAT_FEATURE; }
    virtual uint8_t  * pixelMapping(NvU8 *b)      const { return &ric(b)->pixel_mapping; }
    virtual uint8_t    pixelMapping_PitchLinear() const {     return MAP_PITCH_LINEAR; }
    virtual uint8_t  * batch(NvU8 *b)         const  { return &ric(b)->batch;           }
    virtual uint8_t  * weightFormat(NvU8 *b)       const  { return &ric(b)->weight_format; }
    virtual uint8_t    weightFormat_Uncompressed() const  {     return WEIGHT_FORMAT_UNCOMPRESSED; }
    virtual uint8_t    weightFormat_Compressed()   const  {     return WEIGHT_FORMAT_COMPRESSED;   }
    virtual uint8_t  * dataBank(NvU8 *b)    const  { return &ric(b)->data_bank;    }
    virtual uint8_t  * weightBank(NvU8 *b)  const  { return &ric(b)->weight_bank;  }
    virtual uint32_t * batchStride(NvU8 *b) const  { return &ric(b)->batch_stride; }
    virtual uint16_t * release(NvU8 *b)     const  { return &ric(b)->release;      }
    virtual uint8_t  * postExtension(NvU8 *b) const  { return &ric(b)->post_extension; }
    virtual uint8_t  * reserved1_xxx(NvU8 *)     const  { return 0; } // xxx: not supported for this version
    virtual uint8_t  * pixelOverride(NvU8 *b) const  { return &ric(b)->pixel_override; }
    virtual uint8_t    pixelOverride_UINT()   const  {     return PIXEL_OVERRIDE_UINT; }
    virtual uint8_t    pixelOverride_INT()    const  {     return PIXEL_OVERRIDE_INT; }
    virtual uint8_t  * meanFormat(NvU8 *b)    const  { return &ric(b)->mean_format;      }
    virtual uint8_t    meanFormat_None()      const  {     return 0xFF; }
    virtual uint8_t    meanFormat_Global()    const  {     return 0xFF; }
    virtual uint8_t    meanFormat_PerPixel()  const  {     return 0xFF; }
    virtual uint8_t    meanFormat_Disable()     const  {     return MEAN_FORMAT_DISABLE;     }
    virtual uint8_t    meanFormat_Enable()      const  {     return MEAN_FORMAT_ENABLE;      }
    virtual int16_t  * meanRY(NvU8 *b)  const  { return &ric(b)->mean_ry; }
    virtual int16_t  * meanGU(NvU8 *b)  const  { return &ric(b)->mean_gu; }
    virtual int16_t  * meanBV(NvU8 *b)  const  { return &ric(b)->mean_bv; }
    virtual int16_t  * meanAX(NvU8 *b)  const  { return &ric(b)->mean_ax; }
    virtual uint8_t  * convStrideX(NvU8 *b) const  { return &ric(b)->conv_stride_x; }
    virtual uint8_t  * convStrideY(NvU8 *b) const  { return &ric(b)->conv_stride_y; }
    virtual uint8_t  * padXLeft(NvU8 *b)   const  { return &ric(b)->pad_x_left;   }
    virtual uint8_t  * padXRight(NvU8 *b)  const  { return &ric(b)->pad_x_right;  }
    virtual uint8_t  * padYTop(NvU8 *b)    const  { return &ric(b)->pad_y_top;    }
    virtual uint8_t  * padYBottom(NvU8 *b) const  { return &ric(b)->pad_y_bottom; }
    virtual uint8_t  * dilationX(NvU8 *b) const  { return &ric(b)->dilation_x; }
    virtual uint8_t  * dilationY(NvU8 *b) const  { return &ric(b)->dilation_y; }
    virtual uint8_t  * reserved2(NvU8 *b, size_t i)     const  { return &ric(b)->reserved2[i]; }
    virtual uint8_t  * praTruncate(NvU8 *b)      const  { return &ric(b)->pra_truncate; }
    virtual uint16_t * inputWidthCSC(NvU8 *b)    const  { return &ric(b)->input_width_csc;    }
    virtual uint16_t * inputHeightCSC(NvU8 *b)   const  { return &ric(b)->input_height_csc;   }
    virtual uint16_t * inputChannelCSC(NvU8 *b)  const  { return &ric(b)->input_channel_csc;  }
    virtual uint16_t * kernelWidthCSC(NvU8 *b)   const  { return &ric(b)->kernel_width_csc;   }
    virtual uint16_t * kernelHeightCSC(NvU8 *b)  const  { return &ric(b)->kernel_height_csc;  }
    virtual uint16_t * kernelChannelCSC(NvU8 *b) const  { return &ric(b)->kernel_channel_csc; }
    virtual uint16_t * inputWidthCMAC(NvU8 *b)   const  { return &ric(b)->input_width_cmac;   }
    virtual uint16_t * inputHeightCMAC(NvU8 *b)  const  { return &ric(b)->input_height_cmac;  }
    virtual uint32_t * bytesPerKernel(NvU8 *b)   const  { return &ric(b)->bytes_per_kernel;   }


protected:
    static inline NvU8          *cir(dla_cvt_param *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_conv_op_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_conv_op_desc *>(base); }
};
static DLAConvOpDescA g_dla_conv_op_desc;
const DLAConvOpDesc & DLAInterfaceA::convOpDesc() const { return g_dla_conv_op_desc; }

//
//
//
class DLALUTOffsetA : public DLALUTOffset
{
public:
    virtual ~DLALUTOffsetA() { }

    virtual size_t struct_size()  const { return sizeof(dla_lut_offset);    }
    virtual size_t struct_align() const { return 0; /* unspecified */ }

    virtual uint8_t * expOffset_xxx(NvU8 *) const { return 0; }  // XXX: Not supported for this version
    virtual int8_t * expOffset(NvU8 *b) const { return &ric(b)->exp_offset; }
    virtual uint8_t * fracBits_xxx(NvU8 *)  const { return 0; }  // XXX: Not supported for this version
    virtual int8_t * fracBits(NvU8 *b)  const { return &ric(b)->frac_bits; }
    virtual uint16_t * reserved0(NvU8 *b)  const { return &ric(b)->reserved0; }

protected:
    // static inline NvU8          *cir(dla_lut_offset *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_lut_offset *ric(NvU8 *base)       { return reinterpret_cast<dla_lut_offset *>(base); }
};
static DLALUTOffsetA g_dla_lut_offset;
const DLALUTOffset & DLAInterfaceA::lutOffset() const { return g_dla_lut_offset; }

//
// dla_float_data
//
class DLAFloatDataA : public DLAFloatData
{
public:
    virtual ~DLAFloatDataA() { }

    virtual size_t struct_size()  const { return sizeof(dla_float_data);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual int16_t * scale(NvU8 *b)   const { return &ric(b)->scale; }
    /*
     * below, the compiler caught that shifter changes from uint to int in A->B.
     * because there were no actual usages the original could be moved/renamed _xxx.
     */
    virtual uint8_t * shifter_xxx(NvU8*) const { return 0; }
    virtual int8_t * shifter(NvU8 *b) const { return &ric(b)->shifter; }
    virtual uint8_t * reserved0(NvU8 *b)  const { return &ric(b)->reserved0; }

protected:
    //static inline NvU8          *cir(dla_float_data *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_float_data *ric(NvU8 *base)       { return reinterpret_cast<dla_float_data *>(base); }
};
static DLAFloatDataA g_dla_float_data;
const DLAFloatData & DLAInterfaceA::floatData() const { return g_dla_float_data; }


//
// dla_slope
//
class DLASlopeA : public DLASlope
{
public:
    virtual ~DLASlopeA() { }

    virtual size_t struct_size()  const { return sizeof(dla_slope);    }
    virtual size_t struct_align() const { return 0; /* unspecified */ }

    virtual DLAFloatDataAccessor dataIAccessor(NvU8 *b) const { return DLAFloatDataAccessor(fir(&ric(b)->data_i), g_dla_float_data); }
    virtual uint16_t * dataF(NvU8 *b) const { return &ric(b)->data_f; }

protected:
    static inline NvU8      *fir(dla_float_data *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_slope *ric(NvU8 *base)       { return reinterpret_cast<dla_slope *>(base); }
};
static DLASlopeA g_dla_slope;
const DLASlope & DLAInterfaceA::slope() const { return g_dla_slope; }

//
// dla_lut_param
//
class DLALUTParamA : public DLALUTParam
{
public:
    virtual ~DLALUTParamA() { }

    virtual size_t struct_size()  const { return sizeof(dla_lut_param);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual int16_t  * linearExpTable(NvU8 *b, size_t i)  const { return &ric(b)->linear_exp_table[i]; }
    virtual size_t numLinearExpTable() const { return (1<<LUT_LINEAR_EXP_TABLE_ENTRY_LOG2)+1; }
    virtual int16_t  * linearOnlyTable(NvU8 *b, size_t i) const { return &ric(b)->linear_only_table[i]; }
    virtual size_t numLinearOnlyTable() const { return (1<<LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2)+1; }


    virtual uint8_t  * method(NvU8 *b) const { return &ric(b)->method; }
    virtual uint8_t    method_Exponential() const { return LUT_METHOD_EXPONENTIAL; }
    virtual uint8_t    method_Linear() const { return LUT_METHOD_LINEAR; }
    virtual DLALUTOffsetAccessor linearExpOffsetAccessor(NvU8 *b)  const { return DLALUTOffsetAccessor(lir(&ric(b)->linear_exp_offset), g_dla_lut_offset); }
    virtual DLALUTOffsetAccessor linearOnlyOffsetAccessor(NvU8 *b) const { return DLALUTOffsetAccessor(lir(&ric(b)->linear_only_offset), g_dla_lut_offset); }


    virtual uint64_t * linearExpStart(NvU8 *b)  const { return &ric(b)->linear_exp_start; }
    virtual uint64_t * linearExpEnd(NvU8 *b)  const { return &ric(b)->linear_exp_end; }
    virtual uint64_t * linearOnlyStart(NvU8 *b) const { return &ric(b)->linear_only_start; }
    virtual uint64_t * linearOnlyEnd(NvU8 *b)  const { return &ric(b)->linear_only_end; }

    virtual DLASlopeAccessor linearExpUnderflowSlopeAccessor(NvU8 *b)  const { return DLASlopeAccessor(sir(&ric(b)->linear_exp_underflow_slope), g_dla_slope); }
    virtual DLASlopeAccessor linearExpOverflowSlopeAccessor(NvU8 *b)   const { return DLASlopeAccessor(sir(&ric(b)->linear_exp_overflow_slope), g_dla_slope); }
    virtual DLASlopeAccessor linearOnlyUnderflowSlopeAccessor(NvU8 *b) const { return DLASlopeAccessor(sir(&ric(b)->linear_only_underflow_slope), g_dla_slope); }
    virtual DLASlopeAccessor linearOnlyOverflowSlopeAccessor(NvU8 *b)  const { return DLASlopeAccessor(sir(&ric(b)->linear_only_overflow_slope), g_dla_slope); }
    virtual uint8_t  * hybridPriority(NvU8 *b)    const { return &ric(b)->hybrid_priority;    }
    virtual uint8_t  * underflowPriority(NvU8 *b) const { return &ric(b)->underflow_priority; }
    virtual uint8_t  * overflowPriority(NvU8 *b)  const { return &ric(b)->overflow_priority;  }
    virtual uint8_t    priority_LinearExp() const { return LUT_PRI_LINEAR_EXP; }
    virtual uint8_t    priority_LinearOnly() const { return LUT_PRI_LINEAR_ONLY; }
    virtual int8_t   * inputScaleLog2(NvU8*)      const { return 0;   } // XXX not supported for this version


protected:
    static inline NvU8          *sir(dla_slope *c)     { return reinterpret_cast<NvU8 *>(c);             }
    //    static inline NvU8          *cir(dla_lut_param *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *lir(dla_lut_offset *c) { return reinterpret_cast<NvU8 *>(c);            }
    static inline dla_lut_param *ric(NvU8 *base)       { return reinterpret_cast<dla_lut_param *>(base); }
};
static DLALUTParamA g_dla_lut_param;
const DLALUTParam & DLAInterfaceA::lutParam() const { return g_dla_lut_param; }

//
// dla_sdp_surface_desc
//
class DLASDPSurfaceDescA : public DLASDPSurfaceDesc
{
public:
    virtual ~DLASDPSurfaceDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_sdp_surface_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual DLADataCubeAccessor srcDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->src_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor x1DataAccessor(NvU8 *b)  const { return DLADataCubeAccessor(dir(&ric(b)->x1_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor x2DataAccessor(NvU8 *b)  const { return DLADataCubeAccessor(dir(&ric(b)->x2_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor yDataAccessor(NvU8 *b)   const { return DLADataCubeAccessor(dir(&ric(b)->y_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor dstDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->dst_data), g_dla_data_cube); }


protected:
    static inline NvU8                 *dir(dla_data_cube *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_sdp_surface_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_sdp_surface_desc *>(base); }
};
static DLASDPSurfaceDescA g_dla_sdp_surface_desc;
const DLASDPSurfaceDesc & DLAInterfaceA::sdpSurfaceDesc() const { return g_dla_sdp_surface_desc; }

//
// dla_sdp_cvt
//
class DLASDPCVTA : public DLASDPCVT
{
public:
    virtual ~DLASDPCVTA() { }

    virtual size_t struct_size()  const { return sizeof(dla_sdp_cvt);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual DLACVTParamAccessor aluCVTAccessor(NvU8 *b) const { return DLACVTParamAccessor(cir(&ric(b)->alu_cvt), g_dla_cvt_param); }
    virtual DLACVTParamAccessor mulCVTAccessor(NvU8 *b) const { return DLACVTParamAccessor(cir(&ric(b)->mul_cvt), g_dla_cvt_param); }

protected:
    static inline NvU8        *cir(dla_cvt_param *c) {return reinterpret_cast<NvU8 *>(c); }
    static inline dla_sdp_cvt *ric(NvU8 *base)  { return reinterpret_cast<dla_sdp_cvt *>(base); }
};
static DLASDPCVTA g_dla_sdp_cvt;
const DLASDPCVT & DLAInterfaceA::sdpCVT() const { return g_dla_sdp_cvt; }

//
// dla_sdp_op
//
class DLASDPOpA : public DLASDPOp
{
public:
    virtual ~DLASDPOpA() { }

    virtual size_t struct_size()  const { return sizeof(dla_sdp_op);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }


    //virtual int16_t * index(NvU8 *b)           const { return &ric(b)->index; }
    //    virtual size_t numOpHeads() const { return DLA_OP_NUM; }
    // virtual AnotherAccessor anotherAccessor(NvU8 *b, size_t c) const { return AnotherAccessor(cir(&ric(b)->another[c]), g_another); }

    virtual uint8_t * enable(NvU8 *b)     const { return &ric(b)->enable; }
    virtual uint8_t * ALUType(NvU8 *b)    const { return &ric(b)->alu_type; }
    virtual uint8_t   ALUType_Max() const {     return SDP_ALU_OP_MAX; }
    virtual uint8_t   ALUType_Min() const {     return SDP_ALU_OP_MIN; }
    virtual uint8_t   ALUType_Sum() const {     return SDP_ALU_OP_SUM; }
    virtual uint8_t   ALUType_Eql() const {     return SDP_ALU_OP_EQL; }
    virtual uint8_t * type(NvU8 *b)    const { return &ric(b)->type; }
    virtual uint8_t   type_None() const {     return SDP_OP_NONE; }
    virtual uint8_t   type_Mul() const {     return SDP_OP_MUL; }
    virtual uint8_t   type_Add() const {     return SDP_OP_ADD; }
    virtual uint8_t   type_Both() const {     return SDP_OP_BOTH; }
    virtual uint8_t * mode(NvU8 *b)       const { return &ric(b)->mode; }
    virtual uint8_t   mode_PerLayer()  const {     return SDP_OP_PER_LAYER; }
    virtual uint8_t   mode_PerKernel() const {     return SDP_OP_PER_KERNEL; }
    virtual uint8_t   mode_PerPoint()  const {     return SDP_OP_PER_POINT; }
    virtual uint8_t * act(NvU8 *b)        const { return &ric(b)->act; }
    virtual uint8_t   act_None() const {     return ACTIVATION_NONE; }
    virtual uint8_t   act_RelU() const {     return ACTIVATION_RELU; }
    virtual uint8_t   act_LUT()  const {     return ACTIVATION_LUT; }
    virtual uint8_t * shiftValue(NvU8 *b) const { return &ric(b)->shift_value; }
    virtual int16_t * ALUOperand_xxx(NvU8 *) const { return 0; }   // xxx: not supported for this version
    virtual int16_t * MulOperand_xxx(NvU8 *) const { return 0; }   // xxx: not supported for this version
    virtual int32_t * ALUOperand(NvU8 *b) const { return &ric(b)->alu_operand; }
    virtual int32_t * MulOperand(NvU8 *b) const { return &ric(b)->mul_operand; }
    virtual uint8_t * truncate(NvU8 *b)   const { return &ric(b)->truncate; }
    virtual uint8_t * precision(NvU8 *b)  const { return &ric(b)->precision; }
    virtual DLASDPCVTAccessor cvt(NvU8 *b) const { return DLASDPCVTAccessor(vir(&ric(b)->cvt), g_dla_sdp_cvt); }


protected:
    static inline NvU8          *vir(dla_sdp_cvt *v) { return reinterpret_cast<NvU8 *>(v); }
    static inline NvU8          *cir(dla_sdp_op *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_sdp_op *ric(NvU8 *base)       { return reinterpret_cast<dla_sdp_op *>(base); }
};
static DLASDPOpA g_dla_sdp_op;
const DLASDPOp & DLAInterfaceA::sdpOp() const { return g_dla_sdp_op; }


//
// dla_sdp_op_desc
//
class DLASDPOpDescA : public DLASDPOpDesc
{
public:
    virtual ~DLASDPOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_sdp_op_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t * srcPrecision(NvU8 *b)       const  { return &ric(b)->src_precision;  }
    virtual uint8_t   srcPrecision_Int8()         const { return PRECISION_INT8;  }
    virtual uint8_t   srcPrecision_Int16()        const { return PRECISION_INT16; }
    virtual uint8_t   srcPrecision_FP16()         const { return PRECISION_FP16;  }
    virtual uint8_t * dstPrecision(NvU8 *b)       const  { return &ric(b)->dst_precision; }
    virtual uint8_t   dstPrecision_Int8()         const { return PRECISION_INT8;  }
    virtual uint8_t   dstPrecision_Int16()        const { return PRECISION_INT16; }
    virtual uint8_t   dstPrecision_FP16()         const { return PRECISION_FP16;  }
    virtual int16_t * LUTIndex(NvU8 *b)     const { return &ric(b)->lut_index; }
    virtual DLACVTParamAccessor outCVTAccessor(NvU8 *b) const { return DLACVTParamAccessor(cir(&ric(b)->out_cvt), g_dla_cvt_param); }
    virtual uint8_t  * convMode(NvU8 *b)    const { return &ric(b)->conv_mode; }
    virtual uint8_t    convMode_Direct()    const { return CONV_MODE_DIRECT; }
    virtual uint8_t    convMode_Winograd()  const { return CONV_MODE_WINOGRAD; }
    virtual uint8_t  * batchNum(NvU8 *b)    const { return &ric(b)->batch_num; }
    virtual uint16_t * reserved0(NvU8 *b)    const { return &ric(b)->reserved0; }
    virtual uint32_t * batchStride(NvU8 *b) const { return &ric(b)->batch_stride; }
    virtual DLASDPOpAccessor x1OpAccessor(NvU8 *b) const { return DLASDPOpAccessor(sir(&ric(b)->x1_op), g_dla_sdp_op); }
    virtual DLASDPOpAccessor x2OpAccessor(NvU8 *b) const { return DLASDPOpAccessor(sir(&ric(b)->x2_op), g_dla_sdp_op); }
    virtual DLASDPOpAccessor yOpAccessor(NvU8 *b)  const { return DLASDPOpAccessor(sir(&ric(b)->y_op), g_dla_sdp_op); }

protected:
    static inline NvU8          *cir(dla_cvt_param *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *sir(dla_sdp_op *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_sdp_op_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_sdp_op_desc *>(base); }
};
static DLASDPOpDescA g_dla_sdp_op_desc;
const DLASDPOpDesc & DLAInterfaceA::sdpOpDesc() const { return g_dla_sdp_op_desc; }

//
// dla_pdp_surface_desc
//
class DLAPDPSurfaceDescA : public DLAPDPSurfaceDesc
{
public:
    virtual ~DLAPDPSurfaceDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_pdp_surface_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual DLADataCubeAccessor srcDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->src_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor dstDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->dst_data), g_dla_data_cube); }

protected:
    static inline NvU8                 *dir(dla_data_cube *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_pdp_surface_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_pdp_surface_desc *>(base); }
};
static DLAPDPSurfaceDescA g_dla_pdp_surface_desc;
const DLAPDPSurfaceDesc & DLAInterfaceA::pdpSurfaceDesc() const { return g_dla_pdp_surface_desc; }

//
// dla_pdp_op_desc
//
class DLAPDPOpDescA : public DLAPDPOpDesc
{
public:
    virtual ~DLAPDPOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_pdp_op_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t *precision(NvU8 *b) const { return &ric(b)->precision; }
    virtual uint8_t  precision_Int8()  const  {     return PRECISION_INT8;  }
    virtual uint8_t  precision_Int16() const  {     return PRECISION_INT16; }
    virtual uint8_t  precision_FP16()  const  {     return PRECISION_FP16;  }
    virtual uint8_t *reserved0_xxx(NvU8 *, size_t)     const { return 0; }  // XXX: not supported for this version
    virtual uint8_t *reserved0(NvU8 *b)     const { return &ric(b)->reserved0; }
    virtual int16_t *paddingValue_xxx(NvU8*, size_t)     const { return 0; }  // XXX: not supported for this version
    virtual int32_t *paddingValue(NvU8 *b, size_t i)     const { return &ric(b)->padding_value[i]; }
    virtual uint8_t *splitNum(NvU8 *b) const { return &ric(b)->split_num; }
    virtual uint8_t *reserved1_xxx(NvU8 *, size_t)     const { return 0; }  // XXX: not supported for this version
    virtual uint16_t *partialInWidthFirst(NvU8 *b) const { return &ric(b)->partial_in_width_first; }
    virtual uint16_t *partialInWidthMid(NvU8 *b) const { return &ric(b)->partial_in_width_mid; }
    virtual uint16_t *partialInWidthLast(NvU8 *b) const { return &ric(b)->partial_in_width_last; }

    virtual uint16_t *partialWidthFirst(NvU8 *b) const { return &ric(b)->partial_width_first; }
    virtual uint16_t *partialWidthMid(NvU8 *b) const { return &ric(b)->partial_width_mid; }
    virtual uint16_t *partialWidthLast(NvU8 *b) const { return &ric(b)->partial_width_last; }

    virtual uint8_t *poolMode(NvU8 *b) const { return &ric(b)->pool_mode; }
    virtual uint8_t  poolMode_AVG() const { return POOL_MODE_AVG; }
    virtual uint8_t  poolMode_MAX() const { return POOL_MODE_MAX; }
    virtual uint8_t  poolMode_MIN() const { return POOL_MODE_MIN; }
    virtual uint8_t *poolWidth(NvU8 *b)  const { return &ric(b)->pool_width; }
    virtual uint8_t *poolHeight(NvU8 *b) const { return &ric(b)->pool_height; }
    virtual uint8_t *reserved2_xxx(NvU8 *) const { return 0; }  // XXX: not supported for this version

    virtual uint8_t *strideX(NvU8 *b) const { return &ric(b)->stride_x; }
    virtual uint8_t *strideY(NvU8 *b) const { return &ric(b)->stride_y; }
    virtual uint16_t *strideX_xxx(NvU8*) const { return 0; }    // XXX not supported for this version
    virtual uint16_t *strideY_xxx(NvU8*) const { return 0; }    // XXX not supported for this version
    virtual uint16_t *reserved3_xxx(NvU8 *) const { return 0; } // XXX: not supported for this version

    virtual uint8_t *padLeft(NvU8 *b)   const { return &ric(b)->pad_left; }
    virtual uint8_t *padRight(NvU8 *b)  const { return &ric(b)->pad_right; }
    virtual uint8_t *padTop(NvU8 *b)    const { return &ric(b)->pad_top; }
    virtual uint8_t *padBottom(NvU8 *b) const { return &ric(b)->pad_bottom; }

protected:
    static inline NvU8          *cir(dla_pdp_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_pdp_op_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_pdp_op_desc *>(base); }
};
static DLAPDPOpDescA g_dla_pdp_op_desc;
const DLAPDPOpDesc & DLAInterfaceA::pdpOpDesc() const { return g_dla_pdp_op_desc; }


//
// dla_cdp_surface_desc
//
class DLACDPSurfaceDescA : public DLACDPSurfaceDesc
{
public:
    virtual ~DLACDPSurfaceDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_cdp_surface_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual DLADataCubeAccessor srcDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->src_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor dstDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->dst_data), g_dla_data_cube); }

protected:
    static inline NvU8                 *dir(dla_data_cube *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_cdp_surface_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_cdp_surface_desc *>(base); }
};
static DLACDPSurfaceDescA g_dla_cdp_surface_desc;
const DLACDPSurfaceDesc & DLAInterfaceA::cdpSurfaceDesc() const { return g_dla_cdp_surface_desc; }

//
// dla_cdp_op_desc
//
class DLACDPOpDescA : public DLACDPOpDesc
{
public:
    virtual ~DLACDPOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_cdp_op_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t *inPrecision(NvU8 *b) const { return &ric(b)->in_precision; }
    virtual uint8_t  inPrecision_Int8()  const  {     return PRECISION_INT8;  }
    virtual uint8_t  inPrecision_Int16() const  {     return PRECISION_INT16; }
    virtual uint8_t  inPrecision_FP16()  const  {     return PRECISION_FP16;  }
    virtual uint8_t *outPrecision(NvU8 *b) const { return &ric(b)->out_precision; }
    virtual uint8_t  outPrecision_Int8()  const  {     return PRECISION_INT8;  }
    virtual uint8_t  outPrecision_Int16() const  {     return PRECISION_INT16; }
    virtual uint8_t  outPrecision_FP16()  const  {     return PRECISION_FP16;  }
    virtual DLACVTParamAccessor inCVTAccessor(NvU8 *b)  const  { return DLACVTParamAccessor(cir(&ric(b)->in_cvt), g_dla_cvt_param); }
    virtual DLACVTParamAccessor outCVTAccessor(NvU8 *b) const  { return DLACVTParamAccessor(cir(&ric(b)->out_cvt), g_dla_cvt_param); }
    virtual int16_t *LUTIndex(NvU8 *b) const { return &ric(b)->lut_index; }
    virtual uint8_t *localSize(NvU8 *b) const { return &ric(b)->local_size; }
    virtual uint8_t *bypassSquareSum(NvU8 *b) const { return &ric(b)->bypass_sqsum; }
    virtual uint8_t *bypassOutMul(NvU8 *b) const { return &ric(b)->bypass_out_mul; }
    virtual uint8_t *reserved0(NvU8 *b) const { return &ric(b)->reserved0; }


protected:
    static inline NvU8          *cir(dla_cvt_param *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_cdp_op_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_cdp_op_desc *>(base); }
};
static DLACDPOpDescA g_dla_cdp_op_desc;
const DLACDPOpDesc & DLAInterfaceA::cdpOpDesc() const { return g_dla_cdp_op_desc; }

// --- CDP ---


// Rubik
//
// dla_rubik_surface_desc
//
class DLARubikSurfaceDescA : public DLARubikSurfaceDesc
{
public:
    virtual ~DLARubikSurfaceDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_rubik_surface_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual DLADataCubeAccessor srcDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->src_data), g_dla_data_cube); }
    virtual DLADataCubeAccessor dstDataAccessor(NvU8 *b) const { return DLADataCubeAccessor(dir(&ric(b)->dst_data), g_dla_data_cube); }

protected:
    static inline NvU8                 *dir(dla_data_cube *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_rubik_surface_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_rubik_surface_desc *>(base); }
};
static DLARubikSurfaceDescA g_dla_rubik_surface_desc;
const DLARubikSurfaceDesc & DLAInterfaceA::rubikSurfaceDesc() const { return g_dla_rubik_surface_desc; }

//
// dla_rubik_op_desc
//
class DLARubikOpDescA : public DLARubikOpDesc
{
public:
    virtual ~DLARubikOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(dla_rubik_op_desc);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual uint8_t *mode(NvU8 *b) const { return &ric(b)->mode; }
    virtual uint8_t  mode_Contract() const { return RUBIK_MODE_CONTRACT; }
    virtual uint8_t  mode_Split()    const { return RUBIK_MODE_SPLIT;    }
    virtual uint8_t  mode_Merge()    const { return RUBIK_MODE_MERGE;    }


    virtual uint8_t *precision(NvU8 *b) const { return &ric(b)->precision; }
    virtual uint8_t  precision_Int8()  const  {     return PRECISION_INT8;  }
    virtual uint8_t  precision_Int16() const  {     return PRECISION_INT16; }
    virtual uint8_t  precision_FP16()  const  {     return PRECISION_FP16;  }

    virtual uint8_t *strideX(NvU8 *b) const { return &ric(b)->stride_x; }
    virtual uint8_t *strideY(NvU8 *b) const { return &ric(b)->stride_y; }

protected:
    //    static inline NvU8          *cir(dla_rubik_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_rubik_op_desc *ric(NvU8 *base)       { return reinterpret_cast<dla_rubik_op_desc *>(base); }
};
static DLARubikOpDescA g_dla_rubik_op_desc;
const DLARubikOpDesc & DLAInterfaceA::rubikOpDesc() const { return g_dla_rubik_op_desc; }

// ---RUBIK---

//
// dla_surface_container
//
class DLASurfaceContainerA : public DLASurfaceContainer
{
public:
    virtual ~DLASurfaceContainerA() { }

    virtual size_t struct_size()  const { return sizeof(dla_surface_container);    }
    virtual size_t struct_align() const { return 0; /* unspecified */ }

    virtual DLABDMASurfaceDescAccessor bdmaSurfaceDescAccessor(NvU8 *b, size_t c) const
    {
        return DLABDMASurfaceDescAccessor(bir( &(ric(b)[c]).bdma_surface), g_dla_bdma_surface_desc);
    }
    virtual DLAConvSurfaceDescAccessor convSurfaceDescAccessor(NvU8 *b, size_t c) const
    {
        return DLAConvSurfaceDescAccessor(cir( &(ric(b)[c]).conv_surface), g_dla_conv_surface_desc);
    }
    virtual DLASDPSurfaceDescAccessor  sdpSurfaceDescAccessor (NvU8 *b, size_t c) const
    {
        return DLASDPSurfaceDescAccessor(sir(  &(ric(b)[c]).sdp_surface), g_dla_sdp_surface_desc);
    }
    virtual DLAPDPSurfaceDescAccessor  pdpSurfaceDescAccessor (NvU8 *b, size_t c) const
    {
        return DLAPDPSurfaceDescAccessor(pir(  &(ric(b)[c]).pdp_surface), g_dla_pdp_surface_desc);
    }
    virtual DLACDPSurfaceDescAccessor  cdpSurfaceDescAccessor (NvU8 *b, size_t c) const
    {
        return DLACDPSurfaceDescAccessor(cdir(  &(ric(b)[c]).cdp_surface), g_dla_cdp_surface_desc);
    }
    virtual DLARubikSurfaceDescAccessor  rubikSurfaceDescAccessor (NvU8 *b, size_t c) const
    {
        return DLARubikSurfaceDescAccessor(rir(  &(ric(b)[c]).rubik_surface), g_dla_rubik_surface_desc);
    }


protected:
    static inline NvU8          *bir(dla_bdma_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *cir(dla_conv_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *sir(dla_sdp_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *pir(dla_pdp_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *cdir(dla_cdp_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *rir(dla_rubik_surface_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_surface_container *ric(NvU8 *base)       { return reinterpret_cast<dla_surface_container *>(base); }
};
static DLASurfaceContainerA g_dla_surface_container;
const DLASurfaceContainer & DLAInterfaceA::surfaceContainer() const { return g_dla_surface_container; }


//
// dla_operation_container
//
class DLAOperationContainerA : public DLAOperationContainer
{
public:
    virtual ~DLAOperationContainerA() { }

    virtual size_t struct_size()  const { return sizeof(dla_operation_container);    }
    virtual size_t struct_align() const { return 0; /* unspecified */ }

    virtual DLABDMAOpDescAccessor bdmaOpDescAccessor(NvU8 *b, size_t c) const { return DLABDMAOpDescAccessor(bir(&(ric(b)[c].bdma_op)), g_dla_bdma_op_desc); }
    virtual DLAConvOpDescAccessor convOpDescAccessor(NvU8 *b, size_t c) const { return DLAConvOpDescAccessor(cir(&(ric(b)[c].conv_op)), g_dla_conv_op_desc); }
    virtual DLASDPOpDescAccessor  sdpOpDescAccessor(NvU8  *b, size_t c)  const { return DLASDPOpDescAccessor(sir(&(ric(b)[c].sdp_op)),  g_dla_sdp_op_desc);  }
    virtual DLAPDPOpDescAccessor  pdpOpDescAccessor(NvU8  *b, size_t c)  const { return DLAPDPOpDescAccessor(pir(&(ric(b)[c].pdp_op)),  g_dla_pdp_op_desc);  }
    virtual DLACDPOpDescAccessor  cdpOpDescAccessor(NvU8  *b, size_t c)  const { return DLACDPOpDescAccessor(cdir(&(ric(b)[c].cdp_op)),  g_dla_cdp_op_desc);  }
    virtual DLARubikOpDescAccessor  rubikOpDescAccessor(NvU8  *b, size_t c)  const { return DLARubikOpDescAccessor(rir(&(ric(b)[c].rubik_op)),  g_dla_rubik_op_desc);  }

protected:
    static inline NvU8          *bir(dla_bdma_op_desc *c) { return reinterpret_cast<NvU8 *>(c); }
    static inline NvU8          *cir(dla_conv_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *sir(dla_sdp_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *pir(dla_pdp_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *cdir(dla_cdp_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *rir(dla_rubik_op_desc *c) { return reinterpret_cast<NvU8 *>(c);             }
    static inline dla_operation_container *ric(NvU8 *base)       { return reinterpret_cast<dla_operation_container *>(base); }
};
static DLAOperationContainerA g_dla_operation_container;
const DLAOperationContainer & DLAInterfaceA::operationContainer() const { return g_dla_operation_container; }


//
// interface
//

NvU8 DLAInterfaceA::firmwareTargetVersionMajor()    const { return FIRMWARE_VERSION_MAJOR;    }
NvU8  DLAInterfaceA::firmwareTargetVersionMinor()    const { return FIRMWARE_VERSION_MINOR;    }
NvU8 DLAInterfaceA::firmwareTargetVersionSubminor() const { return FIRMWARE_VERSION_SUBMINOR; }

NvU32 DLAInterfaceA::firmwareTargetVersion() const { return dla_version(); }

NvU8 DLAInterfaceA::firmwareVersionMajor() const
{
    return FIRMWARE_VERSION_MAJOR; // TBD find kmd query?
}

NvU8 DLAInterfaceA::firmwareVersionMinor() const
{
    return FIRMWARE_VERSION_MINOR; // TBD find kmd query?
}

NvU8 DLAInterfaceA::firmwareVersionSubminor() const
{
    return FIRMWARE_VERSION_SUBMINOR; // TBD find kmd query?
}

NvU32 DLAInterfaceA::firmwareVersion() const
{
    return dla_version(); // TBD find kmd query?
}






} // nvdla::priv
} // nvdla
