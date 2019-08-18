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

#include "priv/EMUInterface.h"

#include "priv/emu/emu1/A/emu_version.h"
#include "priv/emu/emu1/A/emu_interface.h"

namespace nvdla
{
namespace priv
{

//
// struct emu_address
//
class EMUAddressA : public EMUAddress
{
public:
    virtual ~EMUAddressA() { }

    virtual size_t struct_size()  const { return sizeof(emu_address);    }
    virtual size_t struct_align() const { return 0; }

    virtual void * hMem(NvU8 *base)   const { return &ric(base)->hMem; }
    virtual NvU32 * offset(NvU8 *base)   const { return &ric(base)->offset; }

protected:
    static inline emu_address *ric(NvU8 *base) { return reinterpret_cast<emu_address *>(base); }
};
static EMUAddressA g_emu_address;
const EMUAddress & EMUInterfaceA::address() const { return g_emu_address; }


//
// struct emu_task_desc
//
class EMUTaskDescA : public EMUTaskDesc
{
public:
    virtual ~EMUTaskDescA() { }

    virtual size_t struct_size()  const { return sizeof(emu_task_desc);    }
    virtual size_t struct_align() const { return 256; }

    virtual NvU32 * numAddresses(NvU8 *base)   const { return &ric(base)->num_addresses; }
    virtual size_t maxBuffersPerTask() const { return NVDLA_EMU_MAX_BUFFERS_PER_TASK; }
    virtual EMUAddressAccessor addressList(NvU8 *base, size_t c)   const { return EMUAddressAccessor(cir(&ric(base)->address_list[c]), g_emu_address); }

protected:
    static inline NvU8          *cir(emu_address *c) { return reinterpret_cast<NvU8 *>(c);           }
    static inline emu_task_desc *ric(NvU8 *base) { return reinterpret_cast<emu_task_desc *>(base); }
};
static EMUTaskDescA g_emu_task_desc;
const EMUTaskDesc & EMUInterfaceA::taskDesc() const { return g_emu_task_desc; }


//
// struct emu_network_desc
//
class EMUNetworkDescA : public EMUNetworkDesc
{
public:
    virtual ~EMUNetworkDescA() { }

    virtual size_t struct_size()  const { return sizeof(emu_network_desc);    }
    virtual size_t struct_align() const { return 256; }

    virtual int16_t  * operationDescIndex(NvU8 *base)   const { return &ric(base)->operation_desc_index;   }
    virtual int16_t  * operationBufferDescIndex(NvU8 *base)  const { return &ric(base)->operation_buffer_desc_index; }
    virtual uint16_t * numOperations(NvU8 *base)        const { return &ric(base)->num_operations;         }

protected:
    static inline emu_network_desc *ric(NvU8 *base) { return reinterpret_cast<emu_network_desc *>(base); }
};
static EMUNetworkDescA g_emu_network_desc;
const EMUNetworkDesc & EMUInterfaceA::networkDesc() const { return g_emu_network_desc; }


//
// struct emu_common_op_desc
//
class EMUCommonOpDescA : public EMUCommonOpDesc
{
public:
    virtual ~EMUCommonOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(emu_common_op_desc);    }
    virtual size_t struct_align() const { return 0; }

    virtual NvU8 * op_type(NvU8 *base) const { return &ric(base)->op_type; }
    virtual NvF32 * input_scale_factor(NvU8 *base) const { return &ric(base)->input_scale_factor; }
    virtual NvF32 * output_scale_factor(NvU8 *base) const { return &ric(base)->output_scale_factor; }

protected:
    static inline emu_common_op_desc *ric(NvU8 *base) { return reinterpret_cast<emu_common_op_desc *>(base); }
};
static EMUCommonOpDescA g_emu_common_op_desc;


//
// struct emu_power_op_desc
//
class EMUPowerOpDescA : public EMUPowerOpDesc
{
public:
    virtual ~EMUPowerOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(emu_power_op_desc);    }
    virtual size_t struct_align() const { return 4; }

    virtual EMUCommonOpDescAccessor commonOpDescAccessor(NvU8 *base) const { return EMUCommonOpDescAccessor(cir(&(ric(base)->common)), g_emu_common_op_desc); }
    virtual NvF32 * power(NvU8 *base) const { return &ric(base)->power; }
    virtual NvF32 * scale(NvU8 *base) const { return &ric(base)->scale; }
    virtual NvF32 * shift(NvU8 *base) const { return &ric(base)->shift; }

protected:
    static inline NvU8          *cir(emu_common_op_desc *c)     { return reinterpret_cast<NvU8 *>(c);             }
    static inline emu_power_op_desc *ric(NvU8 *base) { return reinterpret_cast<emu_power_op_desc *>(base); }
};
static EMUPowerOpDescA g_emu_power_op_desc;


//
// struct emu_softmax_op_desc
//
class EMUSoftmaxOpDescA : public EMUSoftmaxOpDesc
{
public:
    virtual ~EMUSoftmaxOpDescA() { }

    virtual size_t struct_size()  const { return sizeof(emu_softmax_op_desc);    }
    virtual size_t struct_align() const { return 4; }

    virtual EMUCommonOpDescAccessor commonOpDescAccessor(NvU8 *base) const { return EMUCommonOpDescAccessor(cir(&(ric(base)->common)), g_emu_common_op_desc); }
    virtual NvU8 * axis(NvU8 *base) const { return &ric(base)->axis; }

protected:
    static inline NvU8          *cir(emu_common_op_desc *c)     { return reinterpret_cast<NvU8 *>(c);             }
    static inline emu_softmax_op_desc *ric(NvU8 *base) { return reinterpret_cast<emu_softmax_op_desc *>(base); }
};
static EMUSoftmaxOpDescA g_emu_softmax_op_desc;


//
// struct emu_operation_container
//
class EMUOperationContainerA : public EMUOperationContainer
{
public:
    virtual ~EMUOperationContainerA() { }

    virtual size_t struct_size()  const { return sizeof(emu_operation_container);    }
    virtual size_t struct_align() const { return 0; }

    virtual EMUPowerOpDescAccessor powerOpDescAccessor(NvU8 *base, size_t c) const { return EMUPowerOpDescAccessor(sir(&(ric(base)[c].power_op)), g_emu_power_op_desc); }
    virtual EMUSoftmaxOpDescAccessor softmaxOpDescAccessor(NvU8 *base, size_t c) const { return EMUSoftmaxOpDescAccessor(sir(&(ric(base)[c].softmax_op)), g_emu_softmax_op_desc); }

protected:
    static inline NvU8          *sir(emu_power_op_desc *c)       { return reinterpret_cast<NvU8 *>(c);             }
    static inline NvU8          *sir(emu_softmax_op_desc *c)     { return reinterpret_cast<NvU8 *>(c);             }
    static inline emu_operation_container *ric(NvU8 *base)       { return reinterpret_cast<emu_operation_container *>(base); }
};
static EMUOperationContainerA g_emu_operation_container;
const EMUOperationContainer & EMUInterfaceA::operationContainer() const { return g_emu_operation_container; }


//
// struct emu_buffer_desc
//
class EMUBufferDescA : public EMUBufferDesc
{
public:
    virtual ~EMUBufferDescA() { }

    virtual size_t struct_size()  const { return sizeof(emu_buffer_desc);    }
    virtual size_t struct_align() const { return 256; }

    virtual NvS16 * addressIndex(NvU8 *base)    const { return &ric(base)->addressIndex; }
    virtual NvU32 * addressIndexOffset(NvU8 *base)    const { return &ric(base)->addressIndexOffset; }
    virtual NvU32 * size(NvU8 *base)       const { return &ric(base)->size; }
    virtual NvU16 * format(NvU8 *base)     const { return &ric(base)->format; }
    virtual NvU16   format_FF16()          const { return EMU_FORMAT_FF16; }
    virtual NvU16   format_INT8()          const { return EMU_FORMAT_INT8; }
    virtual NvU16   format_INT8_8()        const { return EMU_FORMAT_INT8_8; }
    virtual NvU16   format_UINT8()         const { return EMU_FORMAT_UINT8; }
    virtual NvU16   format_INT16()         const { return EMU_FORMAT_INT16; }
    virtual NvU16   format_UINT16()        const { return EMU_FORMAT_UINT16; }
    virtual NvU16 * width(NvU8 *base)      const { return &ric(base)->width; }
    virtual NvU16 * height(NvU8 *base)     const { return &ric(base)->height; }
    virtual NvU16 * channel(NvU8 *base)    const { return &ric(base)->channel; }
    virtual NvU32 * lineStride(NvU8 *base) const { return &ric(base)->line_stride; }
    virtual NvU32 * surfStride(NvU8 *base) const { return &ric(base)->surf_stride; }

protected:
    static inline emu_buffer_desc *ric(NvU8 *base)       { return reinterpret_cast<emu_buffer_desc *>(base); }
};
static EMUBufferDescA g_emu_buffer_desc;
const EMUBufferDesc & EMUInterfaceA::bufferDesc() const { return g_emu_buffer_desc; }


//
// struct emu_power_buffer_descs
//

class EMUPowerBufferDescsA : public EMUPowerBufferDescs
{
public:
    virtual ~EMUPowerBufferDescsA() { }

    virtual size_t struct_size()  const { return sizeof(emu_power_buffer_descs);    }
    virtual size_t struct_align() const { return 4; /* see __attribute__ */ }

    virtual EMUBufferDescAccessor srcDataAccessor(NvU8 *base) const { return EMUBufferDescAccessor(dir(&ric(base)->src_data), g_emu_buffer_desc); }
    virtual EMUBufferDescAccessor dstDataAccessor(NvU8 *base) const { return EMUBufferDescAccessor(dir(&ric(base)->dst_data), g_emu_buffer_desc); }

protected:
    static inline NvU8          *dir(emu_buffer_desc *d)       { return reinterpret_cast<NvU8 *>(d);             }
    static inline emu_power_buffer_descs *ric(NvU8 *base)    { return reinterpret_cast<emu_power_buffer_descs *>(base); }
};
static EMUPowerBufferDescsA g_emu_power_buffer_descs;


//
// struct emu_softmax_buffer_descs
//

class EMUSoftmaxBufferDescsA : public EMUSoftmaxBufferDescs
{
public:
    virtual ~EMUSoftmaxBufferDescsA() { }

    virtual size_t struct_size()  const { return sizeof(emu_softmax_buffer_descs);    }
    virtual size_t struct_align() const { return 4; }

    virtual EMUBufferDescAccessor srcDataAccessor(NvU8 *base) const { return EMUBufferDescAccessor(dir(&ric(base)->src_data), g_emu_buffer_desc); }
    virtual EMUBufferDescAccessor dstDataAccessor(NvU8 *base) const { return EMUBufferDescAccessor(dir(&ric(base)->dst_data), g_emu_buffer_desc); }

protected:
    static inline NvU8          *dir(emu_buffer_desc *d)       { return reinterpret_cast<NvU8 *>(d);             }
    static inline emu_softmax_buffer_descs *ric(NvU8 *base)    { return reinterpret_cast<emu_softmax_buffer_descs *>(base); }
};
static EMUSoftmaxBufferDescsA g_emu_softmax_buffer_descs;


//
// struct emu_operation_buffer_container
//
class EMUOperationBufferContainerA : public EMUOperationBufferContainer
{
public:
    virtual ~EMUOperationBufferContainerA() { }

    virtual size_t struct_size()  const { return sizeof(emu_operation_buffer_container);    }
    virtual size_t struct_align() const { return 0; }

    virtual EMUPowerBufferDescsAccessor powerBufferDescsAccessor(NvU8 *base, size_t c) const
    {
        return EMUPowerBufferDescsAccessor(sir( &(ric(base)[c]).power_buffers), g_emu_power_buffer_descs);
    }

    virtual EMUSoftmaxBufferDescsAccessor softmaxBufferDescsAccessor(NvU8 *base, size_t c) const
    {
        return EMUSoftmaxBufferDescsAccessor(sir( &(ric(base)[c]).softmax_buffers), g_emu_softmax_buffer_descs);
    }

protected:
    static inline NvU8 *sir(emu_power_buffer_descs *c) { return reinterpret_cast<NvU8 *>(c); }
    static inline NvU8 *sir(emu_softmax_buffer_descs *c) { return reinterpret_cast<NvU8 *>(c); }
    static inline emu_operation_buffer_container *ric(NvU8 *base)  { return reinterpret_cast<emu_operation_buffer_container *>(base); }
};
static EMUOperationBufferContainerA g_emu_operation_buffer_container;
const EMUOperationBufferContainer & EMUInterfaceA::operationBufferContainer() const { return g_emu_operation_buffer_container; }


//
// interface
//

NvU8 EMUInterfaceA::emulatorTargetVersionMajor()    const { return EMULATOR_VERSION_MAJOR;    }
NvU8 EMUInterfaceA::emulatorTargetVersionMinor()    const { return EMULATOR_VERSION_MINOR;    }
NvU8 EMUInterfaceA::emulatorTargetVersionSubminor() const { return EMULATOR_VERSION_SUBMINOR; }

NvU32 EMUInterfaceA::emulatorTargetVersion() const { return emu_version(); }

const std::string EMUInterfaceA::emulatorTargetGerritChange() const { return emu_gerrit_change(); }
const std::string EMUInterfaceA::emulatorTargetGerritReview() const { return emu_gerrit_review(); }

NvU8 EMUInterfaceA::emulatorVersionMajor() const
{
    return EMULATOR_VERSION_MAJOR;
}

NvU8 EMUInterfaceA::emulatorVersionMinor() const
{
    return EMULATOR_VERSION_MINOR;
}

NvU8 EMUInterfaceA::emulatorVersionSubminor() const
{
    return EMULATOR_VERSION_SUBMINOR;
}

NvU32 EMUInterfaceA::emulatorVersion() const
{
    return emu_version();
}



} // nvdla::priv
} // nvdla
