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

#include <stdbool.h>
#include "dlaerror.h"
#include "dlatypes.h"

#include "priv/EMUInterface.h"

namespace nvdla
{
namespace priv
{

//
// emu_address
//
EMUAddressAccessor::EMUAddressAccessor(NvU8 *base, const EMUAddress &n) : _base(base), _n(n) { }

NvU8 * EMUAddressAccessor::struct_base()  const { return _base; }
size_t EMUAddressAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUAddressAccessor::struct_align() const { return _n.struct_align(); }

void * EMUAddressAccessor::hMem()  const { return _n.hMem(_base); }
NvU32 * EMUAddressAccessor::offset()  const { return _n.offset(_base); }


//
// emu_task_desc
//
EMUTaskDescAccessor::EMUTaskDescAccessor(NvU8 *base, const EMUTaskDesc &n) : _base(base), _n(n) { }

NvU8 * EMUTaskDescAccessor::struct_base()  const { return _base; }
size_t EMUTaskDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUTaskDescAccessor::struct_align() const { return _n.struct_align(); }

NvU32 * EMUTaskDescAccessor::numAddresses()  const { return _n.numAddresses(_base); }
size_t EMUTaskDescAccessor::maxBuffersPerTask() const { return _n.maxBuffersPerTask(); }
EMUAddressAccessor EMUTaskDescAccessor::addressList(size_t c) const { return _n.addressList(_base, c); }


//
// emu_network_desc
//
EMUNetworkDescAccessor::EMUNetworkDescAccessor(NvU8 *base, const EMUNetworkDesc &n) : _base(base), _n(n) { }

NvU8 * EMUNetworkDescAccessor::struct_base()  const { return _base;      }
size_t EMUNetworkDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUNetworkDescAccessor::struct_align() const { return _n.struct_align(); }

int16_t  * EMUNetworkDescAccessor::operationDescIndex()   const { return _n.operationDescIndex(_base); }
int16_t  * EMUNetworkDescAccessor::operationBufferDescIndex()     const { return _n.operationBufferDescIndex(_base); }
uint16_t * EMUNetworkDescAccessor::numOperations()        const { return _n.numOperations(_base); }


//
// emu_common_op_desc
//
EMUCommonOpDescAccessor::EMUCommonOpDescAccessor(NvU8 *base, const EMUCommonOpDesc &n) : _base(base), _n(n) { }

NvU8 * EMUCommonOpDescAccessor::struct_base()  const { return _base;      }
size_t EMUCommonOpDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUCommonOpDescAccessor::struct_align() const { return _n.struct_align(); }

NvU8 * EMUCommonOpDescAccessor::op_type()   const { return _n.op_type(_base); }
NvF32 * EMUCommonOpDescAccessor::input_scale_factor() const { return _n.input_scale_factor(_base); }
NvF32 * EMUCommonOpDescAccessor::output_scale_factor() const { return _n.output_scale_factor(_base); }

//
// emu_power_op_desc
//
EMUPowerOpDescAccessor::EMUPowerOpDescAccessor(NvU8 *base, const EMUPowerOpDesc &n) : _base(base), _n(n) { }

NvU8 * EMUPowerOpDescAccessor::struct_base()  const { return _base;      }
size_t EMUPowerOpDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUPowerOpDescAccessor::struct_align() const { return _n.struct_align(); }

EMUCommonOpDescAccessor EMUPowerOpDescAccessor::commonOpDescAccessor() const { return _n.commonOpDescAccessor(_base); }
NvF32 * EMUPowerOpDescAccessor::power()   const { return _n.power(_base); }
NvF32 * EMUPowerOpDescAccessor::scale()   const { return _n.scale(_base); }
NvF32 * EMUPowerOpDescAccessor::shift()   const { return _n.shift(_base); }


//
// emu_softmax_op_desc
//
EMUSoftmaxOpDescAccessor::EMUSoftmaxOpDescAccessor(NvU8 *base, const EMUSoftmaxOpDesc &n) : _base(base), _n(n) { }

NvU8 * EMUSoftmaxOpDescAccessor::struct_base()  const { return _base;      }
size_t EMUSoftmaxOpDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUSoftmaxOpDescAccessor::struct_align() const { return _n.struct_align(); }

EMUCommonOpDescAccessor EMUSoftmaxOpDescAccessor::commonOpDescAccessor() const { return _n.commonOpDescAccessor(_base); }
NvU8 * EMUSoftmaxOpDescAccessor::axis()   const { return _n.axis(_base); }


//
// emu_operation_container
//
EMUOperationContainerAccessor::EMUOperationContainerAccessor(NvU8 *base, const EMUOperationContainer &n) : _base(base), _n(n) { }

NvU8 * EMUOperationContainerAccessor::struct_base()  const { return _base;      }
size_t EMUOperationContainerAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUOperationContainerAccessor::struct_align() const { return _n.struct_align(); }

EMUPowerOpDescAccessor EMUOperationContainerAccessor::powerOpDescAccessor(size_t c) const { return _n.powerOpDescAccessor(_base, c); }
EMUSoftmaxOpDescAccessor EMUOperationContainerAccessor::softmaxOpDescAccessor(size_t c) const { return _n.softmaxOpDescAccessor(_base, c); }


//
// emu_buffer_desc
//
EMUBufferDescAccessor::EMUBufferDescAccessor(NvU8 *base, const EMUBufferDesc &n) : _base(base), _n(n) { }

NvU8 * EMUBufferDescAccessor::struct_base()  const { return _base;      }
size_t EMUBufferDescAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUBufferDescAccessor::struct_align() const { return _n.struct_align(); }

NvS16 * EMUBufferDescAccessor::addressIndex()    const { return _n.addressIndex(_base); }
NvU32 * EMUBufferDescAccessor::addressIndexOffset()    const { return _n.addressIndexOffset(_base); }
NvU32 * EMUBufferDescAccessor::size()       const { return _n.size(_base); }
NvU16 * EMUBufferDescAccessor::format()     const { return _n.format(_base); }
NvU16   EMUBufferDescAccessor::format_FF16()    const { return _n.format_FF16(); }
NvU16   EMUBufferDescAccessor::format_INT8()    const { return _n.format_INT8(); }
NvU16   EMUBufferDescAccessor::format_INT8_8()  const { return _n.format_INT8_8(); }
NvU16   EMUBufferDescAccessor::format_UINT8()   const { return _n.format_UINT8(); }
NvU16   EMUBufferDescAccessor::format_INT16()   const { return _n.format_INT16(); }
NvU16   EMUBufferDescAccessor::format_UINT16()  const { return _n.format_UINT16(); }
NvU16 * EMUBufferDescAccessor::width()      const { return _n.width(_base); }
NvU16 * EMUBufferDescAccessor::height()     const { return _n.height(_base); }
NvU16 * EMUBufferDescAccessor::channel()    const { return _n.channel(_base); }
NvU32 * EMUBufferDescAccessor::lineStride() const { return _n.lineStride(_base); }
NvU32 * EMUBufferDescAccessor::surfStride() const { return _n.surfStride(_base); }


//
// emu_power_buffer_descs
//
EMUPowerBufferDescsAccessor::EMUPowerBufferDescsAccessor(NvU8 *base, const EMUPowerBufferDescs &n) : _base(base), _n(n) { }

NvU8 * EMUPowerBufferDescsAccessor::struct_base()  const { return _base;      }
size_t EMUPowerBufferDescsAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUPowerBufferDescsAccessor::struct_align() const { return _n.struct_align(); }

EMUBufferDescAccessor EMUPowerBufferDescsAccessor::srcDataAccessor() const { return _n.srcDataAccessor(_base); }
EMUBufferDescAccessor EMUPowerBufferDescsAccessor::dstDataAccessor() const { return _n.dstDataAccessor(_base); }


//
// emu_softmax_buffer_descs
//
EMUSoftmaxBufferDescsAccessor::EMUSoftmaxBufferDescsAccessor(NvU8 *base, const EMUSoftmaxBufferDescs &n) : _base(base), _n(n) { }

NvU8 * EMUSoftmaxBufferDescsAccessor::struct_base()  const { return _base;      }
size_t EMUSoftmaxBufferDescsAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUSoftmaxBufferDescsAccessor::struct_align() const { return _n.struct_align(); }

EMUBufferDescAccessor EMUSoftmaxBufferDescsAccessor::srcDataAccessor() const { return _n.srcDataAccessor(_base); }
EMUBufferDescAccessor EMUSoftmaxBufferDescsAccessor::dstDataAccessor() const { return _n.dstDataAccessor(_base); }


//
// emu_operation_buffer_container
//
EMUOperationBufferContainerAccessor::EMUOperationBufferContainerAccessor(NvU8 *base, const EMUOperationBufferContainer &n) : _base(base), _n(n) { }

NvU8 * EMUOperationBufferContainerAccessor::struct_base()  const { return _base;      }
size_t EMUOperationBufferContainerAccessor::struct_size()  const { return _n.struct_size();  }
size_t EMUOperationBufferContainerAccessor::struct_align() const { return _n.struct_align(); }

EMUPowerBufferDescsAccessor EMUOperationBufferContainerAccessor::powerBufferDescsAccessor(size_t c) const { return _n.powerBufferDescsAccessor(_base, c); }
EMUSoftmaxBufferDescsAccessor EMUOperationBufferContainerAccessor::softmaxBufferDescsAccessor(size_t c) const { return _n.softmaxBufferDescsAccessor(_base, c); }


//
// EMUInterface::
//
EMUTaskDescAccessor     EMUInterface::taskDescAccessor(NvU8 *base)     const { return EMUTaskDescAccessor(base, taskDesc()); }
EMUNetworkDescAccessor  EMUInterface::networkDescAccessor(NvU8 *base)  const { return EMUNetworkDescAccessor(base, networkDesc()); }
EMUOperationContainerAccessor EMUInterface::operationContainerAccessor(NvU8 *base) const { return EMUOperationContainerAccessor(base, operationContainer()); }
EMUBufferDescAccessor EMUInterface::bufferDescAccessor(NvU8 *base) const { return EMUBufferDescAccessor(base, bufferDesc()); }
EMUOperationBufferContainerAccessor   EMUInterface::operationBufferContainerAccessor(NvU8 *base)   const { return EMUOperationBufferContainerAccessor(base, operationBufferContainer()); }

} // nvdla::priv
} // nvdla
