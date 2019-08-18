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

#include <queue>

#include "half.h"
#include "priv/Emulator.h"
#include "priv/Check.h"
#include "ErrorMacros.h"

using namespace half_float;

namespace nvdla
{
namespace priv
{

Emulator::Emulator() :
        m_thread(),
        m_threadActive(false),
        m_signalShutdown(false)
{

}

Emulator::~Emulator()
{

}

bool Emulator::ping()
{
    return m_threadActive;
}

NvDlaError Emulator::submit(NvU8* task_mem, bool blocking)
{
    m_taskQueue.push(task_mem);

    if (blocking) {
        // wait until queue becomes empty
        while (!m_taskQueue.empty()) {
            NvDlaThreadYield();
        }
    }

    return NvDlaSuccess;
}

NvDlaError Emulator::start()
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(NvDlaThreadCreate(threadFunction, this, &m_thread), "Failed to create thread");

    return NvDlaSuccess;

fail:
    return e;
}

void Emulator::threadFunction(void* arg)
{
    Emulator* engine = static_cast<Emulator*>(arg);
    engine->run();
}

bool Emulator::stop()
{
    bool ok = true;

    if (m_thread)
    {
        m_signalShutdown = true;
        NvDlaThreadJoin(m_thread);
        m_thread = NULL;
    }

    return ok;
}

bool Emulator::run()
{
    bool ok = true;
    m_threadActive = true;

    EMUInterface* emu_if = new EMUInterfaceA();

    NvDlaDebugPrintf("Emulator starting\n");

    while (true)
    {
        if (!m_taskQueue.empty())
        {
            NvU8* task_mem = m_taskQueue.front();
            NvDlaDebugPrintf("Work Found!\n");

            EMUTaskDescAccessor task_desc = emu_if->taskDescAccessor(task_mem);

            NvU32 numAddresses = *task_desc.numAddresses();
            std::vector<NvU8*> mappedAddressList;
            mappedAddressList.resize(numAddresses);

            // Replace all mem handles with mapped addresses
            for (NvU32 ii=0; ii<numAddresses; ii++)
            {
                void* base = *((void **)task_desc.addressList(ii).hMem());
                NvU32 offset = *task_desc.addressList(ii).offset();

                if (base == 0) {
                    mappedAddressList[ii] = NULL;
                }
                else {
                    mappedAddressList[ii] = (NvU8*)base + offset;
                }
            }

            // Process the task
            processTask(task_mem, mappedAddressList);
            NvDlaDebugPrintf("Work Done\n");

            m_taskQueue.pop();
            continue;
        }

        if (m_signalShutdown)
        {
            NvDlaDebugPrintf("Shutdown signal received, exiting\n");
            break;
        }

        if (m_taskQueue.empty())
        {
            NvDlaSleepMS(500);
        }
    }

    // Cleanup
    while (!m_taskQueue.empty())
    {
        m_taskQueue.pop();
    }

    delete emu_if;
    m_threadActive = false;
    m_signalShutdown = false;

    return ok;
}

NvDlaError Emulator::processTask(NvU8* task_mem, std::vector<NvU8*> addressList)
{
    NvDlaError e = NvDlaSuccess;
    EMUInterface* emu_if = new EMUInterfaceA();
    EMUTaskDescAccessor task_desc = emu_if->taskDescAccessor(task_mem);
    NVDLA_UNUSED(task_desc);

    // 0 - network descriptor
    EMUNetworkDescAccessor network_desc = emu_if->networkDescAccessor(addressList[0]);
    NvU16 numOperations                 = *network_desc.numOperations();

    NvU8*  operation_container_0        = addressList[*network_desc.operationDescIndex()];
    NvU8*  operation_buffer_container_0 = addressList[*network_desc.operationBufferDescIndex()];

    for ( NvU16 op = 0; op < numOperations; ++op)
    {
        // follow the same technique to obtain op_container and buffer_container accessors for each op as the compiler side
        // this short-cut assumes that the op and buffer containers for all batches were placed contiguous in memory

        EMUOperationContainerAccessor operation_container              = emu_if->operationContainerAccessor(operation_container_0);
        EMUOperationBufferContainerAccessor operation_buffer_container = emu_if->operationBufferContainerAccessor(operation_buffer_container_0);

        // HACK: Borrow softmax's accessor to get at the common descriptor
        EMUCommonOpDescAccessor common_op_desc = operation_container.softmaxOpDescAccessor(op).commonOpDescAccessor();

        if (*common_op_desc.op_type() == 0 /* POWER */)
        {
            EMUPowerOpDescAccessor power_op_desc = operation_container.powerOpDescAccessor(op);
            EMUPowerBufferDescsAccessor power_op_buffer_descs = operation_buffer_container.powerBufferDescsAccessor(op);

            PROPAGATE_ERROR_FAIL(executePower(power_op_desc, common_op_desc, power_op_buffer_descs, addressList));

        } else if (*common_op_desc.op_type() == 1 /* SOFTMAX */) {
            EMUSoftmaxOpDescAccessor softmax_op_desc = operation_container.softmaxOpDescAccessor(op);
            EMUSoftmaxBufferDescsAccessor softmax_op_buffer_descs = operation_buffer_container.softmaxBufferDescsAccessor(op);

            PROPAGATE_ERROR_FAIL(executeSoftmax(softmax_op_desc, common_op_desc, softmax_op_buffer_descs, addressList));

        } else {
            NvDlaDebugPrintf("Unknown op type %u\n", *common_op_desc.op_type());
        }
    }
fail:
    return e;
}

NvS8 Emulator::getBpe(EMUBufferDescAccessor buffer)
{
    NvS8 bpe = -1;
    switch(*buffer.format())
    {
        case EMU_FORMAT_FF16:
        case EMU_FORMAT_INT16:
        case EMU_FORMAT_UINT16:
            bpe = 2; break;
        case EMU_FORMAT_INT8:
        case EMU_FORMAT_INT8_8:
        case EMU_FORMAT_UINT8:
            bpe = 1; break;
        default:
            bpe = -1;
    }
    return bpe;
}

NvDlaError Emulator::getAddrOffset(EMUBufferDescAccessor in, NvU32 w, NvU32 h, NvU32 c, NvU32* offset)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 x = 0;
    NvU32 xStride = 0;
    NvU32 cquotient = 0;
    NvU32 cremainder = 0;

    NvS8 bpe = getBpe(in);
    if (bpe < 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    switch(*in.format())
    {
        case EMU_FORMAT_FF16:
        case EMU_FORMAT_INT8:
            x = 32 / bpe;
            xStride = x * bpe;
            cquotient = c / x;
            cremainder = c % x;
            *offset = (cquotient * (*in.surfStride())) + (h * (*in.lineStride())) + (w * xStride) + (cremainder * bpe);
            break;
        case EMU_FORMAT_INT8_8:
            x = 8 / bpe;
            xStride = x * bpe;
            cquotient = c / x;
            cremainder = c % x;
            *offset = (cquotient * (*in.surfStride())) + (h * (*in.lineStride())) + (w * xStride) + (cremainder * bpe);
            break;
        default:
            *offset = 0;
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported input format: %d\n", *in.format());
    }

fail:
    return e;
}

NvDlaError Emulator::executePower
(
    EMUPowerOpDescAccessor opDesc,
    EMUCommonOpDescAccessor commonOpDesc,
    EMUPowerBufferDescsAccessor bufDescs,
    std::vector<NvU8*> addressList
)
{
    NvDlaError e = NvDlaSuccess;
    EMUBufferDescAccessor src = bufDescs.srcDataAccessor();
    EMUBufferDescAccessor dst = bufDescs.dstDataAccessor();

    if ( debugOps() )
    {
        NvDlaDebugPrintf("Processing power [power=%f scale=%f shift=%f]\n", *opDesc.power(), *opDesc.scale(), *opDesc.shift());
        NvDlaDebugPrintf("src format %u\n", *src.format());
        NvDlaDebugPrintf("\taddress[%u][%u] 0x%llx (%ux%ux%u) %uB\n", *src.addressIndex(), *src.addressIndexOffset(),
                addressList[*src.addressIndex()], *src.width(), *src.height(), *src.channel(), *src.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *src.lineStride(), *src.surfStride());
        NvDlaDebugPrintf("\tinput scale factor: %f, output scale factor: %f\n", *commonOpDesc.input_scale_factor(), *commonOpDesc.output_scale_factor());

        NvDlaDebugPrintf("dst format %u\n", *dst.format());
        NvDlaDebugPrintf("\taddress[%u][%u] 0x%llx (%ux%ux%u) %uB\n", *dst.addressIndex(), *dst.addressIndexOffset(),
                addressList[*dst.addressIndex()], *dst.width(), *dst.height(), *dst.channel(), *dst.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *dst.lineStride(), *dst.surfStride());
    }

    if ( *src.format() != *dst.format() )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support EMU Scale operation with different "
            " src (%d) and dst (%d) formats\n", static_cast<NvU32>(*src.format()),
            static_cast<NvU32>(*dst.format()));
    }

    // Execute
    {
        NvU8* pSrc = addressList[*src.addressIndex()] + *src.addressIndexOffset();
        NvU8* pDst = addressList[*dst.addressIndex()] + *dst.addressIndexOffset();

        NvU32 srcoffset = 0;
        NvU32 dstoffset = 0;

        for (NvU32 channel=0; channel<*src.channel(); channel++)
        {
            for (NvU32 height=0; height<*src.height(); height++)
            {
                for (NvU32 width=0; width<*src.width(); width++)
                {
                    PROPAGATE_ERROR_FAIL(getAddrOffset(src, width, height, channel, &srcoffset));
                    PROPAGATE_ERROR_FAIL(getAddrOffset(dst, width, height, channel, &dstoffset));

                    if (*src.format() == EMU_FORMAT_FF16)
                    {
                        NvF32 x = 0;
                        NvF32 y = 0;

                        half_float::half* srchalfp = reinterpret_cast<half_float::half*>(pSrc + srcoffset);
                        half_float::half* dsthalfp = reinterpret_cast<half_float::half*>(pDst + dstoffset);

                        x = float(*srchalfp);
                        y = powf((*opDesc.shift() + (*opDesc.scale() * x)), *opDesc.power());
                        *dsthalfp = half(y);
                    }
                    else if ((*src.format() == EMU_FORMAT_INT8) || (*src.format() == EMU_FORMAT_INT8_8))
                    {
                        NvF32 x = 0;
                        NvF32 y = 0;

                        NvS8* srcint8p = reinterpret_cast<NvS8*>(pSrc + srcoffset);
                        NvS8* dstint8p = reinterpret_cast<NvS8*>(pDst + dstoffset);

                        x = static_cast<NvF32>(*srcint8p);
                        // scale input for executing in FLOAT land
                        x *= *commonOpDesc.input_scale_factor();
                        y = powf((*opDesc.shift() + (*opDesc.scale() * x)), *opDesc.power());
                        // rescale output to write out in INT8 land
                        y /= *commonOpDesc.output_scale_factor();

                        *dstint8p = saturate<NvF32, NvS8>(y);
                    }
                    else
                    {
                        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support EMU scale operation for format: %d\n",
                            static_cast<NvU32>(*src.format()));
                    }
                }
            }
        }
    }

fail:
    return e;
}


NvDlaError Emulator::executeSoftmax
(
    EMUSoftmaxOpDescAccessor opDesc,
    EMUCommonOpDescAccessor commonOpDesc,
    EMUSoftmaxBufferDescsAccessor bufDescs,
    std::vector<NvU8*> addressList
)
{
    NvDlaError e = NvDlaSuccess;

    EMUBufferDescAccessor src = bufDescs.srcDataAccessor();
    EMUBufferDescAccessor dst = bufDescs.dstDataAccessor();

    if ( debugOps() )
    {
        NvDlaDebugPrintf("Processing softmax [axis=%u]\n", *opDesc.axis());
        NvDlaDebugPrintf("src format %u\n", *src.format());
        NvDlaDebugPrintf("\taddress[%u][%u] 0x%llx (%ux%ux%u) %uB\n", *src.addressIndex(), *src.addressIndexOffset(),
                addressList[*src.addressIndex()], *src.width(), *src.height(), *src.channel(), *src.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *src.lineStride(), *src.surfStride());
        NvDlaDebugPrintf("\tinput scale factor: %f, output scale factor: %f\n", *commonOpDesc.input_scale_factor(), *commonOpDesc.output_scale_factor());

        NvDlaDebugPrintf("dst format %u\n", *dst.format());
        NvDlaDebugPrintf("\taddress[%u][%u] 0x%llx (%ux%ux%u) %uB\n", *dst.addressIndex(),  *dst.addressIndexOffset(),
                addressList[*dst.addressIndex()], *dst.width(), *dst.height(), *dst.channel(), *dst.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *dst.lineStride(), *dst.surfStride());
    }

    if ( *src.format() != *dst.format() )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support EMU Scale operation with different "
            " src (%d) and dst (%d) formats\n", static_cast<NvU32>(*src.format()),
            static_cast<NvU32>(*dst.format()));
    }

    // Execute
    if (*src.format() == EMU_FORMAT_FF16)
    {
        half* pSrc = reinterpret_cast<half*>( addressList[*src.addressIndex()] + *src.addressIndexOffset());
        half* pDst = reinterpret_cast<half*>( addressList[*dst.addressIndex()] + *dst.addressIndexOffset());

        NvF32 maxval = -INFINITY;
        for (NvU32 ii=0; ii<*src.channel(); ii++)
        {
            if (float(pSrc[ii]) > maxval)
            {
                maxval = float(pSrc[ii]);
            }
        }
        NvF32 sumexp = 0.0f;
        for (NvU32 ii=0; ii<*src.channel(); ii++)
        {
            sumexp += expf(float(pSrc[ii])-maxval);
        }
        for (NvU32 ii=0; ii<*src.channel(); ii++)
        {
            pDst[ii] = expf(float(pSrc[ii])-maxval) / sumexp;
        }
    }
    else if ((*src.format() == EMU_FORMAT_INT8) || (*src.format() == EMU_FORMAT_INT8_8))
    {
        NvS8* pSrc = reinterpret_cast<NvS8*>( addressList[*src.addressIndex()] + *src.addressIndexOffset() );
        NvS8* pDst = reinterpret_cast<NvS8*>( addressList[*dst.addressIndex()] + *dst.addressIndexOffset() );

        half* pHalfSrc = reinterpret_cast<half*>(malloc(*src.channel() * sizeof(half)));
        half* pHalfDst = reinterpret_cast<half*>(malloc(*dst.channel() * sizeof(half)));

        // scale input for processing in FLOAT land
        for (NvU32 ii = 0; ii < *src.channel(); ii++)
        {
            pHalfSrc[ii] = pSrc[ii] * (*commonOpDesc.input_scale_factor());
        }

        NvF32 maxval = -INFINITY;
        for (NvU32 ii=0; ii<*src.channel(); ii++)
        {
            if (float(pHalfSrc[ii]) > maxval)
            {
                maxval = float(pHalfSrc[ii]);
            }
        }

        NvF32 sumexp = 0.0f;
        for (NvU32 ii=0; ii<*src.channel(); ii++)
        {
            sumexp += expf(float(pHalfSrc[ii])-maxval);
        }
        for (NvU32 ii=0; ii<*src.channel(); ii++)
        {
            pHalfDst[ii] = static_cast<half>(expf(float(pHalfSrc[ii])-maxval) / sumexp);
        }

        // rescale output to write out in INT8 land
        for (NvU32 ii = 0; ii < *dst.channel(); ii++)
        {
            pDst[ii] = saturate<NvF32, NvS8>(pHalfDst[ii] / (*commonOpDesc.output_scale_factor()));
        }

        if (debugPrint())
        {
            NvF32 maxHalfDst = -INFINITY;
            NvU32 maxHalfIndex = -1;
            NvF32 maxIntDst = std::numeric_limits<NvS8>::lowest();
            NvU32 maxIntIndex = -1;

            for (NvU32 ii = 0; ii < *dst.channel(); ii++) {
                if (pHalfDst[ii] > maxHalfDst) {
                    maxHalfDst = pHalfDst[ii];
                    maxHalfIndex = ii;
                }
                if (pDst[ii] > maxIntDst) {
                    maxIntDst = pDst[ii];
                    maxIntIndex = ii;
                }
            }

            NvDlaDebugPrintf("Post-softmax max value: (half) %f, (int) %f\n", maxHalfDst, maxIntDst);
            NvDlaDebugPrintf("at indices (half) %d, (int) %d\n", maxHalfIndex, maxIntIndex);
        }


    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support EMU softmax operation for format: %d\n",
            static_cast<NvU32>(*src.format()));
    }

fail:
    return e;
}


} // nvdla::priv
} // nvdla
