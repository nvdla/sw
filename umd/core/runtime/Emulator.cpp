/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

bool Emulator::processTask(NvU8* task_mem, std::vector<NvU8*> addressList)
{
    EMUInterface* emu_if = new EMUInterfaceA();
    EMUTaskDescAccessor task_desc = emu_if->taskDescAccessor(task_mem);
    NVDLA_UNUSED(task_desc);

    // 0 - network descriptor
    EMUNetworkDescAccessor network_desc = emu_if->networkDescAccessor(addressList[0]);

    EMUOperationContainerAccessor operation_container_0 = emu_if->operationContainerAccessor(addressList[*network_desc.operationDescIndex()]);
    EMUOperationBufferContainerAccessor operation_buffer_container_0 = emu_if->operationBufferContainerAccessor(addressList[*network_desc.operationBufferDescIndex()]);
    EMUCommonOpDescAccessor common_op_desc_0 = operation_container_0.softmaxOpDescAccessor(0).commonOpDescAccessor();

    if (*common_op_desc_0.op_type() == 0 /* POWER */)
    {
        EMUPowerOpDescAccessor power_op_desc = operation_container_0.powerOpDescAccessor(0);
        EMUPowerBufferDescsAccessor power_op_buffer_descs = operation_buffer_container_0.powerBufferDescsAccessor(0);

        executePower(power_op_desc, power_op_buffer_descs, addressList);

    } else if (*common_op_desc_0.op_type() == 1 /* SOFTMAX */) {
        EMUSoftmaxOpDescAccessor softmax_op_desc = operation_container_0.softmaxOpDescAccessor(0);
        EMUSoftmaxBufferDescsAccessor softmax_op_buffer_descs = operation_buffer_container_0.softmaxBufferDescsAccessor(0);

        executeSoftmax(softmax_op_desc, softmax_op_buffer_descs, addressList);

    } else {
        NvDlaDebugPrintf("Unknown op type %u\n", *common_op_desc_0.op_type());
    }

    delete emu_if;

    return true;
}

NvDlaError Emulator::getAddrOffset(EMUBufferDescAccessor in, NvU32 w, NvU32 h, NvU32 c, NvU32* offset)
{
    NvDlaError e = NvDlaSuccess;

    if ((*in.format()) == 2/*NVDLA_FF16_F_FORMAT*/)
    {
        NvU8 bpe = 2;
        NvU32 x = 16;
        NvU32 xStride = x * bpe;
        NvU32 cquotient = c / x;
        NvU32 cremainder = c % x;

        *offset = (cquotient * (*in.surfStride())) + (h * (*in.lineStride())) + (w * xStride) + (cremainder * bpe);
    } else {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    return NvDlaSuccess;

fail:
    return e;
}

bool Emulator::executePower(EMUPowerOpDescAccessor opDesc, EMUPowerBufferDescsAccessor bufDescs, std::vector<NvU8*> addressList)
{

    EMUBufferDescAccessor src = bufDescs.srcDataAccessor();
    EMUBufferDescAccessor dst = bufDescs.dstDataAccessor();

    if ( debugOps() )
    {
        NvDlaDebugPrintf("Processing power [power=%f scale=%f shift=%f]\n", *opDesc.power(), *opDesc.scale(), *opDesc.shift());
        NvDlaDebugPrintf("src format %u\n", *src.format());
        NvDlaDebugPrintf("\taddress[%u] 0x%llx (%ux%ux%u) %uB\n", *src.addressIndex(), addressList[*src.addressIndex()], *src.width(), *src.height(), *src.channel(), *src.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *src.lineStride(), *src.surfStride());

        NvDlaDebugPrintf("dst format %u\n", *dst.format());
        NvDlaDebugPrintf("\taddress[%u] 0x%llx (%ux%ux%u) %uB\n", *dst.addressIndex(), addressList[*dst.addressIndex()], *dst.width(), *dst.height(), *dst.channel(), *dst.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *dst.lineStride(), *dst.surfStride());
    }
    NvU8* pSrc = addressList[*src.addressIndex()];
    NvU8* pDst = addressList[*dst.addressIndex()];

    // Execute
    for (NvU32 channel=0; channel<*src.channel(); channel++)
    {
        for (NvU32 height=0; height<*src.height(); height++)
        {
            for (NvU32 width=0; width<*src.width(); width++)
            {
                NvU32 srcoffset = 0;
                NvU32 dstoffset = 0;
                if (getAddrOffset(src, width, height, channel, &srcoffset) != NvDlaSuccess)
                    return false;
                if (getAddrOffset(dst, width, height, channel, &dstoffset) != NvDlaSuccess)
                    return false;

                half_float::half* srchalfp = reinterpret_cast<half_float::half*>(pSrc + srcoffset);
                half_float::half* dsthalfp = reinterpret_cast<half_float::half*>(pDst + dstoffset);

                NvF32 x = float(*srchalfp);
                NvF32 y = powf((*opDesc.shift() + (*opDesc.scale() * x)), *opDesc.power());
                *dsthalfp = half(y);
            }
        }
    }


    return true;
}


bool Emulator::executeSoftmax(EMUSoftmaxOpDescAccessor opDesc, EMUSoftmaxBufferDescsAccessor bufDescs, std::vector<NvU8*> addressList)
{
    EMUBufferDescAccessor src = bufDescs.srcDataAccessor();
    EMUBufferDescAccessor dst = bufDescs.dstDataAccessor();

    if ( debugOps() )
    {
        NvDlaDebugPrintf("Processing softmax [axis=%u]\n", *opDesc.axis());
        NvDlaDebugPrintf("src format %u\n", *src.format());
        NvDlaDebugPrintf("\taddress[%u] 0x%llx (%ux%ux%u) %uB\n", *src.addressIndex(), addressList[*src.addressIndex()], *src.width(), *src.height(), *src.channel(), *src.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *src.lineStride(), *src.surfStride());

        NvDlaDebugPrintf("dst format %u\n", *dst.format());
        NvDlaDebugPrintf("\taddress[%u] 0x%llx (%ux%ux%u) %uB\n", *dst.addressIndex(), addressList[*dst.addressIndex()], *dst.width(), *dst.height(), *dst.channel(), *dst.size());
        NvDlaDebugPrintf("\tline_stride %uB surface_stride %uB\n", *dst.lineStride(), *dst.surfStride());
    }

    half* pSrc = reinterpret_cast<half*>(addressList[*src.addressIndex()]);
    half* pDst = reinterpret_cast<half*>(addressList[*dst.addressIndex()]);

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

    return true;
}


} // nvdla::priv
} // nvdla
