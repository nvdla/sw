/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "ErrorMacros.h"
#include "RuntimeTest.h"

#include "main.h"

#include "half.h"
#include "nvdla_os_inf.h"

#include <fstream>
#include <sstream>
#include <string>

NvDlaError DIMG2DlaBuffer(const NvDlaImage* image, void** pBuffer)
{
    if (!image || !(*pBuffer))
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    memcpy(*pBuffer, image->m_pData, image->m_meta.size);

    return NvDlaSuccess;
}

NvDlaError DlaBuffer2DIMG(void** pBuffer, NvDlaImage* image)
{
    if (!(*pBuffer) || !image)
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    memcpy(image->m_pData, *pBuffer, image->m_meta.size);

    return NvDlaSuccess;
}

NvDlaError Tensor2DIMG(const nvdla::IRuntime::NvDlaTensor* tensor, NvDlaImage* image)
{
    if (!tensor || !image)
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    // Fill surfaceFormat
    NvDlaImage::PixelFormat surfaceFormat;
    switch (tensor->pixelFormat)
    {
        case TENSOR_PIXEL_FORMAT_FEATURE:
        {
            if (tensor->dataType == TENSOR_DATA_TYPE_HALF) {
                surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_F;
            } else if (tensor->dataType == TENSOR_DATA_TYPE_INT16) {
                surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_I;
            } else if (tensor->dataType == TENSOR_DATA_TYPE_INT8) {
                surfaceFormat = NvDlaImage::D_F8_CxHWx_x32_I;
            } else {
                ORIGINATE_ERROR(NvDlaError_NotSupported);
            }
        }
        break;

        default:
            REPORT_ERROR(NvDlaError_NotSupported, "Unexpected surface format %u, defaulting to D_F16_CxHWx_x16_F", tensor->pixelFormat);
            surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_F;
            break;
    }

    image->m_meta.surfaceFormat = surfaceFormat;
    image->m_meta.width = tensor->w;
    image->m_meta.height = tensor->h;
    image->m_meta.channel = tensor->c;

    image->m_meta.lineStride = tensor->pitchLinear.lineStride;
    image->m_meta.surfaceStride = tensor->pitchLinear.surfStride;
    image->m_meta.size = tensor->bufferSize;

    if ( 0 ) {
        NvDlaDebugPrintf("tensor2dimg: image meta: format=%u width=%u height=%u lineStride=%u surfaceStride=%u size=%u\n",
                        image->m_meta.surfaceFormat,
                        image->m_meta.width,  image->m_meta.height,
                        image->m_meta.lineStride, image->m_meta.surfaceStride,
                        (NvU32)image->m_meta.size);
    }

    // Allocate the buffer
    image->m_pData = NvDlaAlloc(image->m_meta.size);
    if (!image->m_pData)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    // Clear the data
    memset(image->m_pData, 0, image->m_meta.size);

    return NvDlaSuccess;
}

static int roundUp(int num, int mul)
{
    int rem;

    if (mul == 0)
        rem = 0;
    else
        rem = num % mul;

    if (rem == 0)
        return num;

    return num + mul - rem;
}

// This format conversion can be generalized and moved to DlaImageUtils
NvDlaError createFF16ImageCopy(NvDlaImage* in, NvDlaImage* out)
{
    out->m_meta.surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_F;
    out->m_meta.width = in->m_meta.width;
    out->m_meta.height = in->m_meta.height;
    out->m_meta.channel = in->m_meta.channel;

    NvS8 bpe = out->getBpe();
    if (bpe <= 0)
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    // Enforce stride and size alignment of 32B
    NvU32 strideAlign = 32;
    NvU32 sizeAlign = 32;
    // These calculations work for channels <= 16
    if (out->m_meta.channel > 16)
        ORIGINATE_ERROR(NvDlaError_BadParameter);
    out->m_meta.lineStride = roundUp(out->m_meta.width * roundUp(out->m_meta.channel * bpe, strideAlign), strideAlign);
    out->m_meta.surfaceStride = roundUp(out->m_meta.lineStride * out->m_meta.height, strideAlign);
    out->m_meta.size = roundUp(out->m_meta.surfaceStride, sizeAlign);

    // Allocate the buffer
    out->m_pData = NvDlaAlloc(out->m_meta.size);

    // Copy the data
    memset(out->m_pData, 0, out->m_meta.size);

    NvU8* ibuf = static_cast<NvU8*>(in->m_pData);
    NvU8* obuf = static_cast<NvU8*>(out->m_pData);

    for (NvU32 y=0; y < in->m_meta.height; y++)
    {
        for (NvU32 x=0; x < in->m_meta.width; x++)
        {
            for (NvU32 z=0; z < in->m_meta.channel; z++)
            {
                NvS32 ioffset = in->getAddrOffset(x, y, z);
                NvS32 ooffset = out->getAddrOffset(x, y, z);

                if (ioffset < 0)
                    ORIGINATE_ERROR(NvDlaError_BadParameter);
                if (ooffset < 0)
                    ORIGINATE_ERROR(NvDlaError_BadParameter);

                NvU8* inp = ibuf + ioffset;
                half_float::half* outp = reinterpret_cast<half_float::half*>(obuf + ooffset);
                *outp = half_float::half(float(*inp));
            }
        }
    }

    return NvDlaSuccess;
}

