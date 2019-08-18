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

NvDlaError Tensor2DIMG(const TestAppArgs* appArgs, const nvdla::IRuntime::NvDlaTensor* tensor, NvDlaImage* image)
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
        case TENSOR_PIXEL_FORMAT_FEATURE_X8:
        {
            if (tensor->dataType == TENSOR_DATA_TYPE_HALF) {
                surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_F;
            } else if (tensor->dataType == TENSOR_DATA_TYPE_INT16) {
                surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_I;
            } else if (tensor->dataType == TENSOR_DATA_TYPE_INT8) {
                surfaceFormat = NvDlaImage::D_F8_CxHWx_x8_I;
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
    image->m_meta.width = tensor->dims.w;
    image->m_meta.height = tensor->dims.h;
    image->m_meta.channel = tensor->dims.c;

    image->m_meta.lineStride = tensor->stride[1];
    image->m_meta.surfaceStride = tensor->stride[2];
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

NvDlaError createImageCopy(const TestAppArgs* appArgs, const NvDlaImage* in, const nvdla::IRuntime::NvDlaTensor* outTensorDesc, NvDlaImage* out)
{
    NvU8* ibuf = 0;
    NvU8* obuf = 0;

    out->m_meta.width = outTensorDesc->dims.w;
    out->m_meta.height = outTensorDesc->dims.h;
    out->m_meta.channel = outTensorDesc->dims.c;

    if (in->m_meta.width != out->m_meta.width )
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Mismatched width: %u != %u", in->m_meta.width, out->m_meta.width);
    if (in->m_meta.height != out->m_meta.height )
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Mismatched height: %u != %u", in->m_meta.height, out->m_meta.height);
    if (in->m_meta.channel != out->m_meta.channel )
        REPORT_ERROR(NvDlaError_BadParameter, "Mismatched channel: %u != %u", in->m_meta.channel, out->m_meta.channel);

    switch(outTensorDesc->pixelFormat)
    {
    case NVDLA_PIXEL_FORMAT_R8:
        out->m_meta.surfaceFormat = NvDlaImage::T_R8;
        out->m_meta.lineStride    = outTensorDesc->stride[1];
        out->m_meta.surfaceStride = 0;
        out->m_meta.size          = outTensorDesc->bufferSize;
        break;

    case NVDLA_PIXEL_FORMAT_FEATURE:
        if (outTensorDesc->dataType == NVDLA_DATA_TYPE_HALF)
            out->m_meta.surfaceFormat = NvDlaImage::D_F16_CxHWx_x16_F;
        else if (outTensorDesc->dataType == NVDLA_DATA_TYPE_INT8)
        {
            out->m_meta.surfaceFormat = NvDlaImage::D_F8_CxHWx_x32_I;
        }
        else
            ORIGINATE_ERROR(NvDlaError_NotSupported, "Unsupported (pixel, data) combination: (%u, %u)", outTensorDesc->pixelFormat, outTensorDesc->dataType);

        out->m_meta.lineStride    = outTensorDesc->stride[1];
        out->m_meta.surfaceStride = outTensorDesc->stride[2];
        out->m_meta.size          = outTensorDesc->bufferSize;
        break;

    case NVDLA_PIXEL_FORMAT_FEATURE_X8:
        if (outTensorDesc->dataType == NVDLA_DATA_TYPE_INT8)
            out->m_meta.surfaceFormat = NvDlaImage::D_F8_CxHWx_x8_I;
        else
            ORIGINATE_ERROR(NvDlaError_NotSupported, "Unsupported (pixel, data) combination: (%u, %u)", outTensorDesc->pixelFormat, outTensorDesc->dataType);

        out->m_meta.lineStride    = outTensorDesc->stride[1];
        out->m_meta.surfaceStride = outTensorDesc->stride[2];
        out->m_meta.size          = outTensorDesc->bufferSize;
        break;

    case NVDLA_PIXEL_FORMAT_A16B16G16R16_F:
        out->m_meta.surfaceFormat = NvDlaImage::T_A16B16G16R16_F;
        out->m_meta.lineStride    = outTensorDesc->stride[1];
        out->m_meta.surfaceStride = 0;
        out->m_meta.size          = outTensorDesc->bufferSize;
        break;

    case NVDLA_PIXEL_FORMAT_A8B8G8R8:
        out->m_meta.surfaceFormat = NvDlaImage::T_A8B8G8R8;
        out->m_meta.lineStride    = outTensorDesc->stride[1];
        out->m_meta.surfaceStride = 0;
        out->m_meta.size          = outTensorDesc->bufferSize;
        break;
    default:
        ORIGINATE_ERROR(NvDlaError_NotSupported, "Unsupported pixel format: %u", outTensorDesc->pixelFormat);
    }

    if (out->getBpe() <= 0)
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    // These calculations work for channels <= 16
    if (out->m_meta.channel > 16)
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    // Number of input channels should be <= 4
    if (in->m_meta.channel > 4)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Input channel should not be greater than 4");

    if ( 0 )
    {
        NvDlaDebugPrintf("Dims: %d x %d x %d: ", out->m_meta.height, out->m_meta.width, out->m_meta.channel);
        NvDlaDebugPrintf("LS: %d SS: %d Size: %d\n", out->m_meta.lineStride, out->m_meta.surfaceStride, out->m_meta.size);
    }

    // Allocate the buffer
    out->m_pData = NvDlaAlloc(out->m_meta.size);

    // Copy the data
    memset(out->m_pData, 0, out->m_meta.size);

    ibuf = static_cast<NvU8*>(in->m_pData);
    obuf = static_cast<NvU8*>(out->m_pData);

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

                if (outTensorDesc->dataType == NVDLA_DATA_TYPE_HALF)
                {
                    half_float::half* outp = reinterpret_cast<half_float::half*>(obuf + ooffset);
                    *outp = static_cast<half_float::half>((float(*inp) - float(appArgs->mean[z]))/appArgs->normalize_value);
                }
                else if (outTensorDesc->dataType == NVDLA_DATA_TYPE_INT8)
                {
                    char* outp = reinterpret_cast<char*>(obuf + ooffset);
                    //*outp = char(*inp); // no normalization happens
                    // compress the image from [0-255] to [0-127]
                    *outp = static_cast<NvS8>(std::floor((*inp * 127.0/255.0) + 0.5f));
                }
            }
        }
    }

    return NvDlaSuccess;
}

