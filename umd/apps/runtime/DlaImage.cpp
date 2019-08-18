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

#include <math.h>

#include "DlaImage.h"
#include "ErrorMacros.h"
#include "nvdla_os_inf.h"
#include "half.h"

#include <stdio.h> // snprintf

const NvU32 NvDlaImage::ms_version = 0;

NvS8 NvDlaImage::getBpe() const
{
    NvS8 bpe = -1;

    switch(m_meta.surfaceFormat)
    {
    case T_R8:
    case T_R8_I:
    case T_R8G8B8:
    case T_B8G8R8:
    case T_A8B8G8R8:
    case T_A8R8G8B8:
    case T_B8G8R8A8:
    case T_R8G8B8A8:
    case T_X8B8G8R8:
    case T_X8R8G8B8:
    case T_B8G8R8X8:
    case T_R8G8B8X8:
    case D_F8_CHW_I:
    case D_F8_CxHWx_x32_I:
    case D_F8_CxHWx_x8_I:
        bpe = 1;
        break;
    case T_A16B16G16R16_F:
    case D_F16_CHW_I:
    case D_F16_CHW_F:
    case D_F16_CxHWx_x16_I:
    case D_F16_CxHWx_x16_F:
        bpe = 2;
        break;
    case D_F32_CHW_F:
    case D_F32_CxHWx_x8_F:
        bpe = 4;
        break;
    default:
        bpe = -1;
    }

    return bpe;
}

NvDlaImage::PixelFormatType NvDlaImage::getPixelFormatType() const
{
    PixelFormatType type = UNKNOWN;

    switch(m_meta.surfaceFormat)
    {
    case T_R8:
    case T_R10:
    case T_R12:
    case T_R16:
    case T_R8G8B8:
    case T_B8G8R8:
    case T_A16B16G16R16:
    case T_X16B16G16R16:
    case T_A16Y16U16V16:
    case T_V16U16Y16A16:
    case T_A8B8G8R8:
    case T_A8R8G8B8:
    case T_B8G8R8A8:
    case T_R8G8B8A8:
    case T_X8B8G8R8:
    case T_X8R8G8B8:
    case T_B8G8R8X8:
    case T_R8G8B8X8:
    case T_A2B10G10R10:
    case T_A2R10G10B10:
    case T_B10G10R10A2:
    case T_R10G10B10A2:
    case T_A2Y10U10V10:
    case T_V10U10Y10A2:
    case T_A8Y8U8V8:
    case T_V8U8Y8A8:
    case T_Y8___U8V8_N444:
    case T_Y8___V8U8_N444:
    case T_Y10___U10V10_N444:
    case T_Y10___V10U10_N444:
    case T_Y12___U12V12_N444:
    case T_Y12___V12U12_N444:
    case T_Y16___U16V16_N444:
    case T_Y16___V16U16_N444:
        type = UINT;
        break;
    case T_R8_I:
    case T_R16_I:
    case D_F8_CHW_I:
    case D_F16_CHW_I:
    case D_F8_CxHWx_x32_I:
    case D_F8_CxHWx_x8_I:
    case D_F16_CxHWx_x16_I:
        type = INT;
        break;

    case T_R16_F:
    case T_A16B16G16R16_F:
    case T_A16Y16U16V16_F:
    case D_F16_CHW_F:
    case D_F32_CHW_F:
    case D_F16_CxHWx_x16_F:
    case D_F32_CxHWx_x8_F:
        type = IEEEFP;
        break;
    default:
        type = UNKNOWN;
        break;
    }

    return type;
}

NvS32 NvDlaImage::getAddrOffset(NvU32 w, NvU32 h, NvU32 c) const
{
    NvS32 offset = -1;

    if (w >= static_cast<NvU32>(m_meta.width) || h >= static_cast<NvU32>(m_meta.height) || c >= static_cast<NvU32>(m_meta.channel))
    {
        REPORT_ERROR(NvDlaError_BadParameter);
        return -1;
    }

    NvS8 bpe = getBpe();
    if (bpe <= 0)
    {
        REPORT_ERROR(NvDlaError_BadParameter);
        return -1;
    }

    if (m_meta.surfaceFormat == T_R8)
    {
        if (c != 0)
        {
            REPORT_ERROR(NvDlaError_BadParameter);
            return -1;
        }

        offset = ((h * m_meta.lineStride) + w) * bpe;
    }
    else if (m_meta.surfaceFormat == T_B8G8R8 ||
             m_meta.surfaceFormat == T_R8G8B8)
    {
        offset = (h * m_meta.lineStride) + (w * m_meta.channel) + c;
    }
    else if (m_meta.surfaceFormat == T_A8B8G8R8 || m_meta.surfaceFormat == T_A16B16G16R16_F)
    {
        NvU32 x = 4;
        NvU32 xStride = x * bpe;
        offset = (h * m_meta.lineStride) + (w * xStride) + (c * bpe);
    }
    else if (m_meta.surfaceFormat == D_F8_CxHWx_x32_I)
    {
        NvU32 x = 32;
        NvU32 xStride = x * bpe;
        NvU32 cquotient = c / x;
        NvU32 cremainder = c % x;

        offset = (cquotient * m_meta.surfaceStride) + (h * m_meta.lineStride) + (w * xStride) + (cremainder * bpe);
    }
    else if (m_meta.surfaceFormat == D_F8_CxHWx_x8_I)
    {
        NvU32 x = 8;
        NvU32 xStride = x * bpe;
        NvU32 cquotient = c / x;
        NvU32 cremainder = c % x;

        offset = (cquotient * m_meta.surfaceStride) + (h * m_meta.lineStride) + (w * xStride) + (cremainder * bpe);
    }
    else if (m_meta.surfaceFormat == D_F16_CxHWx_x16_I ||
             m_meta.surfaceFormat == D_F16_CxHWx_x16_F)
    {
        NvU32 x = 16;
        NvU32 xStride = x * bpe;
        NvU32 cquotient = c / x;
        NvU32 cremainder = c % x;

        offset = (cquotient * m_meta.surfaceStride) + (h * m_meta.lineStride) + (w * xStride) + (cremainder * bpe);
    }
    else if (m_meta.surfaceFormat == D_F32_CxHWx_x8_F)
    {
        NvU32 x = 8;
        NvU32 xStride = x * bpe;
        NvU32 cquotient = c / x;
        NvU32 cremainder = c % x;

        offset = (cquotient * m_meta.surfaceStride) + (h * m_meta.lineStride) + (w * xStride) + (cremainder * bpe);
    }
    else if (m_meta.surfaceFormat == D_F8_CHW_I ||
             m_meta.surfaceFormat == D_F16_CHW_I ||
             m_meta.surfaceFormat == D_F16_CHW_F ||
             m_meta.surfaceFormat == D_F32_CHW_F)
    {
        offset = (c * m_meta.surfaceStride) + (h * m_meta.lineStride) + (w * bpe);
    }
    else
    {
        REPORT_ERROR(NvDlaError_BadParameter);
        return -1;
    }

    return offset;
}


NvDlaError NvDlaImage::serialize(std::stringstream& sstream, bool stableHash) const
{
    // Serialize header
    sstream << 'D' << 'I' << 'M' << 'G';
    sstream.write(reinterpret_cast<const char*>(&(NvDlaImage::ms_version)), sizeof(NvDlaImage::ms_version));

    // Serialize metadata
    sstream.write(reinterpret_cast<const char*>(&m_meta), sizeof(m_meta));

    // Serialize data
    PROPAGATE_ERROR(packData(sstream, stableHash, false));

    return NvDlaSuccess;
}

NvDlaError NvDlaImage::deserialize(std::stringstream& sstream)
{
    // Deserialize header
    char header[5];
    sstream.read(header, 4);
    header[4] = '\0';

    if (strcmp(header, "DIMG")!=0)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Unknown NvDlaImage header");

    NvU32 version;
    sstream.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != ms_version)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Mismatching NvDlaImage version %u: expected version %u", version, ms_version);

    // Deserialize metadata
    sstream.read(reinterpret_cast<char*>(&m_meta), sizeof(m_meta));

    // Deserialize data
    PROPAGATE_ERROR(unpackData(sstream));

    return NvDlaSuccess;
}

NvDlaError NvDlaImage::packData(std::stringstream& sstream, bool stableHash, bool asRaw) const
{
    NvS8 bpe = getBpe();
    if (bpe <= 0)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Invalid bytes per element %d", bpe);

    PixelFormatType pftype = getPixelFormatType();
    if (pftype == UNKNOWN)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Unknown pixel format type %u\n", pftype);

    /* Force stable hash if packing data need to raw (readable) */
    if (asRaw)
        stableHash = true;

    char* buf = reinterpret_cast<char*>(m_pData);
    for (NvU32 c=0; c<m_meta.channel; c++)
    {
        for (NvU32 y=0; y<m_meta.height; y++)
        {
            for (NvU32 x=0; x<m_meta.width; x++)
            {
                NvS32 offset = getAddrOffset(x, y, c);
                if (offset < 0)
                    ORIGINATE_ERROR(NvDlaError_BadValue, "Invalid getAddrOffset() => %d\n", offset);

                if (offset >= (NvS32)(m_meta.size)) {
                    ORIGINATE_ERROR(NvDlaError_BadValue, "offset > m_meta.size: %d >= %u\n",
                                    offset, m_meta.size);
                }

                if (stableHash)
                {
                    if (pftype == IEEEFP)
                    {
                        if (bpe == 2)
                        {
                            half_float::half tmp = *(reinterpret_cast<half_float::half*>(buf + offset));

                            if (tmp == 0x8000)
                            {
                                // Push negative zeros to positive zeros
                                tmp = 0x0000;
                            }
                            else if (std::isnan(tmp))
                            {
                                // Emit only one version of NaN
                                tmp = 0x7C00;
                            }

                            if (asRaw)
                                sstream << tmp << " ";
                            else
                                sstream.write(reinterpret_cast<const char*>(&tmp), bpe);
                        }
                        else if (bpe == 4)
                        {
                            // FP32
                            float tmp = *(reinterpret_cast<float*>(buf + offset));
                            if (tmp == 0x80000000)
                            {
                                // Push negative zeros to positive zeros
                                tmp = 0x0;
                            }
                            else if (std::isnan(tmp))
                            {
                                // Emit only one version of NaN
                                tmp = 0x7FBFFFFF;
                            }
                            if (asRaw)
                                sstream << tmp << " ";
                            else
                                sstream.write(reinterpret_cast<const char*>(&tmp), bpe);
                        }
                        else
                        {
                            ORIGINATE_ERROR(NvDlaError_NotSupported, "Unspported FP type");
                        }
                    }
                    else if (pftype == UINT)
                    {
                        unsigned tmp;
                        if (bpe == 1)
                            tmp = *(reinterpret_cast<NvU8*>(buf + offset));
                        else
                            tmp = *(reinterpret_cast<NvU16*>(buf + offset));

                        if (asRaw)
                            sstream << tmp << " ";
                        else
                            sstream.write(reinterpret_cast<const char*>(buf + offset), bpe);
                    }
                    else if (pftype == INT)
                    {
                        int tmp;
                        if (bpe == 1)
                            tmp = *(reinterpret_cast<NvS8*>(buf + offset));
                        else
                            tmp = *(reinterpret_cast<NvS16*>(buf + offset));

                        if (asRaw)
                            sstream << tmp << " ";
                        else
                            sstream.write(reinterpret_cast<const char*>(buf + offset), bpe);
                    }
                    else
                    {
                        ORIGINATE_ERROR(NvDlaError_NotSupported, "Unspported type %u", pftype);
                    }
                } else {
                    sstream.write(reinterpret_cast<const char*>(buf + offset), bpe);
                }
            }
        }
    }

    return NvDlaSuccess;
}

NvDlaError NvDlaImage::unpackData(std::stringstream& sstream)
{
    NvS8 bpe = getBpe();
    if (bpe <= 0)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Invalid bytes per element %d", bpe);

    PixelFormatType pftype = getPixelFormatType();
    if (pftype == UNKNOWN)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Unknown pixel format type %u\n", pftype);

    m_pData = NvDlaAlloc(m_meta.size);
    if (!m_pData)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    char* buf = reinterpret_cast<char*>(m_pData);
    for (NvU32 c=0; c<m_meta.channel; c++)
    {
        for (NvU32 y=0; y<m_meta.height; y++)
        {
            for (NvU32 x=0; x<m_meta.width; x++)
            {
                NvS32 offset = getAddrOffset(x, y, c);
                if (offset < 0)
                    ORIGINATE_ERROR(NvDlaError_BadValue, "Invalid getAddrOffset() => %d\n", offset);

                if (offset >= (NvS32)(m_meta.size)) {
                    ORIGINATE_ERROR(NvDlaError_BadValue, "offset > m_meta.size: %d >= %u\n",
                                    offset, m_meta.size);
                }

                sstream.read(reinterpret_cast<char*>(buf + offset), bpe);
            }
        }
    }

    return NvDlaSuccess;
}

NvDlaError NvDlaImage::printInfo() const
{
    NvDlaDebugPrintf("surfaceFormat %u\n", m_meta.surfaceFormat);
    NvDlaDebugPrintf("width %u\n", m_meta.width);
    NvDlaDebugPrintf("height %u\n", m_meta.height);
    NvDlaDebugPrintf("channel %u\n", m_meta.channel);
    NvDlaDebugPrintf("lineStride %u\n", m_meta.lineStride);
    NvDlaDebugPrintf("surfaceStride %u\n", m_meta.surfaceStride);
    NvDlaDebugPrintf("size %u\n", m_meta.size);

    return NvDlaSuccess;
}

NvDlaError NvDlaImage::printBuffer(bool showBorders) const
{
    NvDlaError e = NvDlaSuccess;

    NvS8 bpe = getBpe();
    if (bpe <= 0)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Invalid bytes per element %d", bpe);

    PixelFormatType pftype = getPixelFormatType();
    if (pftype == UNKNOWN)
        ORIGINATE_ERROR(NvDlaError_BadParameter, "Unknown pixel format type %u\n", pftype);

    NvU8 numCharsPerElement = 1;
    if (pftype == IEEEFP && bpe == 2)
    {
        // FP16
        numCharsPerElement = 8;
    }
    else
    {
        numCharsPerElement = 2 * bpe;
    }

    char* buf = reinterpret_cast<char*>(m_pData);
    for (NvU32 c=0; c<m_meta.channel; c++)
    {
        if (showBorders)
        {
            NvDlaDebugPrintf(" ");
            for (NvU32 x=0; x<m_meta.width * numCharsPerElement; x++)
            {
                NvDlaDebugPrintf("-");
            }
            NvDlaDebugPrintf("\n");
        }

        for (NvU32 y=0; y<m_meta.height; y++)
        {
            if (showBorders)
            {
                NvDlaDebugPrintf("|");
            }

            for (NvU32 x=0; x<m_meta.width; x++)
            {
                NvS32 offset = getAddrOffset(x, y, c);
                if (offset < 0)
                    ORIGINATE_ERROR(NvDlaError_BadValue, "Invalid getAddrOffset() => %d\n", offset);

                if (offset >= (NvS32)(m_meta.size)) {
                    NvDlaDebugPrintf("offset %d for (%u, %u, %u)@(%u, %u, %u)\n", offset,
                                    m_meta.channel, m_meta.height, m_meta.width,
                                    x, y, c);
                    ORIGINATE_ERROR(NvDlaError_BadValue,
                                    "offset > m_meta.width: %d >= %u", offset, m_meta.size );
                }

                if (pftype == IEEEFP)
                {
                    if (bpe == 2)
                    {
                        // FP16
                        half_float::half* halfp = reinterpret_cast<half_float::half*>(buf + offset);
                        char str[80];
                        int len = snprintf(str, sizeof(str), "%6.3f,", float(*halfp));

                        int padding_len = numCharsPerElement - len;
                        for (int ii=padding_len; ii>0; ii--)
                            strcat(str, " ");

                        NvDlaDebugPrintf("%s", str);
                    }
                    else if (bpe == 4)
                    {
                        // FP32
                        float* halfp = reinterpret_cast<float*>(buf + offset);
                        char str[80];
                        int len = snprintf(str, sizeof(str), "%6.3f,", float(*halfp));

                        int padding_len = numCharsPerElement - len;
                        for (int ii=padding_len; ii>0; ii--)
                            strcat(str, " ");

                        NvDlaDebugPrintf("%s", str);
                    }
                    else {
                        ORIGINATE_ERROR(NvDlaError_NotSupported);
                    }
                }
                else if (pftype == INT)
                {
                    if (bpe == 1)
                    {
                        // S8
                        NvDlaDebugPrintf("%4d ", *reinterpret_cast<NvS8*>(buf + offset));
                    }
                    else if (bpe == 2)
                    {
                        // S16
                        NvDlaDebugPrintf("%4d", *reinterpret_cast<NvS16*>(buf + offset));
                    }
                    else
                    {
                        ORIGINATE_ERROR(NvDlaError_NotSupported);
                    }
                }
                else if (pftype == UINT)
                {
                    if (bpe == 1)
                    {
                        // U8
                        NvDlaDebugPrintf("%4d", *reinterpret_cast<NvU8*>(buf + offset));
                    }
                    else if (bpe == 2)
                    {
                        // U16
                        NvDlaDebugPrintf("%4d", *reinterpret_cast<NvU16*>(buf + offset));
                    }
                    else
                    {
                        ORIGINATE_ERROR(NvDlaError_NotSupported);
                    }
                }
                else
                {
                    ORIGINATE_ERROR(NvDlaError_NotSupported);
                }
            }
            if (showBorders)
            {
                NvDlaDebugPrintf("|");
            }
            NvDlaDebugPrintf("\n");
        }

        if (showBorders)
        {
            NvDlaDebugPrintf(" ");
            for (NvU32 x=0; x<m_meta.width * numCharsPerElement; x++)
            {
                NvDlaDebugPrintf("-");
            }
            NvDlaDebugPrintf("\n");
        }
    }

    return e;
}
