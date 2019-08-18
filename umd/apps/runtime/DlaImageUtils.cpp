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

#include "DlaImage.h"
#include "DlaImageUtils.h"
#include "ErrorMacros.h"
#include "nvdla_os_inf.h"
#include "half.h"

extern "C" {
#include "jpeglib.h"
}

#include <fstream>
#include <algorithm>

static NvDlaError parsePGMInfo(std::ifstream& hFile, NvDlaImage* image);
static NvDlaError parsePGMData(std::ifstream& hFile, NvDlaImage* image);


static int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

NvDlaError PGM2DIMG(std::string inputfilename, NvDlaImage* output, nvdla::IRuntime::NvDlaTensor *tensorDesc)
{
    std::ifstream hFile(inputfilename.c_str());
    NvS8 bpe = -1;

    if (hFile.fail())
        ORIGINATE_ERROR(NvDlaError_FileOperationFailed, "File operation failed: \"%s\"", inputfilename.c_str());

    // Parse the header
    PROPAGATE_ERROR(parsePGMInfo(hFile, output));

    bpe = output->getBpe();
    if (bpe <= 0)
        ORIGINATE_ERROR(NvDlaError_BadParameter);

    output->m_meta.lineStride = tensorDesc->stride[1];
    output->m_meta.surfaceStride = tensorDesc->stride[2];;
    output->m_meta.size = tensorDesc->bufferSize;

    NvDlaDebugPrintf("pgm2dimg %d %d %d %d %d %d %d\n",
                    output->m_meta.channel,
                    output->m_meta.height,
                    output->m_meta.width,
                    bpe,
                    output->m_meta.lineStride,
                    output->m_meta.surfaceStride,
                    output->m_meta.size);

    // Allocate the buffer
    output->m_pData = NvDlaAlloc(output->m_meta.size);
    if (!output->m_pData)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    // Copy the data
    PROPAGATE_ERROR(parsePGMData(hFile, output));

    hFile.close();

    return NvDlaSuccess;
}

static NvDlaError parsePGMInfo(std::ifstream& hFile, NvDlaImage* image)
{
    NvU32 width;
    NvU32 height;
    NvU32 maxVal;
    std::istringstream iss;
    std::string header[3];
    NvU32 ii = 0;

    // Parse header, read 3 real ASCII lines
    while (ii < 3)
    {
        std::getline(hFile, header[ii]);

        // Pass over comments
        if (header[ii][0] == '#')
            continue;

        ii = ii + 1;
    }

    // Parse the magic word
    if (header[0].compare("P5") != 0)
        ORIGINATE_ERROR(NvDlaError_BadValue, "Unexpected PGM value: %s", header[0].c_str());

    // Gather the width and height parameters
    iss.str(header[1]);
    if (!(iss >> width >> height))
        ORIGINATE_ERROR(NvDlaError_BadValue, "Unexpected PGM value: %s", header[1].c_str());
    iss.clear();

    // Gather the max value
    iss.str(header[2]);
    if (!(iss >> maxVal))
        ORIGINATE_ERROR(NvDlaError_BadValue, "Unexpected PGM value: %s", header[2].c_str());
    iss.clear();

    // We only support .pgm's with maxVal == 255
    if (maxVal != 255)
        ORIGINATE_ERROR(NvDlaError_BadValue, "Unexpected PGM value: %s", header[2].c_str());

    // Transfer information
    image->m_meta.surfaceFormat = NvDlaImage::T_R8;
    image->m_meta.width = width;
    image->m_meta.height = height;
    image->m_meta.channel = 1;

#if 0
    NvDlaDebugPrintf("ppgminfo %d %d %d\n",
                    image->m_meta.channel,
                    image->m_meta.height,
                    image->m_meta.width);
#endif


    return NvDlaSuccess;
}

static NvDlaError parsePGMData(std::ifstream& hFile, NvDlaImage* image)
{
    void* buf = image->m_pData;

    // Clear contents
    memset(buf, 0, image->m_meta.size);

    // Copy contents
    for (NvU32 y=0; y<image->m_meta.height; y++)
    {
        char* linebuf = reinterpret_cast<char*>(buf) + (y * image->m_meta.lineStride);
        hFile.read(linebuf, image->m_meta.width);

        if (hFile.fail())
             ORIGINATE_ERROR(NvDlaError_FileOperationFailed, "File operation failed");
    }

    return NvDlaSuccess;
}

NvDlaError JPEG2DIMG(std::string inputFileName, NvDlaImage* output, nvdla::IRuntime::NvDlaTensor *tensorDesc)
{
    NvDlaError e = NvDlaSuccess;

    NvU8* rowPtr[1];
    struct jpeg_decompress_struct info;
    struct jpeg_error_mgr err;

    FILE* fp = fopen(inputFileName.c_str(), "rb");
    if (!fp) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Cant open file %s", inputFileName.c_str());
    }

    // Prepare for jpeg decomp
    info.err = jpeg_std_error(&err);
    info.dct_method = JDCT_ISLOW;
    jpeg_create_decompress(&info);

    jpeg_stdio_src(&info, fp);
    jpeg_read_header(&info, true);

    // FIXME: right now assuming 8-bit jpegs
    switch(info.jpeg_color_space) {
        case JCS_GRAYSCALE:
            output->m_meta.surfaceFormat = NvDlaImage::T_R8;
            output->m_meta.channel = 1;
            break;
#if defined(JCS_EXTENSIONS)
        case JCS_YCbCr: // FIXME: dont know how to handle compression yet
        case JCS_RGB:
            info.out_color_space = JCS_EXT_BGR;                    // FIXME: currently extracting as BGR (since caffe ref model assumes BGR)
            output->m_meta.surfaceFormat = NvDlaImage::T_B8G8R8;
            output->m_meta.channel = 3;
            break;
        case JCS_EXT_RGB: // upsizing to 4 Chnls from 3 Chnls
        case JCS_EXT_RGBX:
            output->m_meta.surfaceFormat = NvDlaImage::T_R8G8B8X8;
            output->m_meta.channel = 4;
            break;
        case JCS_EXT_BGR: // upsizing to 4 Chnls from 3 Chnls
        case JCS_EXT_BGRX:
            output->m_meta.surfaceFormat = NvDlaImage::T_B8G8R8X8;
            output->m_meta.channel = 4;
            break;
        case JCS_EXT_XBGR:
            output->m_meta.surfaceFormat = NvDlaImage::T_X8B8G8R8;
            output->m_meta.channel = 4;
            break;
        case JCS_EXT_XRGB:
            output->m_meta.surfaceFormat = NvDlaImage::T_X8R8G8B8;
            output->m_meta.channel = 4;
            break;
#if defined(JCS_ALPHA_EXTENSIONS)
        case JCS_EXT_RGBA:
            output->m_meta.surfaceFormat = NvDlaImage::T_R8G8B8A8;
            output->m_meta.channel = 4;
            break;
        case JCS_EXT_BGRA:
            output->m_meta.surfaceFormat = NvDlaImage::T_B8G8R8A8;
            output->m_meta.channel = 4;
            break;
        case JCS_EXT_ABGR:
            output->m_meta.surfaceFormat = NvDlaImage::T_A8B8G8R8;
            output->m_meta.channel = 4;
            break;
        case JCS_EXT_ARGB:
            output->m_meta.surfaceFormat = NvDlaImage::T_A8R8G8B8;
            output->m_meta.channel = 4;
            break;
#endif  // JCS_ALPHA_EXTENSIONS
#endif  // JCS_EXTENSIONS
        case JCS_CMYK:
        case JCS_YCCK:
        default: ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "JPEG color space %d not supported", info.jpeg_color_space);
    }


    jpeg_start_decompress(&info);

    // Read image size
    output->m_meta.height = info.image_height;
    output->m_meta.width = info.image_width;
    output->m_meta.lineStride = tensorDesc->stride[1];
    output->m_meta.surfaceStride = 0;
    output->m_meta.size = tensorDesc->bufferSize;

    NvDlaDebugPrintf("dlaimg height: %d x %d x %d: ", output->m_meta.height, output->m_meta.width, output->m_meta.channel);
    NvDlaDebugPrintf("LS: %d SS: %d Size: %d\n", output->m_meta.lineStride, output->m_meta.surfaceStride, output->m_meta.size);

    // Allocate the buffer
    output->m_pData = NvDlaAlloc(output->m_meta.size);
    NvDlaMemset(output->m_pData, 0, output->m_meta.size);
    if (!output->m_pData)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    // copy the data
    {
        rowPtr[0] = static_cast<NvU8*>(output->m_pData);
        while (info.output_scanline < info.output_height) {
            jpeg_read_scanlines(&info, rowPtr, 1);
            rowPtr[0] += output->m_meta.lineStride;
        }
    }

fail:
    jpeg_finish_decompress(&info);
    fclose(fp);
    return e;
}

NvDlaError DIMG2DIMGFile(const NvDlaImage* input, std::string outputfilename, bool stableHash, bool rawDump)
{
    // Prepare NvDlaImage serialization stream
    std::stringstream sstream;

    if (rawDump)
        PROPAGATE_ERROR(input->packData(sstream, stableHash, true));
    else
        PROPAGATE_ERROR(input->serialize(sstream, stableHash));

    const std::string& str = sstream.str();

    std::ofstream file;
    file.open(outputfilename.c_str(), std::ofstream::binary);
    file.write(str.c_str(), str.length());
    file.close();

    return NvDlaSuccess;
}

NvDlaError DIMGFile2DIMG(std::string inputfilename, NvDlaImage* output)
{
    // Prepare NvDlaImage deserialization stream
    std::stringstream sstream;

    std::ifstream file;
    file.open(inputfilename.c_str(), std::ifstream::binary);
    copy(std::istreambuf_iterator<char>(file),
         std::istreambuf_iterator<char>(),
         std::ostreambuf_iterator<char>(sstream));

    PROPAGATE_ERROR(output->deserialize(sstream));

    file.close();

    return NvDlaSuccess;
}
