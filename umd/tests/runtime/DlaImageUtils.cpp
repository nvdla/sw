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

#include "DlaImage.h"
#include "DlaImageUtils.h"
#include "ErrorMacros.h"
#include "nvdla_os_inf.h"
#include "half.h"

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

NvDlaError PGM2DIMG(std::string inputfilename, NvDlaImage* output)
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

    // Enforce stride and size alignment of 32B
    NvU32 strideAlign = 32;
    NvU32 sizeAlign = 32;
    output->m_meta.lineStride = roundUp(output->m_meta.width * output->m_meta.channel * bpe, strideAlign);
    output->m_meta.surfaceStride = roundUp(output->m_meta.lineStride * output->m_meta.height, strideAlign);
    output->m_meta.size = roundUp(output->m_meta.surfaceStride, sizeAlign);

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

    NvDlaDebugPrintf("ppgminfo %d %d %d\n",
                    image->m_meta.channel,
                    image->m_meta.height,
                    image->m_meta.width);

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

NvDlaError DIMG2DIMGFile(const NvDlaImage* input, std::string outputfilename, bool stableHash)
{
    // Prepare NvDlaImage serialization stream
    std::stringstream sstream;

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
