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

#include "DlaImageUtils.h"
#include "ErrorMacros.h"
#include "RuntimeTest.h"

#include "nvdla/IRuntime.h"

#include "half.h"
#include "main.h"
#include "nvdla_os_inf.h"

#include "dlaerror.h"
#include "dlatypes.h"

#include <cstdio> // snprintf, fopen
#include <string>

using namespace half_float;

static TestImageTypes getImageType(std::string imageFileName)
{
    TestImageTypes it = IMAGE_TYPE_UNKNOWN;
    std::string ext = imageFileName.substr(imageFileName.find_last_of(".") + 1);
    if (ext == "pgm")
    {
        it = IMAGE_TYPE_PGM;
    }

    return it;
}

static NvDlaError copyImageToInputTensor
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pImgBuffer,
    NvDlaImage* pInputImage
)
{
    NvDlaError e = NvDlaSuccess;

    std::string imgPath = /*i->inputImagesPath + */appArgs->inputName;
    NvDlaImage* R8Image = new NvDlaImage();
    NvDlaImage* FF16Image = pInputImage;
    TestImageTypes imageType = getImageType(imgPath);
    if (!R8Image)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);
    if (!FF16Image)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    switch(imageType) {
        case IMAGE_TYPE_PGM:
            PROPAGATE_ERROR(PGM2DIMG(imgPath, R8Image));
            break;
        default:
            //TODO Fix this error condition
//          ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown image type: %s", imgPath.c_str());
            NvDlaDebugPrintf("Unknown image type: %s", imgPath.c_str());
            goto fail;
    }

    PROPAGATE_ERROR(createFF16ImageCopy(R8Image, FF16Image));
    PROPAGATE_ERROR(DIMG2DlaBuffer(FF16Image, pImgBuffer));

fail:
    return e;
}

static NvDlaError prepareOutputTensor
(
    nvdla::IRuntime::NvDlaTensor* pTDesc,
    NvDlaImage* pOutImage,
    void** pOutBuffer
)
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(Tensor2DIMG(pTDesc, pOutImage));
    PROPAGATE_ERROR_FAIL(DIMG2DlaBuffer(pOutImage, pOutBuffer));

fail:
    return e;
}


NvDlaError setupInputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pInputBuffer,
    NvDlaImage* pInputImage
)
{
    NvDlaError e = NvDlaSuccess;
    void *hMem;
    NvS32 numInputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    PROPAGATE_ERROR_FAIL(runtime->getNumInputTensors(&numInputTensors));

    i->numInputs = numInputTensors;

    if (numInputTensors < 1)
        goto fail;

    PROPAGATE_ERROR_FAIL(runtime->getInputTensorDesc(0, &tDesc));
    PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pInputBuffer));
    PROPAGATE_ERROR_FAIL(copyImageToInputTensor(appArgs, i, pInputBuffer, pInputImage));

    if (!runtime->bindInputTensor(0, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindInputTensor() failed");

fail:
    return e;
}

NvDlaError setupOutputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pOutputBuffer,
    NvDlaImage* pOutputImage
)
{
    NVDLA_UNUSED(appArgs);

    NvDlaError e = NvDlaSuccess;
    void *hMem;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    PROPAGATE_ERROR_FAIL(runtime->getNumOutputTensors(&numOutputTensors));

    i->numOutputs = numOutputTensors;

    if (numOutputTensors < 1)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Expected number of output tensors of %u, found %u", 1, numOutputTensors);

    PROPAGATE_ERROR_FAIL(runtime->getOutputTensorDesc(0, &tDesc));
    PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pOutputBuffer));
    PROPAGATE_ERROR_FAIL(prepareOutputTensor(&tDesc, pOutputImage, pOutputBuffer));

    if (!runtime->bindOutputTensor(0, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindOutputTensor() failed");

fail:
    return e;
}

static NvDlaError readLoadable(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(appArgs);
    std::string loadableName;
    NvDlaFileHandle file;
    NvDlaStatType finfo;
    size_t file_size;
    NvU8 *buf = 0;
    size_t actually_read = 0;
    NvDlaError rc;

    // Determine loadable path
    if (appArgs->loadableName == "")
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No loadable found to load");
    }

    loadableName = appArgs->loadableName;

    rc = NvDlaFopen(loadableName.c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "couldn't open %s\n", loadableName.c_str());
    }

    rc = NvDlaFstat(file, &finfo);
    if ( rc != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "couldn't get file stats for %s\n", loadableName.c_str());
    }

    file_size = NvDlaStatGetSize(&finfo);
    if ( !file_size ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "zero-length for %s\n", loadableName.c_str());
    }

    buf = new NvU8[file_size];

    NvDlaFseek(file, 0, NvDlaSeek_Set);

    rc = NvDlaFread(file, buf, file_size, &actually_read);
    if ( rc != NvDlaSuccess )
    {
        NvDlaFree(buf);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read error for %s\n", loadableName.c_str());
    }
    NvDlaFclose(file);
    if ( actually_read != file_size ) {
        NvDlaFree(buf);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read wrong size for buffer? %d\n", actually_read);
    }

    i->pData = buf;

fail:
    return e;
}

NvDlaError loadLoadable(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    if (!runtime->load(i->pData, 0))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->load failed");

fail:
    return e;
}

static NvDlaError preProcessTest(const TestAppArgs* appArgs, TestInfo* i, NvDlaImage* pInput)
{
    NvDlaError e = NvDlaSuccess;
    half* pSrc = reinterpret_cast<half*>(pInput->m_pData);
    half* pDst = reinterpret_cast<half*>(NvDlaAlloc(pInput->m_meta.size));

    // Execute
    for (NvU32 channel=0; channel < pInput->m_meta.channel; channel++)
    {
        for (NvU32 height=0; height < pInput->m_meta.height; height++)
        {
            for (NvU32 width=0; width < pInput->m_meta.width; width++)
            {
                NvS32 srcoffset = pInput->getAddrOffset(width, height, channel);
                NvS32 dstoffset = pInput->getAddrOffset(width, height, channel);

                if (srcoffset == -1 || dstoffset == -1)
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Offsets out of bounds");
                }

                half* srchalfp = reinterpret_cast<half*>(pSrc + srcoffset);
                half* dsthalfp = reinterpret_cast<half*>(pDst + dstoffset);

                NvF32 x = float(*srchalfp);
                NvF32 y = powf(i->imgShift + (i->imgScalingFactor * x), i->imgPowerFactor);
                *dsthalfp = half(y);
            }
        }
    }

    pInput->m_pData = pDst;

fail:
    return e;
}

static NvDlaError postProcessTest(const TestAppArgs* appArgs, TestInfo* i, NvDlaImage* pOutput)
{
    NvDlaError e = NvDlaSuccess;

    if (i->performSoftwareSoftmax)
    {
        half* pSrc = reinterpret_cast<half*>(pOutput->m_pData);
        half* pDst = reinterpret_cast<half*>(NvDlaAlloc(pOutput->m_meta.size));

        NvF32 maxval = -INFINITY;
        for (NvU32 ii=0; ii < pOutput->m_meta.channel; ii++)
        {
            if (float(pSrc[ii]) > maxval)
            {
                maxval = float(pSrc[ii]);
            }
        }
        NvF32 sumexp = 0.0f;
        for (NvU32 ii=0; ii < pOutput->m_meta.channel; ii++)
        {
            sumexp += expf(float(pSrc[ii])-maxval);
        }
        for (NvU32 ii=0; ii < pOutput->m_meta.channel; ii++)
        {
            pDst[ii] = expf(float(pSrc[ii])-maxval) / sumexp;
        }

        pOutput->m_pData = pDst;
    }

    return e;
}

NvDlaError runTest(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    void* pInputBuffer = NULL;
    void* pOutputBuffer = NULL;
    NvDlaImage* pInputImage = NULL;
    NvDlaImage* pOutputImage = NULL;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    pInputImage  = new NvDlaImage();
    pOutputImage = new NvDlaImage();

    i->inputImage = pInputImage;
    i->outputImage = pOutputImage;

    PROPAGATE_ERROR_FAIL(setupInputBuffer(appArgs, i, &pInputBuffer, pInputImage));

    PROPAGATE_ERROR_FAIL(setupOutputBuffer(appArgs, i, &pOutputBuffer, pOutputImage));

    PROPAGATE_ERROR_FAIL(preProcessTest(appArgs, i, pInputImage));

    NvDlaDebugPrintf("submitting tasks...\n");
    if (!runtime->submit())
        ORIGINATE_ERROR(NvDlaError_BadParameter, "runtime->submit() failed");

    PROPAGATE_ERROR_FAIL(DlaBuffer2DIMG(&pOutputBuffer, pOutputImage));

    PROPAGATE_ERROR_FAIL(postProcessTest(appArgs, i, pOutputImage));

fail:
    return e;
}

NvDlaError run(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    NvDlaDebugPrintf("creating new runtime context...\n");
    i->runtime = nvdla::createRuntime();
    if (!i->runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createRuntime() failed");

    if ( !i->dlaServerRunning )
        PROPAGATE_ERROR_FAIL(readLoadable(appArgs, i));

    PROPAGATE_ERROR_FAIL(loadLoadable(appArgs, i));

    PROPAGATE_ERROR_FAIL(runTest(appArgs, i));
fail:
    return e;
}
