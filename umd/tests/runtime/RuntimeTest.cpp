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

#define OUTPUT_DIMG "output.dimg"

using namespace half_float;

static TestImageTypes getImageType(std::string imageFileName)
{
    TestImageTypes it = IMAGE_TYPE_UNKNOWN;
    std::string ext = imageFileName.substr(imageFileName.find_last_of(".") + 1);
    if (ext == "pgm")
    {
        it = IMAGE_TYPE_PGM;
    }
    else if (ext == "jpg")
    {
        it = IMAGE_TYPE_JPG;
    }

    return it;
}

static NvDlaError copyImageToInputTensor
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pImgBuffer
)
{
    NvDlaError e = NvDlaSuccess;

    std::string imgPath = /*i->inputImagesPath + */appArgs->inputName;
    NvDlaImage* R8Image = new NvDlaImage();
    NvDlaImage* FF16Image = NULL;
    TestImageTypes imageType = getImageType(imgPath);
    if (!R8Image)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    switch(imageType) {
        case IMAGE_TYPE_PGM:
            PROPAGATE_ERROR(PGM2DIMG(imgPath, R8Image));
            break;
        case IMAGE_TYPE_JPG:
            PROPAGATE_ERROR(JPEG2DIMG(imgPath, R8Image));
            break;
        default:
            //TODO Fix this error condition
//          ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown image type: %s", imgPath.c_str());
            NvDlaDebugPrintf("Unknown image type: %s", imgPath.c_str());
            goto fail;
    }

    FF16Image = i->inputImage;
    if (FF16Image == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL input Image");
    PROPAGATE_ERROR(createFF16ImageCopy(appArgs, R8Image, FF16Image));
    PROPAGATE_ERROR(DIMG2DlaBuffer(FF16Image, pImgBuffer));

fail:
    if (R8Image != NULL && R8Image->m_pData != NULL)
        NvDlaFree(R8Image->m_pData);
    delete R8Image;

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
    void** pInputBuffer
)
{
    NvDlaError e = NvDlaSuccess;
    void *hMem = NULL;
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
    i->inputHandle = (NvU8 *)hMem;
    PROPAGATE_ERROR_FAIL(copyImageToInputTensor(appArgs, i, pInputBuffer));

    if (!runtime->bindInputTensor(0, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindInputTensor() failed");

fail:
    return e;
}

static void cleanupInputBuffer(const TestAppArgs *appArgs,
                                TestInfo *i)
{
    nvdla::IRuntime *runtime = NULL;
    NvS32 numInputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaError e = NvDlaSuccess;

    if (i->inputImage != NULL && i->inputImage->m_pData != NULL) {
        NvDlaFree(i->inputImage->m_pData);
        i->inputImage->m_pData = NULL;
    }

    runtime = i->runtime;
    if (runtime == NULL)
        return;
    e = runtime->getNumInputTensors(&numInputTensors);
    if (e != NvDlaSuccess)
        return;

    if (numInputTensors < 1)
        return;

    e = runtime->getInputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess)
        return;

    if (i->inputHandle == NULL)
        return;

    /* Free the buffer allocated */
    runtime->freeSystemMemory(i->inputHandle, tDesc.bufferSize);
    i->inputHandle = NULL;
    return;
}

NvDlaError setupOutputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pOutputBuffer
)
{
    NVDLA_UNUSED(appArgs);

    NvDlaError e = NvDlaSuccess;
    void *hMem;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaImage *pOutputImage = NULL;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    PROPAGATE_ERROR_FAIL(runtime->getNumOutputTensors(&numOutputTensors));

    i->numOutputs = numOutputTensors;

    if (numOutputTensors < 1)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Expected number of output tensors of %u, found %u", 1, numOutputTensors);

    PROPAGATE_ERROR_FAIL(runtime->getOutputTensorDesc(0, &tDesc));
    PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pOutputBuffer));
    i->outputHandle = (NvU8 *)hMem;

    pOutputImage = i->outputImage;
    if (i->outputImage == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL Output image");
    PROPAGATE_ERROR_FAIL(prepareOutputTensor(&tDesc, pOutputImage, pOutputBuffer));

    if (!runtime->bindOutputTensor(0, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindOutputTensor() failed");

fail:
    return e;
}

static void cleanupOutputBuffer(const TestAppArgs *appArgs,
                                TestInfo *i)
{
    nvdla::IRuntime *runtime = NULL;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaError e = NvDlaSuccess;

    /* Do not clear outputImage if in server mode */
    if (!i->dlaServerRunning &&
            i->outputImage != NULL &&
            i->outputImage->m_pData != NULL) {
        NvDlaFree(i->outputImage->m_pData);
        i->outputImage->m_pData = NULL;
    }

    runtime = i->runtime;
    if (runtime == NULL)
        return;
    e = runtime->getNumOutputTensors(&numOutputTensors);
    if (e != NvDlaSuccess)
        return;
    e = runtime->getOutputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess)
        return;

    if (i->outputHandle == NULL)
        return;

    /* Free the buffer allocated */
    runtime->freeSystemMemory(i->outputHandle, tDesc.bufferSize);
    i->outputHandle = NULL;
    return;
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

void unloadLoadable(const TestAppArgs* appArgs, TestInfo *i)
{
    NVDLA_UNUSED(appArgs);
    nvdla::IRuntime *runtime = NULL;

    runtime = i->runtime;
    if (runtime != NULL) {
        runtime->unload();
    }
}

NvDlaError runTest(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    void* pInputBuffer = NULL;
    void* pOutputBuffer = NULL;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    i->inputImage = new NvDlaImage();
    i->outputImage = new NvDlaImage();

    PROPAGATE_ERROR_FAIL(setupInputBuffer(appArgs, i, &pInputBuffer));

    PROPAGATE_ERROR_FAIL(setupOutputBuffer(appArgs, i, &pOutputBuffer));
    NvDlaDebugPrintf("submitting tasks...\n");
    if (!runtime->submit())
        ORIGINATE_ERROR(NvDlaError_BadParameter, "runtime->submit() failed");

    PROPAGATE_ERROR_FAIL(DlaBuffer2DIMG(&pOutputBuffer, i->outputImage));

    /* Dump output dimg to a file */
    PROPAGATE_ERROR_FAIL(DIMG2DIMGFile(i->outputImage,
                                        OUTPUT_DIMG,
                                        true,
                                        appArgs->rawOutputDump));

fail:
    cleanupOutputBuffer(appArgs, i);
    /* Do not clear outputImage if in server mode */
    if (!i->dlaServerRunning && i->outputImage != NULL) {
        delete i->outputImage;
        i->outputImage = NULL;
    }

    cleanupInputBuffer(appArgs, i);
    if (i->inputImage != NULL) {
        delete i->inputImage;
        i->inputImage = NULL;
    }

    return e;
}

NvDlaError run(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    /* Create runtime instance */
    NvDlaDebugPrintf("creating new runtime context...\n");
    i->runtime = nvdla::createRuntime();
    if (i->runtime == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createRuntime() failed");

    if (!i->dlaServerRunning)
        PROPAGATE_ERROR_FAIL(readLoadable(appArgs, i));

    /* Load loadable */
    PROPAGATE_ERROR_FAIL(loadLoadable(appArgs, i));

    /* Start emulator */
    if (!i->runtime->initEMU())
        ORIGINATE_ERROR(NvDlaError_DeviceNotFound, "runtime->initEMU() failed");

    /* Run test */
    PROPAGATE_ERROR_FAIL(runTest(appArgs, i));

fail:
    /* Stop emulator */
    if (i->runtime != NULL)
        i->runtime->stopEMU();

    /* Unload loadables */
    unloadLoadable(appArgs, i);

    /* Free if allocated in read Loadable */
    if (!i->dlaServerRunning && i->pData != NULL) {
        delete[] i->pData;
        i->pData = NULL;
    }

    /* Destroy runtime */
    nvdla::destroyRuntime(i->runtime);

    return e;
}
