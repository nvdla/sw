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

#include "dlaerror.h"
#include "dlatypes.h"

#include "nvdla/IRuntime.h"
#include "DlaImageUtils.h"

#include "ErrorMacros.h"

#include "nvdla_inf.h"

#include <string>

enum TestImageTypes
{
    IMAGE_TYPE_PGM = 0,
    IMAGE_TYPE_UNKNOWN = 1,
};

NvDlaError launchTest(const TestAppArgs* appArgs);
NvDlaError testSetup(const TestAppArgs* appArgs, TestInfo* i);

// Test
NvDlaError run(const TestAppArgs* appArgs, TestInfo* i);

// Runtime
NvDlaError runSetup(const TestAppArgs* appArgs, TestInfo* i);
//NvDlaError runTeardown(const TestAppArgs* appArgs, TestInfo* i);
NvDlaError loadLoadable(const TestAppArgs* appArgs, TestInfo* i);
NvDlaError setupBuffers(const TestAppArgs* appArgs, TestInfo* i);
NvDlaError runTest(const TestAppArgs* appArgs, TestInfo* i);

// TestUtils
NvDlaError DIMG2DlaBuffer(const NvDlaImage* image, void** pBuffer);
NvDlaError DlaBuffer2DIMG(void** pBuffer, NvDlaImage* image);
//NvDlaError createSync(NvU32 value, NvU32 condition, nvdla::ISync** sync);
NvDlaError Tensor2DIMG(const nvdla::IRuntime::NvDlaTensor* pTDesc, NvDlaImage* image);
NvDlaError createFF16ImageCopy(NvDlaImage* in, NvDlaImage* out);
