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

#include "DlaImage.h"
#include "ErrorMacros.h"

#include "nvdla_inf.h"

#include <string>
#include <vector>

struct TestAppArgs
{
    std::string inputPath;
    std::string inputName;
    std::string loadableName;
    NvS32 serverPort;
    bool performSoftwareSoftmax;
    NvF32 imgShift;
    NvF32 imgScalingFactor;
    NvF32 imgPowerFactor;

    TestAppArgs() :
        inputPath("./"),
        inputName(""),
        loadableName(""),
        serverPort(6666),
        performSoftwareSoftmax(false),
        imgShift(0.0),
        imgScalingFactor(0.0),
        imgPowerFactor(0.0)
    {}
};

struct TestInfo
{
    TestInfo() :
        runtime(NULL),
        inputLoadablePath(""),
        pData(NULL),
        dlaServerRunning(false),
        performSoftwareSoftmax(false),
        imgShift(0.0f),
        imgScalingFactor(1.0f),
        imgPowerFactor(1.0f),
        dlaRemoteSock(-1),
        dlaServerSock(-1),
        numInputs(0),
        numOutputs(0),
        inputImage(NULL),
        outputImage(NULL)
    {}
    // runtime
    nvdla::IRuntime* runtime;
    std::string inputLoadablePath;
    NvU8 *pData;
    bool dlaServerRunning;
    bool performSoftwareSoftmax;
    NvF32 imgShift;
    NvF32 imgScalingFactor;
    NvF32 imgPowerFactor;
    NvS32 dlaRemoteSock;
    NvS32 dlaServerSock;
    NvU32 numInputs;
    NvU32 numOutputs;
    NvDlaImage* inputImage;
    NvDlaImage* outputImage;
};

// Test
NvDlaError run(const TestAppArgs* appArgs, TestInfo* i);
