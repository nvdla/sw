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

#include "main.h"

#include "nvdla/IProfile.h"
#include "nvdla/IProfiler.h"
#include "nvdla/IWisdom.h"
#include "nvdla/INetwork.h"
#include "nvdla/ICompiler.h"
#include "nvdla/ITargetConfig.h"

#include "ErrorMacros.h"
#include "nvdla_os_inf.h"

NvDlaError compileProfile(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    std::string profileName = "";
    std::string targetConfigName = "";

    NvDlaFileHandle file = 0;
    std::string fileName = "";
    NvU8 *buffer = 0;
    NvU64 size = 0;

    nvdla::ICompiler* compiler = i->wisdom->getCompiler();
    if (!compiler)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->getCompiler() failed");

    if (!(appArgs->configtarget != ""))
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No target config found to load");

    targetConfigName = appArgs->configtarget;

    // Determine profile
    PROPAGATE_ERROR_FAIL(generateProfile(appArgs, &profileName, i));

    // Compile
    NvDlaDebugPrintf("compiling profile \"%s\"... config \"%s\"...\n", profileName.c_str(), targetConfigName.c_str());
    PROPAGATE_ERROR_FAIL(compiler->compile(profileName.c_str(), targetConfigName.c_str(), &i->compiledLoadable));

    // Get loadable buffer and dump it into a file
    PROPAGATE_ERROR_FAIL(compiler->getLoadableImageSize(profileName.c_str(),
                                                    &size));
    if (size == 0) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                            "Invalid size for a loadable");
    }

    buffer = (NvU8 *) NvDlaAlloc(size);
    if (buffer == NULL) {
        ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory,
                            "Failed to allocate buffer for loadable");
    }
    PROPAGATE_ERROR_FAIL(compiler->getLoadableImage(profileName.c_str(),
                                                    buffer));
    fileName = profileName + ".nvdla";
    PROPAGATE_ERROR_FAIL(NvDlaFopen(fileName.c_str(), NVDLA_OPEN_WRITE, &file));
    PROPAGATE_ERROR_FAIL(NvDlaFwrite(file, buffer, size));

fail:
    NvDlaFclose(file);
    if (buffer != NULL)
        NvDlaFree(buffer);
    return e;
}

NvDlaError compile(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    i->compiledLoadable = 0;

    NvDlaDebugPrintf("creating new wisdom context...\n");
    i->wisdom = nvdla::createWisdom();
    if (!i->wisdom)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createWisdom() failed");

    NvDlaDebugPrintf("opening wisdom context...\n");
    if (!i->wisdom->open(i->wisdomPath))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->open() failed to open: \"%s\"", i->wisdomPath.c_str());

    // Compile
    PROPAGATE_ERROR_FAIL(compileProfile(appArgs, i));

    NvDlaDebugPrintf("closing wisdom context...\n");
    i->wisdom->close();

fail:
    if (i->wisdom != NULL) {
        nvdla::destroyWisdom(i->wisdom);
        i->wisdom = NULL;
    }
    return e;
}
