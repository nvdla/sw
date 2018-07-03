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
#include "Server.h"

#include "nvdla_os_inf.h"

#include <cstring>
#include <iostream>
#include <cstdlib> // system

static TestAppArgs defaultTestAppArgs = TestAppArgs();

static NvDlaError testSetup(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    std::string imagePath = "";
    NvDlaStatType stat;

    // Do input paths exist?
    if (std::strcmp(appArgs->inputName.c_str(), "") != 0)
    {
        e = NvDlaStat(appArgs->inputPath.c_str(), &stat);
        if (e != NvDlaSuccess)
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Input path does not exist: \"%s\"", appArgs->inputPath.c_str());

        imagePath = /* appArgs->inputPath + "/images/" + */appArgs->inputName;
        e = NvDlaStat(imagePath.c_str(), &stat);
        if (e != NvDlaSuccess)
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Image path does not exist: \"%s/%s\"", imagePath.c_str());
    }

    return NvDlaSuccess;

fail:
    return e;
}

static NvDlaError launchServer(const TestAppArgs* appArgs)
{
    NvDlaError e = NvDlaSuccess;
    TestInfo testInfo;

    testInfo.dlaServerRunning = false;
    PROPAGATE_ERROR_FAIL(runServer(appArgs, &testInfo));

fail:
    return e;
}

static NvDlaError launchTest(const TestAppArgs* appArgs)
{
    NvDlaError e = NvDlaSuccess;
    TestInfo testInfo;

    testInfo.dlaServerRunning = false;
    PROPAGATE_ERROR_FAIL(testSetup(appArgs, &testInfo));

    PROPAGATE_ERROR_FAIL(run(appArgs, &testInfo));

fail:
    return e;
}

// This is the entry point to the application
int main(int argc, char* argv[])
{
    NvDlaError e = NvDlaError_TestApplicationFailed;
    TestAppArgs testAppArgs = defaultTestAppArgs;
    bool showHelp = false;
    bool unknownArg = false;
    bool missingArg = false;
    bool inputPathSet = false;
    bool serverMode = false;
    NVDLA_UNUSED(inputPathSet);

    NvS32 ii = 1;
    while(true)
    {
        if (ii >= argc)
            break;

        const char* arg = argv[ii];

        if (std::strcmp(arg, "-h") == 0) // help
        {
            // Print usage
            showHelp = true;
            break;
        }
        if (std::strcmp(arg, "-s") == 0) // server mode
        {
            // Print usage
            serverMode = true;
            break;
        }
        else if (std::strcmp(arg, "-i") == 0) // input path
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.inputPath = std::string(argv[++ii]);
            inputPathSet = true;
        }
        else if (std::strcmp(arg, "--image") == 0) // imagename
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.inputName = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--loadable") == 0)
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.loadableName = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--normalize") == 0) // normalize value
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.normalize_value = atoi(argv[++ii]);
        }
        else if (std::strcmp(arg, "--mean") == 0) // Mean
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }
            char *token;
            int i = 0;
            token = strtok(argv[++ii], ",\n");
            while( token != NULL ) {
                if (i > 3) {
                    NvDlaDebugPrintf("Number of mean values should not be greater than 4 \n");
                    showHelp = true;
                    break;
                }
                testAppArgs.mean[i] = atof(token);
                token = strtok(NULL, ",\n");
                i++;
            }
        }
        else if (std::strcmp(arg, "--rawdump") == 0)
        {
            testAppArgs.rawOutputDump = true;
        }
        else // unknown
        {
            // Unknown argument
            unknownArg = true;
            showHelp = true;
            break;
        }

        ii++;
    }

    /* Check if any mandatory arguments are missing */
    if (strcmp(testAppArgs.loadableName.c_str(), "") == 0 && !serverMode) {
        showHelp = true;
        missingArg = true;
    }

    if (showHelp)
    {
        NvDlaDebugPrintf("Usage: %s [-options] --loadable <loadable_file>\n", argv[0]);
        NvDlaDebugPrintf("where options include:\n");
        NvDlaDebugPrintf("    -h                    print this help message\n");
        NvDlaDebugPrintf("    -s                    launch test in server mode\n");
        NvDlaDebugPrintf("    --image <file>        input jpg/pgm file\n");
        NvDlaDebugPrintf("    --normalize <value>   normalize value for input image\n");
        NvDlaDebugPrintf("    --mean <value>        comma separated mean value for input image\n");
        NvDlaDebugPrintf("    --rawdump             dump raw dimg data\n");

        if (unknownArg || missingArg)
            return EXIT_FAILURE;
        else
            return EXIT_SUCCESS;
    }

    if (serverMode)
    {
        e = launchServer(&testAppArgs);
    }
    else
    {
        // Launch
        e = launchTest(&testAppArgs);
    }

    if (e != NvDlaSuccess)
    {
        return EXIT_FAILURE;
    }
    else
    {
        NvDlaDebugPrintf("Test pass\n");
        return EXIT_SUCCESS;
    }

    return EXIT_SUCCESS;
}
