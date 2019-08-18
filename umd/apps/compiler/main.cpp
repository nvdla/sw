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
#include "ErrorMacros.h"

#include "nvdla_os_inf.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <cstdlib> // system

#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_DATA_FMT nvdla::DataFormat::NHWC
#define DEFAULT_QUANT_MODE nvdla::QuantizationMode::NONE
#define TARGET_CONFIG_NAME "nv_full"

static TestAppArgs defaultTestAppArgs =
{
    /* .project = */ "OpenDLA",
    /* .inputPath = */ "./",
    /* .inputName = */ "",
    /* .outputPath = */ "./",
    /* .testname = */ "",
    /* .testArgs = */ "",
    /* .prototxt = */ "",
    /* .caffemodel = */ "",
    /* .cachemodel = */ "",
    /* .profileName = */ "fast-math",
    /* .profileFile = */ "",
    /* .configtarget = */ TARGET_CONFIG_NAME,
    /* .calibtable = */ "",
    /* .quantizationMode = */ DEFAULT_QUANT_MODE,
    /* .numBatches = */ DEFAULT_BATCH_SIZE,
    /* .inDataFormat = */ DEFAULT_DATA_FMT,
    /* .computePrecision = */ nvdla::DataType::INT8
};

NvDlaError testSetup(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    std::string wisdomPath = appArgs->outputPath + "wisdom.dir/";
    std::string removeCmd = "";
    std::string imagePath = "";
    NvDlaStatType stat;
    int ii = 0;

    // Do input paths exist?
    e = NvDlaStat(appArgs->inputPath.c_str(), &stat);
    if (e != NvDlaSuccess)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Input path does not exist: \"%s\"", appArgs->inputPath.c_str());

    // Do output paths exist?
    e = NvDlaStat(appArgs->outputPath.c_str(), &stat);
    if (e != NvDlaSuccess)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Output path does not exist: \"%s\"", appArgs->outputPath.c_str());

    // Clear wisdomPath if any exist
    removeCmd += "rm -rf " + wisdomPath;
    ii = std::system(removeCmd.c_str()); // This is pretty awful
    if (ii != 0)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "system command failed: \"%s\"", removeCmd.c_str());

    PROPAGATE_ERROR_FAIL(NvDlaMkdir(const_cast<char *>(wisdomPath.c_str())));

    // Initialize TestInfo
    i->wisdom = NULL;
    i->wisdomPath = wisdomPath;
    i->pData = NULL;

    return NvDlaSuccess;

fail:
    return e;
}

NvDlaError launchTest(const TestAppArgs* appArgs)
{
    NvDlaError e = NvDlaSuccess;
    TestInfo testInfo;

    PROPAGATE_ERROR_FAIL(testSetup(appArgs, &testInfo));

    PROPAGATE_ERROR_FAIL(parseAndCompile(appArgs, &testInfo));

    return NvDlaSuccess;

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
    bool outputPathSet = false;
    bool testnameSet = false;
    NVDLA_UNUSED(inputPathSet);
    NVDLA_UNUSED(outputPathSet);
    NVDLA_UNUSED(testnameSet);

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
        else if (std::strcmp(arg, "-P") == 0) // project
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.project = std::string(argv[++ii]);
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
        else if (std::strcmp(arg, "-o") == 0) // output path
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.outputPath = std::string(argv[++ii]);
            outputPathSet = true;
        }
        else if (std::strcmp(arg, "-t") == 0) // testname
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.testname = std::string(argv[++ii]);
            testnameSet = true;
        }
        else if (std::strcmp(arg, "--prototxt") == 0) // prototxt file
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.prototxt = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--caffemodel") == 0) // caffemodel file
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.caffemodel = std::string(argv[++ii]);
        }
        else if (strcmp(arg, "--cachemodel") == 0) // cachemodel file
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.cachemodel = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--configtarget") == 0) // configtargetname
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.configtarget = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--profile") == 0) // named profile
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }
            if (testAppArgs.profileFile != "") {
                NvDlaDebugPrintf("Profile already set by --profilecfg argument (%s)\n", testAppArgs.profileFile.c_str());
                showHelp = true;
                unknownArg = true;
                break;
            }
            testAppArgs.profileName = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--profilecfg") == 0) // profile from file
        {
            if (ii+1 >= argc)
            {
                // expecting another parameter
                showHelp = true;
                break;
            }
            if (testAppArgs.profileName != "") {
                NvDlaDebugPrintf("Profile already set by --profile argument (%s)\n", testAppArgs.profileName.c_str());
                showHelp = true;
                unknownArg = true;
                break;
            }
            testAppArgs.profileFile = std::string(argv[++ii]);
        }
        else if (std::strcmp(arg, "--batch") == 0)
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            testAppArgs.numBatches = atoi(argv[++ii]);
        }
        else if (std::strcmp(arg, "--quantizationMode") == 0)
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            std::string quantMode = std::string(argv[++ii]);
            std::transform(quantMode.begin(), quantMode.end(), quantMode.begin(), ::tolower);

            if (quantMode == "per-kernel")
            {
                testAppArgs.quantizationMode = nvdla::QuantizationMode::PER_KERNEL;
            }
            else if (quantMode == "per-filter")
            {
                testAppArgs.quantizationMode = nvdla::QuantizationMode::PER_FILTER;
            }
            else
            {
                testAppArgs.quantizationMode = nvdla::QuantizationMode::NONE;
            }
        }
        else if (std::strcmp(arg, "--informat") == 0)
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            std::string inDataFormat = std::string(argv[++ii]);
            std::transform(inDataFormat.begin(), inDataFormat.end(), inDataFormat.begin(), ::tolower);
            testAppArgs.inDataFormat = inDataFormat == "ncxhwx" ? nvdla::DataFormat::NCxHWx :
                                       inDataFormat == "nchw" ? nvdla::DataFormat::NCHW :
                                       inDataFormat == "nhwc" ? nvdla::DataFormat::NHWC :
                                                                nvdla::DataFormat::UNKNOWN;
        }
        else if (std::strcmp(arg, "--cprecision") == 0)
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            std::string computePrecision = std::string(argv[++ii]);
            std::transform(computePrecision.begin(), computePrecision.end(), computePrecision.begin(), ::tolower);
            testAppArgs.computePrecision = computePrecision == "fp16" ? nvdla::DataType::HALF :
                                           computePrecision == "int8" ? nvdla::DataType::INT8 :
                                                                        nvdla::DataType::UNKNOWN;
        }
        else if (std::strcmp(arg, "--calibtable") == 0)
        {
            if (ii+1 >= argc)
            {
                // Expecting another parameter
                showHelp = true;
                break;
            }

            std::string calibTable = std::string(argv[++ii]);
            std::transform(calibTable.begin(), calibTable.end(), calibTable.begin(), ::tolower);
            testAppArgs.calibTable = calibTable;
        }
        else // unknown
        {
            // Unknown argument
            NvDlaDebugPrintf("unknown argument: %s\n", arg);
            unknownArg = true;
            showHelp = true;
            break;
        }

        ii++;
    }

    if (std::strcmp(testAppArgs.prototxt.c_str(), "") == 0 || std::strcmp(testAppArgs.caffemodel.c_str(), "") == 0) {
        missingArg = true;
        showHelp = true;
    }

    if (showHelp)
    {
        NvDlaDebugPrintf("Usage: %s [-options] --prototxt <prototxt_file> --caffemodel <caffemodel_file>\n", argv[0]);
        NvDlaDebugPrintf("where options include:\n");
        NvDlaDebugPrintf("    -h                                                          print this help message\n");
        NvDlaDebugPrintf("    -o <outputpath>                                             outputs wisdom files in 'outputpath' directory\n");
        NvDlaDebugPrintf("    --profile <basic|default|performance|fast-math>             computation profile (default: fast-math)\n");
        NvDlaDebugPrintf("    --cprecision <fp16|int8>                                    compute precision (default: fp16)\n");
        NvDlaDebugPrintf("    --configtarget <opendla-full|opendla-large|opendla-small>   target platform (default: nv_full)\n");
        NvDlaDebugPrintf("    --calibtable <int8 calib file>                              calibration table for INT8 networks (default: 0.00787)\n");
        NvDlaDebugPrintf("    --quantizationMode <per-kernel|per-filter>                  quantization mode for INT8 (default: per-kernel)\n");
        NvDlaDebugPrintf("    --batch                                                     batch size (default: 1)\n");
        NvDlaDebugPrintf("    --informat <ncxhwx|nchw|nhwc>                               input data format (default: nhwc)\n");

        if (unknownArg)
            return EXIT_FAILURE;
        else
            return EXIT_SUCCESS;
    }

    // Launch
    e = launchTest(&testAppArgs);
    if (e != NvDlaSuccess)
        return -1;
    else
        return 0;
}
