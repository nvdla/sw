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

#include "nvdla/IWisdom.h"
#include "nvdla/INetwork.h"
#include "nvdla/caffe/ICaffeParser.h"
#include "nvdla/ILayer.h"
#include "nvdla/ICompiler.h"
#include "nvdla/IRuntime.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

#include "ErrorMacros.h"
#include "nvdla_os_inf.h"

#include <cstdlib> // system

NvDlaError parseSetup(const TestAppArgs* appArgs, TestInfo* i)
{
    return NvDlaSuccess;
}

NvDlaError parseTensorScales(const TestAppArgs* appArgs, TestInfo *i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;
    std::string calibTableFile = /*i->calibTablesPath + "/" + */appArgs->calibTable;

    PROPAGATE_ERROR_FAIL(NvDlaStat(calibTableFile.c_str(), &stat));

    // populate the scaling factor/dynamic range of each of the tensors on the network
    {
        FILE* fp = fopen(calibTableFile.c_str(), "r");
        char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};

        rapidjson::Document doc;
        rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));

        doc.ParseStream(inStr);
        if (doc.HasParseError())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "JSON parsing error: %s", GetParseError_En(doc.GetParseError()));
        }

        {
            std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
            std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

            std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
            std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

            // set scaling factor for the network input tensors
            for (; nii != networkInputs.end(); ++nii)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string tName = (*nii)->getName();
                if (doc[tName.c_str()].HasMember("scale")) {
                    scale = doc[tName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[tName.c_str()].HasMember("min") && doc[tName.c_str()].HasMember("max")) {
                    min = doc[tName.c_str()]["min"].GetFloat();
                    max = doc[tName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", tName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( (*nii)->setChannelDynamicRange(-1, min, max) );
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
            }

            for (; li != networkLayers.end(); ++li)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string lName = (*li)->getName();
                nvdla::ITensor* outTensor = (*li)->getOutput(0);

                if (doc[lName.c_str()].HasMember("scale")) {
                    scale = doc[lName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[lName.c_str()].HasMember("min") && doc[lName.c_str()].HasMember("max")) {
                    min = doc[lName.c_str()]["min"].GetFloat();
                    max = doc[lName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", lName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( outTensor->setChannelDynamicRange(-1, min, max) );
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
            }
        }

        fclose(fp);
    }

fail:
    return e;
}

static NvDlaError parseCaffeNetwork(const TestAppArgs* appArgs, TestInfo* i)
{
    NVDLA_UNUSED(appArgs);
    NvDlaError e = NvDlaSuccess;

    nvdla::INetwork* network = NULL;

    const nvdla::caffe::IBlobNameToTensor* b = NULL;
    nvdla::caffe::ICaffeParser* parser = nvdla::caffe::createCaffeParser();
    std::string caffePrototxtFile = appArgs->prototxt.c_str();
    std::string caffeModelFile = appArgs->caffemodel.c_str();

    network = nvdla::createNetwork();
    if (!network)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createNetwork() failed");

    NvDlaDebugPrintf("parsing caffe network...\n");
    b = parser->parse(caffePrototxtFile.c_str(), caffeModelFile.c_str(), network);
    if (!b)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unable to parse caffemodel: \"%s\"", caffePrototxtFile.c_str());

    // if the application has so far not marked the network's outputs, allow the parser to do so now
    if (network->getNumOutputs() <= 0)
    {
        int outs = parser->identifyOutputs(network);
        NvDlaDebugPrintf("Marking total %d outputs\n", outs);
        if (outs <= 0)
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unable to identify outputs for the network: %d", outs);
    }

    if (appArgs->computePrecision == nvdla::DataType::INT8)
    {
        if (appArgs->calibTable != "")  // parse and set tensor scales
        {
            NvDlaDebugPrintf("parsing calibration table...\n");
            PROPAGATE_ERROR_FAIL(parseTensorScales(appArgs, i, network));
        }
        else    // use default or const scaling factors
        {
            NvDlaDebugPrintf("initialize all tensors with const scaling factors of 127...\n");
            PROPAGATE_ERROR_FAIL(generateTensorScales(appArgs, i, network));
        }
    }

    NvDlaDebugPrintf("attaching parsed network to the wisdom...\n");
    if (!i->wisdom->setNetworkTransient(network))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->setNetworkTransient() failed");

    return NvDlaSuccess;

fail:
    return e;
}

NvDlaError parse(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    bool isCaffe = appArgs->caffemodel != "";

    PROPAGATE_ERROR_FAIL(parseSetup(appArgs, i));

    NvDlaDebugPrintf("creating new wisdom context...\n");
    i->wisdom = nvdla::createWisdom();
    if (!i->wisdom)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createWisdom() failed");

    NvDlaDebugPrintf("opening wisdom context...\n");
    if (!i->wisdom->open(i->wisdomPath))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->open() failed to open: \"%s\"", i->wisdomPath.c_str());

    // Parse
    if (isCaffe)
        PROPAGATE_ERROR_FAIL(parseCaffeNetwork(appArgs, i));
    else
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown network type encountered");

    /* Destroy network before closing wisdom context */
    nvdla::destroyNetwork(i->wisdom->getNetwork());

    NvDlaDebugPrintf("closing wisdom context...\n");
    i->wisdom->close();

fail:
    if (i->wisdom != NULL) {
        nvdla::destroyWisdom(i->wisdom);
        i->wisdom = NULL;
    }
    return e;
}

NvDlaError parseAndCompile(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    bool isCaffe = appArgs->caffemodel != "";

    PROPAGATE_ERROR_FAIL(parseSetup(appArgs, i));

    NvDlaDebugPrintf("creating new wisdom context...\n");
    i->wisdom = nvdla::createWisdom();
    if (!i->wisdom)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createWisdom() failed");

    NvDlaDebugPrintf("opening wisdom context...\n");
    if (!i->wisdom->open(i->wisdomPath))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->open() failed to open: \"%s\"", i->wisdomPath.c_str());

    // Parse
    if (isCaffe)
        PROPAGATE_ERROR_FAIL(parseCaffeNetwork(appArgs, i));
    else
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown network type encountered");

    // Compile
    PROPAGATE_ERROR_FAIL(compileProfile(appArgs, i));

    /* Destroy network before closing wisdom context */
    nvdla::destroyNetwork(i->wisdom->getNetwork());

    NvDlaDebugPrintf("closing wisdom context...\n");
    i->wisdom->close();

fail:
    if (i->wisdom != NULL) {
        nvdla::destroyWisdom(i->wisdom);
        i->wisdom = NULL;
    }
    return e;
}
