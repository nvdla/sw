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

#include "half.h"
#include "main.h"

#include "nvdla/ILayer.h"
#include "nvdla/INetwork.h"
#include "nvdla/IProfile.h"
#include "nvdla/IProfiler.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

#include "ErrorMacros.h"
#include "nvdla_os_inf.h"

static NvDlaError beginWithNamedProfile(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IProfiler* profiler;
    nvdla::IProfile* profile;

    profiler = i->wisdom->getProfiler();
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profiler not initialized");
    }

    profile = profiler->getProfile(appArgs->profileName.c_str());
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profile %s not initialized", appArgs->profileName.c_str());
    }

fail:
    return e;
}

static NvDlaError beginWithCfgProfile(const TestAppArgs* appArgs, TestInfo* i, nvdla::DataFormat& inDataFormat)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;
    std::string profileCfgFile;
    std::string profileName;
    nvdla::IProfiler* profiler;
    nvdla::IProfile* profile;

    profiler = i->wisdom->getProfiler();
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profiler not initialized");
    }

    profileName = appArgs->profileFile;
    profileName = profileName.substr(0, profileName.find_last_of("."));
    profile = profiler->getProfile(profileName.c_str());
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profile %s not initialized", profileName.c_str());
    }

    profileCfgFile = i->profilesPath + appArgs->profileFile;
    PROPAGATE_ERROR_FAIL(NvDlaStat(profileCfgFile.c_str(), &stat));

    // first use settings from default profile
    profile->initWithDefaultProfile();

    // then populate the existing profile with params in the cfg file (overriding as necessary)
    {
        FILE* fp = fopen(profileCfgFile.c_str(), "r");
        char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};

        rapidjson::Document doc;
        rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));

        doc.ParseStream(inStr);
        if (doc.HasParseError())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "JSON parsing error: %s", GetParseError_En(doc.GetParseError()));
        }

        {
            nvdla::PixelFormat pf;

            /* Gather compile params of the profile */
            if (doc["profile"].HasMember("compute_precision")) {
                rapidjson::Value& compPrecision = doc["profile"]["compute_precision"];
                std::string prec = compPrecision.GetString();
                nvdla::DataType dt = nvdla::DataType::getEnum(prec);

                if (dt.v() == nvdla::DataType::UNKNOWN) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Precision %s not supported", prec.c_str());
                }
                profile->setComputePrecision(dt);
            }

            if (doc["profile"].HasMember("weight_packing")) {
                rapidjson::Value& weightPacking = doc["profile"]["weight_packing"];
                std::string wtPacking = weightPacking.GetString();

                if ( wtPacking == "COMPRESSED" )
                    profile->setCanCompressWeights(true);
                else
                    profile->setCanCompressWeights(false);
            }

            if (doc["profile"].HasMember("sdp_pdp_on_fly")) {
                rapidjson::Value& sdpPdpOnFly = doc["profile"]["sdp_pdp_on_fly"];
                profile->setCanSDPPDPOnFly(sdpPdpOnFly.GetBool());
            }

            if (doc["profile"].HasMember("sdp_merge_math_ops")) {
                rapidjson::Value& sdpMergeMathOps = doc["profile"]["sdp_merge_math_ops"];
                profile->setCanSDPMergeMathOps(sdpMergeMathOps.GetBool());
            }

            if (doc["profile"].HasMember("sdp_fuse_subengine_ops")) {
                rapidjson::Value& sdpFuseSubEngineOps = doc["profile"]["sdp_fuse_subengine_ops"];
                profile->setCanSDPFuseSubEngineOps(sdpFuseSubEngineOps.GetBool());
            }

            if (doc["profile"].HasMember("can_winograd")) {
                rapidjson::Value& canWinograd = doc["profile"]["can_winograd"];
                profile->setCanWinograd(canWinograd.GetBool());
            }

            /* Gather global params of the profile */
            if (doc["profile"]["network_input"].HasMember("format")) {
                rapidjson::Value& inFormat = doc["profile"]["network_input"]["format"];
                pf = nvdla::PixelFormat::getEnum(inFormat.GetString());
                if (pf.v() == nvdla::PixelFormat::UNKNOWN) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Pixel format %s not supported", inFormat.GetString());
                }
                profile->setNetworkInputSurfaceFormat(pf);
                if (pf < nvdla::PixelFormat::FEATURE) {
                    inDataFormat = nvdla::DataFormat::NHWC;
                }
                else if ((pf == nvdla::PixelFormat::FEATURE) || (pf == nvdla::PixelFormat::FEATURE_X8)) {
                    inDataFormat = nvdla::DataFormat::NCxHWx;
                }
                else {
                    PROPAGATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support input pixel format: %s", pf.c_str());
                }
            }

            if (doc["profile"]["network_input"].HasMember("pixel_offset_x")) {
                rapidjson::Value& pxOffX = doc["profile"]["network_input"]["pixel_offset_x"];
                profile->setNetworkInputPixelOffX(pxOffX.GetInt());
            }
            if (doc["profile"]["network_input"].HasMember("pixel_offset_y")) {
                rapidjson::Value& pxOffY = doc["profile"]["network_input"]["pixel_offset_y"];
                profile->setNetworkInputPixelOffY(pxOffY.GetInt());
            }

            if (doc["profile"]["network_output"].HasMember("format")) {
                rapidjson::Value& outFormat = doc["profile"]["network_output"]["format"];
                pf = nvdla::PixelFormat::getEnum(outFormat.GetString());
                if (pf.v() == nvdla::PixelFormat::UNKNOWN) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Pixel format %s not supported", outFormat.GetString());
                }
                profile->setNetworkOutputSurfaceFormat(pf);
            }
        }

        fclose(fp);
    }

fail:
    return e;
}

static NvDlaError updateProfileWithCmdLineArgs
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    const char* profileName,
    nvdla::DataFormat inDataFormat
)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IProfiler* profiler;
    nvdla::IProfile* profile;

    profiler = i->wisdom->getProfiler();
    if (!profiler)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->getProfiler() failed");
    profile   = profiler->getProfile(profileName);
    if (!profile)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "profiler->getProfile() failed");

    PROPAGATE_ERROR_FAIL(profile->setComputePrecision(appArgs->computePrecision));
    PROPAGATE_ERROR_FAIL(profile->setNetworkInputDataFormat(inDataFormat));

    // determine input surface format
    switch(inDataFormat)
    {
        case nvdla::DataFormat::NHWC:

            if (appArgs->computePrecision == nvdla::DataType::HALF)
            {
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A16B16G16R16_F));
            }
            else if (appArgs->computePrecision == nvdla::DataType::INT8)
            {
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A8B8G8R8));
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "NHWC and compute precision %u is not yet supported",
                                     appArgs->computePrecision.v());
            }
            break;
        case nvdla::DataFormat::NCxHWx:
        case nvdla::DataFormat::NCHW:
        case nvdla::DataFormat::UNKNOWN:    // atleast start the test with feature data format
        default:
            if (std::strcmp(appArgs->configtarget.c_str(), "opendla-small") == 0)
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8));
            else
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE));
    }

    // determine int8 cfgs
    if (appArgs->computePrecision == nvdla::DataType::INT8)
    {
        PROPAGATE_ERROR_FAIL(profile->setTensorScalingMode(nvdla::TensorScalingMode::PER_TENSOR));
        switch(appArgs->quantizationMode)
        {
            case nvdla::QuantizationMode::PER_FILTER:
                PROPAGATE_ERROR_FAIL(profile->setQuantizationMode(nvdla::QuantizationMode::PER_FILTER));
                break;
            case nvdla::QuantizationMode::PER_KERNEL:
            case nvdla::QuantizationMode::NONE: // default to per-kernel; find a way to run int8 tests w/ NONE qtzMode cleanly
            default:
                PROPAGATE_ERROR_FAIL(profile->setQuantizationMode(nvdla::QuantizationMode::PER_KERNEL));
        }
    }
    else
    {
        PROPAGATE_ERROR_FAIL(profile->setTensorScalingMode(nvdla::TensorScalingMode::NONE));
        PROPAGATE_ERROR_FAIL(profile->setQuantizationMode(nvdla::QuantizationMode::NONE));
    }

    PROPAGATE_ERROR_FAIL(profile->setNetworkOutputDataFormat(nvdla::DataFormat::NCxHWx));

    if (std::strcmp(appArgs->configtarget.c_str(), "opendla-small") == 0)
        PROPAGATE_ERROR_FAIL(profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8));
    else
        PROPAGATE_ERROR_FAIL(profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE));

    if (appArgs->numBatches > 0)
        PROPAGATE_ERROR_FAIL(profile->setMultiBatchSize(appArgs->numBatches));

fail:
    return e;
}

NvDlaError generateProfile(const TestAppArgs* appArgs, std::string* profileName, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::DataFormat inDataFormat = nvdla::DataFormat::UNKNOWN;

    if (appArgs->profileName != "")
    {
        // init named profile (basic/default/performance) with default params in its constructor and exit
        PROPAGATE_ERROR_FAIL(beginWithNamedProfile(appArgs, i));
        *profileName = appArgs->profileName;
    }
    else if (appArgs->profileFile != "")
    {
        // if named profile is absent, create a default profile
        // and then populate it with params from the cfg file (overriding as necessary)
        PROPAGATE_ERROR_FAIL(beginWithCfgProfile(appArgs, i, inDataFormat));
        *profileName = appArgs->profileFile;
        *profileName = profileName->substr(0, profileName->find_last_of("."));
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No profile supplied to load");
    }

    // capture profile params from command line (override the existing ones as necessary)
    inDataFormat = inDataFormat == nvdla::DataFormat::UNKNOWN ? appArgs->inDataFormat : inDataFormat;
    PROPAGATE_ERROR_FAIL(updateProfileWithCmdLineArgs(appArgs, i, profileName->c_str(), inDataFormat));

fail:
    return e;
}

NvDlaError generateTensorScales(const TestAppArgs* appArgs, TestInfo* i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;

    std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
    std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

    std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
    std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

    // set scaling factor for the network input tensors
    for (; nii != networkInputs.end(); ++nii)
    {
        NvF32 scale = 1;
        NvF32 min = scale * -127.0f;
        NvF32 max = scale * 127.0f;
        std::string tName = (*nii)->getName();

        // set same dynamic range for all channels of the tensor (cIndex = -1)
        PROPAGATE_ERROR_FAIL( (*nii)->setChannelDynamicRange(-1, min, max) );
        const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
        if (0)
            NvDlaDebugPrintf("setting dynamic range of: %s to %f\n", tName.c_str(), scale);
    }

    for (; li != networkLayers.end(); ++li)
    {
        NvF32 scale = 127;
        NvF32 min = scale * -127.0f;
        NvF32 max = scale * 127.0f;
        std::string lName = (*li)->getName();
        nvdla::ITensor* outTensor = (*li)->getOutput(0);

        // set same dynamic range for all channels of the tensor (cIndex = -1)
        PROPAGATE_ERROR_FAIL( outTensor->setChannelDynamicRange(-1, min, max) );
        const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
        if (0)
            NvDlaDebugPrintf("setting dynamic range of: %s to %f\n", lName.c_str(), scale);
    }

fail:
    return e;
}
