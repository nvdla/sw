/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <string>

#include "priv/Type.h"

#include "priv/TestPointParameter.h"

#include "ErrorMacros.h"


namespace nvdla
{

namespace priv
{

ENUM_PARAMETER_STATIC(BatchModeParameter, BATCH_MODE_ENUMS, "BatchMode")
ENUM_PARAMETER_STATIC(CVSRamSizeParameter, CVSRAM_SIZE_ENUMS, "CVSRamSize")
ENUM_PARAMETER_STATIC(HWLayerTuningParameter, HW_LAYER_TUNING_ENUMS, "HWLayerTuning")
ENUM_PARAMETER_STATIC(MappingWeightsParameter, MAPPING_WEIGHTS_ENUMS, "MappingWeights")
ENUM_PARAMETER_STATIC(PaddingParameter, PADDING_ENUMS, "Padding")
ENUM_PARAMETER_STATIC(OutputSequenceParameter, OUTPUT_SEQUENCE_ENUMS, "OutputSequence")
ENUM_PARAMETER_STATIC(DilationParameter, DILATION_ENUMS, "Dilation")
ENUM_PARAMETER_STATIC(WeightDensityParameter, WEIGHT_DENSITY_ENUMS, "WeightDensity")
ENUM_PARAMETER_STATIC(FeatureDensityParameter, FEATURE_DENSITY_ENUMS, "")
ENUM_PARAMETER_STATIC(ChannelExtensionParameter, CHANNEL_EXTENSION_ENUMS, "ChannelExtension")
ENUM_PARAMETER_STATIC(ConvMACRedundancyParameter, CONV_MAC_REDUNDANCY_ENUMS, "")
ENUM_PARAMETER_STATIC(ConvBufBankMgmtParameter, CONV_BUF_BANK_MGMT_ENUMS, "ConvBufBankMgmt")
ENUM_PARAMETER_STATIC(PDPOpModeParameter, PDP_OP_MODE_ENUMS, "PDPOpMode")
ENUM_PARAMETER_STATIC(OffFlyingOpModeParameter, OFF_FLYING_OP_MODE_ENUMS, "OffFlyingOpMode")
ENUM_PARAMETER_STATIC(AXIFSchedParameter, AXIF_SCHED_ENUMS, "AXIFSched")
ENUM_PARAMETER_STATIC(PixelDataFormatParameter, PIXEL_DATA_FORMAT_ENUMS, "PixelDataFormat")
ENUM_PARAMETER_STATIC(NetworkForksParameter, NETWORK_FORKS_ENUMS, "NetworkForks")




static void trythis()
{
    NVDLA_UNUSED(&trythis);
    CVSRamSizeParameter f( CVSRamSize::ZERO_MB  );
    const char * fstr = f.c_str();
    CVSRamSizeParameter::underlying_type v = f.v();
    NVDLA_UNUSED(fstr);
    NVDLA_UNUSED(v);

    switch (f.e())
    {
    case CVSRamSize::ZERO_MB:
        break;

    case CVSRamSize::TWO_MB:
        break;

    case CVSRamSize::FOUR_MB:
        break;

    default:
        break;
    };
}


} // nvdla::priv

} // nvdla
