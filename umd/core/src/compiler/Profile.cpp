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

#include "priv/Check.h"

#include "priv/Wisdom.h"
#include "priv/WisdomContainer.h"
#include "priv/Profile.h"
#include "priv/Profiler.h"

using std::endl;
using std::string;

namespace nvdla
{


IProfile::IProfile(){ }
IProfile::~IProfile() { }

namespace priv
{

ProfileFactory::ProfilePrivPair ProfileFactory::newProfile()
{
    IProfile *profile;
    Profile *profile_priv;
    profile = profile_priv = new priv::Profile();
    if (profile) {
        s_priv.insert(profile,profile_priv);
        s_self.insert(profile,profile);
    }
    return ProfilePrivPair(profile, profile_priv);
}

Profile *ProfileFactory::priv(IProfile *profile)
{
    BiMap<IProfile *, Profile *>::left_iterator f = s_priv.find_left(profile);
    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

IProfile *ProfileFactory::i(Profile *profile)
{
    BiMap<IProfile *, Profile *>::right_iterator f = s_priv.find_right(profile);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

IProfile *ProfileFactory::self(void *s)
{
    BiMap<void *, IProfile *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}



IProfile *ProfileFactory::deserializeFrom(WisdomContainerEntry *entry)
{
    bool ok = false;
    IProfile *profile = NULL;

    // only one type of profile right now (IProfile/Profile)... but go through the motions
    WisdomContainerEntry factory_type_entry;
    // ProfileType factory_type;
    NvU32 v;

    if ( entry->type() != IWisdomContainerEntry::ENTRY_TYPE_OBJECT ) {
        goto done;
    }

    ok = entry->getEntry("factory_type", IWisdomContainerEntry::ENTRY_TYPE_UINT32, &factory_type_entry);

    ok = ok && factory_type_entry.readUInt32(v);
    if ( !ok ) {
        goto done;
    }

    profile = deserializeProfile(entry);

 done:
    return profile;
}

BiMap<IProfile *, Profile*> ProfileFactory::s_priv;
BiMap<void *, IProfile*> ProfileFactory::s_self;


// there's only one type of "Profile" for now. so only one of these... so it looks
// silly.  see the same paths in "LayerFactory::deserialize*" for why it makes sense
// to organize this way preemptively.
IProfile *ProfileFactory::deserializeProfile(WisdomContainerEntry *entry)
{
    ProfileFactory::ProfilePrivPair t = newProfile();
    if ( !t ) {
        return NULL;
    }
    t.priv()->deserializeFrom(entry);
    return t.i();
}


NvU16 Profile::getFactoryType() const
{
    return 0; // only one type of profile so far, not complicated by factory splits
}


bool Profile::serializeTo(WisdomContainerEntry *e) const
{
    bool ok = true;
    WisdomContainerEntry name_entry;

    // gLogError << __func__ << " name=[" << getName() << "]" << endl;

    ok = ok && e->writeUInt32("factory_type", getFactoryType());
    ok = ok && e->writeString("name", getName());

    ok = ok && e->writeUInt32("pixel_offset_x", m_globalParams.m_NwInPixelOffX);
    ok = ok && e->writeUInt32("pixel_offset_y", m_globalParams.m_NwInPixelOffY);
    ok = ok && e->writeUInt8Enum("input_data_format", m_globalParams.m_NwInDataFormat);
    ok = ok && e->writeUInt8Enum("output_data_format", m_globalParams.m_NwOutDataFormat);
    ok = ok && e->writeUInt8Enum("network_input_format", m_globalParams.m_NwInSurfFormat.v());
    ok = ok && e->writeUInt8Enum("network_output_format", m_globalParams.m_NwOutSurfFormat.v());
    ok = ok && e->writeUInt8Enum("input_pixel_mapping", m_globalParams.m_NwInPixelMapping.v());

    ok = ok && e->writeUInt32("weight_compression", m_compileParams.m_canCompressWeights);
    ok = ok && e->writeUInt32("winograd_enabled", m_compileParams.m_canWinograd);
    ok = ok && e->writeUInt32("conv_weight_banks_allotted", m_compileParams.m_CONVWeightBanksAllotted);
    ok = ok && e->writeUInt32("conv_data_banks_allotted", m_compileParams.m_CONVDataBanksAllotted);
    ok = ok && e->writeUInt32("sdp_pdp_onfly_enabled", m_compileParams.m_canSDPPDPOnFly);
    ok = ok && e->writeUInt32("sdp_merge_math_ops_enabled", m_compileParams.m_canSDPMergeMathOps);
    ok = ok && e->writeUInt32("sdp_fuse_subengine_ops_enabled", m_compileParams.m_canSDPFuseSubEngineOps);
    ok = ok && e->writeUInt32("sdp_bust_nops_enabled", m_compileParams.m_canSDPBustNOPs);
    ok = ok && e->writeUInt32("sdp_op_fusion_enabled", m_compileParams.m_canSDPFuseVerticalOps);
    ok = ok && e->writeUInt32("memory_pooling_enabled", m_compileParams.m_useMemPool);
    ok = ok && e->writeUInt32("reuse_pooled_memory", m_compileParams.m_useReusePooledMemory);
    ok = ok && e->writeUInt32("cvsram_alloc_enabled", m_compileParams.m_useCVSRAMAllocate);
    ok = ok && e->writeUInt32("copy_out_debug_surfaces", m_compileParams.m_copyOutDebugSurfaces);
    ok = ok && e->writeUInt64("global_dram_pool_size", m_compileParams.m_globalDRAMSize);
    ok = ok && e->writeUInt64("local_dram_pool_size", m_compileParams.m_localDRAMSize);
    ok = ok && e->writeUInt64("local_cvsram_pool_size", m_compileParams.m_localCVSRAMSize);
    ok = ok && e->writeUInt32("multi_batch_size", m_compileParams.m_multiBatchSize);
    ok = ok && e->writeUInt32("img_post_chnl_extn_enabled", m_compileParams.m_canIMGPostChnlExtend);
    ok = ok && e->writeUInt8Enum("compute_precision", m_compileParams.m_computePrecision.v());
    ok = ok && e->writeUInt8Enum("tensor_scaling_mode", m_compileParams.m_tensorScalingMode.v());
    ok = ok && e->writeUInt8Enum("quantization_mode", m_compileParams.m_quantizationMode.v());

    return ok;
}

bool Profile::deserializeFrom(WisdomContainerEntry *e)
{
    WisdomContainerEntry name_entry;
    string name;
    Dims3 dims;
    NvS32 data_format, data_type, tensor_type;
    NvU32 boolProxy = 0;
    NvU8 charProxy = 0;
    NVDLA_UNUSED(dims);
    NVDLA_UNUSED(data_format);
    NVDLA_UNUSED(data_type);
    NVDLA_UNUSED(tensor_type);

    bool ok = e->getEntry(string("name"), IWisdomContainerEntry::ENTRY_TYPE_STRING, &name_entry);
    ok = e->readString("name", name);

    setName(name.c_str()); // always make this first.  if we make any choices based upon name
                           // they shouldn't override how it was actually saved off.

    ok = ok && e->readUInt32("pixel_offset_x", m_globalParams.m_NwInPixelOffX);
    ok = ok && e->readUInt32("pixel_offset_y", m_globalParams.m_NwInPixelOffY);

    ok = ok && e->readUInt8Enum("input_data_format", charProxy);
    m_globalParams.m_NwInDataFormat = nvdla::DataFormat(charProxy);
    ok = ok && e->readUInt8Enum("output_data_format", charProxy);
    m_globalParams.m_NwOutDataFormat = nvdla::DataFormat(charProxy);
    ok = ok && e->readUInt8Enum("network_input_format", charProxy);
    m_globalParams.m_NwInSurfFormat = surface::SurfaceFormat(charProxy);
    ok = ok && e->readUInt8Enum("network_output_format", charProxy);
    m_globalParams.m_NwOutSurfFormat = surface::SurfaceFormat(charProxy);
    ok = ok && e->readUInt8Enum("input_pixel_mapping", charProxy);
    m_globalParams.m_NwInPixelMapping = surface::PixelMapping(charProxy);

    ok = ok && e->readUInt32("memory_pooling_enabled", boolProxy);
    m_compileParams.m_useMemPool = bool(boolProxy);
    ok = ok && e->readUInt32("reuse_pooled_memory", boolProxy);
    m_compileParams.m_useReusePooledMemory = bool(boolProxy);
    ok = ok && e->readUInt32("cvsram_alloc_enabled", boolProxy);
    m_compileParams.m_useCVSRAMAllocate = bool(boolProxy);
    ok = ok && e->readUInt32("copy_out_debug_surfaces", boolProxy);
    m_compileParams.m_copyOutDebugSurfaces = bool(boolProxy);
    ok = ok && e->readUInt32("weight_compression", boolProxy);
    m_compileParams.m_canCompressWeights = bool(boolProxy);
    ok = ok && e->readUInt32("winograd_enabled", boolProxy);
    m_compileParams.m_canWinograd = bool(boolProxy);
    ok = ok && e->readUInt32("sdp_pdp_onfly_enabled", boolProxy);
    m_compileParams.m_canSDPPDPOnFly = bool(boolProxy);
    ok = ok && e->readUInt32("sdp_merge_math_ops_enabled", boolProxy);
    m_compileParams.m_canSDPMergeMathOps = bool(boolProxy);
    ok = ok && e->readUInt32("sdp_fuse_subengine_ops_enabled", boolProxy);
    m_compileParams.m_canSDPFuseSubEngineOps = bool(boolProxy);
    ok = ok && e->readUInt32("sdp_bust_nops_enabled", boolProxy);
    m_compileParams.m_canSDPBustNOPs = bool(boolProxy);
    ok = ok && e->readUInt32("sdp_op_fusion_enabled", boolProxy);
    m_compileParams.m_canSDPFuseVerticalOps = bool(boolProxy);
    ok = ok && e->readUInt32("img_post_chnl_extn_enabled", boolProxy);
    m_compileParams.m_canIMGPostChnlExtend = bool(boolProxy);

    ok = ok && e->readUInt64("global_dram_pool_size", m_compileParams.m_globalDRAMSize);
    ok = ok && e->readUInt64("local_dram_pool_size", m_compileParams.m_localDRAMSize);
    ok = ok && e->readUInt64("local_cvsram_pool_size", m_compileParams.m_localCVSRAMSize);
    ok = ok && e->readUInt32("conv_weight_banks_allotted", m_compileParams.m_CONVWeightBanksAllotted);
    ok = ok && e->readUInt32("conv_data_banks_allotted", m_compileParams.m_CONVDataBanksAllotted);
    ok = ok && e->readUInt32("multi_batch_size", m_compileParams.m_multiBatchSize);

    ok = ok && e->readUInt8Enum("compute_precision", charProxy);
    m_compileParams.m_computePrecision = surface::SurfacePrecision(charProxy);
    ok = ok && e->readUInt8Enum("tensor_scaling_mode", charProxy);
    m_compileParams.m_tensorScalingMode = nvdla::TensorScalingMode(charProxy);
    ok = ok && e->readUInt8Enum("quantization_mode", charProxy);
    m_compileParams.m_quantizationMode = nvdla::QuantizationMode(charProxy);

    return ok;
}



const char* Profile::getName() const
{
    return m_name.c_str();
}

void Profile::setBasicProfile()
{
    setUseCVSRAMAllocate(false);
    setUseMemPool(false);
    setUseReusePooledMemory(false);
    setUseGreedyEviction(false);
    setCanSDPBustNOPs(false);
    setCanSDPMergeMathOps(false);
    setCanSDPFuseSubEngineOps(false);
}

void Profile::setDefaultProfile()
{
    setBasicProfile();
    setUseMemPool(true);
    setUseReusePooledMemory(true);
    setUseGreedyEviction(true);
    setCanSDPBustNOPs(true);
}

void Profile::setPerformanceProfile()
{
    setDefaultProfile();
//    setUseCVSRAMAllocate(true);
}

void Profile::setFastMathProfile()
{
    setPerformanceProfile();
    setCanSDPMergeMathOps(true);
    setCanSDPFuseSubEngineOps(true);
//    setCanWinograd(true);
}

void Profile::setName(const char* n)
{
    API_CHECK_NULL(n);
    m_name = n;

    if ( isBasicProfile() )
    {
        setBasicProfile();
    }

    if ( isDefaultProfile() )
    {
        setDefaultProfile();
    }

    if ( isPerformanceProfile() )
    {
        setPerformanceProfile();
    }

    if ( isFastMathProfile() )
    {
        setFastMathProfile();
    }
}


NvDlaError Profile::getNumLoadables(int *n) const
{
    NvDlaError e = NvDlaSuccess;
    if ( !n )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    *n = 1;
 fail:
    return e;
}

// if name.length() == 0 or index < 0 then they are ignored for the purpose of lookup
NvDlaError Profile::getLoadable(const std::string &name, int index, ILoadable **i_loadable)
{
    bool nameSpecified = name.length() > 0;
    bool indexSpecified = index >= 0;

    NvDlaError e = NvDlaSuccess;
    if ( !i_loadable )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    *i_loadable = 0;

    if ( (nameSpecified && indexSpecified) || !(nameSpecified || indexSpecified) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "must specify (only) one of index or name for loadable association with profile");
    }

    if ( nameSpecified )
    {
        if ( m_loadablesByName.find(name) != m_loadablesByName.end() )
        {
            *i_loadable = m_loadablesByName[name];
        }
        if ( debug() )
        {
            gLogInfo << "profile getLoadable looked for loadable with name " << name << endl;
        }
    }
    else if ( indexSpecified )
    {
        size_t u_index(index); // note sign checked above.
        if ( u_index < m_loadables.size() )
        {
            *i_loadable = m_loadables[u_index];
        }
    }


 fail:
    return e;
}

// if name.length() == 0 or index < 0 then be sure the same one comes back during a lookup.
NvDlaError Profile::insertLoadable(const std::string & name, int index, ILoadable *i_loadable)
{
    NvDlaError e = NvDlaSuccess;
    bool nameSpecified = name.length() > 0;
    bool indexSpecified = index >= 0;

    if ( !i_loadable )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    if ( (nameSpecified && indexSpecified) || !(nameSpecified || indexSpecified) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "must specify (only) one of index or name for loadable association with profile");
    }

    if ( nameSpecified )
    {
        if ( debug() )
        {
            gLogInfo << "profile insertLoadable saving loadable with name " << name << endl;
        }
        m_loadablesByName[name] = i_loadable;
    }
    else if ( indexSpecified )
    {
        size_t u_index(index); // note sign checked above.
        if ( u_index >= m_loadables.size() )
        {
            m_loadables.resize(u_index + 1, 0);
        }
        m_loadables[index] = i_loadable;
    }

 fail:
    return e;
}

static NvDlaError setSurfaceFormat(nvdla::PixelFormat pf, surface::SurfacePrecision cp, surface::SurfaceFormat& sf)
{
    NvDlaError e = NvDlaSuccess;

    switch(pf) {
        case nvdla::PixelFormat::R8:                sf = surface::SurfaceFormatEnum::NVDLA_IMG_R8; break;
        case nvdla::PixelFormat::R10:               sf = surface::SurfaceFormatEnum::NVDLA_IMG_R10; break;
        case nvdla::PixelFormat::R12:               sf = surface::SurfaceFormatEnum::NVDLA_IMG_R12; break;
        case nvdla::PixelFormat::R16:               sf = surface::SurfaceFormatEnum::NVDLA_IMG_R16; break;
        case nvdla::PixelFormat::R16_I:             sf = surface::SurfaceFormatEnum::NVDLA_IMG_R16_I; break;
        case nvdla::PixelFormat::R16_F:             sf = surface::SurfaceFormatEnum::NVDLA_IMG_R16_F; break;
        case nvdla::PixelFormat::A16B16G16R16:      sf = surface::SurfaceFormatEnum::NVDLA_IMG_A16B16G16R16; break;
        case nvdla::PixelFormat::X16B16G16R16:      sf = surface::SurfaceFormatEnum::NVDLA_IMG_X16B16G16R16; break;
        case nvdla::PixelFormat::A16B16G16R16_F:    sf = surface::SurfaceFormatEnum::NVDLA_IMG_A16B16G16R16_F; break;
        case nvdla::PixelFormat::A16Y16U16V16:      sf = surface::SurfaceFormatEnum::NVDLA_IMG_A16Y16U16V16; break;
        case nvdla::PixelFormat::V16U16Y16A16:      sf = surface::SurfaceFormatEnum::NVDLA_IMG_V16U16Y16A16; break;
        case nvdla::PixelFormat::A16Y16U16V16_F:    sf = surface::SurfaceFormatEnum::NVDLA_IMG_A16Y16U16V16_F; break;
        case nvdla::PixelFormat::A8B8G8R8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_A8B8G8R8; break;
        case nvdla::PixelFormat::A8R8G8B8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_A8R8G8B8; break;
        case nvdla::PixelFormat::B8G8R8A8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_B8G8R8A8; break;
        case nvdla::PixelFormat::R8G8B8A8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_R8G8B8A8; break;
        case nvdla::PixelFormat::X8B8G8R8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_X8B8G8R8; break;
        case nvdla::PixelFormat::X8R8G8B8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_X8R8G8B8; break;
        case nvdla::PixelFormat::B8G8R8X8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_B8G8R8X8; break;
        case nvdla::PixelFormat::R8G8B8X8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_R8G8B8X8; break;
        case nvdla::PixelFormat::A2B10G10R10:       sf = surface::SurfaceFormatEnum::NVDLA_IMG_A2B10G10R10; break;
        case nvdla::PixelFormat::A2R10G10B10:       sf = surface::SurfaceFormatEnum::NVDLA_IMG_A2R10G10B10; break;
        case nvdla::PixelFormat::B10G10R10A2:       sf = surface::SurfaceFormatEnum::NVDLA_IMG_B10G10R10A2; break;
        case nvdla::PixelFormat::R10G10B10A2:       sf = surface::SurfaceFormatEnum::NVDLA_IMG_R10G10B10A2; break;
        case nvdla::PixelFormat::Y8___U8V8_N444:    sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y8___U8V8_N444; break;
        case nvdla::PixelFormat::Y8___V8U8_N444:    sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y8___V8U8_N444; break;
        case nvdla::PixelFormat::Y10___U10V10_N444: sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y10___U10V10_N444; break;
        case nvdla::PixelFormat::Y10___V10U10_N444: sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y10___V10U10_N444; break;
        case nvdla::PixelFormat::Y12___U12V12_N444: sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y12___U12V12_N444; break;
        case nvdla::PixelFormat::Y12___V12U12_N444: sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y12___V12U12_N444; break;
        case nvdla::PixelFormat::Y16___U16V16_N444: sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y16___U16V16_N444; break;
        case nvdla::PixelFormat::Y16___V16U16_N444: sf = surface::SurfaceFormatEnum::NVDLA_IMG_Y16___V16U16_N444; break;
        case nvdla::PixelFormat::A2Y10U10V10:       sf = surface::SurfaceFormatEnum::NVDLA_IMG_A2Y10U10V10; break;
        case nvdla::PixelFormat::V10U10Y10A2:       sf = surface::SurfaceFormatEnum::NVDLA_IMG_V10U10Y10A2; break;
        case nvdla::PixelFormat::A8Y8U8V8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_A8Y8U8V8; break;
        case nvdla::PixelFormat::V8U8Y8A8:          sf = surface::SurfaceFormatEnum::NVDLA_IMG_V8U8Y8A8; break;
        case nvdla::PixelFormat::FEATURE : {
            switch(cp.v()) {
                case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:  sf = surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_FP16; break;
                case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16: sf = surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT16; break;
                case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:  sf = surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8; break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Cant determine surface format FEATURE_ with compute precision: %d", (int)cp.v());
            }
        }; break;
        case nvdla::PixelFormat::FEATURE_X8: {
            switch(cp.v()) {
                case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:  sf = surface::SurfaceFormatEnum::NVDLA_FEATURE_DATA_INT8; break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Cant determine surface format FEATURE_ with compute precision: %d", (int)cp.v());
            }
        }; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized input format: %d", pf.v());
    }

fail:
    return e;
}

NvDlaError Profile::setNetworkInputSurfaceFormat(nvdla::PixelFormat nisf)
{
    NvDlaError e = NvDlaSuccess;
    PROPAGATE_ERROR_FAIL(setSurfaceFormat(nisf, m_compileParams.m_computePrecision, m_globalParams.m_NwInSurfFormat));

fail:
    return e;
}

NvDlaError Profile::setNetworkOutputSurfaceFormat(nvdla::PixelFormat nosf)
{
    NvDlaError e = NvDlaSuccess;
    PROPAGATE_ERROR_FAIL(setSurfaceFormat(nosf, m_compileParams.m_computePrecision, m_globalParams.m_NwOutSurfFormat));

fail:
    return e;
}

NvDlaError Profile::setNetworkInputPixelMapping(nvdla::PixelMapping pm)
{
    NvDlaError e = NvDlaSuccess;

    switch(pm) {
        case nvdla::PixelMapping::PITCH_LINEAR: m_globalParams.m_NwInPixelMapping = surface::PixelMappingEnum::PITCH_LINEAR; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "unrecognized pixel mapping: %d", pm.v());
    }

fail:
    return e;
}

NvDlaError Profile::initGlobalParams(IProfile::IGlobalParams* globalParams)
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(setNetworkInputPixelOffX(globalParams->pixelOffsetX));
    PROPAGATE_ERROR_FAIL(setNetworkInputPixelOffY(globalParams->pixelOffsetY));
    PROPAGATE_ERROR_FAIL(setNetworkInputDataFormat(globalParams->inputDataFormat));
    PROPAGATE_ERROR_FAIL(setNetworkOutputDataFormat(globalParams->outputDataFormat));
    PROPAGATE_ERROR_FAIL(setNetworkInputPixelMapping(globalParams->inputPixelMapping));
    PROPAGATE_ERROR_FAIL(setNetworkInputSurfaceFormat(globalParams->inputPixelFormat));
    PROPAGATE_ERROR_FAIL(setNetworkOutputSurfaceFormat(globalParams->outputPixelFormat));

fail:
    return e;
}

NvDlaError Profile::setComputePrecision(nvdla::DataType cp)
{
    NvDlaError e = NvDlaSuccess;
    switch (cp) {
        case nvdla::DataType::HALF:  m_compileParams.m_computePrecision = surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16; break;
        case nvdla::DataType::INT16: m_compileParams.m_computePrecision = surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16; break;
        case nvdla::DataType::INT8:  m_compileParams.m_computePrecision = surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized compute precision: %d", cp.v());
    }

fail:
    return e;
}

NvDlaError Profile::setTensorScalingMode(nvdla::TensorScalingMode tsm)
{
    NvDlaError e = NvDlaSuccess;

    if (tsm.v() > EnumMax<TensorScalingMode>())
    {
        e = NvDlaError_BadParameter;
        goto fail;
    }

    switch (tsm.v()) {
        case nvdla::TensorScalingMode::NONE:
        case nvdla::TensorScalingMode::PER_TENSOR:
            m_compileParams.m_tensorScalingMode = tsm;
            break;
        default:
            e = NvDlaError_NotSupported;
    }

fail:
    return e;
}

NvDlaError Profile::setQuantizationMode(nvdla::QuantizationMode qm)
{
    NvDlaError e = NvDlaSuccess;

    if (qm.v() > EnumMax<QuantizationMode>())
    {
        e = NvDlaError_BadParameter;
        goto fail;
    }

    switch (qm.v()) {
        case nvdla::QuantizationMode::NONE:
        case nvdla::QuantizationMode::PER_KERNEL:
        case nvdla::QuantizationMode::PER_FILTER:
            m_compileParams.m_quantizationMode = qm;
            break;
        default:
            e = NvDlaError_NotSupported;
    }

fail:
    return e;
}

NvDlaError Profile::initCompileParams(IProfile::ICompileParams* compileParams)
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(setCanCompressWeights(compileParams->canCompressWeights));
    PROPAGATE_ERROR_FAIL(setCanWinograd(compileParams->canWinograd));
    PROPAGATE_ERROR_FAIL(setCONVWeightBanksAllotted(compileParams->convWeightBanksAllotted));
    PROPAGATE_ERROR_FAIL(setCONVDataBanksAllotted(compileParams->convDataBanksAllotted));
    PROPAGATE_ERROR_FAIL(setCanSDPPDPOnFly(compileParams->canSdpPdpOnFly));
    PROPAGATE_ERROR_FAIL(setCanSDPMergeMathOps(compileParams->canSdpMergeMathOps));
    PROPAGATE_ERROR_FAIL(setCanSDPFuseSubEngineOps(compileParams->canSdpFuseSubEngineOps));
    PROPAGATE_ERROR_FAIL(setCanSDPBustNOPs(compileParams->canSdpBustNOPs));
    PROPAGATE_ERROR_FAIL(setCanSDPFuseVerticalOps(compileParams->canSdpFuseVerticalOps));
    PROPAGATE_ERROR_FAIL(setUseCVSRAMAllocate(compileParams->useCvsramAllocate));
    PROPAGATE_ERROR_FAIL(setUseMemPool(compileParams->useMemPool));
    PROPAGATE_ERROR_FAIL(setUseReusePooledMemory(compileParams->useReusePooledMemory));
    PROPAGATE_ERROR_FAIL(setCopyOutDebugSurfaces(compileParams->copyOutDebugSurfaces));
    PROPAGATE_ERROR_FAIL(setGlobalDRAMSize(compileParams->globalDramSize));
    PROPAGATE_ERROR_FAIL(setLocalDRAMSize(compileParams->localDramSize));
    PROPAGATE_ERROR_FAIL(setLocalCVSRAMSize(compileParams->localCvsramSize));
    PROPAGATE_ERROR_FAIL(setMultiBatchSize(compileParams->multiBatchSize));
    PROPAGATE_ERROR_FAIL(setCanIMGPostChnlExtend(compileParams->canImgPostChnlExtend));
    PROPAGATE_ERROR_FAIL(setComputePrecision(compileParams->computePrecision));
    PROPAGATE_ERROR_FAIL(setTensorScalingMode(compileParams->tensorScalingMode));
    PROPAGATE_ERROR_FAIL(setQuantizationMode(compileParams->quantizationMode));

    setName(m_name.c_str()); //XXX hack to force use to reevaluate perf/default stuff just clobbered.

fail:
    return e;
}

void Profile::initWithDefaultProfile()
{
    setDefaultProfile();
}

} // nvdla::priv

} // nvdla::
