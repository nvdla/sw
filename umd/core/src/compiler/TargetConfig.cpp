/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "priv/TargetConfig.h"
#include "priv/Profiler.h"

using std::endl;
using std::string;

namespace nvdla
{


ITargetConfig::ITargetConfig(){ }
ITargetConfig::~ITargetConfig() { }

namespace priv
{

TargetConfigFactory::TargetConfigPrivPair TargetConfigFactory::newTargetConfig()
{
    ITargetConfig *target_config;
    TargetConfig *target_config_priv;
    target_config = target_config_priv = new priv::TargetConfig();
    if (target_config) {
        s_priv.insert(target_config,target_config_priv);
        s_self.insert(target_config,target_config);
    }
    return TargetConfigPrivPair(target_config, target_config_priv);
}

TargetConfig *TargetConfigFactory::priv(ITargetConfig *target_config)
{
    BiMap<ITargetConfig *, TargetConfig *>::left_iterator f = s_priv.find_left(target_config);
    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ITargetConfig *TargetConfigFactory::i(TargetConfig *target_config)
{
    BiMap<ITargetConfig *, TargetConfig *>::right_iterator f = s_priv.find_right(target_config);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

ITargetConfig *TargetConfigFactory::self(void *s)
{
    BiMap<void *, ITargetConfig *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}

BiMap<ITargetConfig *, TargetConfig*> TargetConfigFactory::s_priv;
BiMap<void *, ITargetConfig*> TargetConfigFactory::s_self;

const char* TargetConfig::getName() const
{
    return m_instance_name.c_str();
}

void TargetConfig::setName(const char* n)
{
    API_CHECK_NULL(n);
    m_instance_name = n;

    if ( isFullConfig() )
    {
        m_targetConfigParams.m_atomicCSize = 64;
        m_targetConfigParams.m_atomicKSize = 32;
        m_targetConfigParams.m_memoryAtomicSize = 32;
        m_targetConfigParams.m_numConvBufBankAllotted = 16;
        m_targetConfigParams.m_numConvBufEntriesPerBank = 256;
        m_targetConfigParams.m_numConvBufEntryWidth = 128;
        m_targetConfigParams.m_maxBatchSize = 32;
        m_targetConfigParams.m_isPDPCapable = true;
        m_targetConfigParams.m_isCDPCapable = true;
        m_targetConfigParams.m_isSDPBiasCapable = true;
        m_targetConfigParams.m_isSDPBatchNormCapable = true;
        m_targetConfigParams.m_isSDPEltWiseCapable = true;
        m_targetConfigParams.m_isSDPLutCapable = true;
        m_targetConfigParams.m_isBDMACapable = true;
        m_targetConfigParams.m_isRubikCapable = true;
        m_targetConfigParams.m_isWinogradCapable = false;
        m_targetConfigParams.m_isCompressWeightsCapable = false;
        m_targetConfigParams.m_isBatchModeCapable = true;
    }
    else if ( isLargeConfig() )
    {
        m_targetConfigParams.m_atomicCSize = 64;
        m_targetConfigParams.m_atomicKSize = 32;
        m_targetConfigParams.m_memoryAtomicSize = 32;
        m_targetConfigParams.m_numConvBufBankAllotted = 16;
        m_targetConfigParams.m_numConvBufEntriesPerBank = 512;
        m_targetConfigParams.m_numConvBufEntryWidth = 64;
        m_targetConfigParams.m_maxBatchSize = 32;
        m_targetConfigParams.m_isPDPCapable = true;
        m_targetConfigParams.m_isCDPCapable = true;
        m_targetConfigParams.m_isSDPBiasCapable = true;
        m_targetConfigParams.m_isSDPBatchNormCapable = true;
        m_targetConfigParams.m_isSDPEltWiseCapable = true;
        m_targetConfigParams.m_isSDPLutCapable = true;
        m_targetConfigParams.m_isBDMACapable = false;
        m_targetConfigParams.m_isRubikCapable = false;
        m_targetConfigParams.m_isWinogradCapable = false;
        m_targetConfigParams.m_isCompressWeightsCapable = false;
        m_targetConfigParams.m_isBatchModeCapable = false;
    }
    else if ( isSmallConfig() )
    {
        m_targetConfigParams.m_atomicCSize = 8;
        m_targetConfigParams.m_atomicKSize = 8;
        m_targetConfigParams.m_memoryAtomicSize = 8;
        m_targetConfigParams.m_numConvBufBankAllotted = 32;
        m_targetConfigParams.m_numConvBufEntriesPerBank = 512;
        m_targetConfigParams.m_numConvBufEntryWidth = 8;
        m_targetConfigParams.m_maxBatchSize = 0;
        m_targetConfigParams.m_isPDPCapable = true;
        m_targetConfigParams.m_isCDPCapable = true;
        m_targetConfigParams.m_isSDPBiasCapable = true;
        m_targetConfigParams.m_isSDPBatchNormCapable = true;
        m_targetConfigParams.m_isSDPEltWiseCapable = false;
        m_targetConfigParams.m_isSDPLutCapable = false;
        m_targetConfigParams.m_isBDMACapable = false;
        m_targetConfigParams.m_isRubikCapable = false;
        m_targetConfigParams.m_isWinogradCapable = false;
        m_targetConfigParams.m_isCompressWeightsCapable = false;
        m_targetConfigParams.m_isBatchModeCapable = false;
    }
    else
    {
        gLogError << "Invalid target config" << std::endl;
    }
}

NvDlaError TargetConfig::initTargetConfigParams(ITargetConfig::ITargetConfigParams* targetConfigParams)
{
    NvDlaError e = NvDlaSuccess;

    m_targetConfigParams.m_atomicCSize                = targetConfigParams->atomicCSize;
    m_targetConfigParams.m_atomicKSize                = targetConfigParams->atomicKSize;
    m_targetConfigParams.m_memoryAtomicSize           = targetConfigParams->memoryAtomicSize;
    m_targetConfigParams.m_numConvBufBankAllotted     = targetConfigParams->numConvBufBankAllotted;
    m_targetConfigParams.m_numConvBufEntriesPerBank   = targetConfigParams->numConvBufEntriesPerBank;
    m_targetConfigParams.m_numConvBufEntryWidth       = targetConfigParams->numConvBufEntryWidth;
    m_targetConfigParams.m_maxBatchSize               = targetConfigParams->maxBatchSize;

    m_targetConfigParams.m_isWinogradCapable          = targetConfigParams->isWinogradCapable;
    m_targetConfigParams.m_isCompressWeightsCapable   = targetConfigParams->isCompressWeightsCapable;
    m_targetConfigParams.m_isBatchModeCapable         = targetConfigParams->isBatchModeCapable;
    m_targetConfigParams.m_isPDPCapable               = targetConfigParams->isPDPCapable;
    m_targetConfigParams.m_isCDPCapable               = targetConfigParams->isCDPCapable;
    m_targetConfigParams.m_isSDPBiasCapable           = targetConfigParams->isSDPBiasCapable;
    m_targetConfigParams.m_isSDPBatchNormCapable      = targetConfigParams->isSDPBatchNormCapable;
    m_targetConfigParams.m_isSDPEltWiseCapable        = targetConfigParams->isSDPEltWiseCapable;
    m_targetConfigParams.m_isSDPLutCapable            = targetConfigParams->isSDPLutCapable;
    m_targetConfigParams.m_isBDMACapable              = targetConfigParams->isBDMACapable;
    m_targetConfigParams.m_isRubikCapable             = targetConfigParams->isRubikCapable;

    return e;
}

} // nvdla::priv

} // nvdla::
