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

#include <map>
#include <string>

#include "priv/Check.h"

#include "priv/Profile.h"
#include "priv/Profiler.h"
#include "priv/TargetConfig.h"


using std::string;
using std::endl;


namespace nvdla
{

IProfiler::IProfiler() { }
IProfiler::~IProfiler() { }

namespace priv
{


ProfilerFactory::ProfilerPrivPair ProfilerFactory::newProfiler()
{
    IProfiler *profiler;
    Profiler *profiler_priv;
    profiler = profiler_priv = new priv::Profiler();
    if (profiler) {
        s_priv.insert(profiler, profiler_priv);
        s_self.insert(profiler, profiler);
    }
    return ProfilerPrivPair(profiler, profiler_priv);
}

NvDlaError ProfilerFactory::deleteProfiler(IProfiler *profiler)
{
    if (profiler != NULL) {
        Profiler *profiler_priv = priv(profiler);
        if (profiler_priv != NULL) {
            delete profiler_priv;
        }

        s_priv.remove(profiler);
        s_priv.remove(profiler);
    }

    return NvDlaSuccess;
}

Profiler *ProfilerFactory::priv(IProfiler *profiler)
{
    BiMap<IProfiler *, Profiler *>::left_iterator f = s_priv.find_left(profiler);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

IProfiler *ProfilerFactory::i(Profiler *profiler)
{
    BiMap<IProfiler *, Profiler *>::right_iterator f = s_priv.find_right(profiler);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

IProfiler *ProfilerFactory::self(void *s)
{
    BiMap<void *, IProfiler *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}



BiMap<IProfiler *, Profiler*> ProfilerFactory::s_priv;
BiMap<void *, IProfiler*> ProfilerFactory::s_self;

Profiler::Profiler() :
    IProfiler(),
    m_wisdom(0)
{

}

Profiler::~Profiler()
{

}

IWisdom *Profiler::wisdom()
{
    return (IWisdom*)m_wisdom; // Wisdom->IWisdom ok, but tbd: hook up WisdomFactory...
}

NvU16 Profiler::getFactoryType() const
{
    return 0; // only one kind so far
}

IProfile *Profiler::createProfile(const char *profile_name)
{
    return getProfile(profile_name);
}

IProfile *Profiler::getProfile(const char *name)
{
    ProfileFactory::ProfilePrivPair pp;
    std::map<std::string, ProfileFactory::ProfilePrivPair>::iterator f;
    IProfile *iprofile = NULL;

    if (name == NULL)
    {
        goto fail;
    }
    f = m_profiles.find(std::string(name));

    if ( f != m_profiles.end() )
    {
        pp = f->second;
    }
    else
    {
        pp = ProfileFactory::newProfile();
        m_profiles[name] = pp;
        pp.priv()->setName(name);
    }

    iprofile = pp.i();

fail:
    return iprofile;

}

ITargetConfig *Profiler::getTargetConfig(const char *name)
{
    TargetConfigFactory::TargetConfigPrivPair pp;
    std::map<std::string, TargetConfigFactory::TargetConfigPrivPair>::iterator f;
    ITargetConfig *itargetconfig = NULL;

    if (name == NULL)
    {
        goto fail;
    }
    f = m_targetConfigs.find(std::string(name));

    if ( m_targetConfigs.find(name) != m_targetConfigs.end() )
    {
        pp = f->second;
    }
    else
    {
        pp = TargetConfigFactory::newTargetConfig();
        m_targetConfigs[name] = pp;
        pp.priv()->setName(name);
    }

    itargetconfig = pp.i();

fail:
    return itargetconfig;
}

} // nvdla::priv

} // nvdla
