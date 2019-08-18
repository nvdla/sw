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

#include "priv/Check.h"

#include "priv/Setup.h"

using std::string;
using std::endl;


namespace nvdla
{

ISetup::ISetup() { }
ISetup::~ISetup() { }



ISetup *createSetup()
{
    priv::SetupFactory::SetupPrivPair p = priv::SetupFactory::newSetup();
    return p.i();
}

NvDlaError destroySetup(ISetup *setup)
{
    return priv::SetupFactory::deleteSetup(setup);
}

namespace priv
{


SetupFactory::SetupPrivPair SetupFactory::newSetup()
{
    ISetup *setup;
    Setup *setup_priv;
    setup = setup_priv = new priv::Setup();
    if (setup) {
        s_priv.insert(setup, setup_priv);
    }
    return SetupPrivPair(setup, setup_priv);
}

NvDlaError SetupFactory::deleteSetup(ISetup *setup)
{
    if (setup != NULL) {
        Setup *setup_priv = priv(setup);
        if (setup_priv != NULL)
            delete setup_priv;

        s_priv.remove(setup);
    }

    return NvDlaSuccess;
}

Setup *SetupFactory::priv(ISetup *setup)
{
    BiMap<ISetup *, Setup *>::left_iterator f = s_priv.find_left(setup);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ISetup *SetupFactory::i(Setup *setup)
{
    BiMap<ISetup *, Setup *>::right_iterator f = s_priv.find_right(setup);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}


BiMap<ISetup *, Setup*> SetupFactory::s_priv;

Setup::Setup() :
    ISetup(),
    m_wisdom(0)
{

}

Setup::~Setup()
{

}

IWisdom *Setup::wisdom()
{
    return (IWisdom*)m_wisdom; // Wisdom->IWisdom ok, but tbd: hook up WisdomFactory...
}

NvU16 Setup::getFactoryType() const
{
    return 0; // only one kind so far
}



} // nvdla::priv

} // nvdla
