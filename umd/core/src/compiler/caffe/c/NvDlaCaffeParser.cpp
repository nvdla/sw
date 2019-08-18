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

#include "priv/c/API.h"

#include "ErrorMacros.h"

using nvdla::caffe::ICaffeParser;
using nvdla::INetwork;
using nvdla::IWisdom;

#define CHECK_SELF()       \
    NvDlaError e = NvDlaSuccess; \
    ICaffeParser *parser; \
    e = nvdla::priv::checkSelf<NvDlaCaffeParser, ICaffeParser, nvdla::caffe::priv::CaffeParserFactory>(cpt, parser); \
    PROPAGATE_ERROR_FAIL( e )





namespace nvdla {

namespace priv {


static NvDlaError caffeParserParse(NvDlaCaffeParser cpt, const char *d, const char *m, NvDlaNetwork nt)
{
    INetwork *network;
    CHECK_SELF();
    PROPAGATE_ERROR_FAIL( checkNetworkSelf(nt, network) );
    if ( ! parser->parse(d, m, network) ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
 fail:
    return e;
}

static NvDlaError caffeParserIdentifyOutputs(NvDlaCaffeParser cpt, NvDlaNetwork nt)
{
    INetwork *network;
    CHECK_SELF();
    PROPAGATE_ERROR_FAIL( checkNetworkSelf(nt, network) );
    if ( parser->identifyOutputs(network) != 0 ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
 fail:
    return e;
}

const struct NvDlaCaffeParserI caffeParserI =
{
    /* .parse = */ caffeParserParse,
    /* .identifyOutputs = */ caffeParserIdentifyOutputs,
};


} // nvdla::priv

} // nvdla


extern "C"
{

    NvDlaError NvDlaCreateCaffeParser(NvDlaCaffeParser *rcpt)
    {
        NvDlaError e = NvDlaSuccess;
        if ( !rcpt ) {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
        }
        rcpt->self = nvdla::caffe::createCaffeParser();
        rcpt->i = &nvdla::priv::caffeParserI;

    fail:
        return e;
    }

    NvDlaError NvDlaDestroyCaffeParser(NvDlaCaffeParser cpt)
    {
        CHECK_SELF();
        PROPAGATE_ERROR_FAIL(nvdla::caffe::destroyCaffeParser(parser));

    fail:
        return e;
    }
}
