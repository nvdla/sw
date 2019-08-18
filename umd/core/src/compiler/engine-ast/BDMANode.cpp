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

#include <iostream>
#include <math.h>

#include "priv/EngineAST.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{

//----------------------------------------------------------------------
//                           Code Emission Utils
//----------------------------------------------------------------------
engine_ast::BDMAEngineParams engine_ast::BDMANode::calcTransferParams
(
    surface::TensorSurfaceDesc* in_tsd,
    surface::TensorSurfaceDesc* out_tsd
)
{
    NvDlaError e = NvDlaError_Success;
    NVDLA_UNUSED(e);
    engine_ast::BDMAEngineParams transfer_params;
    int bpe;

    if (in_tsd->surfaceFormat().bytesPerElement() != out_tsd->surfaceFormat().bytesPerElement())
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Input and Output surfaces have different precisions, %d and %d",
                                                   (int)in_tsd->surfaceFormat().bytesPerElement(),
                                                   (int)out_tsd->surfaceFormat().bytesPerElement());
    }

    bpe = in_tsd->surfaceFormat().bytesPerElement();


    if (in_tsd->dimensions().c != out_tsd->dimensions().c ||
        in_tsd->dimensions().h != out_tsd->dimensions().h ||
        in_tsd->dimensions().w != out_tsd->dimensions().w)
    {
        REPORT_ERROR(NvDlaError_BadParameter, "Input and Output surfaces have different dimensions:");
        REPORT_ERROR(NvDlaError_BadParameter, "Src[%d x %d x %d]", (int)in_tsd->dimensions().c, (int)in_tsd->dimensions().h, (int)in_tsd->dimensions().w);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Dst[%d x %d x %d]", (int)out_tsd->dimensions().c, (int)out_tsd->dimensions().h, (int)out_tsd->dimensions().w);
    }

    gLogInfo << "bdma xfr params bpe=" << bpe << endl;

#if 0
    {
        NvU32 line_size = in_tsd->dimensions().w * bpe;
        NvU32 padded_line_size = 32 * ((line_size + 31) / 32); // tbd: get lineStride from surface
        transfer_params.setLineSize( 32 * ((line_size + 31)/32));
        transfer_params.setLineRepeat( in_tsd->dimensions().h );
        transfer_params.setSrcLine( padded_line_size );
        transfer_params.setDestLine( padded_line_size );

        if ( in_tsd->surfaceFormat().category().v() == surface::SurfaceCategoryEnum::FEATURE_DATA )
        {
            transfer_params.setSrcSurface( padded_line_size * in_tsd->dimensions().h );
            transfer_params.setDestSurface( padded_line_size * out_tsd->dimensions().h );
            transfer_params.setSurfaceRepeat(in_tsd->dimensions().c);
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_SurfaceNotSupported, "Surface Format Not Supported!!");
        }
    }
#else
    // hack for transfer debug.  just move 32B per.
    {
        NvU32 line_size = in_tsd->dimensions().w * bpe;
        NvU32 padded_line_size = 32 * ((line_size + 31) / 32);
        NVDLA_UNUSED(padded_line_size);
        transfer_params.setLineSize( 32 );
        transfer_params.setLineRepeat( 1 );
        transfer_params.setSrcLine( 32 );
        transfer_params.setDestLine( 32 );

        if ( in_tsd->surfaceFormat().category().v() == surface::SurfaceCategoryEnum::FEATURE_DATA )
        {
            transfer_params.setSrcSurface( 32 );
            transfer_params.setDestSurface( 32 );
            transfer_params.setSurfaceRepeat( 1 );
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_SurfaceNotSupported, "Surface Format Not Supported!!");
        }


    }
#endif




fail:
    return transfer_params;
}

};  // nvdla::priv::
};  // nvdla::
