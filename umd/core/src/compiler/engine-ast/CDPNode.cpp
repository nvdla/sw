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

#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

using std::endl;

namespace nvdla
{
namespace priv
{


void engine_ast::CDPNode::captureCanonicalParams()
{

}

NvDlaError engine_ast::CDPNode::verifySurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;

    bool isSrcTSD = false;
    bool isDstTSD = false;
    surface::TensorSurfaceDesc* srcTSD = NULL;
    surface::TensorSurfaceDesc* dstTSD = NULL;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    isSrcTSD = inputEdges()[0]->tensorSurfaceDesc() == tsd;
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;

    if (!isSrcTSD && !isDstTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }

    srcTSD = isSrcTSD ? tsd : inputEdges()[0]->tensorSurfaceDesc();
    dstTSD = isDstTSD ? tsd : outputEdges()[0]->tensorSurfaceDesc();

    if (srcTSD->dimensions() != dstTSD->dimensions())
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "(%s) Input and Output tensors should have the same dimensions",
                                                name().c_str());
    }

fail:
    return e;
}

/*------------------------------Handle Multi-Batch---------------------*/
NvDlaError engine_ast::CDPNode::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;

    //Handle operation parameters for the multi-batch operations
    NvU32 numBatches = graph()->profile()->multiBatchSize();
    for (NvU32 nn = 1; nn < numBatches; ++nn)
    {
        params(nn) = params(0);
    }

    return e;
}

};  // nvdla::priv::
};  // nvdla::
