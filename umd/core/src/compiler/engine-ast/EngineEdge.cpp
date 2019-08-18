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

#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

#include <cmath>

using std::map;
using std::vector;
using std::endl;


namespace nvdla
{
namespace priv
{

//----------------------------------------------------------------------
//                           Compiler Utils
//----------------------------------------------------------------------
/*--------------------------Register Surface Desc---------------------*/
/* idempotent */
NvDlaError engine_ast::Edge::registerSurface()
{
    NvDlaError e = NvDlaError_Success;

    surface::TensorSurfaceDesc* tsd = NULL;
    TensorType tt;
    NvU16 numBatches = graph()->profile()->multiBatchSize();

    if ( !isDataEdge() )
    {
        goto fail;
    }

    tt = originalTensor() ? originalTensor()->getTensorType() : TensorType::kDEBUG;

    if ( tt == TensorType::kUNKNOWN )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unidentified TensorType '%d' in edge '%s'", tt, id().c_str());
    }

    tsd = tensorSurfaceDesc();
    if ( !tsd )
    {
        tsd = graph()->resourceMgr()->regTensorSurfaceDesc(tt, numBatches);
        tsd->setName(std::string(originalTensor()->getName()));
        tsd->setBufferOffset(0);        // default offset
        tsd->setDimensions(originalTensor()->getDimensions());
        tsd->setCopyOutDebugSurface(tt == TensorType::kDEBUG);
        tsd->setDataFormat(originalTensor()->getDataFormat());
        tsd->setParentEdge(this);
        setTensorSurfaceDesc(tsd);

        //
        // is this edge related to a bindable resource?
        // if so we need to maintain that connection.
        //
        if ( bindable() )
        {
            enum IOD bindDomain;
            NvS16 bid = bindId(bindDomain);
            tsd->setBindId(bid, bindDomain);
            if ( debugBinding() )
            {
                gLogInfo << "set bind id " << bid << " for " << id() << " " << tsd->id() << endl;
            }

            // tbd: theoretically could be cvsram as well?
            // choosing not to support it for now.
            //tsd->setMemoryLoc(memory::LocationEnum::lDRAM);
        }

        if ( graph()->debugSurfaces() )
        {
            gLogInfo << ((tt == TensorType::kDEBUG) ? "(debug) ":"" ) <<
                "edge: " << id() << " tsd: " << tsd->id() <<
                " registered" << endl;
        }
    }

fail:
    return e;
}

/*----------------------Determine Surface Clients----------------------*/
NvDlaError engine_ast::Edge::determineSurfaceClients()
{
    NvDlaError e = NvDlaSuccess;
    Graph::NodeSequence upNodes, downNodes;

    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();

    if (!isDataEdge())
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Surface Desc not yet registered for %s", id().c_str());
    }

    tsd->clearProducers();
    upNodes = graph()->upstreamNodes(this);
    for (vector< engine_ast::Node* >::const_iterator ni = upNodes.begin();
         ni != upNodes.end(); ++ni)
    {
        tsd->addProducer((*ni));
    }

    tsd->clearConsumers();
    downNodes = graph()->downstreamNodes(this);
    for (vector< engine_ast::Node* >::const_iterator ni = downNodes.begin();
         ni != downNodes.end(); ++ni)
    {
        tsd->addConsumer((*ni));
    }

fail:
    return e;
}

/*----------------------Determine Surface Format-----------------------*/
NvDlaError engine_ast::Edge::determineSurfaceFormat()
{
    NvDlaError e = NvDlaSuccess;
    TensorType tt;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    bool isAUXSurface, isInterimSurface, isBindableSurface;
    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    std::set<surface::SurfaceFormat> producerProposedSFs;
    std::set<surface::SurfaceFormat> consumerProposedSFs;
    std::set<surface::SurfaceFormat> proposedSFs;
    std::vector<surface::SurfaceFormat> suggestedSFs;
    std::vector<surface::SurfaceFormat> supportedSFs;

    if (!isDataEdge())
    {
        goto fail;
    }

    tt = originalTensor()->getTensorType();
    isAUXSurface      = (tt == TensorType::kWEIGHT || tt == TensorType::kBIAS ||
                         tt == TensorType::kBATCH_NORM || tt == TensorType::kSCALE);
    isInterimSurface  = (tt == TensorType::kIO || tt == TensorType::kSTREAM ||
                         tt == TensorType::kDEBUG);
    isBindableSurface = (tt == TensorType::kNW_INPUT || tt == TensorType::kNW_OUTPUT);

    if (!isAUXSurface && !isInterimSurface && !isBindableSurface)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unsupported tensor type %d", (int)tt);
    }

    if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Surface Desc not yet registered for %s", id().c_str());
    }
    else if (tsd->surfaceFormat().v() != surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT)
    {
        if ( graph()->debugSurfaces() )
        {
            gLogInfo << id() << " edge already has set surface format " <<
                    tsd->surfaceFormat().c_str() << endl;
        }
        goto fail;
    }

    producers = tsd->producers();
    consumers = tsd->consumers();

    // Step-1: Capture surf formats suggested by consumer nodes
    for (Graph::NodeUnorderedSetIterator ci = consumers.begin(); ci != consumers.end(); ++ci)
    {
        suggestedSFs.clear();
        if (isAUXSurface)
        {
            suggestedSFs = (*ci)->suggestAuxSurfaceFormats(this);
        }
        else if (isInterimSurface)
        {
            suggestedSFs = (*ci)->suggestInputSurfaceFormats();
        }
        else if (isBindableSurface)
        {
            suggestedSFs = std::vector<surface::SurfaceFormat>(1, graph()->suggestNwSurfaceFormat(TensorType::kNW_INPUT));
        }
        std::copy(suggestedSFs.begin(), suggestedSFs.end(), std::inserter(consumerProposedSFs, consumerProposedSFs.end()));
    }

    // Step-2: Capture surf formats suggested by producer nodes
    for (Graph::NodeUnorderedSetIterator pi = producers.begin(); pi != producers.end(); ++pi)
    {
        suggestedSFs.clear();
        if (isInterimSurface)
        {
            suggestedSFs = (*pi)->suggestOutputSurfaceFormats();
        }
        else if (isBindableSurface)
        {
            suggestedSFs = std::vector<surface::SurfaceFormat>(1, graph()->suggestNwSurfaceFormat(TensorType::kNW_OUTPUT));
        }
        std::copy(suggestedSFs.begin(), suggestedSFs.end(), std::inserter(producerProposedSFs, producerProposedSFs.end()));
    }

    // Step-3: Find intersection of suggested surf formats from producers and consumers
    if (isBindableSurface)
    {
        std::copy(producerProposedSFs.begin(), producerProposedSFs.end(), std::inserter(proposedSFs, proposedSFs.end()));
        std::copy(consumerProposedSFs.begin(), consumerProposedSFs.end(), std::inserter(proposedSFs, proposedSFs.end()));
    }
    else if (isAUXSurface)
    {
        std::copy(consumerProposedSFs.begin(), consumerProposedSFs.end(), std::inserter(proposedSFs, proposedSFs.end()));
    }
    else if (isInterimSurface)
    {
        for (std::set<surface::SurfaceFormat>::iterator csfi = consumerProposedSFs.begin(); csfi != consumerProposedSFs.end(); ++csfi)
        {
            if (producerProposedSFs.find(*csfi) != producerProposedSFs.end())
            {
                proposedSFs.insert(*csfi);
            }
        }
        for (std::set<surface::SurfaceFormat>::iterator psfi = producerProposedSFs.begin(); psfi != producerProposedSFs.end(); ++psfi)
        {
            if (consumerProposedSFs.find(*psfi) != consumerProposedSFs.end())
            {
                proposedSFs.insert(*psfi);
            }
        }
    }

    // prune the proposed surface formats based on criteria
    // Criteria-1: prune those SFs that don't fit with #chnls of the TSD
    for (std::set<surface::SurfaceFormat>::iterator sfi = proposedSFs.begin(); sfi != proposedSFs.end(); )
    {
        // allow surface formats that work with any #channels
        if ((*sfi).channelsPerAtom() == -1)
            ++sfi;
        else if ((*sfi).channelsPerAtom() != tsd->dimensions().c)
            proposedSFs.erase(sfi++);
        else
            ++sfi;
    }

    if (proposedSFs.size() == 0)
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Couldn't narrow down to suitable SF for %s/%s",
                id().c_str(), tsd->id().c_str());
    }
    else if (proposedSFs.size() > 1)
    {
        gLogInfo << "More than 1 proposed SFs for " << id() << "/" << tsd->id() << endl;
        for (std::set<surface::SurfaceFormat>::iterator sfi = proposedSFs.begin(); sfi != proposedSFs.end(); ++sfi)
            gLogInfo << (*sfi).c_str() << endl;
    }

    //TODO: in future, loop over all proposed sfs and fork graphs
    for (Graph::NodeUnorderedSetIterator ci = consumers.begin(); ci != consumers.end(); ++ci)
    {
        supportedSFs.clear();
        if (isAUXSurface)
        {
            supportedSFs = (*ci)->supportedAuxSurfFormats();
        }
        else if (isInterimSurface || isBindableSurface)
        {
            supportedSFs = (*ci)->supportedInSurfFormats();
        }
        PROPAGATE_ERROR_FAIL((*ci)->supportsSurfaceFormat(*(proposedSFs.begin()), supportedSFs));
    }

    for (Graph::NodeUnorderedSetIterator pi = producers.begin(); pi != producers.end(); ++pi)
    {
        supportedSFs.clear();
        if (isInterimSurface || isBindableSurface)
        {
            supportedSFs = (*pi)->supportedOutSurfFormats();
        }
        PROPAGATE_ERROR_FAIL((*pi)->supportsSurfaceFormat(*(proposedSFs.begin()), supportedSFs));
    }

    ASSERT(proposedSFs.size() >= 1);

    tsd->setSurfaceFormat(*(proposedSFs.begin()));

    if ( graph()->debugSurfaces() )
    {
        gLogInfo << id() << " edge setting new surface format "
                 << tsd->surfaceFormat().c_str() << endl;
    }

fail:
    return e;
}

/*----------------------Determine Surface Strides----------------------*/
NvDlaError engine_ast::Edge::determineSurfaceStrides()
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    NvU32 commonLS  = 0;
    NvU32 commonSS  = 0;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    tsd->resetLineStride();
    tsd->resetSurfaceStride();

    /*
     * The readers and writers of a tensor could have differing stride alignment
     * limitations. DLA supports reading/writing of smaller cubes within a larger cube
     * provided the strides and dimensions are programmed correctly.
     * To allow that, the strides for the surface should always represent the larger cube.
     */
    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        commonLS = std::max<NvU32>(commonLS, (*cli)->suggestLineStride(tsd));
        commonSS = std::max<NvU32>(commonSS, (*cli)->suggestSurfaceStride(tsd));
    }

    tsd->setLineStride(commonLS);
    tsd->setSurfaceStride(commonSS);

fail:
    return e;
}

NvDlaError engine_ast::Edge::determineSurfaceSize()
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    NvU64 commonSize  = 0;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    tsd->resetSize();

    /*
     * The readers and writers of a tensor could have differing stride and size
     * requirements. DLA supports reading/writing of smaller cubes within a larger cube.
     * To allow that, the size for the surface should always represent the larger cube.
     */
    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        commonSize = std::max<NvU64>(commonSize, (*cli)->suggestSurfaceSize(tsd));
    }

    tsd->setSize(commonSize);

fail:
    return e;
}

NvDlaError engine_ast::Edge::determineSurfaceOffsetInBuffer()
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    NvU64 bufferOffset  = 0;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    tsd->resetBufferOffset();

    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        // arguably, all client nodes should report the same surface offset in buffer
        bufferOffset = std::max<NvU64>(bufferOffset, (*cli)->suggestSurfaceOffsetInBuffer(tsd));
    }

    tsd->setBufferOffset(bufferOffset);

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceClients()
{
    NvDlaError e = NvDlaSuccess;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeSequence upNodes;
    Graph::NodeSequence downNodes;
    Graph::NodeUnorderedSet producerNodes;
    Graph::NodeUnorderedSet consumerNodes;

    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }


    upNodes   = graph()->upstreamNodes(this);
    producers = tensorSurfaceDesc()->producers();
    for (Graph::NodeSequenceIterator uni = upNodes.begin(); uni != upNodes.end(); ++uni)
        producerNodes.insert(*uni);

    downNodes = graph()->downstreamNodes(this);
    consumers = tensorSurfaceDesc()->consumers();
    for (Graph::NodeSequenceIterator dni = downNodes.begin(); dni != downNodes.end(); ++dni)
        consumerNodes.insert(*dni);

    if (producers != producerNodes)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Producer nodes (%d) != upstream nodes (%d) for %s",
                producerNodes.size(), producers.size(), tsd->id().c_str());
    }
    else if (consumers != consumerNodes)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer nodes (%d) != downstream nodes (%d) for %s",
                consumerNodes.size(), consumers.size(), tsd->id().c_str());
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceFormat()
{
    NvDlaError e = NvDlaSuccess;
    TensorType tt;
    surface::SurfaceFormat sf;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    bool isAUXSurface, isInterimSurface, isBindableSurface;
    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    std::vector<surface::SurfaceFormat> supportedSFs;

    if ( !isDataEdge() )
    {
        goto fail;
    }

    tt = originalTensor()->getTensorType();
    isAUXSurface      = (tt == TensorType::kWEIGHT || tt == TensorType::kBIAS ||
                         tt == TensorType::kBATCH_NORM || tt == TensorType::kSCALE);
    isInterimSurface  = (tt == TensorType::kIO || tt == TensorType::kSTREAM ||
                         tt == TensorType::kDEBUG);
    isBindableSurface = (tt == TensorType::kNW_INPUT || tt == TensorType::kNW_OUTPUT);

    if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    sf = tsd->surfaceFormat();
    if (sf.v() == surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT)
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Surface format not yet determined for %s",
                tsd->id().c_str());
    }

    producers = tsd->producers();
    consumers = tsd->consumers();
    for (Graph::NodeUnorderedSetIterator ci = consumers.begin(); ci != consumers.end(); ++ci)
    {
        supportedSFs.clear();
        if (isAUXSurface)
        {
            supportedSFs = (*ci)->supportedAuxSurfFormats();
        }
        else if (isInterimSurface || isBindableSurface)
        {
            supportedSFs = (*ci)->supportedInSurfFormats();
        }
        PROPAGATE_ERROR_FAIL((*ci)->supportsSurfaceFormat(sf, supportedSFs));
    }

    for (Graph::NodeUnorderedSetIterator pi = producers.begin(); pi != producers.end(); ++pi)
    {
        supportedSFs.clear();
        if (isInterimSurface || isBindableSurface)
        {
            supportedSFs = (*pi)->supportedOutSurfFormats();
        }
        PROPAGATE_ERROR_FAIL((*pi)->supportsSurfaceFormat(sf, supportedSFs));
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceDims()
{
    NvDlaError e = NvDlaSuccess;
    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        PROPAGATE_ERROR_FAIL( (*cli)->verifySurfaceDims(tsd) );
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceStrides()
{
    NvDlaError e = NvDlaSuccess;
    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    /*
     * The readers and writers of a tensor could have differing stride alignment
     * limitations. DLA supports reading/writing of smaller cubes within a larger cube
     * provided the strides and dimensions are programmed correctly.
     * Verify that, the strides for the surface should always represent the larger cube.
     */
    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        ASSERT(tsd->lineStride() >= (*cli)->suggestLineStride(tsd));
        ASSERT(tsd->surfaceStride() >= (*cli)->suggestSurfaceStride(tsd));
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceSize()
{
    NvDlaError e = NvDlaSuccess;
    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    /*
     * The readers and writers of a tensor could have differing stride alignment
     * limitations. DLA supports reading/writing of smaller cubes within a larger cube
     * provided the strides, size and dimensions are programmed correctly.
     * Verify that, the size for the surface should always represent the larger cube.
     */
    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        ASSERT(tsd->size() >= (*cli)->suggestSurfaceSize(tsd));
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceOffsetInBuffer()
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        // all client nodes should report the same surface offset in buffer
        ASSERT(tsd->bufferOffset() == (*cli)->suggestSurfaceOffsetInBuffer(tsd));
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurfaceTensorScales()
{
    NvDlaError e = NvDlaSuccess;
    Tensor* tensor = originalTensor();
    Dims4 tensorDims;
    std::vector<NvF32> tensorScales;
    NvF32 perTensorScale;

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if ( isAuxEdge() )
    {
        goto fail;
    }
    else if (graph()->profile()->computePrecision().v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        goto fail;
    }
    else if (!tensor)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Tensor not yet registered for edge %s", id().c_str());
    }
    if (graph()->profile()->tensorScalingMode().v() != nvdla::TensorScalingMode::PER_TENSOR)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support any tensor scaling mode other "
                                                 " than PER_TENSOR for this network\n");
    }

    tensorDims = tensor->getDimensions();
    tensorScales = tensor->getChannelScales();

    ASSERT (tensorScales.size() == static_cast<NvU32>(tensorDims.c));

    perTensorScale = tensorScales.at(0);
    for (int cc = 0; cc < tensorDims.c; ++cc)
    {
        ASSERT (tensorScales.at(cc) != 0);
        ASSERT (!std::isnan(tensorScales.at(cc)));
        ASSERT (!std::isinf(tensorScales.at(cc)));
        ASSERT (tensorScales.at(cc) == perTensorScale);
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifySurface()
{
    NvDlaError e = NvDlaSuccess;

    /* Verify that the producers and consumers of the tsd are
     * the same as the physical upstream and downstream nodes
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceClients() );

    /* Verify that the surface format determined for the tsd are
     * compatible with all the node(s) operating on it.
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceFormat() );

    /* Verify that none of the node(s) changed the dims of any tsd,
     * such that the node(s) on the other end of that tsd couln't
     * operate on it anymore
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceDims() );

    /* Verify that the surface strides determined for the tsd are
     * compatible with all the node(s) operating on it.
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceStrides() );

    /* Verify that the size determined for the tsd are
     * compatible with all the node(s) operating on it.
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceSize() );

    /* Verify that the surface offset determined in the buffer
     * for the tsd is compatible with all the node(s) operating on it.
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceOffsetInBuffer() );

    /* Verify that each tensor has channel scales set and
     * that they are valid
     */
    PROPAGATE_ERROR_FAIL( verifySurfaceTensorScales() );

fail:
    return e;
}

/*----------------------I/O Buffer Descriptor Registration------------*/
NvDlaError engine_ast::Edge::registerBuffer()
{
    NvDlaError e = NvDlaError_Success;
    typedef memory::TensorBufferDesc TBD;

    surface::TensorSurfaceDesc *tsd;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    TBD* currTBD = NULL;
    TBD* commonTBD = NULL;
    std::map<Node*, TBD*> clientBufferMap;

    tsd = tensorSurfaceDesc();

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }
    else if ( tsd->tensorCategory().v() == memory::TensorCategoryEnum::UNKNOWN_TENSOR )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Edge %s has 'unknown' tensor category",
                             tsd->id().c_str());
    }

    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    commonTBD = tsd->tensorBufferDesc();
    if ( !commonTBD )
    {
        Graph::NodeUnorderedSetIterator cli;

        for (cli = clients.begin(); cli != clients.end(); ++cli)
        {
            if ((*cli)->isSoftwareNode())
            {
                commonTBD = (*cli)->suggestBuffer(tsd);
                break;
            }
        }

        // Step-1: If there's a software client, prefer its suggested TBD
        //         don't bother querying the TBDs from non-software clients
        if (commonTBD)
        {
            tsd->setTensorBufferDesc(commonTBD);
        }
        // Step-2: If there's no software client, assert that all clients suggested 1 common TBD
        else
        {
            for (cli = clients.begin(); cli != clients.end(); ++cli)
            {
                currTBD = (*cli)->suggestBuffer(tsd);
                if (cli == clients.begin())
                {
                    commonTBD = currTBD;
                    tsd->setTensorBufferDesc(commonTBD);
                }
                else
                {
                    ASSERT(currTBD == commonTBD);
                }
            }
        }
    }

    PROPAGATE_ERROR_FAIL( commonTBD->addSurface(tsd) );
    if ( graph()->debugBuffers() )
    {
        gLogInfo << commonTBD->id() << " for " << tsd->id() << " for " << id() << " with " << tsd->surfaceFormat().c_str() << endl;
    }

fail:
    return e;
}

NvDlaError engine_ast::Edge::verifyBuffer()
{
    NvDlaError e = NvDlaError_Success;

    surface::TensorSurfaceDesc *tsd;
    Graph::NodeUnorderedSet producers;
    Graph::NodeUnorderedSet consumers;
    Graph::NodeUnorderedSet clients;

    tsd = tensorSurfaceDesc();

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }
    else if ( tsd->tensorCategory().v() == memory::TensorCategoryEnum::UNKNOWN_TENSOR )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Edge %s has 'unknown' tensor category",
                             tsd->id().c_str());
    }

    producers = tsd->producers();
    consumers = tsd->consumers();
    clients.insert(producers.begin(), producers.end());
    clients.insert(consumers.begin(), consumers.end());

    for (Graph::NodeUnorderedSetIterator cli = clients.begin(); cli != clients.end(); ++cli)
    {
        // all client nodes should report the same buffer
        ASSERT(tsd->tensorBufferDesc() == (*cli)->suggestBuffer(tsd));
    }

fail:
    return e;
}

/*----------------------I/O Buffer Reservation-------------------------*/
/* idempotent */
NvDlaError engine_ast::Edge::reserveBuffer()
{
    NvDlaError e = NvDlaError_Success;

    NvU64 existingSize = 0;
    NvU64 proposedSize = 0;
    memory::TensorCategory tc;
    memory::TensorBufferDesc* tbd;
    surface::TensorSurfaceDesc* tsd = tensorSurfaceDesc();
    NvU16 numBatches = graph()->profile()->multiBatchSize();

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!tsd)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }
    else if (!tsd->size())
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "surface size == 0 for %s", tsd->id().c_str());
    }

    tbd = tsd->tensorBufferDesc();
    tc  = tsd->tensorCategory();
    existingSize = tbd->size();
    proposedSize = tsd->size();
    switch(tc.v())
    {
        case memory::TensorCategoryEnum::GLOBAL_TENSOR:
        case memory::TensorCategoryEnum::LOCAL_TENSOR:
            tbd->setSize( std::max<NvU64>(existingSize, proposedSize) );
            break;
        case memory::TensorCategoryEnum::EXTERNAL_TENSOR:
            ASSERT( bindable() );
            // adjust buffer size for multiple batches on the bindable tensor
            if ( existingSize )
            {
                tbd->setSize( std::max<NvU64>(existingSize, proposedSize * numBatches) );
            }
            else
            {
                tbd->setSize(proposedSize * numBatches);
            }
            break;
        case memory::TensorCategoryEnum::STREAM_TENSOR:
            tbd->setMemoryLoc(memory::LocationEnum::lSTREAM);
            break;

        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Tensor Category:%s not recognized", tc.c_str());
    }

fail:
    return e;
}

/*----------------------Handle Multi Batch-----------------------------*/
NvDlaError engine_ast::Edge::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;

    surface::TensorSurfaceDesc* mbTSD = tensorSurfaceDesc();
    NvU32 numBatches = graph()->profile()->multiBatchSize();

    if ( !isDataEdge() )
    {
        goto fail;
    }
    else if (!mbTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD not yet registered for edge %s", id().c_str());
    }

    if ( bindable() )
    {
        NvU64 offsetInBindableBuffer = 0;
        if ( mbTSD->dimensions().n != (NvS32)numBatches )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Nw edge %s doesn't represent a multi batch bindable tensor", id().c_str());
        }
        for (NvU16 nn = 0; nn < numBatches; ++nn)
        {
            // for a bindable tensor surf desc, different batches scribble at different offsets in the same buffer
            mbTSD->setBufferOffset(offsetInBindableBuffer, nn);
            offsetInBindableBuffer += mbTSD->size();
        }
    }
    else
    {
        memory::Location firstBatchMemLoc = mbTSD->tensorBufferDesc()->memoryLoc(0);
        NvU64 offsetInNonBindableBuffer = mbTSD->bufferOffset(/*batchId*/0);

        if ( !isAuxEdge() && mbTSD->dimensions().n != 1 )
        {
            // FIXME: allow true NCHW contiguous multibatch for intermediate tensors as well. atleast for FC first
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Contiguous multi-batch non-bindable tensor at %s/%s is not yet supported", id().c_str(),
                    mbTSD->id().c_str());
        }
        for ( NvU16 nn = 0; nn < numBatches; ++nn )
        {
            /* For a non-bindable tensor surf desc, different batches scribble at same offset in different buffers
             * If a non-bindable tensor shares the buffer with a bindable tensor, then account for the existing offsets
             * of each batch in the NCHW bindable tensor.
             */
            NvU64 batchOffsetInBindableBuffer = mbTSD->tensorBufferDesc()->boundSurface(0) ?
                                                mbTSD->tensorBufferDesc()->boundSurface(0)->bufferOffset(nn) : 0;
            mbTSD->setBufferOffset(batchOffsetInBindableBuffer + offsetInNonBindableBuffer, nn);
            mbTSD->tensorBufferDesc()->setMemoryLoc(firstBatchMemLoc, nn);
        }
    }
fail:
    return e;
}

};  // nvdla::priv::
};  // nvdla::
