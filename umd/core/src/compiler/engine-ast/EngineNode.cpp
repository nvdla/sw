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

#include <list>

#include "half.h"
#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"

#include "ErrorMacros.h"

using half_float::half;
using std::set;
using std::map;
using std::vector;
using std::endl;
using std::list;
using std::pair;
using std::make_pair;
using std::find;
using std::string;

namespace nvdla
{
namespace priv
{

//----------------------------------------------------------------------
//                           Generic Node Utils
//----------------------------------------------------------------------
// idempotent
NvDlaError engine_ast::Node::populateEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    EdgeSequence inputEdges = graph()->upstreamDataEdges(this);
    EdgeSequence outputEdges = graph()->downstreamDataEdges(this);

    /**
     * should be min 1 upstream edge;
     * if only 1 upstream edge, it should be the data input
     * if >1 upstream edges, find input and/or aux edges
     */
    if (inputEdges.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s has 0 input edges", name().c_str());
    }
    else if (inputEdges.size() == 1)
    {
        markInputEdge(inputEdges[0]);
    }
    else
    {
        for (EdgeSequenceIterator iei = inputEdges.begin(); iei != inputEdges.end(); ++iei)
        {
            if ((*iei)->isAuxEdge())
            {
                markAuxEdge(*iei);
            }
            else
            {
                markInputEdge(*iei);
            }
        }
    }

    /**
     * should be exactly only 1 output edge, it should be the data output,
     * none of the engine nodes is capable of >1 outputs, fail if so since
     * concat and split nodes are handled separately
     */
    if (outputEdges.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s has 0 output edges", name().c_str());
    }
    else if (outputEdges.size() == 1)
    {
        markOutputEdge(outputEdges[0]);
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s has >1 output edges", name().c_str());
    }

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

fail:
    return e;
}

// get number of operations whose various events directly update the dependency count of this node
NvU16 engine_ast::DependencyParams::getDependencyCount()
{
    NvU16 num_producers = 0;
    for (size_t cc = 0; cc < EngineType::num_elements(); ++cc)
    {
        if (cc == EngineTypeEnum::CPU || cc == EngineTypeEnum::SPLIT || cc == EngineTypeEnum::CONCATENATION)
        {
            // dont count cpu/emu ops as a producer of dla ops
            continue;
        }

        if (producer(cc).nodeAnnId() != -1)
        {
            num_producers++;
        }
    }
    // if a fused downstream op exists, then the combo is enabled in the reverse order -
    // thus the fused node becomes a signalling operation
    num_producers += fusedNode(IODirectionEnum::OUTPUT) ? 1 : 0;
    return num_producers;
}


void engine_ast::Node::emitDependencyParams(DLAInterface* target_dla, DLACommonOpDescAccessor dep, NvU32 batchId)
{
    DLAConsumerAccessor fused_acc  = dep.fusedParentAccessor();
    Node *inputFusedNode = dependencyParams(batchId).fusedNode(engine_ast::IODirectionEnum::INPUT);

    *dep.index()  = dependencyParams(batchId).annotationId();
    *dep.opType() = ASTToDLAInterface::getEngineType(target_dla, engineType());
    *dep.dependencyCount() = dependencyParams(batchId).getDependencyCount(); // number of dependent units

    for ( size_t c = 0; c < EngineType::num_elements(); ++c )
    {
        NvS8 fw_op_index = ASTToDLAInterface::getEngineType(target_dla, c);
        if ( fw_op_index < 0 )
        {
            continue;
        }

        DLAConsumerAccessor cons_acc = dep.consumerAccessor(fw_op_index);

        *cons_acc.index()  = dependencyParams(batchId).consumer(c).nodeAnnId();
        *cons_acc.event()  = ASTToDLAInterface::getOperationEventType(target_dla, dependencyParams(batchId).consumer(c).opEvent());
    }

    *fused_acc.index() = inputFusedNode ? inputFusedNode->dependencyParams(batchId).annotationId() : -1;
    *fused_acc.event() = inputFusedNode ? fused_acc.event_OpEnabled() : fused_acc.event_OpCompleted();
}

void engine_ast::Node::setDataCubeAccessor(DLADataCubeAccessor acc, surface::TensorSurfaceDesc* tsd, IODirection iod, NvU32 batchId)
{
    memory::Location location = tsd->tensorBufferDesc()->memoryLoc(batchId).e();
    Node *fusedNode   = dependencyParams(batchId).fusedNode(iod);
    if ( fusedNode )
    {
        *acc.type()       = acc.type_HW();
        *acc.address()    = -1;
    }
    else
    {
        *acc.type()       = location.e() == memory::LocationEnum::lCVSRAM ? acc.type_CV() : acc.type_MC();
        *acc.address()    = tsd->addressId(batchId);
        *acc.offset()     = tsd->addressIdOffset(batchId);
        if ( graph()->debugOps() || graph()->debugMemoryLayout() )
        {
            gLogInfo << "data cube access by tsd:batch=" << tsd->id() << ":" << batchId << " id[offs]=" << *acc.address() << endl;
        }
    }

    *acc.size()       = tsd->size();
    *acc.width()      = tsd->dimensions().w;
    *acc.height()     = tsd->dimensions().h;
    *acc.channel()    = tsd->dimensions().c;
    if ( !fusedNode && tsd->bindable() )
    {
        NvS16 addrId = *acc.address();
        uintptr_t lineOffs = uintptr_t(acc.lineStride()); // - uintptr_t(surf_acc.struct_base());
        uintptr_t surfOffs = uintptr_t(acc.surfStride()); // - uintptr_t(surf_acc.struct_base());
        graph()->insertRelocEntry(ILoadable::RelocEntry(addrId,
                                                        lineOffs,
                                                        NVDLA_LOADABLE_INTERFACE_DLA1,
                                                        NVDLA_LOADABLE_SUB_INTERFACE_DLA1_SURFS,
                                                        ELST_Line));
        graph()->insertRelocEntry(ILoadable::RelocEntry(addrId,
                                                        surfOffs,
                                                        NVDLA_LOADABLE_INTERFACE_DLA1,
                                                        NVDLA_LOADABLE_SUB_INTERFACE_DLA1_SURFS,
                                                        ELST_Surf));
    }
    *acc.lineStride() = tsd->lineStride();
    *acc.surfStride() = tsd->surfaceStride();
    *acc.planeStride()= 0;
}

NvDlaError engine_ast::Node::nodeDataEdge(const std::vector<surface::SurfaceCategory>& types, ast::EdgeSideEnum dir, engine_ast::Edge** retEdge)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Edge* matchedEdge = NULL;
    EdgeSequence allEdges = graph()->nodeEdges(this, dir);
    for (EdgeSequenceIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        if ( !(*ei)->isDataEdge() )
        {
            continue;
        }
        surface::SurfaceCategory sc = (*ei)->tensorSurfaceDesc()->surfaceFormat().category();
        for (size_t sci = 0; sci < types.size(); ++sci)
        {
            if (sc.v() == types[sci].v())
            {
                if (matchedEdge != NULL)
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, ">1 data edge (%s, %s) of same type",
                            matchedEdge->id().c_str(), (*ei)->id().c_str());
                }
                matchedEdge = *ei;
            }
        }
    }
    *retEdge = matchedEdge;
fail:
    return e;
}

NvDlaError engine_ast::Node::nodeDataEdge(TensorType raw_tt, ast::EdgeSideEnum dir, engine_ast::Edge** retEdge)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Edge* matchedEdge = NULL;
    EdgeSequence allEdges = graph()->nodeEdges(this, dir);
    for (EdgeSequenceIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        if ( !(*ei)->isDataEdge() )
        {
            continue;
        }
        if ( (*ei)->originalTensor() )
        {
            TensorType tt = (*ei)->originalTensor()->getTensorType();
            if (tt == raw_tt)
            {
                if (matchedEdge != NULL)
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, ">1 data edge (%s, %s) of same TensorType %d",
                                         matchedEdge->id().c_str(), (*ei)->id().c_str(), (int)raw_tt);
                }
                matchedEdge = *ei;
            }
        }
    }
    *retEdge = matchedEdge;
fail:
    return e;
}

//----------------------------------------------------------------------
//                           Compiler Utils
//----------------------------------------------------------------------

/*----------------------Surface Descriptor Registration------------------*/
//TODO: collapse the following 3 into 1 with switch-case
const std::vector<surface::SurfaceCategory> engine_ast::Node::supportedInSurfCategories() const
{
    std::vector<surface::SurfaceCategory> supported_scs_v;
    std::set<surface::SurfaceCategory, SequenceEnumCompare<surface::SurfaceCategory>> supported_scs;
    for (size_t ss = 0; ss < supportedInSurfFormats().size(); ++ss)
    {
        supported_scs.insert(supportedInSurfFormats()[ss].f().category());
    }
    supported_scs_v.resize(supported_scs.size());
    std::copy(supported_scs.begin(), supported_scs.end(), supported_scs_v.begin());
    return supported_scs_v;
}

const std::vector<surface::SurfaceCategory> engine_ast::Node::supportedAuxSurfCategories() const
{
    std::vector<surface::SurfaceCategory> supported_scs_v;
    std::set<surface::SurfaceCategory, SequenceEnumCompare<surface::SurfaceCategory>> supported_scs;

    for (size_t ss = 0; ss < supportedAuxSurfFormats().size(); ++ss)
    {
        supported_scs.insert(supportedAuxSurfFormats()[ss].f().category());
    }

    supported_scs_v.resize(supported_scs.size());
    std::copy(supported_scs.begin(), supported_scs.end(), supported_scs_v.begin());

    return supported_scs_v;
}

const std::vector<surface::SurfaceCategory> engine_ast::Node::supportedOutSurfCategories() const
{
    std::vector<surface::SurfaceCategory> supported_scs_v;
    std::set<surface::SurfaceCategory, SequenceEnumCompare<surface::SurfaceCategory>> supported_scs;
    for (size_t ss = 0; ss < supportedOutSurfFormats().size(); ++ss)
    {
        supported_scs.insert(supportedOutSurfFormats()[ss].f().category());
    }
    supported_scs_v.resize(supported_scs.size());
    std::copy(supported_scs.begin(), supported_scs.end(), supported_scs_v.begin());
    return supported_scs_v;
}

NvDlaError engine_ast::Node::supportsSurfaceFormat(surface::SurfaceFormat proposed_sf,
                                                std::vector<surface::SurfaceFormat> supported_sfs)
{
    NvDlaError e = NvDlaError_Success;
    size_t ss = 0;

    if (proposed_sf.e() == surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT)
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Unknown surface format proposed for the node:%s", name().c_str());
    }

    for (ss = 0; ss < supported_sfs.size(); ++ss)
    {
        if (proposed_sf.e() == supported_sfs[ss].e())
            break;
    }

    if(ss == supported_sfs.size())
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_BadParameter, "Proposed surf format:%s is not supported by node:%s",
                proposed_sf.c_str(), name().c_str());
    }

fail:
    return e;
}

std::vector<surface::SurfaceFormat> engine_ast::Node::suggestInputSurfaceFormats()
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    std::vector<surface::SurfaceFormat> supportedInSFs = supportedInSurfFormats();
    std::vector<surface::SurfaceFormat> suggestedInSFs;
    std::vector<surface::SurfaceFormat>::iterator inSFItr;
    surface::SurfacePrecision compPrec = graph()->profile()->computePrecision();

    if (supportedInSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported in surface formats for %s", name().c_str());
    }
    else if (supportedInSFs.size() == 1)
    {
        // most engines support only 1 input format
        // prefer that since there's no other choice (usually, it would be FEATURE_DATA)
        suggestedInSFs = supportedInSFs;
        goto fail;
    }

    // Weed out the suggested inSFs whose precision don't match compute precision of the model
    for (inSFItr = supportedInSFs.begin(); inSFItr != supportedInSFs.end(); ++inSFItr)
    {
        if ((*inSFItr).precision().v() != compPrec.v())
        {
            continue;
        }
        else
        {
            suggestedInSFs.push_back(*inSFItr);
        }
    }

    if (suggestedInSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported input surface formats for node:%s", name().c_str());
    }

fail:
    return suggestedInSFs;
}


std::vector<surface::SurfaceFormat> engine_ast::Node::suggestAuxSurfaceFormats(engine_ast::Edge* auxEdge)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    std::vector<surface::SurfaceFormat> supportedAuxSFs = supportedAuxSurfFormats();
    std::vector<surface::SurfaceFormat> suggestedAuxSFs;
    std::vector<surface::SurfaceFormat>::iterator auxSFItr;

    if (supportedAuxSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported aux surface formats for %s", name().c_str());
    }
    else if (supportedAuxSFs.size() == 1)
    {
        suggestedAuxSFs = supportedAuxSFs;
        goto fail;
    }
    else
    {
        surface::SurfacePrecision compPrec = graph()->profile()->computePrecision();
        for (auxSFItr = supportedAuxSFs.begin(); auxSFItr != supportedAuxSFs.end(); ++auxSFItr)
        {
            if ((*auxSFItr).precision().v() != compPrec.v())
            {
                continue;
            }
            else
            {
                suggestedAuxSFs.push_back(*auxSFItr);
            }
        }
    }

    if (suggestedAuxSFs.size() == 0) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No suggested aux surface formats for node:%s", name().c_str());
    }

fail:
    return suggestedAuxSFs;
}

std::vector<surface::SurfaceFormat> engine_ast::Node::suggestOutputSurfaceFormats()
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    surface::SurfaceFormat inSF;
    std::vector<surface::SurfaceFormat> supportedOutSFs;
    std::vector<surface::SurfaceFormat> suggestedOutSFs;
    std::vector<surface::SurfaceFormat>::iterator outSFItr;
    surface::SurfacePrecision compPrec;

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    inSF = inputEdges()[0]->tensorSurfaceDesc()->surfaceFormat();
    supportedOutSFs = supportedOutSurfFormats();
    compPrec = graph()->profile()->computePrecision();

    if (supportedOutSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported out surface formats for %s", name().c_str());
    }
    else if (supportedOutSFs.size() == 1)
    {
        // most engines support only 1 output format (eventhough they may support >1 input formats (e.g., conv)
        // prefer that since there's no other choice (usually, it would be FEATURE_DATA)
        suggestedOutSFs = supportedOutSFs;
        goto fail;
    }
    else
    {
        if (std::find_if(supportedOutSFs.begin(), supportedOutSFs.end(), surface::SurfaceFormat(inSF)) !=
                supportedOutSFs.end())
        {
            // most engines have only 1 supported in/out surf-formats. hence their out_sf = in_sf
            suggestedOutSFs.clear();
            suggestedOutSFs.push_back(inSF);
            goto fail;
        }
    }

    // Weed out the suggested out_sfs whose precision don't match compute precision of the model
    // or that of the inputSF (until we figure out the CVT business - todo)
    for (outSFItr = supportedOutSFs.begin(); outSFItr != supportedOutSFs.end(); ++outSFItr)
    {
        if ((*outSFItr).precision().v() != compPrec.v() ||
            (*outSFItr).precision().v() != inSF.precision().v())
        {
            continue;
        }
        else
        {
            suggestedOutSFs.push_back(*outSFItr);
        }
    }

    // last try, desperately attempt to find the closest possible output surface format, before giving up
    if (suggestedOutSFs.size() == 0)
    {
        gLogInfo << "Last attempt to determine out SF for " << name() << endl;
        supportedOutSFs = supportedOutSurfFormats();
        for (outSFItr = supportedOutSFs.begin(); outSFItr != supportedOutSFs.end(); ++outSFItr)
        {
            if ((*outSFItr).precision().v() != compPrec.v())
            {
                continue;
            }
            else
            {
                gLogInfo << "last try adding " << (*outSFItr).c_str() << endl;
                suggestedOutSFs.push_back(*outSFItr);
            }
        }
    }

    if (suggestedOutSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No suggested output surface formats for node:%s", name().c_str());
    }

fail:
    return suggestedOutSFs;
}

Dims4 engine_ast::Node::suggestSurfaceDims(surface::TensorSurfaceDesc* tsd)
{
    return tsd->dimensions();
}

NvU32 engine_ast::Node::suggestLineStride(surface::TensorSurfaceDesc* tsd)
{
    return tsd->lineStride();
}

NvU32 engine_ast::Node::suggestSurfaceStride(surface::TensorSurfaceDesc* tsd)
{
    return tsd->surfaceStride();
}

NvU64 engine_ast::Node::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
{
    return tsd->size();
}

NvU64 engine_ast::Node::suggestSurfaceOffsetInBuffer(surface::TensorSurfaceDesc* tsd)
{
    return tsd->bufferOffset();
}

memory::TensorBufferDesc* engine_ast::Node::suggestBuffer(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    bool isDstTSD = false;
    bool isSrcTSD = false;
    bool isAuxTSD = false;
    EdgeSequence outEdges;
    EdgeSequence inEdges;
    EdgeSequence auxlEdges;
    memory::TensorBufferDesc* tbd = NULL;
    NvU16 numBatches = graph()->profile()->multiBatchSize();

    PROPAGATE_ERROR_FAIL(verifyEdgePorts());

    outEdges  = outputEdges();
    inEdges   = inputEdges();
    auxlEdges = auxEdges();

    for (EdgeSequenceIterator iei = inEdges.begin(); iei != inEdges.end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isSrcTSD = true;
            break;
        }
    }
    isDstTSD = outputEdges()[0]->tensorSurfaceDesc() == tsd;
    for (EdgeSequence::const_iterator iei = auxEdges().begin(); iei != auxEdges().end(); ++iei)
    {
        if ((*iei)->tensorSurfaceDesc() == tsd)
        {
            isAuxTSD = true;
            break;
        }
    }

    if (!isSrcTSD && !isDstTSD && !isAuxTSD)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "TSD %s doesn't belong to %s", tsd->id().c_str(), name().c_str());
    }
    else if ( tsd->tensorCategory().v() == memory::TensorCategoryEnum::UNKNOWN_TENSOR )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Edge %s has 'unknown' tensor category",
                             tsd->id().c_str());
    }

    tbd = tsd->tensorBufferDesc();
    if ( !tbd )
    {
        tbd = graph()->resourceMgr()->regTensorBufferDesc(numBatches);
    }

fail:
    return tbd;
}

vector< surface::TensorSurfaceDesc *> engine_ast::Node::inputSurfaces() const
{
    vector< surface::TensorSurfaceDesc *> r;
    surface::TensorSurfaceDesc* i0 =
        graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    if ( i0 )
    {
        r.push_back(i0);
    }
    return r;
}

vector< surface::TensorSurfaceDesc *> engine_ast::Node::auxSurfaces() const
{
    vector< surface::TensorSurfaceDesc *> r;
    surface::TensorSurfaceDesc* a0 = NULL;
    //Get surface info for all aux edges
    for (size_t i=0; i<auxEdges().size(); i++)
    {
        a0 = graph()->nodeInputTensorSurface(this, i, supportedAuxSurfCategories());
        if ( a0 )
        {
            r.push_back(a0);
        }
    }
    return r;
}

vector< surface::TensorSurfaceDesc *> engine_ast::Node::outputSurfaces() const
{
    vector< surface::TensorSurfaceDesc *> r;

    surface::TensorSurfaceDesc *o0 =
        graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

    if ( o0 )
    {
        r.push_back(o0);
    }
    return r;
}

//
// 'depends' denotes an upstream search.
//
// search with respect to the edge types given in the 'via's to determine
// if this node is connected to the 'of' node.
//
// . nodes never depend upon themselves.
//
// . if requiredVia is specified (non-zero sized) then connection paths are
//   valid iff at some point one of the requiredVia edge types arises.
//
// . if 'allowedVia' is specified (non-zero sized) then paths are allowed
//   to also have the allowedVia edge types present.
//
// . requiredVia -> nil: && allowedVia -> nil: simple connection test.
// . requiredVia -> !nil  && allowedVia -> nil: conntection test where
//   all paths must consist of only requiredVia types
//
bool engine_ast::Node::dependsOn(Node *of,
                                 const vector<EdgeType> &requiredVia,
                                 const vector<EdgeType> &allowedVia) const
{
    bool done = false, depends = false;

    map< const Node *, bool> followNodes;
    map< const Node *, bool> followedNodes;

    bool requires = requiredVia.size() != 0;
    bool allows   = allowedVia.size()  != 0;

    if ( of == this )
    {
        goto done;
    }

    followNodes[this] = false;

    while ( followNodes.size() && !(done || depends) )
    {

        const Node *followingNode = followNodes.begin()->first;
        bool followingSatisfied = followNodes.begin()->second;
        EdgeSequence followingEdges;

        followNodes.erase(followingNode);
        followedNodes[followingNode] = followingSatisfied;

        if ( followingNode == of )
        {
            done = true;
            depends = followingSatisfied;
            continue;
        }

        followingEdges = graph()->upstreamEdges(followingNode);

        for ( size_t fei = 0, FEI = followingEdges.size(); fei != FEI; ++fei )
        {
            Edge *testEdge    = followingEdges[fei];
            EdgeType testType = testEdge->edgeType();
            bool testEdgeSatisfies, satisfied;
            bool follow = (!allows) || ( find(allowedVia.begin(), allowedVia.end(), testType) != allowedVia.end() );
            NodeSequence upstreamNodes;

            if ( !follow )
            {
                continue; // unfollowable path (edge type not allowed)
            }

            testEdgeSatisfies = (!requires) || ( find(requiredVia.begin(), requiredVia.end(), testType) != requiredVia.end() ) ||
                                               ( find(allowedVia.begin(), allowedVia.end(), testType) != allowedVia.end() );
            satisfied = followingSatisfied || testEdgeSatisfies;

            upstreamNodes = graph()->upstreamNodes(testEdge);

            for ( size_t uni = 0, UNI = upstreamNodes.size(); uni != UNI; ++uni )
            {
                Node *upstreamNode = upstreamNodes[uni];
                map<const Node *, bool>::iterator fun = followedNodes.find(upstreamNode);

                // if we've never encountered this node before
                // then follow it.  if we've seen it before only
                // follow it again if it was not a satisfied path
                // then, but now is.
                if ( fun == followedNodes.end() )
                {
                    followNodes[upstreamNode] = satisfied;
                }
                else
                {
                    if ( satisfied && (fun->second == false) )
                    {
                        followNodes[upstreamNode] = true;
                        followedNodes[upstreamNode] = true;
                    }
                }
            }
        }
    }

 done:
    return depends;

}

/*---------------------------Resolve Data Dependencies-----------------*/
NvDlaError engine_ast::Node::resolveDataDependencies(engine_ast::Node* next)
{
    NvDlaError e = NvDlaSuccess;
    NvU8 this_type;
    NodeSequence consumer_nodes;
    EdgeSequence output_data_edges = graph()->downstreamDataEdges(this);
    EdgeSequence hazardEdges = graph()->upstreamHazardEdges(this);
    NodeSequence hazardNodes, realHazardNodes;

    this_type = engineType().v();

    if ( hazardEdges.size() )
    {
        set<Node *> uniqHazardNodes;

        // gLogInfo << "hazard: resolveDataDependencies sees " << output_hazard_edges.size() << " hazards" << endl;
        //        gLogInfo << "hazard: this depends on that: " << endl;

        for ( size_t hei = 0, HEI = hazardEdges.size(); hei != HEI; ++hei )
        {
            NodeSequence edgeHazardNodes = graph()->upstreamNodes( hazardEdges[hei] );

            for ( size_t hni=0, HNI = edgeHazardNodes.size(); hni != HNI; ++hni )
            {
                if ( uniqHazardNodes.find( edgeHazardNodes[hni] ) == uniqHazardNodes.end() )
                {
                    hazardNodes.push_back(edgeHazardNodes[hni]);
                    uniqHazardNodes.insert(edgeHazardNodes[hni]);
                }
            }
        }
    }
    if ( hazardNodes.size() )
    {

        // gLogInfo << "rdd hazard: found " << hazardNodes.size() << " hazard nodes downstream" << endl;
        // each of these represents a memory hazard.  if we happen to be on the same unit there's
        // no actual problem.  check that.

        for ( size_t hni=0, HNI = hazardNodes.size(); hni != HNI; ++hni )
        {
            Node *otherOp = hazardNodes[hni];
            if ( otherOp->engineType().v() == this_type )
            {
                continue;
            }
            realHazardNodes.push_back(otherOp);
        }
    }
    if ( realHazardNodes.size() )
    {
        if ( graph()->debugDepGraph() )
        {
            gLogInfo << "info: node=" << id() << " found " << realHazardNodes.size() << " actionable hazard nodes." << endl;
        }
    }

    if (next)
    {
        Node * previous = this;
        // 'currNode' may be connected to fusedNodes of 'previous'
        Node * fusedNode = previous;
        while (fusedNode) {
            if  (graph()->adjacentNodes(fusedNode,next)
                 // uncomment this line to for faster enabling same engine typed nodes from threads to threads
                 // when no register group race conditions
                 // e.g. diamonds from group convolutions
                 // || fusedNode->engineType() == next->engineType()
                )
            {
                break;
            }
            fusedNode = fusedNode->dependencyParams().fusedNode(IODirectionEnum::INPUT);
        }

        if (!fusedNode)
        {
            // no fusedNodes connect to currNode
            //      currNode is the head of a new thread
            Edge* DFSthreadsCompEdge = graph()->addComputeEdge(previous, next);
            NVDLA_UNUSED(DFSthreadsCompEdge);
            if ( debugResolveDependencies() )
            {
                gLogInfo << "adding compute edge " << DFSthreadsCompEdge->id()
                        << " from " << (previous)->name() << " -> " << (next)->name() << endl;

                //printGraph(graph(), true);
            }
        }
        else if (fusedNode->engineType() == next->engineType() && !graph()->adjacentNodes(fusedNode,next))
        {
            // if fusedNode and next are non-adjacent nodes with same engine type
            //      we need this edge for mem resolve
            Edge* DFSthreadsCompEdge = graph()->addComputeEdge(fusedNode, next);
            NVDLA_UNUSED(DFSthreadsCompEdge);
            if ( debugResolveDependencies() )
            {
                gLogInfo << "adding compute edge " << DFSthreadsCompEdge->id()
                        << " from " << (fusedNode)->name() << " -> " << (next)->name() << endl;

                //printGraph(graph(), true);
            }
            // fusedNodes is connected to currNode
            previous = fusedNode;
        }
        else
        {
            // fusedNodes is connected to currNode
            previous = fusedNode;
        }

        if (!fusedNode || (graph()->adjacentNodes(previous,next) && graph()->connectedDataNodes(previous, next)))
        {
            // if previous is connected to currNode with data edge, or
            // if previous is not connected to currNode but there's no fusedNodes connection either
            //      then there's a computeEdge added
            // need to attach OP_COMPLETED

            if (previous->isSoftwareNode())
            {
                for (size_t et = 0; et < EngineType::num_elements(); ++et)
                {
                    Node * consumer = previous->dependencyParams(0).consumer(et).node();
                    NvU8 producerType = (previous)->engineType().v();

                    if (consumer) consumer->dependencyParams(0).producer(producerType).setNode(NULL);
                    previous->dependencyParams(0).consumer(et).setNode(NULL);
                }
            }
            if (next->isSoftwareNode())
            {
                for (size_t et = 0; et < EngineType::num_elements(); ++et)
                {
                    Node * producer = next->dependencyParams(0).producer(et).node();
                    NvU8 consumerType = (next)->engineType().v();

                    if (producer) producer->dependencyParams(0).consumer(consumerType).setNode(NULL);
                    next->dependencyParams(0).producer(et).setNode(NULL);
                }
            }

            NvU8 consumerType = (next)->engineType().v();
            NvU8 producerType = (previous)->engineType().v();
            (previous)->dependencyParams(/*batchId*/0).consumer(consumerType).setNode(next);
            (previous)->dependencyParams(/*batchId*/0).consumer(consumerType).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
            (next)->dependencyParams(/*batchId*/0).producer(producerType).setNode(previous);
            (next)->dependencyParams(/*batchId*/0).producer(producerType).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
        }

    }


    //
    // realHazardNodes ops must complete before this/ op can begin.
    // so this node is a consumer of it/them.
    //
    for ( size_t rhi = 0, RHI = realHazardNodes.size(); rhi != RHI; ++rhi )
    {
        Node *hazardProducer = realHazardNodes[rhi];
        EngineType hazardProducerEngineType = hazardProducer->engineType();
        NvU8 hazardProducerEngine = hazardProducerEngineType.v();
        Node *existingEngineProducer = dependencyParams(/*batchId*/0).producer(hazardProducerEngine).node();

        // don't do split or concat nodes...
        if ( hazardProducerEngineType == EngineTypeEnum::CONCATENATION ||
             hazardProducerEngineType == EngineTypeEnum::SPLIT )
        {
            continue;
        }

        if ( !existingEngineProducer )
        {
            // the operation which needs to wait (this, the consumer)
            if ( this->dependsOn(existingEngineProducer, viaComputeData, allowAll) )
            {
                if ( graph()->debugDepGraph() )
                {
                    // no worries.  the hazard will be cleared due to existing serialization.
                    gLogInfo <<"info: hazard cleared due to existing order on engine " <<
                        hazardProducerEngineType.c_str() << endl;
                }
            }
            else
            {
                // treat as normal producer-consumer
                // update consumer of this node
                dependencyParams(/*batchId*/0).producer(hazardProducerEngine).setNode(hazardProducer);
                dependencyParams(/*batchId*/0).producer(hazardProducerEngine).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);

                // at the same time update producer of consumer node
                hazardProducer->dependencyParams(/*batchId*/0).consumer(this_type).setNode(this);
                hazardProducer->dependencyParams(/*batchId*/0).consumer(this_type).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
            }
        }
        // else
        // {
        //     /*there's already a wait on that engine.  don't need to worry about it.*/
        // }
    }

    return e;
}

/*---------------------------Resolve Compute Dependencies--------------*/
NvDlaError engine_ast::Node::resolveComputeDependencies(const NodeSequence &ordered_nodes)
{
    NvDlaError e = NvDlaError_Success;
    NodeSequence::const_iterator nin, ni;

    if ( isSoftwareNode() )
    {
        goto fail;
    }

    ni = std::find(ordered_nodes.begin(), ordered_nodes.end(), this);

    for (nin = (ni == ordered_nodes.end() ? ordered_nodes.end() : ni+1); nin != ordered_nodes.end(); ++nin )
    {
        Node *other_node = *nin;
        EngineType curr_eng_type = engineType();
        /* Treat a directly connected or distant operation of same engineType as a consumer except for the software nodes. */
        if (other_node->isEngineType(curr_eng_type))
        {
            dependencyParams(/*batchId*/0).consumer( curr_eng_type.v() ).setNode(other_node);
            other_node->dependencyParams(/*batchId*/0).producer( curr_eng_type.v() ).setNode(this);

            /* If 2 adjacent nodes of the same engine type relay a data tensor between them, then
             * the downstream op should wait until the upstream op is completed otherwise the
             * downstream op could start independently.
             */
            if ( graph()->adjacentNodes(this, other_node))
            {
                if ( other_node->dependsOn(this, viaCompute, allowAll) )
                {
                    dependencyParams(/*batchId*/0).consumer( curr_eng_type.v() ).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
                    other_node->dependencyParams(/*batchId*/0).producer( curr_eng_type.v() ).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
                }
                else
                {
                    dependencyParams(/*batchId*/0).consumer( curr_eng_type.v() ).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
                    other_node->dependencyParams(/*batchId*/0).producer( curr_eng_type.v() ).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
                }
            }
            else
            {
                /* However, if 2 nodes of the same engineType are distant in the graph
                 * OR
                 * 2 adjacent nodes need different sub-engines (i.e. they are fused) -
                 * (ONLY happens when different types of SDP use different sub-engines)
                 * then the consumer can be programmed ASA producer is programmed.
                 * But, it will be waiting on some other node to complete before executing.
                 */
                dependencyParams(/*batchId*/0).consumer( curr_eng_type.v() ).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
                other_node->dependencyParams(/*batchId*/0).producer( curr_eng_type.v() ).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
            }
            break;
        }
    }

fail:
    return e;
}

/*---------------------Resolve Software Dependencies-------------------*/
NvDlaError engine_ast::Node::resolveSoftwareDependencies()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence producerNodes;
    NodeSequence consumerNodes;
    Node* prodToSwNode = NULL;
    Node* consOfSwNode = NULL;
    Node* existingConsOfProd = NULL;
    Node* existingProdOfCons = NULL;
    EngineType prodEngType;
    EngineType consEngType;
    OperationEventType prodEvent;
    OperationEventType consEvent;

    if (!isSoftwareNode())
    {
        goto fail;
    }

    for (size_t et = 0; et < EngineType::num_elements(); ++et)
    {
        Node* prod = dependencyParams(0).producer(et).node();
        Node* cons = dependencyParams(0).consumer(et).node();

        if (prod)
        {
            producerNodes.push_back(prod);
        }
        if (cons)
        {
            consumerNodes.push_back(cons);
        }
    }

    if ( producerNodes.size() > 1 )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Can't handle >1 producers "
                "to software node %s", name().c_str());
    }
    else if ( consumerNodes.size() > 1 )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Can't handle >1 consumers "
                "to software node %s", name().c_str());
    }
    if ( (producerNodes.size() == 0) || (consumerNodes.size() == 0) )
    {
        // no need to cross-relegate dependencies in case of a missing producer or consumer
        goto fail;
    }

    prodToSwNode = producerNodes.at(0);
    prodEngType  = prodToSwNode->engineType();
    prodEvent    = dependencyParams(0).producer(prodEngType.v()).opEvent();

    consOfSwNode = consumerNodes.at(0);
    consEngType  = consOfSwNode->engineType();
    consEvent    = dependencyParams(0).consumer(consEngType.v()).opEvent();

    if (prodEvent.v() != consEvent.v())
    {
        gLogError << "Prod " << dependencyParams(0).producer(prodEngType.v()).node()->name()
                 << " event " << prodEvent.c_str() << endl;
        gLogError << "Cons " << dependencyParams(0).consumer(consEngType.v()).node()->name()
                << " event " << consEvent.c_str() << endl;
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't connect producer and consumer of "
                "%s with different op events", name().c_str());
    }

    existingConsOfProd = prodToSwNode->dependencyParams(0).consumer(consEngType.v()).node();
    existingProdOfCons = consOfSwNode->dependencyParams(0).producer(prodEngType.v()).node();
    if ( existingConsOfProd &&
        !existingConsOfProd->isSoftwareNode() &&
        (existingConsOfProd != consOfSwNode) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't override existing consumer (%s) of %s with %s",
                existingConsOfProd->name().c_str(), prodToSwNode->name().c_str(), consOfSwNode->name().c_str());
    }
    else if ( existingProdOfCons &&
             !existingProdOfCons->isSoftwareNode() &&
             (existingProdOfCons != prodToSwNode) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Can't override existing producer (%s) of %s with %s",
                existingProdOfCons->name().c_str(), consOfSwNode->name().c_str(), prodToSwNode->name().c_str());
    }
    else
    {
        prodToSwNode->dependencyParams(0).consumer(consEngType.v()).setNode(consOfSwNode);
        prodToSwNode->dependencyParams(0).consumer(consEngType.v()).setOpEvent(consEvent);
        consOfSwNode->dependencyParams(0).producer(prodEngType.v()).setNode(prodToSwNode);
        consOfSwNode->dependencyParams(0).producer(prodEngType.v()).setOpEvent(prodEvent);

        // at the end, remove the existing software node from the dependency graph of
        // both its producer and consumer
        prodToSwNode->dependencyParams(0).consumer(this->engineType().v()).setNode(NULL);
        prodToSwNode->dependencyParams(0).consumer(this->engineType().v()).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
        consOfSwNode->dependencyParams(0).producer(this->engineType().v()).setNode(NULL);
        consOfSwNode->dependencyParams(0).producer(this->engineType().v()).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
    }

fail:
    return e;
}

/*--------------------------------Self Annotation----------------------*/
NvDlaError engine_ast::Node::selfAnnotate(NvS16& lastUsedAnnId)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 numBatches = graph()->profile()->multiBatchSize();

    for (NvU32 nn = 0; nn < numBatches; ++nn)
    {
        dependencyParams(nn).setAnnotationId(++lastUsedAnnId);
    }

    return e;
}

/*-------------------------Resolve Multi-Batch Dependencies------------*/
/*
 * Once the dependency graph is prepared for all the operations of a single batch,
 * its time to introduce dependencies that will chain executing 'N' batches together.
 *
 * Remember, same operation executed for different batches never has a data dependency
 * between the copies, but only compute dependency. So introduce compute dependency
 * between the operations of the batches and shift the producer events calculated for
 * the single batch to batch-0 and consumer events to batch-N
 */
NvDlaError engine_ast::Node::resolveMultiBatchDependencies()
{
    NvDlaError e = NvDlaSuccess;

    NvU32 numBatches = graph()->profile()->multiBatchSize();
    NvU32 firstBatch = 0;
    NvU32 lastBatch  = numBatches - 1;
    EngineType currEngType = engineType();

    if (numBatches == 1)
    {
        goto fail;
    }

    //TODO Keep parallel execution disable until it is verified
    if (0)
    {
        for (size_t et = 0; et < EngineType::num_elements(); ++et)
        {
            Node* skipFusedConsumer = dependencyParams(firstBatch).fusedNode(IODirectionEnum::OUTPUT);
            if (skipFusedConsumer && skipFusedConsumer->engineType() == et)
            {
                continue;
            }

            Node* consumer                   = dependencyParams(firstBatch).consumer(et).node();
            OperationEventType consumerEvent = dependencyParams(firstBatch).consumer(et).opEvent();
            NvS16 consumerAnnId              = consumer ? consumer->dependencyParams(firstBatch).annotationId() : -1;
            NvS16 lastBatchAnnId             = dependencyParams(lastBatch).annotationId();
            dependencyParams(lastBatch).consumer(et).setNode(consumer);
            dependencyParams(lastBatch).consumer(et).setNodeAnnId(consumerAnnId);
            dependencyParams(lastBatch).consumer(et).setOpEvent(consumerEvent);
            if ( consumer )
            {
                consumer->dependencyParams(firstBatch).producer(currEngType.v()).setNode(this);
                consumer->dependencyParams(firstBatch).producer(currEngType.v()).setNodeAnnId(lastBatchAnnId);
                consumer->dependencyParams(firstBatch).producer(currEngType.v()).setOpEvent(consumerEvent);
            }

            // and finally clear the dependency params of the 1st batch
            dependencyParams(firstBatch).consumer(et).setNode(NULL);
            dependencyParams(firstBatch).consumer(et).setNodeAnnId(-1);
            dependencyParams(firstBatch).consumer(et).setOpEvent(OperationEventTypeEnum::OP_COMPLETED);
        }


        for (NvU16 iod = 0; iod < IODirection::num_elements(); ++iod)
        {
            Node* fusedNode = dependencyParams(firstBatch).fusedNode(iod);
            if (fusedNode)
            {
                EngineType         fusedNodeEngType = fusedNode->engineType();
                OperationEventType fusedNodeOpEvent = iod == IODirectionEnum::INPUT ?
                                dependencyParams(firstBatch).producer(fusedNodeEngType.e()).opEvent() :
                                dependencyParams(firstBatch).consumer(fusedNodeEngType.e()).opEvent();
                for (NvU32 nn = firstBatch + 1; nn < numBatches; ++nn)
                {
                    dependencyParams(nn).setFusedNode(iod, fusedNode);
                    NvS16 fusedNodeAnnId = fusedNode->dependencyParams(nn).annotationId();
                    if (iod == IODirectionEnum::INPUT)
                    {
                        dependencyParams(nn).producer(fusedNodeEngType.e()).setNode(fusedNode);
                        dependencyParams(nn).producer(fusedNodeEngType.e()).setNodeAnnId(fusedNodeAnnId);
                        dependencyParams(nn).producer(fusedNodeEngType.e()).setOpEvent(fusedNodeOpEvent);
                    }
                    else
                    {
                        dependencyParams(nn).consumer(fusedNodeEngType.e()).setNode(fusedNode);
                        dependencyParams(nn).consumer(fusedNodeEngType.e()).setNodeAnnId(fusedNodeAnnId);
                        dependencyParams(nn).consumer(fusedNodeEngType.e()).setOpEvent(fusedNodeOpEvent);
                    }
                }
            }
        }
    }
    else
    {
        /* Inherit dependency graph from batch-0 into other batches, such that they inherit
         * producer/consumer details from the operation chain of the relevant batch
         */
        for (NvU32 nn = 1; nn < numBatches; ++nn)
        {
            for (size_t et = 0; et < EngineType::num_elements(); ++et)
            {
                Node* consumer                   = dependencyParams(0).consumer(et).node();
                OperationEventType consumerEvent = dependencyParams(0).consumer(et).opEvent();
                NvS16 consumerAnnId              = consumer ? consumer->dependencyParams(nn).annotationId() : -1;
                dependencyParams(nn).consumer(et).setNode(consumer);
                dependencyParams(nn).consumer(et).setNodeAnnId(consumerAnnId);
                dependencyParams(nn).consumer(et).setOpEvent(consumerEvent);

                Node* producer                   = dependencyParams(0).producer(et).node();
                OperationEventType producerEvent = dependencyParams(0).producer(et).opEvent();
                NvS16 producerAnnId              = producer ? producer->dependencyParams(nn).annotationId() : -1;
                dependencyParams(nn).producer(et).setNode(producer);
                dependencyParams(nn).producer(et).setNodeAnnId(producerAnnId);
                dependencyParams(nn).producer(et).setOpEvent(producerEvent);
            }

            for (NvU16 iod = 0; iod < IODirection::num_elements(); ++iod)
            {
                dependencyParams(nn).setFusedNode(iod, dependencyParams(0).fusedNode(iod));
            }
        }

        /* Delegate the compute consumer of the node to the dependency graph of its last Batch */
        Node*              computeConsumerNode  = dependencyParams(/*batchId*/0).consumer(currEngType.v()).node();
        OperationEventType computeEvent         = dependencyParams(/*batchId*/0).consumer(currEngType.v()).opEvent();
        NvS16 consumerAnnId                     = computeConsumerNode ? computeConsumerNode->dependencyParams(/*batchId*/0).annotationId() : -1;
        NvS16 lastBatchAnnId                    = dependencyParams(lastBatch).annotationId();

        /* Connect the last batch of this node to the first batch of the consumer node */
        dependencyParams(/*batchId*/lastBatch).consumer(currEngType.v()).setNode(computeConsumerNode);
        dependencyParams(/*batchId*/lastBatch).consumer(currEngType.v()).setNodeAnnId(consumerAnnId);
        dependencyParams(/*batchId*/lastBatch).consumer(currEngType.v()).setOpEvent(computeEvent);
        if ( computeConsumerNode )
        {
            computeConsumerNode->dependencyParams(/*firstBatchId*/0).producer(currEngType.v()).setNode(this);
            computeConsumerNode->dependencyParams(/*firstBatchId*/0).producer(currEngType.v()).setNodeAnnId(lastBatchAnnId);
            computeConsumerNode->dependencyParams(/*firstBatchId*/0).producer(currEngType.v()).setOpEvent(computeEvent);
        }
    }


    /* Chain the operations of the same type within the batches with soft stops */
    for (NvU32 currBatch = firstBatch; currBatch < lastBatch; ++currBatch)
    {
        NvU32 nextBatch = currBatch + 1;
        NvS16 nextBatchAnnId = dependencyParams(nextBatch).annotationId();
        NvS16 currBatchAnnId = dependencyParams(currBatch).annotationId();

        /* batches executing the same operation are connected by OP_COMPLETE events */
        dependencyParams(/*batchId*/currBatch).consumer(currEngType.v()).setNode(this);
        dependencyParams(/*batchId*/currBatch).consumer(currEngType.v()).setNodeAnnId(nextBatchAnnId);
        dependencyParams(/*batchId*/currBatch).consumer(currEngType.v()).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);

        dependencyParams(/*batchId*/nextBatch).producer(currEngType.v()).setNode(this);
        dependencyParams(/*batchId*/nextBatch).producer(currEngType.v()).setNodeAnnId(currBatchAnnId);
        dependencyParams(/*batchId*/nextBatch).producer(currEngType.v()).setOpEvent(OperationEventTypeEnum::OP_PROGRAMMED);
    }

fail:
    return e;
}

/*-------------------------Verify Dependency Params--------------------*/
NvDlaError engine_ast::Node::verifyDependencyParams()
{
    NvDlaError e = NvDlaSuccess;

    NvU32 numBatches = graph()->profile()->multiBatchSize();
    NvU32 firstBatch = 0;
    NvU32 lastBatch  = numBatches - 1;
    EngineType currEngType = engineType();

    if (isSoftwareNode())
    {
        goto fail;
    }

    for (NvU32 currBatch = firstBatch + 1; currBatch < lastBatch; ++currBatch)
    {
        NvU32 prevBatch = currBatch - 1;
        NvU32 nextBatch = currBatch + 1;
        Node* consumer = NULL;
        Node* producer = NULL;
        OperationEventType consOpEvent;
        OperationEventType prodOpEvent;

        for(size_t et = 0; et < EngineType::num_elements(); ++et)
        {
            if (et == EngineTypeEnum::CONCATENATION || et == EngineTypeEnum::SPLIT)
            {
                continue;
            }

            consumer    = dependencyParams(currBatch).consumer(et).node();
            consOpEvent = dependencyParams(currBatch).consumer(et).opEvent();
            if (consumer == this)
            {
                if(consumer->dependencyParams(nextBatch).producer(currEngType.v()).nodeAnnId() !=
                        dependencyParams(currBatch).annotationId())
                {
                    gLogInfo << "consumer id " << consumer->dependencyParams(nextBatch).producer(currEngType.v()).nodeAnnId()
                             << " and producer id " << dependencyParams(currBatch).annotationId() << endl;
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair", consumer->name().c_str(), name().c_str());
                }
                if(consumer->dependencyParams(nextBatch).producer(currEngType.v()).opEvent().e() != consOpEvent.e())
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair with same op events", consumer->name().c_str(), name().c_str());
                }
            }
            else if (consumer != NULL)
            {
                if(consumer->dependencyParams(currBatch).producer(currEngType.v()).nodeAnnId() !=
                        dependencyParams(currBatch).annotationId())
                {
                    gLogInfo << "consumer id " << consumer->dependencyParams(currBatch).producer(currEngType.v()).nodeAnnId()
                             << " and producer id " << dependencyParams(currBatch).annotationId() << endl;
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair", consumer->name().c_str(), name().c_str());
                }
                if(consumer->dependencyParams(currBatch).producer(currEngType.v()).opEvent().e() !=
                        consOpEvent.e())
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair with same op events", consumer->name().c_str(), name().c_str());
                }
            }
            producer    = dependencyParams(currBatch).producer(et).node();
            prodOpEvent = dependencyParams(currBatch).producer(et).opEvent();
            if (producer == this)
            {
                if(producer->dependencyParams(prevBatch).consumer(currEngType.v()).nodeAnnId() !=
                        dependencyParams(currBatch).annotationId())
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair", name().c_str(), producer->name().c_str());
                }
                if(producer->dependencyParams(prevBatch).consumer(currEngType.v()).opEvent().e() !=
                        prodOpEvent.e())
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair with same op events", name().c_str(), producer->name().c_str());
                }
            }
            else if (producer != NULL)
            {
                if(producer->dependencyParams(currBatch).consumer(currEngType.v()).nodeAnnId() !=
                        dependencyParams(currBatch).annotationId())
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair", name().c_str(), producer->name().c_str());
                }
                if(producer->dependencyParams(currBatch).consumer(currEngType.v()).opEvent().e() !=
                        prodOpEvent.e())
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Consumer %s and Producer %s don't form "
                            "the right pair with same op events", name().c_str(), producer->name().c_str());
                }
            }
        }
    }

fail:
    return e;
}

//----------------------------------------------------------------------
//                           Code Emission Utils
//----------------------------------------------------------------------
NvDlaError engine_ast::Node::emitOp(engine_ast::Graph *g,
                             DLAInterface *dla_if,
                             NvU32 op_slot, NvU32 batch_id,
                             DLACommonOpDescAccessor dep_acc,
                             DLAOperationContainerAccessor op_acc,
                             DLASurfaceContainerAccessor surf_acc)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(g);
    NVDLA_UNUSED(dep_acc);
    NVDLA_UNUSED(dla_if);
    NVDLA_UNUSED(op_slot);
    NVDLA_UNUSED(batch_id);
    NVDLA_UNUSED(op_acc);
    NVDLA_UNUSED(surf_acc);

    // If you see this message then something is tbd or bugged.
    ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState,
                         "hit bare Node::emitOp! engine_type=%s engine_op_type=%s\n",
                         m_engine_type.c_str(), m_engine_op_type.c_str());
 fail:
    return e;
}

NvDlaError engine_ast::Node::emitOp(engine_ast::Graph *g,
                             EMUInterface *emu_if,
                             NvU32 op_slot, NvU32 batch_id,
                             EMUOperationContainerAccessor op_acc,
                             EMUOperationBufferContainerAccessor buf_acc)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(g);
    NVDLA_UNUSED(emu_if);
    NVDLA_UNUSED(op_slot);
    NVDLA_UNUSED(batch_id);
    NVDLA_UNUSED(op_acc);
    NVDLA_UNUSED(buf_acc);

    // If you see this message then something is tbd or bugged.
    ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState,
                         "hit bare Node::emitOp! engine_type=%s engine_op_type=%s\n",
                         m_engine_type.c_str(), m_engine_op_type.c_str());

 fail:
    return e;
}


};  // nvdla::privs::
};  // nvdla::
