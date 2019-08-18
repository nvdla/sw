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

#include "priv/Type.h"
#include "priv/EngineAST.h"
#include "priv/Profile.h"
#include "priv/Tensor.h"
#include "priv/Compiler.h"
#include "priv/WeightTranslationUnit.h"

#include "ErrorMacros.h"

using std::endl;
using std::max;

namespace nvdla
{
namespace priv
{

NvDlaError engine_ast::SDPSuperOpNode::captureCanonicalData(SDPSubEngineType xN, TensorType tN)
{
    NvDlaError e = NvDlaError_Success;
    Dims4             dataDims;
    Tensor*           dataTensor = NULL;
    Edge* xEdge = NULL;

    if (tN != TensorType::kIO)
    {
        dataDims = params().dlaDataDims(xN);
        dataTensor = graph()->addAuxTensor(graph()->newAuxTensorName(), dataDims, tN);
        xEdge = graph()->addDataEdge((canonical_ast::Edge*)NULL, (engine_ast::Node*)NULL, this, dataTensor);
        markSdpAuxEdge(xN, xEdge);
    }
    return e;
}

void engine_ast::SDPSuperOpNode::inheritParamsForSubEngine(SDPSuperOpNode* otherSuperOp, SDPSubEngineType xN)
{
    params().setAuxDataType(xN, otherSuperOp->params().auxDataType(xN));
    params().setAuxSurfaceFormats(xN, otherSuperOp->params().auxSurfaceFormats(xN));

    params().setMultiplierDims(xN, otherSuperOp->params().multiplierDims(xN));
    params().setAdderDims(xN, otherSuperOp->params().adderDims(xN));
    params().setDLADataDims(xN, otherSuperOp->params().dlaDataDims(xN));

    params().setRawMultiplierData(xN, otherSuperOp->params().rawMultiplierData(xN));
    params().setRawAdderData(xN, otherSuperOp->params().rawAdderData(xN));
    params().setDLAData(xN, otherSuperOp->params().dlaData(xN));

    engine_ast::Edge* xEdge = NULL;
    otherSuperOp->auxEdgeBySubEngine(xN, &xEdge);
    markSdpAuxEdge(xN, xEdge);
}

void engine_ast::SDPSuperOpNode::inheritParams(Node* otherNode)
{
    SDPSuperOpNode* otherSuperOp = NodeFactory::nodeCast<SDPSuperOpNode*>(otherNode);
    params().setX1Params(otherSuperOp->params().x1Params());
    if (params().x1Params().enabled())
    {
        inheritParamsForSubEngine(otherSuperOp, SDP_ENGINE_X1);
    }

    params().setX2Params(otherSuperOp->params().x2Params());
    if (params().x2Params().enabled())
    {
        inheritParamsForSubEngine(otherSuperOp, SDP_ENGINE_X2);
    }

    // FIXME
    /*
    params().setYParams(otherSuperOp->params().yParams());
    if (params().yParams().enabled())
    {
        inheritParamsForSubEngine(otherSuperOp, SDP_ENGINE_Y);
    }
    */

    params().setConvMode(otherSuperOp->params().convMode());
    params().setWinogradParams(otherSuperOp->params().winogradParams());
    params().setNumGroups(otherSuperOp->params().numGroups());
}

engine_ast::SDPSubEngineParams* engine_ast::SDPSuperOpNode::subEngineParams(SDPSubEngineType xN)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    engine_ast::SDPSubEngineParams* retParams = NULL;
    switch(xN.e())
    {
        case SDP_ENGINE_X1:
            retParams = &params().x1Params();
            break;
        case SDP_ENGINE_X2:
            retParams = &params().x2Params();
            break;
        case SDP_ENGINE_Y:
            retParams = &params().yParams();
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Invalid subEngineType");
    }
fail:
    return retParams;
}

/* Identify input, output and aux(x1,x2,y) edges for SuperOp.
   Update internal map of SDPSubEngineToAuxEdge with any edge updates.
*/
NvDlaError engine_ast::SDPSuperOpNode::populateEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    //gLogInfo << name() << " : populateEdgePorts: " << endl;
    //printSdpXEdgeMap();

    EdgeSequence inputEdges = graph()->upstreamDataEdges(this);
    EdgeSequence outputEdges = graph()->downstreamDataEdges(this);

    // upstream Edges are input + aux edges
    size_t upstreamEdgesExpected = 1 + m_sdpXengineToAuxEdgeMap.size();

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

    if (inputEdges.size() != upstreamEdgesExpected)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s has (%d) input+aux edges. expected (%d)",
            name().c_str(), inputEdges.size(), upstreamEdgesExpected);
    }

    {
        NvU8 foundEdges = 0;
        NvU8 foundX1Edge = 1 << 0;
        NvU8 foundX2Edge = 1 << 1;
        NvU8 foundInputEdge = 1 << 2;
        NvU8 foundAllEdges = foundX1Edge | foundX2Edge | foundInputEdge;

        EdgeSequence newEdges;
        for (EdgeSequenceIterator iei = inputEdges.begin(); iei != inputEdges.end(); ++iei)
        {
            SdpXengineToEdgeMapIterator xEdge = findSdpAuxEdge(*iei);
            if ( xEdge != m_sdpXengineToAuxEdgeMap.end() )
            {
                if (xEdge->first == SDP_ENGINE_X1)
                    foundEdges |= foundX1Edge;
                else if (xEdge->first == SDP_ENGINE_X2)
                    foundEdges |= foundX2Edge;
                else
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s Aux Edge Y %s not supported. Invalid state",
                        name().c_str(), engine_ast::Edge::prettyId(*iei).c_str());
                }
            }

            if ( (*iei)->isAuxEdge() )
            {
                // just make sure edge is already in the map
                if ( xEdge == m_sdpXengineToAuxEdgeMap.end() )
                {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s Aux Edge %s not in xEdge map. Invalid state",
                        name().c_str(), engine_ast::Edge::prettyId(*iei).c_str());
                }
                markAuxEdge(*iei);
            }
            else if ( xEdge != m_sdpXengineToAuxEdgeMap.end() )
            {
                // this is a kIO tensor dataedge linked as Aux edge
                markAuxEdge(*iei);
            }
            else if ( (*iei) == m_InputEdge )
            {
                // source input edge
                markInputEdge(*iei);
                foundEdges |= foundInputEdge;
            }
            else
            {
                // Its data edge, but its new one.
                markInputEdge(*iei);
                newEdges.push_back(*iei);
            }
        }

        // Did we find all expected edges? if not, update internal structures.
        if ( foundEdges != foundAllEdges )
        {
            if (newEdges.size() != 1)
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s multiple new edges found. Invalid state", name().c_str());
            }

            if (!(foundEdges & foundInputEdge))
            {
                m_InputEdge = newEdges[0];
                foundEdges |= foundInputEdge;
            }
            else if (!(foundEdges & foundX1Edge))
            {
                m_sdpXengineToAuxEdgeMap[SDP_ENGINE_X1] = newEdges[0];
                foundEdges |= foundX1Edge;
            }
            else
            {
                m_sdpXengineToAuxEdgeMap[SDP_ENGINE_X2] = newEdges[0];
                foundEdges |= foundX2Edge;
            }

            if (!(foundEdges & foundAllEdges))
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s multiple edges missing from the map. Invalid state", name().c_str());
            }
        }

        PROPAGATE_ERROR_FAIL( verifyEdgePorts() );
    }
    fail:
        return e;
}

NvDlaError engine_ast::SDPSuperOpNode::verifyEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    //gLogInfo << name() << " : verifyEdgePorts: " << endl;
    //printSdpXEdgeMap();

    if (    (inputEdges().size() != 1) ||
            (auxEdges().size() != m_sdpXengineToAuxEdgeMap.size()) ||
            (outputEdges().size() != 1) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s: node in invalid state. edge counts invalid",
            name().c_str());
    }

    // verify input source edge
    if (inputEdges()[0] != m_InputEdge)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s: source input edge invalid",
            name().c_str());
    }

    // verify expected Aux edges
    for (auto it = auxEdges().begin(); it != auxEdges().end(); ++it)
    {
        SdpXengineToEdgeMapIterator xEdge = findSdpAuxEdge(*it);
        if ( xEdge == m_sdpXengineToAuxEdgeMap.end() )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "%s: aux edge invalid.",
                name().c_str());
        }
    }
fail:
    return e;
}

/********** Internal xN-edge map management functions ***************/
NvDlaError engine_ast::SDPSuperOpNode::auxEdgeBySubEngine(SDPSubEngineType xN, engine_ast::Edge **ret_edge)
{
    NvDlaError e = NvDlaSuccess;

    SdpXengineToEdgeMapIterator xEdgeElem = m_sdpXengineToAuxEdgeMap.find(xN.e());
    if (xEdgeElem == m_sdpXengineToAuxEdgeMap.end())
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue,
            "SDP SuperOp %s xN edge lookup failed for %d", name().c_str(), xN.e());

    }
    *ret_edge = xEdgeElem->second;

fail:
    return e;
}

engine_ast::SDPSuperOpNode::SdpXengineToEdgeMapIterator engine_ast::SDPSuperOpNode::findSdpAuxEdge(engine_ast::Edge* edge)
{
    SdpXengineToEdgeMapIterator it;
    for (it = m_sdpXengineToAuxEdgeMap.begin(); it != m_sdpXengineToAuxEdgeMap.end(); ++it)
    {
        if (it->second == edge)
        {
            break;
        }
    }
    return it;
}

void engine_ast::SDPSuperOpNode::printSdpXEdgeMap()
{
    SdpXengineToEdgeMapIterator it;
    gLogInfo << name() << ": SdpXEdgeMap:" << endl;
    if (m_InputEdge != NULL)
        gLogInfo << "\tInput : " << engine_ast::Edge::prettyId(m_InputEdge) << endl;
    else
        gLogInfo << "\tInput : " << "null" << endl;
    for (it = m_sdpXengineToAuxEdgeMap.begin(); it != m_sdpXengineToAuxEdgeMap.end(); ++it)
    {
        gLogInfo << "\t" << NvU16(it->first) << " : " << engine_ast::Edge::prettyId(it->second) << endl;
    }

    EdgeSequence edges = inputEdges();
    for (EdgeSequenceIterator iei = edges.begin(); iei != edges.end(); ++iei)
    {
        gLogInfo << "\tinput : " << engine_ast::Edge::prettyId(*iei) << endl;
    }
    edges = outputEdges();
    for (EdgeSequenceIterator iei = edges.begin(); iei != edges.end(); ++iei)
    {
        gLogInfo << "\toutput : " << engine_ast::Edge::prettyId(*iei) << endl;
    }
    edges = auxEdges();
    for (EdgeSequenceIterator iei = edges.begin(); iei != edges.end(); ++iei)
    {
        gLogInfo << "\taux : " << engine_ast::Edge::prettyId(*iei) << endl;
    }
}

/* Returns pre-stored Aux Surface formats */
std::vector<surface::SurfaceFormat> engine_ast::SDPSuperOpNode::suggestAuxSurfaceFormats(engine_ast::Edge* auxEdge)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    std::vector<surface::SurfaceFormat> supportedAuxSFs;
    std::vector<surface::SurfaceFormat> suggestedAuxSFs;
    SdpXengineToEdgeMapIterator xEdge;
    SDPSubEngineTypeEnum xN;
    TensorType ttN;
    //surface::SurfacePrecision compPrec = graph()->profile()->computePrecision();

    // is auxEdge valid
    if (auxEdge == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Invalid auxEdge for node %s, required to suggest aux surface formats", name().c_str());
    }
    xEdge = findSdpAuxEdge(auxEdge);
    if ( xEdge == m_sdpXengineToAuxEdgeMap.end() )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Invalid auxEdge for node %s, required to suggest aux surface formats", name().c_str());
    }

    // Get tensortype from AuxEdge subengine type
    xN = SDPSubEngineTypeEnum(xEdge->first);
    ttN = params().auxDataType(xN);
    if ( auxEdge->originalTensor()->getTensorType() != ttN )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Invalid auxEdge for node %s, tensor types dont match", name().c_str());
    }

    // Get surface formats for the aux data on subengine
    suggestedAuxSFs = params().auxSurfaceFormats(xN);

    if (suggestedAuxSFs.size() == 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "No supported aux surface formats for %s", name().c_str());
    }

fail:
    return suggestedAuxSFs;
}

/* Return aux data associated with the aux edge */
const void* engine_ast::SDPSuperOpNode::getAuxData(engine_ast::Edge* auxEdge)
{
    NvDlaError e = NvDlaSuccess;
    void* data = NULL;
    SdpXengineToEdgeMapIterator xEdge;
    SDPSubEngineTypeEnum xN;
    NVDLA_UNUSED(e);

    // is auxEdge valid
    if (auxEdge == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Invalid auxEdge for node %s", name().c_str());
    }
    xEdge = findSdpAuxEdge(auxEdge);
    if ( xEdge == m_sdpXengineToAuxEdgeMap.end() )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Invalid auxEdge for node %s", name().c_str());
    }
    xN = SDPSubEngineTypeEnum(xEdge->first);
    data = const_cast<void*>(params().dlaData(xN).values);
fail:
    return data;
}

/********************************Process Aux Data*****************************/
/*
 * SDP SuperOp is created by combining other SDP ops. Hence pre processing
 * of aux data should have already be handled by each indivisual SDP op and
 * not really by SuperOp.
 * preProcessAuxData would remain as an empty op.
 */
NvDlaError engine_ast::SDPSuperOpNode::preProcessAuxData()
{
    NvDlaError e = NvDlaSuccess;
    return e;
}

NvDlaError engine_ast::SDPSuperOpNode::translateAuxDataInternal(SDPSubEngineType xType, SDPSubEngineParams& xParams)
{
    NvDlaError e = NvDlaSuccess;
    Weights trnsData;
    engine_ast::Edge* auxEdge;
    NvU32 channelsPerGroup = 0;

    surface::SurfacePrecision computePrecision = graph()->profile()->computePrecision();
    TensorType dt = params().auxDataType(xType);

    if (dt == TensorType::kIO)
    {
        // No translation required for feature data
        goto fail;
    }

    PROPAGATE_ERROR_FAIL(auxEdgeBySubEngine(xType, &auxEdge));

    if (graph()->profile()->computePrecision().v() == surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        channelsPerGroup = graph()->target_config()->atomicKSize();
    }

    if ( graph()->debugWeights() )
    {
        gLogInfo << "translating weights for " << name() << " data-dims = " <<
            auxEdge->tensorSurfaceDesc()->dimensions().n << "," <<
            auxEdge->tensorSurfaceDesc()->dimensions().c << "," <<
            auxEdge->tensorSurfaceDesc()->dimensions().h << "," <<
            auxEdge->tensorSurfaceDesc()->dimensions().w << endl;
    }

    switch(dt) {
        case TensorType::kBATCH_NORM: {
            Weights meanData = params().rawAdderData(xType);
            Weights varData  = params().rawMultiplierData(xType);

            WeightTrns::WeightDims dims (meanData.count,
                                        auxEdge->tensorSurfaceDesc()->dimensions().n,
                                        auxEdge->tensorSurfaceDesc()->dimensions().c,
                                        auxEdge->tensorSurfaceDesc()->dimensions().w,
                                        auxEdge->tensorSurfaceDesc()->dimensions().h,
                                        1,   //strides dont matter for BN
                                        1);

            PRECISION_SWITCH(meanData.type.v(), computePrecision.v(), trnsData, WeightTrns::translateDataForBatchNorm,
                                                                                xParams.mode(),
                                                                                dims,
                                                                                meanData,
                                                                                varData);
            break;
        }
        case TensorType::kSCALE: {
            Weights rawScaleData  = params().rawMultiplierData(xType);
            WeightTrns::WeightDims dims (rawScaleData.count,
                                    auxEdge->tensorSurfaceDesc()->dimensions().n,
                                    auxEdge->tensorSurfaceDesc()->dimensions().c,
                                    auxEdge->tensorSurfaceDesc()->dimensions().w,
                                    auxEdge->tensorSurfaceDesc()->dimensions().h,
                                    1,   //strides dont matter for Scale
                                    1);

            PRECISION_SWITCH(rawScaleData.type.v(), computePrecision.v(), trnsData, WeightTrns::translateDataForScale,
                                                                                xParams.mode(),
                                                                                dims,
                                                                                rawScaleData,
                                                                                channelsPerGroup);


            break;
        }
        case TensorType::kBIAS: {
            Weights rawBiasData  = params().rawAdderData(xType);
            WeightTrns::WeightDims dims (rawBiasData.count,
                                    auxEdge->tensorSurfaceDesc()->dimensions().n,
                                    auxEdge->tensorSurfaceDesc()->dimensions().c,
                                    auxEdge->tensorSurfaceDesc()->dimensions().w,
                                    auxEdge->tensorSurfaceDesc()->dimensions().h,
                                    1,   //strides ??
                                    1);


            PRECISION_SWITCH(rawBiasData.type.v(), computePrecision.v(), trnsData, WeightTrns::translateDataForBias,
                                                                                xParams.mode(),
                                                                                dims,
                                                                                rawBiasData,
                                                                                channelsPerGroup);
            break;
        }
        case TensorType::kIO:
        default: {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue,
                "SDP SuperOp data translation failed for node '%s'. Unsupported Tensor type: %d",
                name().c_str(), dt);
            break;
        }
    }

    if (trnsData.values == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue,
            "SDP SuperOp data translation failed for node '%s'. Unsupported Tensor type: %d",
            name().c_str(), dt);
    }
    params().setDLAData(xType, trnsData);

fail:
    return e;
}

NvDlaError engine_ast::SDPSuperOpNode::translateAuxData()
{
    NvDlaError e = NvDlaSuccess;

    // Translate X1 aux edge data
    PROPAGATE_ERROR_FAIL(translateAuxDataInternal(SDP_ENGINE_X1, params().x1Params()));
    // Translate X2 aux edge data
    PROPAGATE_ERROR_FAIL(translateAuxDataInternal(SDP_ENGINE_X2, params().x2Params()));

fail:
    return e;
}

NvU64 engine_ast::SDPSuperOpNode::suggestSurfaceSize(surface::TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 size = 0;
    bool isAuxEdge = false;
    SDPSubEngineTypeEnum xN;
    engine_ast::Edge* xEdge;

    PROPAGATE_ERROR_FAIL( verifyEdgePorts() );

    if (m_nodeTSDSurfaceSize.find(tsd) != m_nodeTSDSurfaceSize.end())
    {
        size = m_nodeTSDSurfaceSize[tsd];
        goto fail;
    }

    for (SdpXengineToEdgeMapIterator it = m_sdpXengineToAuxEdgeMap.begin(); it != m_sdpXengineToAuxEdgeMap.end(); ++it)
    {
        xN = SDPSubEngineTypeEnum(it->first);
        xEdge = it->second;

        // check if edge has Aux type tensor.
        if (xEdge->isAuxEdge() && xEdge->tensorSurfaceDesc() == tsd)
        {
            isAuxEdge = true;
            break;
        }
    }

    if (isAuxEdge)
    {
        surface::TensorSurfaceDesc probeTSD = *tsd;
        Dims4 surfDims = suggestSurfaceDims(tsd);
        probeTSD.setDimensions(surfDims);
        probeTSD.resetSize();
        size = probeTSD.size();
        SDPSubEngineParams* prms = subEngineParams(xN);

        // if the op does int8 rescaling, it has both the
        // aux data for the op and the rescaling factors
        if (prms->isINT8Rescaling())
        {
            size *= 2;
        }

        m_nodeTSDSurfaceSize[tsd] = size;
    }

fail:
    return size;
}

NvDlaError engine_ast::SDPSuperOpNode::emitOp(Graph *g,
                                               DLAInterface *target_dla,
                                               NvU32 op_slot, NvU32 batch_id,
                                               DLACommonOpDescAccessor       dep,
                                               DLAOperationContainerAccessor op,
                                               DLASurfaceContainerAccessor   surf)
{
    NvDlaError e = NvDlaSuccess;
    DLASDPOpDescAccessor  sdp_op = op.sdpOpDescAccessor(0);
    DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
    DLASDPOpAccessor x1_op_acc = sdp_op.x1OpAccessor();
    DLASDPOpAccessor x2_op_acc = sdp_op.x2OpAccessor();
    DLASDPOpAccessor y_op_acc  = sdp_op.yOpAccessor();
    DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
    DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
    DLADataCubeAccessor y_data_acc     = surf_acc.yDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(y_data_acc);
    NVDLA_UNUSED(fused_acc);

    surface::TensorSurfaceDesc *src_tsd  = m_InputEdge->tensorSurfaceDesc();
    surface::TensorSurfaceDesc *dst_tsd  = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *x1_tsd = NULL;
    surface::TensorSurfaceDesc *x2_tsd = NULL;
    engine_ast::Edge* auxEdge;
    PROPAGATE_ERROR_FAIL(auxEdgeBySubEngine(SDP_ENGINE_X1, &auxEdge));
    x1_tsd   = auxEdge->tensorSurfaceDesc();
    PROPAGATE_ERROR_FAIL(auxEdgeBySubEngine(SDP_ENGINE_X2, &auxEdge));
    x2_tsd   = auxEdge->tensorSurfaceDesc();

    *sdp_op.srcPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, src_tsd->surfaceFormat().precision());
    *sdp_op.dstPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, dst_tsd->surfaceFormat().precision());
    *sdp_op.LUTIndex()       = -1;
    *sdp_op.batchNum()       = 1;
    *sdp_op.batchStride()    = 0;

    *out_cvt_acc.scale()     = params().outCVT().scale();
    *out_cvt_acc.truncate()  = params().outCVT().truncate();
    *out_cvt_acc.offset()    = params().outCVT().offset();
    *out_cvt_acc.enable()    = static_cast<NvU8>(params().outCVT().isEnable());

    *x1_op_acc.enable()      = params(batch_id).x1Params().enabled();
    *x1_op_acc.ALUType()     = ASTToDLAInterface::getSDPALUType(target_dla, params(batch_id).x1Params().aluType());
    *x1_op_acc.type()        = ASTToDLAInterface::getSDPOpType(target_dla, params(batch_id).x1Params().opType());
    *x1_op_acc.mode()        = ASTToDLAInterface::getSDPMode(target_dla, params(batch_id).x1Params().mode());
    *x1_op_acc.act()         = ASTToDLAInterface::getSDPActType(target_dla, params(batch_id).x1Params().actType());
    // FIXME
    *x1_op_acc.shiftValue()  = params(batch_id).x1Params().shiftValue();
    *x1_op_acc.ALUOperand()  = params(batch_id).x1Params().aluOperand();
    *x1_op_acc.MulOperand()  = /*params(batch_id).x1Params().mulOperand();*/ 1;
    *x1_op_acc.truncate()    = params(batch_id).x1Params().truncate();
    *x1_op_acc.precision()   = ASTToDLAInterface::getSDPPrecision(target_dla, x1_tsd->surfaceFormat().precision());

    *x2_op_acc.enable()      = params(batch_id).x2Params().enabled();
    *x2_op_acc.ALUType()     = ASTToDLAInterface::getSDPALUType(target_dla, params(batch_id).x2Params().aluType());
    *x2_op_acc.type()        = ASTToDLAInterface::getSDPOpType(target_dla, params(batch_id).x2Params().opType());
    *x2_op_acc.mode()        = ASTToDLAInterface::getSDPMode(target_dla, params(batch_id).x2Params().mode());
    *x2_op_acc.act()         = ASTToDLAInterface::getSDPActType(target_dla, params(batch_id).x2Params().actType());
    // FIXME
    *x2_op_acc.shiftValue()  = params(batch_id).x2Params().shiftValue();
    *x2_op_acc.ALUOperand()  = params(batch_id).x2Params().aluOperand();
    *x2_op_acc.MulOperand()  = /*params(batch_id).x2Params().mulOperand();*/ 1;
    *x2_op_acc.truncate()    = params(batch_id).x2Params().truncate();
    *x2_op_acc.precision()   = ASTToDLAInterface::getSDPPrecision(target_dla, x2_tsd->surfaceFormat().precision());

    // FIXME
    *y_op_acc.enable()  = 0;

    emitDependencyParams(target_dla, dep, batch_id);
    setDataCubeAccessor(src_data_acc, src_tsd, IODirectionEnum::INPUT, batch_id);
    setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);
    setDataCubeAccessor(x1_data_acc, x1_tsd, IODirectionEnum::UNKNOWN, batch_id);
    setDataCubeAccessor(x2_data_acc, x2_tsd, IODirectionEnum::UNKNOWN, batch_id);

    if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD )
    {
        *sdp_op.convMode() = sdp_op.convMode_Winograd();

        *src_data_acc.width() = params().winogradParams().ioDims.w;
        *src_data_acc.height() = params().winogradParams().ioDims.h;

        *dst_data_acc.width() = params().winogradParams().ioDims.w;
        *dst_data_acc.height() = params().winogradParams().ioDims.h;
    }
    else if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_DIRECT )
    {
        *sdp_op.convMode() = sdp_op.convMode_Direct();
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Conv mode for %s", name().c_str());
    }

    if ( g->debugOps() )
    {
        gLogInfo << "SDP Super Op node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
        gLogInfo << "\tsrc precision " << (int)*sdp_op.srcPrecision() << endl;
        gLogInfo << "\tdst precision " << (int)*sdp_op.dstPrecision() << endl;
        gLogInfo << "\tx1 enable " << (int)*x1_op_acc.enable() << endl;
        if (*x1_op_acc.enable())
        {
            gLogInfo << "\tx1 precision " << (int)*x1_op_acc.precision() << endl;
            gLogInfo << "\tx1 aluType " << (int)*x1_op_acc.ALUType() << endl;
            gLogInfo << "\tx1 type " << (int)*x1_op_acc.type() << endl;
            gLogInfo << "\tx1 mode " << (int)*x1_op_acc.mode() << endl;
            gLogInfo << "\tx1 act " << (int)*x1_op_acc.act() << endl;
            gLogInfo << "\tx1 shiftValue " << (int)*x1_op_acc.shiftValue() << endl;
            gLogInfo << "\tx1 aluOperand " << (int)*x1_op_acc.ALUOperand() << endl;
            gLogInfo << "\tx1 mulOperand " << (int)*x1_op_acc.MulOperand() << endl;
            gLogInfo << "\tx1 truncate " << (int)*x1_op_acc.truncate() << endl;
        }
        gLogInfo << "\tx2 enable " << (int)*x2_op_acc.enable() << endl;
        if (*x2_op_acc.enable())
        {
            gLogInfo << "\tx2 precision " << (int)*x2_op_acc.precision() << endl;
            gLogInfo << "\tx2 aluType " << (int)*x2_op_acc.ALUType() << endl;
            gLogInfo << "\tx2 type " << (int)*x2_op_acc.type() << endl;

            gLogInfo << "\tx2 mode " << (int)*x2_op_acc.mode() << endl;

            gLogInfo << "\tx2 act " << (int)*x2_op_acc.act() << endl;

            gLogInfo << "\tx2 shiftValue " << (int)*x2_op_acc.shiftValue() << endl;

            gLogInfo << "\tx2 aluOperand " << (int)*x2_op_acc.ALUOperand() << endl;
            gLogInfo << "\tx2 mulOperand " << (int)*x2_op_acc.MulOperand() << endl;
            gLogInfo << "\tx2 truncate " << (int)*x2_op_acc.truncate() << endl;
        }
        gLogInfo << "\ty enable " << (int)*y_op_acc.enable() << endl;
        if (*y_op_acc.enable())
        {
            gLogInfo << "\ty precision " << (int)*y_op_acc.precision() << endl;
            gLogInfo << "\ty aluType " << (int)*y_op_acc.ALUType() << endl;
            gLogInfo << "\ty type " << (int)*y_op_acc.type() << endl;
            gLogInfo << "\ty mode " << (int)*y_op_acc.mode() << endl;
            gLogInfo << "\ty act " << (int)*y_op_acc.act() << endl;
            gLogInfo << "\ty shiftValue " << (int)*y_op_acc.shiftValue() << endl;
            gLogInfo << "\ty aluOperand " << (int)*y_op_acc.ALUOperand() << endl;
            gLogInfo << "\ty mulOperand " << (int)*y_op_acc.MulOperand() << endl;
            gLogInfo << "\ty truncate " << (int)*y_op_acc.truncate() << endl;
        }
        gLogInfo << "\tsrc tsd:" << src_tsd->id() << endl;
        gLogInfo << "\tdst tsd:" << dst_tsd->id() << endl;
        gLogInfo << "\tx1 tsd:" << x1_tsd->id() << endl;
        gLogInfo << "\tx2 tsd:" << x2_tsd->id() << endl;
        gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
        gLogInfo << "\tconv_mode " << (int)*sdp_op.convMode() << endl;
        gLogInfo << "\tsrc addr=" << *src_data_acc.address() << endl;
        gLogInfo << "\tsrc type=" << (int)*src_data_acc.type() << endl;
        gLogInfo << "\tsrc size " << *src_data_acc.size()    << endl;
        gLogInfo << "\tsrc width " << *src_data_acc.width()   << endl;
        gLogInfo << "\tsrc height " << *src_data_acc.height()   << endl;
        gLogInfo << "\tsrc channel " << *src_data_acc.channel()  << endl;
        gLogInfo << "\tsrc linestride " << *src_data_acc.lineStride() << endl;
        gLogInfo << "\tsrc surfstride " << *src_data_acc.surfStride()  << endl;
        gLogInfo << "\tx1 addr=" << *x1_data_acc.address() << endl;
        gLogInfo << "\tx1 type=" << (int)*x1_data_acc.type() << endl;
        gLogInfo << "\tx1 size " << *x1_data_acc.size()    << endl;
        gLogInfo << "\tx1 width " << *x1_data_acc.width()   << endl;
        gLogInfo << "\tx1 height " << *x1_data_acc.height()   << endl;
        gLogInfo << "\tx1 channel " << *x1_data_acc.channel()  << endl;
        gLogInfo << "\tx1 linestride " << *x1_data_acc.lineStride() << endl;
        gLogInfo << "\tx1 surfstride " << *x1_data_acc.surfStride()  << endl;
        gLogInfo << "\tx2 addr=" << *x2_data_acc.address() << endl;
        gLogInfo << "\tx2 type=" << (int)*x2_data_acc.type() << endl;
        gLogInfo << "\tx2 size " << *x2_data_acc.size()    << endl;
        gLogInfo << "\tx2 width " << *x2_data_acc.width()   << endl;
        gLogInfo << "\tx2 height " << *x2_data_acc.height()   << endl;
        gLogInfo << "\tx2 channel " << *x2_data_acc.channel()  << endl;
        gLogInfo << "\tx2 linestride " << *x2_data_acc.lineStride() << endl;
        gLogInfo << "\tx2 surfstride " << *x2_data_acc.surfStride()  << endl;
        gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
        gLogInfo << "\tdst type=" << (int)*dst_data_acc.type() << endl;
        gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
        gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
        gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
        gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
        gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
        gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
        gLogInfo << "\tout_cvt enable " << (int)*out_cvt_acc.enable() << endl;
        gLogInfo << "\tout_cvt scale " << (int)*out_cvt_acc.scale() << endl;
        gLogInfo << "\tout_cvt offset " << (int)*out_cvt_acc.offset() << endl;
        gLogInfo << "\tout_cvt truncate " << (int)*out_cvt_acc.truncate() << endl;
    }

fail:
    return e;
}

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
static nvdla_prototest_interface::SDPOp_SDPOpMode opMode2InfOpMode(engine_ast::SDPMode om)
{
    nvdla_prototest_interface::SDPOp_SDPOpMode iom = nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL;
    switch(om.v())
    {
        case engine_ast::SDPModeEnum::SDP_MODE_PER_CHANNEL: iom =
                nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL; break;
        case engine_ast::SDPModeEnum::SDP_MODE_PER_ELEMENT: iom =
                nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_POINT; break;
        case engine_ast::SDPModeEnum::SDP_MODE_PER_LAYER: iom =
                nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_LAYER; break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown op mode: %s", om.c_str());
    }

    return iom;
}

static nvdla_prototest_interface::SDPOpType opType2InfOpType(engine_ast::SDPOpType ot)
{
    nvdla_prototest_interface::SDPOpType iot = nvdla_prototest_interface::SDP_OP_ADD;
    switch(ot.v())
    {
        case engine_ast::SDPOpTypeEnum::SDP_OP_TYPE_ADD : iot =
                nvdla_prototest_interface::SDP_OP_ADD; break;
        case engine_ast::SDPOpTypeEnum::SDP_OP_TYPE_MUL: iot =
                nvdla_prototest_interface::SDP_OP_MUL; break;
        case engine_ast::SDPOpTypeEnum::SDP_OP_TYPE_BOTH: iot =
                nvdla_prototest_interface::SDP_OP_BOTH; break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown op mode: %s", ot.c_str());
    }

    return iot;
}

static nvdla_prototest_interface::SDPActivation actType2InfActType(engine_ast::SDPActType at)
{
    nvdla_prototest_interface::SDPActivation iat = nvdla_prototest_interface::SDPActivation::ACT_NONE;
    switch(at.v())
    {
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_NONE: iat =
                nvdla_prototest_interface::SDPActivation::ACT_NONE; break;
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_RELU: iat =
                nvdla_prototest_interface::SDPActivation::ACT_RELU; break;
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_TANH:
        case engine_ast::SDPActTypeEnum::SDP_ACT_TYPE_SIGMOID:iat =
                nvdla_prototest_interface::SDPActivation::ACT_LUT; break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown act type: %s", at.c_str());
    }

    return iat;
}

static nvdla_prototest_interface::ALUType aluType2InfAluType(engine_ast::SDPALUType at)
{
    nvdla_prototest_interface::ALUType iat = nvdla_prototest_interface::ALU_SUM;
    switch(at.v())
    {
        case engine_ast::SDPALUTypeEnum::SDP_ALU_TYPE_MAX : iat =
                nvdla_prototest_interface::ALU_MAX; break;
        case engine_ast::SDPALUTypeEnum::SDP_ALU_TYPE_MIN: iat =
                nvdla_prototest_interface::ALU_MIN; break;
        case engine_ast::SDPALUTypeEnum::SDP_ALU_TYPE_SUM: iat =
                nvdla_prototest_interface::ALU_SUM; break;
        case engine_ast::SDPALUTypeEnum::SDP_ALU_TYPE_EQL: iat =
                nvdla_prototest_interface::ALU_EQL; break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown op mode: %s", at.c_str());
    }

    return iat;
}

NvDlaError engine_ast::SDPSuperOpNode::emitOp(NvU32 op_slot, NvU32 batch_id,
                                               DLAInterface* target_dla,
                                               DLACommonOpDescAccessor&        dep,
                                               DLAOperationContainerAccessor&  op,
                                               DLASurfaceContainerAccessor&    surf,
                                               nvdla_prototest_interface::Layer* protoLayer)
{
    NvDlaError e = NvDlaSuccess;

    NvU8 numConsumers = 0;

    DLASDPOpDescAccessor  sdp_op    = op.sdpOpDescAccessor(0);
    DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
    DLASDPOpAccessor x1_op_acc      = sdp_op.x1OpAccessor();
    DLASDPOpAccessor x2_op_acc      = sdp_op.x2OpAccessor();
    DLASDPOpAccessor y_op_acc       = sdp_op.yOpAccessor();
    DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
    DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
    DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
    DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
    DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
    DLADataCubeAccessor y_data_acc      = surf_acc.yDataAccessor();
    DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
    NVDLA_UNUSED(out_cvt_acc);
    NVDLA_UNUSED(y_data_acc);
    NVDLA_UNUSED(batch_id);

    surface::TensorSurfaceDesc *src_tsd     = graph()->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
    surface::TensorSurfaceDesc *dst_tsd     = graph()->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());
    surface::TensorSurfaceDesc *x1_tsd = NULL;
    surface::TensorSurfaceDesc *x2_tsd = NULL;

    nvdla_prototest_interface::SDPOpDesc* protoSDPOpDesc        = protoLayer->mutable_op_config()->mutable_sdp_op();
    nvdla_prototest_interface::SDPSurfaceDesc* protoSDPSurfDesc = protoLayer->mutable_surface()->mutable_sdp_surface();
    nvdla_prototest_interface::SDPOp*          protoSDPX1OpDesc = protoSDPOpDesc->mutable_x1_op();
    nvdla_prototest_interface::SDPOp*          protoSDPX2OpDesc = protoSDPOpDesc->mutable_x2_op();
    nvdla_prototest_interface::SDPOp*          protoSDPYOpDesc  = protoSDPOpDesc->mutable_y_op();
    nvdla_prototest_interface::DataCube* protoSrcDataCube       = protoSDPSurfDesc->mutable_src_data();
    nvdla_prototest_interface::DataCube* protoDstDataCube       = protoSDPSurfDesc->mutable_dst_data();
    nvdla_prototest_interface::DataCube* protoX1DataCube        = protoSDPSurfDesc->mutable_x1_data();
    nvdla_prototest_interface::DataCube* protoX2DataCube        = protoSDPSurfDesc->mutable_x2_data();
    nvdla_prototest_interface::DataPrecision protoSrcPrec, protoDstPrec;

    engine_ast::Edge* auxEdge;
    PROPAGATE_ERROR_FAIL(auxEdgeBySubEngine(SDP_ENGINE_X1, &auxEdge));
    x1_tsd   = auxEdge->tensorSurfaceDesc();
    PROPAGATE_ERROR_FAIL(auxEdgeBySubEngine(SDP_ENGINE_X2, &auxEdge));
    x2_tsd   = auxEdge->tensorSurfaceDesc();

    protoLayer->set_index(op_slot);
    protoLayer->set_roi_index(0);
    protoLayer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP);
    protoLayer->set_dependency_count(*dep.dependencyCount());

    /* consumers */
    for (size_t c = 0; c < EngineType::num_elements(); c++)
    {
        NvS8 fw_op_index = ASTToDLAInterface::getEngineType(target_dla, c);
        if ( fw_op_index < 0 )
        {
            continue;
        }

        DLAConsumerAccessor cons_acc = dep.consumerAccessor(fw_op_index);
        if (*cons_acc.index() != -1) {
            numConsumers++;
            nvdla_prototest_interface::Consumer* protoConsumer = protoLayer->add_bottom();
            protoConsumer->set_index(*cons_acc.index());
            switch(c) {
                case EngineTypeEnum::BDMA : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_BDMA); break;
                case EngineTypeEnum::CONVOLUTION : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
                case EngineTypeEnum::SDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_SDP); break;
                case EngineTypeEnum::PDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_PDP); break;
                case EngineTypeEnum::CDP  : protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CDP); break;
                case EngineTypeEnum::RUBIK: protoConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_RUBIK); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer");
            }
            switch(dependencyParams().consumer(c).opEvent().v()) {
                case OperationEventTypeEnum::OP_CDMA_WEIGHT_DONE : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_CDMA_WT_DONE); break;
                case OperationEventTypeEnum::OP_CDMA_DATA_DONE   : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_CDMA_DT_DONE); break;
                case OperationEventTypeEnum::OP_COMPLETED        : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_COMPLETED); break;
                case OperationEventTypeEnum::OP_ENABLED          : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_ENABLED); break;
                case OperationEventTypeEnum::OP_PROGRAMMED       : protoConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_PROGRAMMED); break;
                default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized consumer event");
            }
        }
    }

    /* fused node */
    if (dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT))
    {
        nvdla_prototest_interface::Consumer* protoFusedConsumer = protoLayer->mutable_fused();
        protoFusedConsumer->set_index(*fused_acc.index());
        protoFusedConsumer->set_event(nvdla_prototest_interface::Consumer_EventType::Consumer_EventType_OP_ENABLED);
        switch(dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT)->engineType().v()) {
            case EngineTypeEnum::CONVOLUTION: protoFusedConsumer->set_type(nvdla_prototest_interface::LayerType::DLA_OP_CONV); break;
            default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "SDP can have only Conv op as its fused partner on input side");
        }
    }

    switch(src_tsd->surfaceFormat().precision().v()) {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : protoSrcPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : protoSrcPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : protoSrcPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized src precision");
    }

    switch(dst_tsd->surfaceFormat().precision().v()) {
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16  : protoDstPrec = nvdla_prototest_interface::DataPrecision::PRECISION_FP16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16 : protoDstPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT16; break;
        case  surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8  : protoDstPrec = nvdla_prototest_interface::DataPrecision::PRECISION_INT8; break;
        default: ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unrecognized dst precision");
    }

    protoSDPOpDesc->set_src_precision(protoSrcPrec);
    protoSDPOpDesc->set_dst_precision(protoDstPrec);
    protoSDPOpDesc->set_lut_index(-1);

    protoSDPOpDesc->mutable_out_cvt()->set_enable(1);
    protoSDPOpDesc->mutable_out_cvt()->set_offset(0);
    protoSDPOpDesc->mutable_out_cvt()->set_scale(1);
    protoSDPOpDesc->mutable_out_cvt()->set_truncate(0);

    protoSDPOpDesc->set_conv_mode(nvdla_prototest_interface::ConvMode::DIRECT);
    protoSDPOpDesc->set_batch_num(1);
    protoSDPOpDesc->set_batch_stride(0);

    protoSDPX1OpDesc->set_enable(*x1_op_acc.enable());
    protoSDPX1OpDesc->set_alu_type(aluType2InfAluType(params(batch_id).x1Params().aluType()));
    protoSDPX1OpDesc->set_type(opType2InfOpType(params(batch_id).x1Params().opType()));
    protoSDPX1OpDesc->set_mode(opMode2InfOpMode(params().x1Params().mode()));
    protoSDPX1OpDesc->set_act(actType2InfActType(params().x1Params().actType()));
    protoSDPX1OpDesc->set_shift_value(*x1_op_acc.shiftValue());
    protoSDPX1OpDesc->set_alu_operand(*x1_op_acc.ALUOperand());
    protoSDPX1OpDesc->set_mul_operand(*x1_op_acc.MulOperand());
    protoSDPX1OpDesc->set_truncate(*x1_op_acc.truncate());
    protoSDPX1OpDesc->set_precision(protoSrcPrec);
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*x1_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*x1_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*x1_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPX1OpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*x1_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*x1_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*x1_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*x1_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPX1OpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*x1_op_acc.cvt().mulCVTAccessor().offset());

    protoSDPX2OpDesc->set_enable(*x2_op_acc.enable());
    protoSDPX2OpDesc->set_alu_type(aluType2InfAluType(params(batch_id).x2Params().aluType()));
    protoSDPX2OpDesc->set_type(opType2InfOpType(params(batch_id).x2Params().opType()));
    protoSDPX2OpDesc->set_mode(opMode2InfOpMode(params().x2Params().mode()));
    protoSDPX2OpDesc->set_act(actType2InfActType(params().x2Params().actType()));
    protoSDPX2OpDesc->set_shift_value(*x2_op_acc.shiftValue());
    protoSDPX2OpDesc->set_alu_operand(*x2_op_acc.ALUOperand());
    protoSDPX2OpDesc->set_mul_operand(*x2_op_acc.MulOperand());
    protoSDPX2OpDesc->set_truncate(*x2_op_acc.truncate());
    protoSDPX2OpDesc->set_precision(protoSrcPrec);
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*x2_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*x2_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*x2_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPX2OpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*x2_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*x2_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*x2_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*x2_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPX2OpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*x2_op_acc.cvt().mulCVTAccessor().offset());


    protoSDPYOpDesc->set_enable(*y_op_acc.enable());
    protoSDPYOpDesc->set_alu_type(nvdla_prototest_interface::ALUType::ALU_SUM);
    protoSDPYOpDesc->set_type(nvdla_prototest_interface::SDPOpType::SDP_OP_ADD);
    protoSDPYOpDesc->set_mode(nvdla_prototest_interface::SDPOp_SDPOpMode::SDPOp_SDPOpMode_SDP_OP_PER_KERNEL);
    protoSDPYOpDesc->set_act(nvdla_prototest_interface::SDPActivation::ACT_NONE);
    protoSDPYOpDesc->set_shift_value(*y_op_acc.shiftValue());
    protoSDPYOpDesc->set_alu_operand(*y_op_acc.ALUOperand());
    protoSDPYOpDesc->set_mul_operand(*y_op_acc.MulOperand());
    protoSDPYOpDesc->set_truncate(*y_op_acc.truncate());
    protoSDPYOpDesc->set_precision(protoSrcPrec);
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_enable(*y_op_acc.cvt().aluCVTAccessor().enable());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_truncate(*y_op_acc.cvt().aluCVTAccessor().truncate());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_scale(*y_op_acc.cvt().aluCVTAccessor().scale());
    protoSDPYOpDesc->mutable_cvt()->mutable_alu_cvt()->set_offset(*y_op_acc.cvt().aluCVTAccessor().offset());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_enable(*y_op_acc.cvt().mulCVTAccessor().enable());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_truncate(*y_op_acc.cvt().mulCVTAccessor().truncate());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_scale(*y_op_acc.cvt().mulCVTAccessor().scale());
    protoSDPYOpDesc->mutable_cvt()->mutable_mul_cvt()->set_offset(*y_op_acc.cvt().mulCVTAccessor().offset());


    if (*fused_acc.index() != -1) {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_HW);
    } else {
        protoSrcDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    }
    protoSrcDataCube->set_address(*src_data_acc.address());
    protoSrcDataCube->set_size(src_tsd->tensorBufferDesc()->size() - src_tsd->bufferOffset());
    protoSrcDataCube->set_width(*src_data_acc.width());
    protoSrcDataCube->set_height(*src_data_acc.height());
    protoSrcDataCube->set_channel(*src_data_acc.channel());
    protoSrcDataCube->set_line_stride(*src_data_acc.lineStride());
    protoSrcDataCube->set_surf_stride(*src_data_acc.surfStride());
    protoSrcDataCube->set_plane_stride(*src_data_acc.planeStride());
    protoSrcDataCube->mutable_mem_info()->set_mem_id(src_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoSrcDataCube->mutable_mem_info()->set_mem_size(src_tsd->tensorBufferDesc()->size());
    protoSrcDataCube->mutable_mem_info()->set_offset(src_tsd->bufferOffset());

    protoDstDataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoDstDataCube->set_address(*dst_data_acc.address());
    protoDstDataCube->set_size(dst_tsd->tensorBufferDesc()->size() - dst_tsd->bufferOffset());
    protoDstDataCube->set_width(*dst_data_acc.width());
    protoDstDataCube->set_height(*dst_data_acc.height());
    protoDstDataCube->set_channel(*dst_data_acc.channel());
    protoDstDataCube->set_line_stride(*dst_data_acc.lineStride());
    protoDstDataCube->set_surf_stride(*dst_data_acc.surfStride());
    protoDstDataCube->set_plane_stride(*dst_data_acc.planeStride());
    protoDstDataCube->mutable_mem_info()->set_mem_id(dst_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoDstDataCube->mutable_mem_info()->set_mem_size(dst_tsd->tensorBufferDesc()->size());
    protoDstDataCube->mutable_mem_info()->set_offset(dst_tsd->bufferOffset());
    if (numConsumers == 0) {
        protoDstDataCube->mutable_mem_info()->set_fill_type(nvdla_prototest_interface::FillerType::FILL_NONE);
        protoDstDataCube->mutable_mem_info()->set_flag(nvdla_prototest_interface::MemFlag::DLA_MEM_OUTPUT);
        protoDstDataCube->mutable_mem_info()->set_precision(protoDstPrec);
    }

    protoX1DataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoX1DataCube->set_address(*x1_data_acc.address());
    protoX1DataCube->set_size(x1_tsd->tensorBufferDesc()->size());
    protoX1DataCube->set_width(*x1_data_acc.width());
    protoX1DataCube->set_height(*x1_data_acc.height());
    protoX1DataCube->set_channel(*x1_data_acc.channel());
    protoX1DataCube->set_line_stride(*x1_data_acc.lineStride());
    protoX1DataCube->set_surf_stride(*x1_data_acc.surfStride());
    protoX1DataCube->set_plane_stride(*x1_data_acc.planeStride());
    protoX1DataCube->mutable_mem_info()->set_mem_id(x1_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoX1DataCube->mutable_mem_info()->set_mem_size(x1_tsd->tensorBufferDesc()->size());
    protoX1DataCube->mutable_mem_info()->set_offset(x1_tsd->bufferOffset());

    protoX2DataCube->set_type(nvdla_prototest_interface::MemType::DLA_MEM_MC);
    protoX2DataCube->set_address(*x2_data_acc.address());
    protoX2DataCube->set_size(x2_tsd->tensorBufferDesc()->size());
    protoX2DataCube->set_width(*x2_data_acc.width());
    protoX2DataCube->set_height(*x2_data_acc.height());
    protoX2DataCube->set_channel(*x2_data_acc.channel());
    protoX2DataCube->set_line_stride(*x2_data_acc.lineStride());
    protoX2DataCube->set_surf_stride(*x2_data_acc.surfStride());
    protoX2DataCube->set_plane_stride(*x2_data_acc.planeStride());
    protoX2DataCube->mutable_mem_info()->set_mem_id(x2_tsd->tensorBufferDesc()->memoryId(batch_id));
    protoX2DataCube->mutable_mem_info()->set_mem_size(x2_tsd->tensorBufferDesc()->size());
    protoX2DataCube->mutable_mem_info()->set_offset(x2_tsd->bufferOffset());
fail:
    return e;
}
#endif

};  // nvdla::priv

};  // nvdla::
