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

using std::endl;

namespace nvdla
{
namespace priv
{

//----------------------------------------------------------------------
//                           Multi Ops Node Utils
//----------------------------------------------------------------------
NvDlaError engine_ast::MultiOpsNode::plugNestedGraph()
{
    NvDlaError e = NvDlaSuccess;

    MultiOpsNode* superNode = this;

    Graph* outerGraph                = superNode->graph();
    NestedGraph* nestedGraph         = superNode->nestedGraph();
    EdgeSequence superNodeInEdges    = outerGraph->upstreamEdges(this);
    EdgeSequence superNodeOutEdges   = outerGraph->downstreamEdges(this);
    EdgeSequence nestedGrInEdges     = nestedGraph->inputEdges();
    EdgeSequence nestedGrOutEdges    = nestedGraph->outputEdges();
    NodeSequence nestedGrHeadNodes   = nestedGraph->topNodes();
    NodeSequence nestedGrTailNodes   = nestedGraph->bottomNodes();
    NodeSequence nestedGrAllNodes    = nestedGraph->orderedNodes();
    EdgeSequence nestedGrAllEdges;
    EdgeSequence treatedEdges;
    Node* nestedNode = NULL;
    std::unordered_map< Edge*, Edge* > extToIntEdgeMap = isomorphicEdgeMap();
    std::unordered_map< Edge*, Edge* >::iterator edgeMapItr;

    if ( nestedGrInEdges.size() != superNodeInEdges.size() ||
        nestedGrOutEdges.size() != superNodeOutEdges.size() )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Edge Asymmetry between inside and outside of the super node: %s", name().c_str());
    }
    else if ( nestedGrInEdges.size() != nestedGrHeadNodes.size() ||
              nestedGrOutEdges.size() != nestedGrTailNodes.size() )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Edge-Node Asymmetry within the super node: %s", name().c_str());
    }

    /* Connect outside world of the super Node to the nested graph */
    for (EdgeSequence::const_iterator sniei = superNodeInEdges.begin(); sniei != superNodeInEdges.end(); ++sniei)
    {
        edgeMapItr = extToIntEdgeMap.find(*sniei);

        // if the edge is a data edge, it should always connect to the nested head node;
        // for other types, the edge should connect to the prevalent sink node
        if ((*sniei)->isDataEdge())
        {
            nestedNode = nestedGrHeadNodes[0];
        }
        else
        {
            if (edgeMapItr != extToIntEdgeMap.end())
            {
                nestedNode = nestedGraph->downstreamNodes(edgeMapItr->second)[0];
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Edge %s belongs to none of the nested nodes of %s",
                        (*sniei)->id().c_str(), name().c_str());
            }
        }

        (*sniei)->graph()->replaceEdgeNodes(*sniei, ast::EdgeSideEnum::SECOND, superNode, nestedNode);
        (*sniei)->graph()->replaceNodeEdges(nestedNode, ast::EdgeSideEnum::SECOND, edgeMapItr->second, *sniei);

        if ( nestedGraph->debugNestedGraph() )
        {
            gLogInfo << "[PLUG/FLATTEN Nested Graph] replace " << edgeMapItr->second->id()
                     << "(" << edgeMapItr->second->edgeType().c_str() << ") with "
                     << (*sniei)->id() << "(" << (*sniei)->edgeType().c_str() << ") "
                     << "as input edge to " << nestedNode->name() << endl;
        }
    }

    for (EdgeSequence::const_iterator snoei = superNodeOutEdges.begin(); snoei != superNodeOutEdges.end(); ++snoei)
    {
        edgeMapItr = extToIntEdgeMap.find(*snoei);

        // if the edge is a data edge, it should always connect to the nested tail node;
        // for other types, the edge should connect to the prevalent src node
        if ((*snoei)->isDataEdge())
        {
            nestedNode = nestedGrTailNodes[0];
        }
        else
        {
            if (edgeMapItr != extToIntEdgeMap.end())
            {
                nestedNode = nestedGraph->upstreamNodes(edgeMapItr->second)[0];
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Edge %s belongs to none of the nested nodes of %s",
                        (*snoei)->id().c_str(), name().c_str());
            }
        }

        (*snoei)->graph()->replaceEdgeNodes(*snoei, ast::EdgeSideEnum::FIRST, superNode, nestedNode);
        (*snoei)->graph()->replaceNodeEdges(nestedNode, ast::EdgeSideEnum::FIRST, edgeMapItr->second, *snoei);

        if ( nestedGraph->debugNestedGraph() )
        {
            gLogInfo << "[PLUG/FLATTEN Nested Graph] replace " << edgeMapItr->second->id()
                     << "(" << edgeMapItr->second->edgeType().c_str() << ") with "
                     << (*snoei)->id() << "(" << (*snoei)->edgeType().c_str() << ") "
                     << "as output edge from " << nestedNode->name() << endl;
        }
    }

    /* Finally let the nested nodes point to the outer graph */
    for (NodeSequence::const_iterator ni = nestedGrAllNodes.begin(); ni != nestedGrAllNodes.end(); ++ni)
    {
        (*ni)->setGraph(outerGraph);
    }

    // at this point multi-ops node is disconnected from outer graph
    superNode->setIsOnline(false);

    PROPAGATE_ERROR_FAIL( repopulateEdgePorts() );

fail:
    return e;
}

NvDlaError engine_ast::MultiOpsNode::unplugNestedGraph()
{
    NvDlaError e = NvDlaSuccess;

    Node* superNode = this;
    Graph* outerGraph = superNode->graph();
    NodeSequence nestedGrHeadNodes   = nestedGraph()->topNodes();
    NodeSequence nestedGrTailNodes   = nestedGraph()->bottomNodes();
    EdgeSequence nestedGrInEdges     = nestedGraph()->inputEdges();
    EdgeSequence nestedGrOutEdges    = nestedGraph()->outputEdges();
    std::unordered_map< Edge*, Edge* > extToIntEdgeMap = isomorphicEdgeMap();
    std::unordered_map< Edge*, Edge* >::iterator edgeMapItr;

    std::unordered_set< Edge* > exteriorEdges;
    for (EdgeSequenceIterator iei = nestedGrInEdges.begin(); iei != nestedGrInEdges.end(); ++iei)
    {
        exteriorEdges.insert(*iei);
    }
    for (EdgeSequenceIterator oei = nestedGrOutEdges.begin(); oei != nestedGrOutEdges.end(); ++oei)
    {
        exteriorEdges.insert(*oei);
    }

    Graph::NodeSet nestedGrAllNodes = nestedGraph()->nodes();
    for (EdgeSequenceIterator iei = nestedGrInEdges.begin(); iei != nestedGrInEdges.end(); ++iei)
    {
        Node* nestedNode = NULL;
        Edge* outerGrIsomorphicInEdge = NULL;
        Edge* innerGrIsomorphicInEdge = *iei;
        NodeSequence tempNodeSequence;

        outerGrIsomorphicInEdge = outerGraphIsomorphEdgeOf(innerGrIsomorphicInEdge);

        // if the edge is a data edge, it should always connect to the nested head node
        // for other types, the edge should connect to the prevalent sink node
        if (innerGrIsomorphicInEdge->isDataEdge())
        {
            nestedNode = nestedGrHeadNodes[0];
        }
        else
        {
            if (outerGrIsomorphicInEdge)
            {
                nestedNode = outerGraph->downstreamNodes(outerGrIsomorphicInEdge)[0];
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Edge %s belongs to none of the nested nodes of %s",
                        (*iei)->id().c_str(), name().c_str());
            }
        }

        tempNodeSequence.push_back(nestedNode);

        /*
         * Transfer everything in between these 2 boundary edges to the nested graph
         * and replace them with the super node
         */
        outerGrIsomorphicInEdge->graph()->replaceEdgeNodes(outerGrIsomorphicInEdge,
                                                           ast::EdgeSideEnum::SECOND,
                                                           nestedNode,
                                                           superNode);
        nestedGraph()->setEdgeNodes(*iei, NodeSequence(), tempNodeSequence);

        if ( nestedGraph()->debugNestedGraph() )
        {
            gLogInfo << "[UNPLUG/RE-NEST Nested Graph] replace " << innerGrIsomorphicInEdge->id() << "("
                     << innerGrIsomorphicInEdge->edgeType().c_str() << ") with "
                     << outerGrIsomorphicInEdge->id() << "("
                     << outerGrIsomorphicInEdge->edgeType().c_str() << ") as input edge to "
                     << nestedNode->name() << endl;
        }
    }

    for (EdgeSequenceIterator oei = nestedGrOutEdges.begin(); oei != nestedGrOutEdges.end(); ++oei)
    {
        Node* nestedNode = NULL;
        Edge* outerGrIsomorphicOutEdge = NULL;
        Edge* innerGrIsomorphicOutEdge = *oei;
        NodeSequence tempNodeSequence;

        outerGrIsomorphicOutEdge = outerGraphIsomorphEdgeOf(innerGrIsomorphicOutEdge);

        // if the edge is a data edge, it should always connect to the nested tail node
        // for other types, the edge should connect to the prevalent src node
        if (innerGrIsomorphicOutEdge->isDataEdge())
        {
            nestedNode = nestedGrTailNodes[0];
        }
        else
        {
            if (outerGrIsomorphicOutEdge)
            {
                nestedNode = outerGraph->upstreamNodes(outerGrIsomorphicOutEdge)[0];
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Edge %s belongs to none of the nested nodes of %s",
                        (*oei)->id().c_str(), name().c_str());
            }
        }

        tempNodeSequence.push_back(nestedNode);

        outerGrIsomorphicOutEdge->graph()->replaceEdgeNodes(outerGrIsomorphicOutEdge,
                                                            ast::EdgeSideEnum::FIRST,
                                                            nestedNode,
                                                            superNode);
        nestedGraph()->setEdgeNodes(*oei, tempNodeSequence, NodeSequence());
        if (nestedGraph()->debugNestedGraph())
        {
            gLogInfo << "[UNPLUG/RE-NEST Nested Graph] replace " << innerGrIsomorphicOutEdge->id() << "("
                     << innerGrIsomorphicOutEdge->edgeType().c_str() << ") with "
                     << outerGrIsomorphicOutEdge->id() << "("
                     << outerGrIsomorphicOutEdge->edgeType().c_str() << ") as output edge from "
                     << nestedNode->name() << endl;
        }
    }

    /* Finally let the siphoned-off nodes point to the nested graph */
    for (Graph::NodeSet::const_iterator ni = nestedGrAllNodes.begin(); ni != nestedGrAllNodes.end(); ++ni)
    {
        (*ni)->setGraph(nestedGraph());
    }

    // at this point multi-ops node is re-connected in the outer graph
    setIsOnline(true);

    PROPAGATE_ERROR_FAIL( repopulateEdgePorts() );
fail:
    return e;
}

bool engine_ast::MultiOpsNode::isEngineType(EngineType et)
{
    bool match = false;
    Graph::NodeSet nestedNodes = nestedGraph()->nodes();

    if (engineType() == et)
    {
        match = true;
        goto done;
    }

    for (Graph::NodeSetIterator ni = nestedNodes.begin(); ni != nestedNodes.end(); ++ni)
    {
        if ((*ni)->engineType() == et)
        {
            match = true;
            break;
        }
    }

done:
    return match;
}

engine_ast::Edge* engine_ast::MultiOpsNode::outerGraphIsomorphEdgeOf(Edge* nestedEdge)
{
    struct IsEdgeIsomorphicToOuterEdge
    {
        Edge* _nestedEdge;
        IsEdgeIsomorphicToOuterEdge(Edge* ne) { _nestedEdge = ne; }
        bool operator() (const std::pair<Edge*, Edge*>& mapIter) { return mapIter.second == _nestedEdge; }
    };

    std::unordered_map<Edge*, Edge*> isoEdgeMap = isomorphicEdgeMap();
    std::unordered_map<Edge*, Edge*>::iterator oeIter = std::find_if(isoEdgeMap.begin(), isoEdgeMap.end(), IsEdgeIsomorphicToOuterEdge(nestedEdge));

    return oeIter != isoEdgeMap.end() ? oeIter->first : 0;
}

NvDlaError engine_ast::MultiOpsNode::populateEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    Graph::NodeSet nestedNodes = nestedGraph()->nodes();

    if (!isOnline())
    {
        goto fail;
    }

    for (Graph::NodeSetIterator ni = nestedNodes.begin(); ni != nestedNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->repopulateEdgePorts());
    }

    Node::unpopulateEdgePorts();
    PROPAGATE_ERROR_FAIL(Node::populateEdgePorts());

fail:
    return e;
}

NvDlaError engine_ast::MultiOpsNode::repopulateEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    Graph::NodeSet nestedNodes = nestedGraph()->nodes();

    if (!isOnline())
    {
        goto fail;
    }

    Node::unpopulateEdgePorts();
    for (Graph::NodeSetIterator ni = nestedNodes.begin(); ni != nestedNodes.end(); ++ni)
    {
        (*ni)->unpopulateEdgePorts();
    }

    PROPAGATE_ERROR_FAIL(populateEdgePorts());

fail:
    return e;
}

NvDlaError engine_ast::MultiOpsNode::verifyEdgePorts()
{
    NvDlaError e = NvDlaSuccess;

    Graph::NodeSet nestedNodes = nestedGraph()->nodes();

    if (!isOnline())
    {
        goto fail;
    }

    if (inputEdges().size() != 1 ||
        auxEdges().size() != 0 ||
        outputEdges().size() != 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue);
    }

    for (Graph::NodeSetIterator ni = nestedNodes.begin(); ni != nestedNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->verifyEdgePorts());
    }

fail:
    return e;
}

/*-------------------------------Split Nodes---------------------------*/
NvDlaError engine_ast::MultiOpsNode::splitNodes()
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(nestedGraph()->splitNodes());

fail:
    return e;
}

/*-------------------------------Handle Multi Batch--------------------*/
NvDlaError engine_ast::MultiOpsNode::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(nestedGraph()->handleMultiBatch());

fail:
    return e;
}

/*--------------------------------Self Annotation----------------------*/
NvDlaError engine_ast::MultiOpsNode::selfAnnotate(NvS16& lastUsedAnnId)
{
    NvDlaError e = NvDlaSuccess;

    /*
     * Determine task boundaries within the nested graph
     */
    NodeSequence topological_order;

    // multi ops node annotates are not expected.
    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Do not expect MultiOpsNode annotation.");
    
    // But it can be handled with this code
    nestedGraph()->topologicalSort(topological_order);
    PROPAGATE_ERROR_FAIL(nestedGraph()->determineTaskBoundaries(topological_order));
    if (nestedGraph()->graphlets().size() > 1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Can't have more than 1 graphlet inside a super node's nested graph"
                " i.e. A nested graph in a super-node cannot have hybrid engine operations DLA & EMU");
    }

    /*
     * Annotate nodes within the nested graph
     */
    PROPAGATE_ERROR_FAIL(nestedGraph()->annotateNodes(lastUsedAnnId));

fail:
    return e;
}

/*-------------------------Resolve Multi-Batch Dependencies------------*/
NvDlaError engine_ast::MultiOpsNode::resolveMultiBatchDependencies()
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(nestedGraph()->resolveMultiBatchDependencies());

fail:
    return e;
}

};  // nvdla::priv

};  // nvdla::
