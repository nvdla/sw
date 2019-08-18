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

NvDlaError engine_ast::NestedGraph::populateNestedGraph(NodeSequence& groupedOps)
{
    NvDlaError e = NvDlaSuccess;

    EdgeSequence graphDataInEdges = groupedOps.front()->inputEdges();
    EdgeSequence graphDataOutEdges= groupedOps.back()->outputEdges();
    EdgeSequence nestedGrClonedInEdges;
    EdgeSequence nestedGrClonedOutEdges;
    Node* nestedGrHeadNode = groupedOps.front();
    Node* nestedGrTailNode = groupedOps.back();
    EdgeSet allEdges;
    std::unordered_map< Edge*, Edge* > extToIntEdgeMap;

    setScoredOrdering( new ScoredDependencyOrdering(this) );
    setOrdering(new DependencyOrdering(this->scoredOrdering()));

    engine_ast::Graph* outerGraph = containingSuperNode()->graph();
    engine_ast::Graph* innerGraph = this;
    Node* superNode = containingSuperNode();

    /*
     * Step-1: Cache all edges
     */
    for (NodeSequence::const_iterator ni = groupedOps.begin(); ni != groupedOps.end(); ++ni)
    {
        EdgeSequence nodeEdges = outerGraph->nodeEdges((*ni), ast::EdgeSideEnum::BOTH);
        for (EdgeSequence::const_iterator ei = nodeEdges.begin(); ei != nodeEdges.end(); ++ei)
        {
            allEdges.insert(*ei);
        }
    }

    /*
     * Step-2: Share ownership of nodes in the outer graph with the nested one
     */
    for (NodeSequence::const_iterator ni = groupedOps.begin(); ni != groupedOps.end(); ++ni)
    {
        innerGraph->insertNode(*ni);
    }

    /*
     * Step-3: Disconnect graph input and output edges from inner node and connect them to super node,
     * clone them and associate the clones with inner nodes,
     *
     * Step-3.1: Handle data input edges to the head nodes of the nested graph
     */
    for (EdgeSequence::const_iterator giei = graphDataInEdges.begin(); giei != graphDataInEdges.end(); ++giei )
    {
        /*
         * Edges in nested graph are only for internal local connectivity;
         * they must really use tensors, tsd, tbd from edges of the containing super node
         */
        Edge* clonedInput = innerGraph->addDataEdge(*giei, 0, nestedGrHeadNode, (*giei)->originalTensor());
        clonedInput->setTensorSurfaceDesc((*giei)->tensorSurfaceDesc());
        (*giei)->tensorSurfaceDesc()->setParentEdge(clonedInput);

        if ( debugNestedGraph() )
        {
            gLogInfo << "\tcloning input " << (*giei)->id() << " to " << clonedInput->id()
                     << " type " << (*giei)->edgeType().c_str()
                     << " tsd[" << clonedInput->tensorSurfaceDesc()->id() << "]" << endl;
        }

        /* replace the old downstream node that just got nested, with the superNode */
        (*giei)->graph()->replaceEdgeNodes(*giei, ast::EdgeSideEnum::SECOND, nestedGrHeadNode, superNode);

        nestedGrClonedInEdges.push_back(clonedInput);
        extToIntEdgeMap.insert(std::pair<Edge*, Edge*>(*giei, clonedInput));
    }

    /*
     *  Step-3.2: Handle data output edges from the tail nodes of the nested graph
     */
    for (EdgeSequence::const_iterator goei = graphDataOutEdges.begin(); goei != graphDataOutEdges.end(); ++goei )
    {
        /*
         * Edges in nested graph are only for internal local connectivity;
         * they must really use tensors, tsd, tbd from edges of the containing super node
         */
        Edge* clonedOutput = innerGraph->addDataEdge(*goei, nestedGrTailNode, 0, (*goei)->originalTensor());
        clonedOutput->setTensorSurfaceDesc((*goei)->tensorSurfaceDesc());
        (*goei)->tensorSurfaceDesc()->setParentEdge(clonedOutput);

        if ( debugNestedGraph() )
        {
            gLogInfo << "\tcloning output " << (*goei)->id() << " to " <<  clonedOutput->id()
                     << " type " << (*goei)->edgeType().c_str()
                     << " tsd[" << clonedOutput->tensorSurfaceDesc()->id() << "]" << endl;
        }

        /* replace the old upstream node that just got nested, with the superNode */
        (*goei)->graph()->replaceEdgeNodes(*goei, ast::EdgeSideEnum::FIRST, nestedGrTailNode, superNode);

        nestedGrClonedOutEdges.push_back(clonedOutput);
        extToIntEdgeMap.insert(std::pair<Edge*, Edge*>(*goei, clonedOutput));
    }


    for (NodeSequence::const_iterator ni = groupedOps.begin(); ni != groupedOps.end(); ++ni)
    {
        /*
         * Step-3.3: Handle all non-data input edges to all the nodes of the nested graph
         */
        EdgeSequence nodeInEdges = outerGraph->upstreamEdges(*ni);
        for (EdgeSequenceIterator iei = nodeInEdges.begin(); iei != nodeInEdges.end(); ++iei)
        {
            Edge* clonedInput = NULL;
            if ((*iei)->isComputeEdge())
            {
                clonedInput = innerGraph->addComputeEdge(0, *ni);
            }
            else if ((*iei)->isHazardEdge())
            {
                clonedInput = innerGraph->addHazardEdge(0, *ni);
            }
            else if ((*iei)->isDataEdge())
            {
                continue;
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown edge type %s", (*iei)->id().c_str());
            }

            if ( debugNestedGraph() )
            {
                gLogInfo << "\tcloning input " << (*iei)->id() << " to " << clonedInput->id()
                         << " type " << (*iei)->edgeType().c_str() << endl;
            }

            (*iei)->graph()->replaceEdgeNodes(*iei, ast::EdgeSideEnum::SECOND, *ni, superNode);

            nestedGrClonedInEdges.push_back(clonedInput);
            extToIntEdgeMap.insert(std::pair<Edge*, Edge*>(*iei, clonedInput));
        }

        /*
         * Step-3.4: Handle all non-data output edges from all the nodes of the nested graph
         */
        EdgeSequence nodeOutEdges = outerGraph->downstreamEdges(*ni);
        for (EdgeSequenceIterator oei = nodeOutEdges.begin(); oei != nodeOutEdges.end(); ++oei)
        {
            Edge* clonedOutput = NULL;
            if ((*oei)->isComputeEdge())
            {
                clonedOutput = innerGraph->addComputeEdge(*ni, 0);
            }
            else if ((*oei)->isHazardEdge())
            {
                clonedOutput = innerGraph->addHazardEdge(*ni, 0);
            }
            else if ((*oei)->isDataEdge())
            {
                continue;
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown edge type %s", (*oei)->id().c_str());
            }

            if ( debugNestedGraph() )
            {
                gLogInfo << "\tcloning output " << (*oei)->id() << " to " << clonedOutput->id()
                         << " type " << (*oei)->edgeType().c_str() << endl;
            }

            (*oei)->graph()->replaceEdgeNodes(*oei, ast::EdgeSideEnum::FIRST, *ni, superNode);

            nestedGrClonedOutEdges.push_back(clonedOutput);
            extToIntEdgeMap.insert(std::pair<Edge*, Edge*>(*oei, clonedOutput));
        }
    }

    /*
     * Step-4: Share ownership of internal edges between grouped nodes between both outer graph and the nested one
     */
    for (EdgeSetIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        NodeSequence firstNodes, secondNodes;

        std::unordered_map< Edge*, Edge* >::iterator mapi = extToIntEdgeMap.find(*ei);

        if ( mapi != extToIntEdgeMap.end())
        {
            firstNodes = innerGraph->edgeNodes(mapi->second, ast::EdgeSideEnum::FIRST);
            secondNodes = innerGraph->edgeNodes(mapi->second, ast::EdgeSideEnum::SECOND);
            innerGraph->setEdgeNodes(mapi->second, firstNodes, secondNodes);
            continue;
        }

        firstNodes = outerGraph->edgeNodes(*ei, ast::EdgeSideEnum::FIRST);
        secondNodes = outerGraph->edgeNodes(*ei, ast::EdgeSideEnum::SECOND);
        innerGraph->setEdgeNodes(*ei, firstNodes, secondNodes);

        innerGraph->insertEdge(*ei);
    }

    /*
     * Step-5: Now, finally let the grouped ops be point to the nested graph
     */
    for (NodeSequence::const_iterator ni = groupedOps.begin(); ni != groupedOps.end(); ++ni)
    {
        (*ni)->setGraph(innerGraph);
    }

    innerGraph->setInputEdges(nestedGrClonedInEdges);
    innerGraph->setOutputEdges(nestedGrClonedOutEdges);
    NodeFactory::nodeCast<MultiOpsNode*>(superNode)->setIsomorphicEdgeMap(extToIntEdgeMap);

    ordering()->generate();
    markClean();

    refreshGraphState();

fail:
    return e;
}


void engine_ast::NestedGraph::checkDirty()
{
    if ( dirty() )
    {
        m_ng_ordering->generate();
        markClean();
    }
}

const engine_ast::Graph::NodeSequence &engine_ast::NestedGraph::orderedNodes() { checkDirty(); return m_ng_ordering->nodeOrder(); }
const engine_ast::Graph::EdgeSequence &engine_ast::NestedGraph::orderedEdges() { checkDirty(); return m_ng_ordering->edgeOrder(); }
const engine_ast::Graph::ElemSequence &engine_ast::NestedGraph::orderedElems() { checkDirty(); return m_ng_ordering->elemOrder(); }

//----------------------------------------------------------------------
//                           Nested Graph Utils
//----------------------------------------------------------------------

bool engine_ast::NestedGraph::connectNodesWithEdge(Edge* e, Node* fromNode, Node* toNode)
{
    bool ok = true;

    ok &= containingSuperNode()->graph()->connectNodesWithEdge(e, fromNode, toNode);
    ok &= Graph::connectNodesWithEdge(e, fromNode, toNode);

    return ok;
}

bool engine_ast::NestedGraph::insertEdge(Edge* e)
{
    bool ok = true;

    ok &= containingSuperNode()->graph()->insertEdge(e);
    ok &= Graph::insertEdge(e);

    return ok;
}

bool engine_ast::NestedGraph::insertNode(Node* n)
{
    bool ok = true;
    ok &= containingSuperNode()->graph()->insertNode(n);
    ok &= Graph::insertNode(n);

    return ok;
}

bool engine_ast::NestedGraph::removeEdge(Edge* e)
{
    bool ok = true;
    ok &= containingSuperNode()->graph()->removeEdge(e);
    ok &= Graph::removeEdge(e);

    return ok;
}

bool engine_ast::NestedGraph::removeNode(Node* n)
{
    bool ok = true;
    ok &= containingSuperNode()->graph()->removeNode(n);
    ok &= Graph::removeNode(n);

    return ok;
}

bool engine_ast::NestedGraph::removeEdgeFromNode(Edge *edge, ast::EdgeSide side, Node *node)
{
    bool ok = true;
    ok &= containingSuperNode()->graph()->removeEdgeFromNode(edge, side, node);
    ok &= Graph::removeEdgeFromNode(edge, side, node);

    return ok;
}

bool engine_ast::NestedGraph::removeNodeFromEdge(Edge *edge, ast::EdgeSide side, Node *node)
{
    bool ok = true;
    ok &= containingSuperNode()->graph()->removeNodeFromEdge(edge, side, node);
    ok &= Graph::removeNodeFromEdge(edge, side, node);

    return ok;
}

bool engine_ast::NestedGraph::appendNodeToEdge(Edge *edge, ast::EdgeSide side, Node *node)
{
    bool ok = true;
    ok &= containingSuperNode()->graph()->appendNodeToEdge(edge, side, node);
    ok &= Graph::appendNodeToEdge(edge, side, node);

    return ok;
}

engine_ast::Graph::NodeSequence engine_ast::NestedGraph::topNodes()
{
    NodeSequence sinkNodes;
    EdgeSequence nestedGrInputEdges = inputEdges();
    for (EdgeSequence::const_iterator ei = nestedGrInputEdges.begin(); ei != nestedGrInputEdges.end(); ++ei)
    {
        NodeSequence dsNodes = downstreamNodes(*ei);
        sinkNodes.insert(sinkNodes.end(), dsNodes.begin(), dsNodes.end());
    }
    return sinkNodes;
}

engine_ast::Graph::NodeSequence engine_ast::NestedGraph::bottomNodes()
{
    NodeSequence srcNodes;
    EdgeSequence nestedGrOutputEdges = outputEdges();
    for (EdgeSequence::const_iterator ei = nestedGrOutputEdges.begin(); ei != nestedGrOutputEdges.end(); ++ei)
    {
        NodeSequence usNodes = upstreamNodes(*ei);
        srcNodes.insert(srcNodes.end(), usNodes.begin(), usNodes.end());
    }
    return srcNodes;
}


};  // nvdla::priv

};  // nvdla::
