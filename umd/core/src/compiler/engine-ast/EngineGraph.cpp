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

#include <algorithm>
#include <stack>

#include "priv/EngineAST.h"
#include "priv/Loadable.h"
#include "priv/Profile.h"
#include "priv/TargetConfig.h"
#include "priv/Tensor.h"

#include "ErrorMacros.h"

#include "math.h" // ceilf

using std::endl;
using std::pair;
using std::map;
using std::set;
using std::string;
using std::stringstream;
using std::vector;
using std::list;

namespace nvdla
{
namespace priv
{

//----------------------------------------------------------------------
//                           Constructors
//----------------------------------------------------------------------
engine_ast::Graph::Graph(Profile *profile, TargetConfig *targetconfig) :
    m_scored_ordering(0),
    m_ordering(0),
    m_next_node_id(0),
    m_next_edge_id(0),
    m_profile(profile),
    m_targetconfig(targetconfig),
    m_memoryResolver(0),
    m_lutManager()
{ }



// Graph copy constructor for clone
// XXX: this is at best mildly broken and likely worse.
engine_ast::Graph::Graph(const engine_ast::Graph &other_g) :
    GraphBase(other_g),
    m_next_node_id(other_g.m_next_node_id),
    m_next_edge_id(other_g.m_next_edge_id),
    m_graph_input_edges(vector<engine_ast::Edge*>()),
    m_graph_output_edges(vector<engine_ast::Edge*>()),
    m_resource_mgr(other_g.m_resource_mgr),
    m_profile(other_g.m_profile),
    m_targetconfig(other_g.m_targetconfig),
    m_memoryResolver(other_g.m_memoryResolver),
    m_lutManager(other_g.m_lutManager)
{
    NvDlaError e = NvDlaSuccess; // for throw macro
    THROW_ERROR(NvDlaError_InvalidState);
}

//----------------------------------------------------------------------
//                           Generic Graph Utils
//----------------------------------------------------------------------

NvDlaError engine_ast::Graph::initGraphResources()
{
    NvDlaError e = NvDlaSuccess;
    vector<memory::Pool> *memPools = m_resource_mgr.memoryPools();
    // Init all pools
    if ( profile()->useMemPool() )
    {
        PROPAGATE_ERROR_THROW( (*memPools)[memory::PoolTypeEnum::GLOBAL_DRAM_POOL].
                               init(memory::PoolTypeEnum::GLOBAL_DRAM_POOL,
                                    m_profile->globalDRAMPoolSize(), 4 * 1024) );

        PROPAGATE_ERROR_THROW( (*memPools)[memory::PoolTypeEnum::LOCAL_DRAM_POOL].
                               init(memory::PoolTypeEnum::LOCAL_DRAM_POOL,
                                    m_profile->localDRAMPoolSize(), 4 * 1024) );

        if( profile()->useCVSRAMAllocate() )
        {
            PROPAGATE_ERROR_THROW( (*memPools)[memory::PoolTypeEnum::LOCAL_CVSRAM_POOL].
                    init(memory::PoolTypeEnum::LOCAL_CVSRAM_POOL,
                         m_profile->localCVSRAMPoolSize(), 4 * 1024) );
        }
    }
    m_lutManager = new LutManager();

    return e;
}

vector<engine_ast::Edge *> engine_ast::Graph::upstreamDataEdges(engine_ast::Node * node)
{
    EdgeSequence r, d = upstreamEdges(node);
    for (EdgeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ( (*di)->isDataEdge() )
        {
            r.push_back(*di);
        }
    }
    return r;
}

vector<engine_ast::Edge *> engine_ast::Graph::downstreamDataEdges(engine_ast::Node * node)
{
    EdgeSequence r, d = downstreamEdges(node);
    for (EdgeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ( (*di)->isDataEdge() )
        {
            r.push_back(*di);
        }
    }
    return r;
}

/**
 * We call two data edges to be siblings, if they share same parent/upstream node.
 * returns set of edges that are siblings to given edge.
 * consider below case, with edges
 *   e-0: node-0-->node-2
 *   e-1: node-1-->node-2
 *   e-2: node-1-->node-3
 *
 *           (node-0)    (node-1)
 *                  \    /       \
 *              e-0  \  / e-1     \ e-2
 *                    \/           \
 *                 (node-2)      (node-3)
 *
 *  e-1 and e-2 are siblings while e-0 is not sibling to any other edge.
 **/
vector<engine_ast::Edge *> engine_ast::Graph::siblingDataEdges(engine_ast::Edge* edge)
{
    EdgeSequence es;
    NodeSequence inputNodes = upstreamNodes(edge);

    for (NodeSequenceIterator ni = inputNodes.begin(); ni != inputNodes.end(); ++ni)
    {
        EdgeSequence outputEdges = downstreamDataEdges(*ni);
        for (EdgeSequenceIterator ei = outputEdges.begin(); ei != outputEdges.end(); ++ei)
        {
            if ((*ei) != edge)
            {
                es.push_back(*ei);
            }
        }
    }

    return es;
}

engine_ast::Edge* engine_ast::Graph::connectingDataEdge(engine_ast::Node* fromNode, engine_ast::Node * toNode, ast::EdgeSide fromDir)
{
    engine_ast::Edge* retEdge = NULL;
    EdgeSequence fromNodeEdges;
    ast::EdgeSide toDir = ast::EdgeSideEnum::SECOND;
    if (fromDir == ast::EdgeSideEnum::FIRST)
    {
        fromNodeEdges = downstreamDataEdges(fromNode);
        toDir = ast::EdgeSideEnum::SECOND;
    }
    else
    {
        fromNodeEdges = upstreamDataEdges(fromNode);
        toDir = ast::EdgeSideEnum::FIRST;
    }
    for (EdgeSequenceIterator fi = fromNodeEdges.begin(); fi != fromNodeEdges.end(); ++fi)
    {
        NodeSequence nodes = edgeNodes(*fi, toDir);
        for (NodeSequenceIterator ni = nodes.begin(); ni != nodes.end(); ++ni)
        {
            if ( (*ni) == toNode )
            {
                retEdge = (*fi);
                goto done;
            }
        }
    }
done:
    return retEdge;
}

vector<engine_ast::Edge *> engine_ast::Graph::upstreamAuxEdges(engine_ast::Node * node)
{
    EdgeSequence r;
    EdgeSequence d = upstreamEdges(node);
    for (EdgeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ((*di)->isAuxEdge())
        {
            r.push_back(*di);
        }
    }
    return r;
}

engine_ast::Edge* engine_ast::Graph::getUpstreamAuxEdge(engine_ast::Node * node, NvU8 id)
{
    Edge* e = NULL;
    EdgeSequence edges = upstreamAuxEdges(node);
    if ( id < edges.size() )
    {
        e = edges[id];
    }
    return e;
}

vector<engine_ast::Edge *> engine_ast::Graph::downstreamComputeEdges(engine_ast::Node * node)
{
    EdgeSequence r, d = downstreamEdges(node);
    for (EdgeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ( (*di)->isComputeEdge() )
        {
            r.push_back(*di);
        }
    }
    return r;
}


vector<engine_ast::Edge *> engine_ast::Graph::downstreamHazardEdges(engine_ast::Node * node)
{
    EdgeSequence r, d = downstreamEdges(node);
    for (EdgeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ( (*di)->isHazardEdge() )
        {
            r.push_back(*di);
        }
    }
    return r;
}

vector<engine_ast::Node* > engine_ast::Graph::downstreamDataNodes(engine_ast::Node* node)
{
/*    NodeSequence r, d = downstreamNodes(node);
    gLogInfo << "out of " << d.size() << " consumers " << endl;
    for(NodeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ( connectedDataNodes(node, *di) && std::find(r.begin(), r.end(), *di) == r.end())
        {
            gLogInfo << (*di)->name() << " is a consumer of " << node->name() << std::endl;
            r.push_back(*di);
        }
    }
    return r;
    */
    NodeSequence r;
    EdgeSequence d = downstreamDataEdges(node);
    for ( EdgeSequenceIterator ei = d.begin(); ei != d.end(); ++ei )
    {
        NodeSequence consumerNodes = downstreamNodes(*ei);
        for ( NodeSequenceIterator ni = consumerNodes.begin(); ni != consumerNodes.end(); ++ni )
        {
            if (std::find(r.begin(), r.end() , *ni) == r.end())
            {
                r.push_back(*ni);
            }
        }
    }
    return r;
}

vector<engine_ast::Node* > engine_ast::Graph::downstreamComputeNodes(engine_ast::Node* node)
{
    NodeSequence r;
    EdgeSequence d = downstreamComputeEdges(node);
    for ( EdgeSequenceIterator ei = d.begin(); ei != d.end(); ++ei )
    {
        NodeSequence consumerNodes = downstreamNodes(*ei);
        for ( NodeSequenceIterator ni = consumerNodes.begin(); ni != consumerNodes.end(); ++ni )
        {
            if (std::find(r.begin(), r.end() , *ni) == r.end())
            {
                r.push_back(*ni);
            }
        }
    }
    return r;
}

vector<engine_ast::Node* > engine_ast::Graph::downstreamHazardNodes(engine_ast::Node* node)
{
    NodeSequence r;
    EdgeSequence d = downstreamHazardEdges(node);
    for ( EdgeSequenceIterator ei = d.begin(); ei != d.end(); ++ei )
    {
        NodeSequence consumerNodes = downstreamNodes(*ei);
        for ( NodeSequenceIterator ni = consumerNodes.begin(); ni != consumerNodes.end(); ++ni )
        {
            if (std::find(r.begin(), r.end() , *ni) == r.end())
            {
                r.push_back(*ni);
            }
        }
    }
    return r;
}

vector<engine_ast::Node* > engine_ast::Graph::upstreamDataNodes(engine_ast::Node* node)
{
/*    NodeSequence r, u = upstreamNodes(node);
    gLogInfo << "out of " << u.size() << " producers " << endl;
    for(NodeSequenceIterator ui = u.begin(); ui != u.end(); ++ui)
    {
        if ( connectedDataNodes(*ui, node) && std::find(r.begin(), r.end(), *ui) == r.end())
        {
            gLogInfo << (*ui)->name() << " is a producer of " << node->name() << std::endl;
            r.push_back(*ui);
        }
    }
    return r;
    */
    NodeSequence r;
    EdgeSequence u = upstreamDataEdges(node);
    for ( EdgeSequenceIterator ei = u.begin(); ei != u.end(); ++ei )
    {
        NodeSequence producerNodes = upstreamNodes(*ei);
        for ( NodeSequenceIterator ni = producerNodes.begin(); ni != producerNodes.end(); ++ni )
        {
            if (std::find(r.begin(), r.end() , *ni) == r.end())
            {
                r.push_back(*ni);
            }
        }
    }
    return r;
}

vector<engine_ast::Edge *> engine_ast::Graph::upstreamHazardEdges(engine_ast::Node * node)
{
    EdgeSequence r, d = upstreamEdges(node);
    for (EdgeSequenceIterator di = d.begin(); di != d.end(); ++di)
    {
        if ( (*di)->isHazardEdge() )
        {
            r.push_back(*di);
        }
    }
    return r;
}

Tensor* engine_ast::Graph::addAuxTensor(const string &s, const Dims4 dims, TensorType tt)
{
    Tensor* at = NULL;
    TensorFactory::TensorPrivPair t = TensorFactory::newTensor();
    if ( !t ) {
        goto done;
    }
    t.i()->setName(s.c_str());
    t.i()->setDimensions(dims);
    t.i()->setTensorType(tt);
    m_aux_tensors.push_back(t.priv());
    at = t.priv();
done:
    return at;
}

string engine_ast::Graph::newAuxTensorName()
{
    stringstream ss;
    ss << "tensor-aux-" << m_aux_tensors.size();
    return ss.str();
}

// tbd: remove the recursion from this.
bool engine_ast::Graph::connectedComputeNodes(engine_ast::Node *upStream, engine_ast::Node* downStream)
{
    return downStream->dependsOn(upStream, viaCompute, allowAll);
}

bool engine_ast::Graph::connectedDataNodes(engine_ast::Node *upStream, engine_ast::Node* downStream)
{
    return downStream->dependsOn(upStream, viaData, allowAll);
}

void engine_ast::Graph::replaceEdgeNodes(Edge* edge, ast::EdgeSide dir, Node* oldNode, Node* newNode)
{
    removeEdgeFromNode(edge, dir, oldNode);
    removeNodeFromEdge(edge, dir, oldNode);
    appendNodeToEdge(edge, dir, newNode);
}

void engine_ast::Graph::replaceNodeEdges(Node* node, ast::EdgeSide dir, Edge* oldEdge, Edge* newEdge)
{
    removeEdgeFromNode(oldEdge, dir, node);
    removeNodeFromEdge(oldEdge, dir, node);
    appendNodeToEdge(newEdge, dir, node);
}

bool engine_ast::Graph::connectNodesWithEdge(Edge* e, Node* fromNode, Node* toNode)
{
    bool ok = true;
    insertEdge(e);

    if ( fromNode )
    {
        ok &= appendNodeToEdge(e, ast::EdgeSideEnum::FIRST, fromNode);
    }
    if ( toNode )
    {
        ok &= appendNodeToEdge(e, ast::EdgeSideEnum::SECOND, toNode);
    }

    return ok;
}

/* Remove the given node from the AST by detaching and deleting its
 * edge(s) in the specified I/O direction, aux edge (if any) and
 * reconnecting the edge(s) on the other I/O side to its neighboring
 * upstream or downstream node(s) accordingly
 */
NvDlaError engine_ast::Graph::removeNodeFromAST(Node* killNode, IODirection iod)
{
    NvDlaError e = NvDlaSuccess;

    NodeSequence ioSideNodes;
    EdgeSequence ioSideEdges;
    EdgeSequence oppSideEdges;
    Edge* killNodeAuxEdge = NULL;

    if (iod.v() == IODirectionEnum::UNKNOWN)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Can't remove node unless the "
                "I/O direction of edges to trim is specified: %s", iod.c_str());
    }

    ioSideNodes  = iod.v() == IODirectionEnum::INPUT ? upstreamNodes(killNode) : downstreamNodes(killNode);
    ioSideEdges  = iod.v() == IODirectionEnum::INPUT ? upstreamEdges(killNode) : downstreamEdges(killNode);
    oppSideEdges = iod.v() == IODirectionEnum::INPUT ? downstreamEdges(killNode) : upstreamEdges(killNode);

    killNode->nodeAuxEdge(&killNodeAuxEdge);

    /* Transfer that set of edge(s) of the node_to_be_removed which are going to stay in the graph -
     * to the node(s) on the opposite side
     */
    for (EdgeSequenceIterator oppsei = oppSideEdges.begin(); oppsei != oppSideEdges.end(); ++oppsei)
    {
        if ((*oppsei)->isAuxEdge())
        {
            continue;
        }

        for (NodeSequenceIterator iosni = ioSideNodes.begin(); iosni != ioSideNodes.end(); ++iosni)
        {
            if (iod.v() == IODirectionEnum::INPUT)
            {
                replaceEdgeNodes(*oppsei, ast::EdgeSideEnum::FIRST, killNode, *iosni);
            }
            else
            {
                replaceEdgeNodes(*oppsei, ast::EdgeSideEnum::SECOND, killNode, *iosni);
            }
        }
    }

    /* Detach and delete the specified I/O side edge(s) from the node_to_be_removed and the
     * node(s) on the other side of those edge(s)
     */
    for (EdgeSequenceIterator iosei = ioSideEdges.begin(); iosei != ioSideEdges.end(); ++iosei)
    {
        if ((*iosei)->isAuxEdge())
        {
            continue;
        }

        for (NodeSequenceIterator iosni = ioSideNodes.begin(); iosni != ioSideNodes.end(); ++iosni)
        {
            if (iod.v() == IODirectionEnum::INPUT)
            {
                removeEdgeFromNode(*iosei, ast::EdgeSideEnum::FIRST, *iosni);
                removeNodeFromEdge(*iosei, ast::EdgeSideEnum::FIRST, *iosni);
            }
            else
            {
                removeEdgeFromNode(*iosei, ast::EdgeSideEnum::SECOND, *iosni);
                removeNodeFromEdge(*iosei, ast::EdgeSideEnum::SECOND, *iosni);
            }
        }

        if (iod.v() == IODirectionEnum::INPUT)
        {
            removeEdgeFromNode(*iosei, ast::EdgeSideEnum::SECOND, killNode);
            removeNodeFromEdge(*iosei, ast::EdgeSideEnum::SECOND, killNode);
        }
        else
        {
            removeEdgeFromNode(*iosei, ast::EdgeSideEnum::FIRST, killNode);
            removeNodeFromEdge(*iosei, ast::EdgeSideEnum::FIRST, killNode);
        }

        resourceMgr()->unregTensorSurfaceDesc((*iosei)->tensorSurfaceDesc());
        resourceMgr()->unregTensorBufferDesc((*iosei)->tensorBufferDesc());
        removeEdge(*iosei);
    }

    /* Repopulate the edge ports of the affected node(s) after this edge upheaval */
    for (NodeSequenceIterator iosni = ioSideNodes.begin(); iosni != ioSideNodes.end(); ++iosni)
    {
        PROPAGATE_ERROR_FAIL((*iosni)->repopulateEdgePorts());
    }

    /* Detach and delete the aux edge (if any) of the node_to_be_removed */
    if (killNodeAuxEdge)
    {
        removeEdgeFromNode(killNodeAuxEdge, ast::EdgeSideEnum::SECOND, killNode);
        removeNodeFromEdge(killNodeAuxEdge, ast::EdgeSideEnum::SECOND, killNode);
        resourceMgr()->unregTensorSurfaceDesc(killNodeAuxEdge->tensorSurfaceDesc());
        resourceMgr()->unregTensorBufferDesc(killNodeAuxEdge->tensorBufferDesc());
        removeEdge(killNodeAuxEdge);
        delete killNodeAuxEdge;
    }

    /* Finally remove the node */
    removeNode(killNode);
    delete killNode;
    killNode = NULL;

fail:
    return e;
}

/* Substitute a node in AST with a chain of nodes by delegating all its input
 * (except the aux edge since they are non-transferable) and output edge(s)
 * to the substitute nodes' chain
 */
NvDlaError engine_ast::Graph::substituteNodeInAST(Node* origNode, NodeSequence subNodes)
{
    NvDlaError e = NvDlaSuccess;

    Node* headSubNode = subNodes.front();
    Node* tailSubNode = subNodes.back();
    NodeSequence origNodeSrcNodes  = upstreamNodes(origNode);
    EdgeSequence origNodeSrcEdges  = upstreamEdges(origNode);
    EdgeSequence origNodeSinkEdges = downstreamEdges(origNode);

    /* Delegate all input edge(s) to the substitute node */
    for (EdgeSequenceIterator uei = origNodeSrcEdges.begin(); uei != origNodeSrcEdges.end(); ++uei)
    {
        if ( (*uei)->isAuxEdge() )
        {
            continue;
        }
        replaceEdgeNodes(*uei, ast::EdgeSideEnum::SECOND, origNode, headSubNode);
    }

    /* Delegate all output edge(s) to the substitute node */
    for (EdgeSequenceIterator dei = origNodeSinkEdges.begin(); dei != origNodeSinkEdges.end(); ++dei)
    {
        replaceEdgeNodes(*dei, ast::EdgeSideEnum::FIRST, origNode, tailSubNode);
    }

    /* Repopulate the edge ports of the substitute node after this edge upheaval */
    for (NodeSequenceIterator sni = subNodes.begin(); sni != subNodes.end(); ++sni)
    {
        PROPAGATE_ERROR_FAIL( (*sni)->repopulateEdgePorts() );
    }

fail:
    return e;
}

/* Substitute an edge in AST with another by delegating its input
 * and output side nodes to the substitute edge
 */
NvDlaError engine_ast::Graph::substituteEdgeInAST(Edge* origEdge, Edge* subEdge)
{
    NvDlaError e = NvDlaSuccess;

    NodeSequence origEdgeSrcNodes  = upstreamNodes(origEdge);
    NodeSequence origEdgeSinkNodes = downstreamNodes(origEdge);

    /* Delegate all input side nodes to the substitute edge */
    for (NodeSequenceIterator uni = origEdgeSrcNodes.begin(); uni != origEdgeSrcNodes.end(); ++uni)
    {
        replaceNodeEdges(*uni, ast::EdgeSideEnum::FIRST, origEdge, subEdge);
    }

    /* Delegate all output side nodes to the substitute edge */
    for (NodeSequenceIterator dni = origEdgeSinkNodes.begin(); dni != origEdgeSinkNodes.end(); ++dni)
    {
        replaceNodeEdges(*dni, ast::EdgeSideEnum::SECOND, origEdge, subEdge);
    }

    /* Repopulate the edge ports of the all the nodes after this edge upheaval */
    for (NodeSequenceIterator uni = origEdgeSrcNodes.begin(); uni != origEdgeSrcNodes.end(); ++uni)
    {
        PROPAGATE_ERROR_FAIL((*uni)->repopulateEdgePorts());
    }
    for (NodeSequenceIterator dni = origEdgeSinkNodes.begin(); dni != origEdgeSinkNodes.end(); ++dni)
    {
        PROPAGATE_ERROR_FAIL((*dni)->repopulateEdgePorts());
    }

fail:
    return e;
}

engine_ast::Edge *engine_ast::Graph::addComputeEdge(Node *fromNode, Node *toNode)
{
    NvDlaError e = NvDlaSuccess;
    Edge *edge       = new engine_ast::Edge((canonical_ast::Edge*)0); // no canonical

    edge->setGraph(this);
    edge->setId(nextEdgeId());
    edge->setComputeEdge();
    edge->setOriginalTensor(0); // no tensor

    if ( !connectNodesWithEdge(edge, fromNode, toNode) )
    {
        THROW_ERROR(NvDlaError_BadValue, "failed to insert compute edge %s between %s and %s",
                                                edge->id().c_str(), fromNode->name().c_str(), toNode->name().c_str());
    }

    return edge;
}

engine_ast::Edge *engine_ast::Graph::addHazardEdge(Node *fromNode, Node *toNode)
{
    NvDlaError e = NvDlaSuccess;
    Edge *edge       = new engine_ast::Edge((canonical_ast::Edge*)0); // no canonical

    edge->setGraph(this);
    edge->setId(nextEdgeId());
    edge->setHazardEdge();
    edge->setOriginalTensor(0); // no tensor

    if ( !connectNodesWithEdge(edge, fromNode, toNode) )
    {
        THROW_ERROR(NvDlaError_BadValue, "failed to insert hazard edge %s between %s and %s",
                                                edge->id().c_str(), fromNode->name().c_str(), toNode->name().c_str());
    }

    return edge;
}

engine_ast::Edge *engine_ast::Graph::addDataEdge(canonical_ast::Edge *canEdge, Node *fromNode, Node *toNode, Tensor *origTensor)
{
    NvDlaError e = NvDlaSuccess;
    Edge *edge       = new engine_ast::Edge(canEdge); // 0 is ok for canEdge

    edge->setGraph(this);
    edge->setId(nextEdgeId());
    edge->setDataEdge();
    if ( origTensor)
    {
        edge->setOriginalTensor(origTensor);
    }

    if ( !connectNodesWithEdge(edge, fromNode, toNode) )
    {
        THROW_ERROR(NvDlaError_BadValue, "failed to insert edge %s between %s and %s",
                                                edge->id().c_str(), fromNode->name().c_str(), toNode->name().c_str());
    }

    return edge;
}

engine_ast::Edge *engine_ast::Graph::addDataEdge(engine_ast::Edge *cloneEdge, Node *fromNode, Node *toNode, Tensor *origTensor)
{
    NvDlaError e = NvDlaSuccess;
    Edge *edge;
    if ( cloneEdge )
    {
        edge = new engine_ast::Edge(*cloneEdge);
    }
    else
    {
        edge = new engine_ast::Edge((canonical_ast::Edge*)0);
    }

    edge->setGraph(this);
    edge->setId(nextEdgeId());
    edge->setDataEdge();
    if ( origTensor )
    {
        edge->setOriginalTensor(origTensor);
    }

    if ( !connectNodesWithEdge(edge, fromNode, toNode) )
    {
        THROW_ERROR(NvDlaError_BadValue, "failed to insert edge %s between %s and %s",
                                                edge->id().c_str(), fromNode->name().c_str(), toNode->name().c_str());
    }

    return edge;
}


void engine_ast::Graph::checkDirty()
{
    if ( dirty() )
    {
        m_ordering->generate();
        markClean();
    }
}


const engine_ast::Graph::NodeSequence &engine_ast::Graph::orderedNodes() { checkDirty(); return m_ordering->nodeOrder(); }
const engine_ast::Graph::EdgeSequence &engine_ast::Graph::orderedEdges() { checkDirty(); return m_ordering->edgeOrder(); }
const engine_ast::Graph::EdgeSequence &engine_ast::Graph::orderedDataEdges() { checkDirty(); return m_ordering->dataEdgeOrder(); }
const engine_ast::Graph::EdgeSequence &engine_ast::Graph::orderedComputeEdges() { checkDirty(); return m_ordering->computeEdgeOrder(); }
const engine_ast::Graph::ElemSequence &engine_ast::Graph::orderedElems() { checkDirty(); return m_ordering->elemOrder(); }

void engine_ast::DependencyOrdering::clear()
{
    ast::GraphOrdering< engine_ast::Graph >::clear();

    m_data_edge_order.clear();
    m_compute_edge_order.clear();
    m_hazard_edge_order.clear();
}

NvDlaError engine_ast::DependencyOrdering::generate()
{
    NvDlaError e = NvDlaSuccess;

    clear();

    e = m_sdo->generate();

    const std::vector<ast::ScoredGraphOrdering<engine_ast::Graph>::ElemScoresIterator> &elemScoreOrder = m_sdo->elemScoreOrder();
    for ( size_t i = 0; i < elemScoreOrder.size(); ++i )
    {
        Elem elem = elemScoreOrder[i]->first;
        // ast::ScoredGraphOrdering<engine_ast::Graph>::Score s = elemScoreOrder[i]->second;
        m_elem_order.push_back(elem);

        if ( !elem.first == !elem.second )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "illegal element");
        }

        if ( elem.first )
        {
            m_node_order.push_back(elem.first);
        }
        else if ( elem.second )
        {
            m_edge_order.push_back(elem.second);
            if ( elem.second->isDataEdge() )
            {
                m_data_edge_order.push_back(elem.second);
            }
            else if ( elem.second->isComputeEdge() )
            {
                m_compute_edge_order.push_back(elem.second);
            }
            else if ( elem.second->isHazardEdge() )
            {
                m_hazard_edge_order.push_back(elem.second);
            }
        }
    }
 fail:
    return e;
}


// returns the i'th surface tensor which *matches* the condition given
surface::TensorSurfaceDesc *
engine_ast::Graph::nodeTensorSurface(const engine_ast::Node *n, size_t i,
                                    const vector<surface::SurfaceCategory> &types,
                                    ast::EdgeSideEnum dir)
{
    size_t matched = 0;
    surface::TensorSurfaceDesc* matched_tsd = NULL;
    vector<engine_ast::Edge *> test_edges = nodeEdges(n, dir);
    for ( size_t tei = 0, TEI = test_edges.size(); tei != TEI; ++tei )
    {
        if ( !test_edges[tei]->isDataEdge() )
        {
            continue;
        }
        surface::TensorSurfaceDesc * e_tsd = test_edges[tei]->tensorSurfaceDesc();
        surface::SurfaceCategoryEnum sc = e_tsd->surfaceFormat().category().e();
        for ( size_t tt = 0, TT = types.size(); tt != TT; ++tt)
        {
            if ( sc == types[tt].e() )
            {
                if ( matched == i )
                {
                    matched_tsd = e_tsd;
                    goto done;
                }
                matched++;
            }
        }
    }

done:
    return matched_tsd;
}

surface::TensorSurfaceDesc *engine_ast::Graph::nodeInputTensorSurface(const engine_ast::Node *n, size_t i, const vector<surface::SurfaceCategory> &types)
{
    return nodeTensorSurface(n, i, types, ast::EdgeSideEnum::SECOND);
}

surface::TensorSurfaceDesc *engine_ast::Graph::nodeOutputTensorSurface(const engine_ast::Node *n, size_t i, const vector<surface::SurfaceCategory> &types)
{
    return nodeTensorSurface(n, i, types, ast::EdgeSideEnum::FIRST);
}

surface::SurfaceFormat engine_ast::Graph::suggestNwSurfaceFormat(TensorType nst)
{
    surface::SurfaceFormat sf = surface::SurfaceFormatEnum::NVDLA_UNKNOWN_FORMAT;
    if (nst == TensorType::kNW_INPUT)
    {
        sf = profile()->networkInputSurfaceFormat();
    }
    else if (nst == TensorType::kNW_OUTPUT)
    {
        sf = profile()->networkOutputSurfaceFormat();
    }
    else
    {
        REPORT_ERROR(NvDlaError_BadParameter, "Bad Network surface format:%d", (int)nst);
    }
    return sf;
}

void engine_ast::Graph::printGraph(engine_ast::Graph* g, bool nested, std::string graphName)
{
    typedef engine_ast::Graph::EdgeSequence::const_iterator ESI;
    gLogInfo << "printGraph: " << graphName << std::endl;
    engine_ast::Graph::NodeSequence allNodes = g->orderedNodes();
    for (engine_ast::Graph::NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        gLogInfo << (nested ? "\t\t" : "\t") << (*ni)->id() << ":" << (*ni)->name() << "/"
                 << ((*ni)->canonicalNode() ? (*ni)->canonicalNode()->name() : "" ) << ":";
        engine_ast::Graph::EdgeSequence inputEdges = g->nodeEdges(*ni, ast::EdgeSideEnum::SECOND);
        engine_ast::Graph::EdgeSequence outputEdges = g->nodeEdges(*ni, ast::EdgeSideEnum::FIRST);
        for (ESI ii = inputEdges.begin(); ii != inputEdges.end(); ++ii)
        {
            if ((*ii)->isAuxEdge())
                gLogInfo << "\t(Aux)";
            else
                gLogInfo << "\t(in)";
            gLogInfo << (*ii)->id();
            if ((*ii)->tensorSurfaceDesc())
            {
                Dims4 dims = (*ii)->tensorSurfaceDesc()->dimensions();
                gLogInfo << "[" << dims.n << "x" << dims.c << "x" << dims.h << "x" << dims.w << "]";
                gLogInfo << "[" << (*ii)->tensorSurfaceDesc()->id() << "]";
            }
            if ((*ii)->originalTensor())
            {
                gLogInfo << "[" << "tt-" << (*ii)->originalTensor()->getTensorType() << "],";
            }
            gLogInfo << " ";
        }
        for (ESI ii = outputEdges.begin(); ii != outputEdges.end(); ++ii)
        {
            gLogInfo << "\t(out)" << (*ii)->id();
            if ((*ii)->tensorSurfaceDesc())
            {
                Dims4 dims = (*ii)->tensorSurfaceDesc()->dimensions();
                gLogInfo << "[" << dims.n << "x" << dims.c << "x" << dims.h << "x" << dims.w << "]";
                gLogInfo << "[" << (*ii)->tensorSurfaceDesc()->id() << "]";
            }
            if ((*ii)->originalTensor())
            {
                gLogInfo << "[" << "tt-" << (*ii)->originalTensor()->getTensorType() << "],";
            }
            gLogInfo << " ";
        }
        gLogInfo << std::endl;
        if ((*ni)->engineType() == engine_ast::MULTI_OPS)
        {
            printGraph(engine_ast::NodeFactory::nodeCast<engine_ast::MultiOpsNode*>(*ni)->nestedGraph(), true);
        }
    }
}

static void printDependencyGraph(engine_ast::Graph* g)
{
    using namespace engine_ast;
    for (vector< Graph::Graphlet * >::iterator gli = g->graphlets().begin(); gli != g->graphlets().end(); ++gli)
    {
        for (Graph::NodeSequenceIterator ni = (*gli)->nodeList().begin(); ni != (*gli)->nodeList().end(); ++ni)
        {
            for (NvU32 nn = 0; nn < g->profile()->multiBatchSize(); ++nn)
            {
                gLogInfo << "annid=" << (*ni)->dependencyParams(nn).annotationId()
                         << " node=" << (*ni)->name() << ".B" << nn
                         << " deps=" << (*ni)->dependencyParams(nn).getDependencyCount() << endl;
                gLogInfo << "\tproducer: [";
                for (size_t ii = 0; ii < EngineType::num_elements(); ++ii)
                {
                    if ( (*ni)->dependencyParams(nn).producer(ii).nodeAnnId() != -1 )
                    {
                        gLogInfo << (*ni)->dependencyParams(nn).producer(ii).node()->name()
                                 << "(annId:" << (*ni)->dependencyParams(nn).producer(ii).nodeAnnId() << ")"
                                 << ":" << (*ni)->dependencyParams(nn).producer(ii).opEvent().c_str() << ", ";
                    }
                    else
                    {
                        gLogInfo << ", ";
                    }
                }
                gLogInfo << "]" << endl << "\tconsumer: [";
                for (size_t ii = 0; ii < EngineType::num_elements(); ++ii)
                {
                    if ( (*ni)->dependencyParams(nn).consumer(ii).nodeAnnId() != -1 )
                    {
                        gLogInfo << (*ni)->dependencyParams(nn).consumer(ii).node()->name()
                                 << "(annId:" << (*ni)->dependencyParams(nn).consumer(ii).nodeAnnId() << ")"
                                 << ":" << (*ni)->dependencyParams(nn).consumer(ii).opEvent().c_str() << ", ";
                    }
                    else
                    {
                        gLogInfo << ", ";
                    }
                }
                gLogInfo << "]" << endl;
            }
        }
    }
}

/*----------------------Surface Descriptor Registration------------------*/
NvDlaError engine_ast::Graph::registerAllSurfaces()
{
    NvDlaError e = NvDlaSuccess;
    EdgeSequence allEdges   = orderedEdges();
    NodeSequence allNodes   = orderedNodes();

    FOR_EACH(allNodes, NodeSequenceIterator, clearNodeTSDStateMapping);

    FOR_EACH(allEdges, EdgeSequenceIterator, registerSurface);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceClients);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceFormat);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceStrides);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceSize);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceOffsetInBuffer);

    PROPAGATE_ERROR_FAIL( verifyAllSurfaces() );

fail:
    return e;
}

/*----------------------Verify Surface Descriptor Params---------------------*/
NvDlaError engine_ast::Graph::verifyAllSurfaces()
{
    NvDlaError e = NvDlaSuccess;
    EdgeSequence allEdges = orderedEdges();
    NodeSequence allNodes = orderedNodes();

    FOR_EACH(allNodes, NodeSequenceIterator, clearNodeTSDStateMapping);

    /*
     * A shared tensor between 2 nodes can be interpreted differently based on
     * the corresponding engine's limitations/requirements.
     * Since DLA can read a small cube from a larger cube, verify in this
     * API that a tensor honors all requirements of all nodes it is catering to
     */
    for (EdgeSequenceIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->verifySurface());
    }

    FOR_EACH(allNodes, NodeSequenceIterator, clearNodeTSDStateMapping);

fail:
    return e;
}

/*----------------------Buffer Descriptor Registration------------------*/
NvDlaError engine_ast::Graph::registerAllBuffers()
{
    NvDlaError e = NvDlaSuccess;
    EdgeSequence allEdges   = orderedEdges();

    for (EdgeSequenceIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->registerBuffer());
    }

    PROPAGATE_ERROR_FAIL( verifyAllBuffers() );

fail:
    return e;
}

/*----------------------Verify Buffer  Descriptor Params---------------------*/
NvDlaError engine_ast::Graph::verifyAllBuffers()
{
    NvDlaError e = NvDlaSuccess;
    EdgeSequence allEdges = orderedEdges();

    for (EdgeSequenceIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->verifyBuffer());
    }

fail:
    return e;
}

NvDlaError engine_ast::Graph::refreshGraphState()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    EdgeSequence allEdges = orderedEdges();

    FOR_EACH(allNodes, NodeSequenceIterator, repopulateEdgePorts);
    FOR_EACH(allNodes, NodeSequenceIterator, clearNodeTSDStateMapping);

    FOR_EACH(allEdges, EdgeSequenceIterator, registerSurface);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceClients);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceFormat);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceStrides);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceSize);
    FOR_EACH(allEdges, EdgeSequenceIterator, determineSurfaceOffsetInBuffer);
    FOR_EACH(allEdges, EdgeSequenceIterator, registerBuffer);
    FOR_EACH(allEdges, EdgeSequenceIterator, reserveBuffer);

    PROPAGATE_ERROR_FAIL( verifyAllSurfaces() );

fail:
    return e;
}

/*----------------------Pre-process Aux Data------------------------------*/
NvDlaError engine_ast::Graph::preProcessAuxData()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();

    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->preProcessAuxData());
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*-----------------------Merge Unit Scale Operations-------------------------*/
/*
 * Unit scale operations are introduced for int8 scaling when there is no
 * adjacent SDP operation to perform compression to int8 space. But these
 * unit scale operations can be merged if another adjacent SDP is available.
 * Remove such unit scale operations before start merging other SDP ops.
 */
NvDlaError engine_ast::Graph::mergeUnitScaleOperations()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    NodeSequenceIterator ni = allNodes.begin();
    NodeSequenceIterator startNodeIter = ni;
    bool maxOptimized = false;
    Node* currNode   = NULL;
    Node* prevNode   = NULL;
    Node* removeNode = NULL;

    do
    {
        for (ni = startNodeIter; ni != allNodes.end(); ++ni)
        {
            if ((*ni)->engineType().v() != EngineTypeEnum::SDP)
            {
                continue;
            }

            NodeSequence sinkNodes = downstreamDataNodes((*ni));
            /* Attempt to combine only those sdp nodes which are 1:1 connected and
             * not part of a multi-way junction
             */
            if ( sinkNodes.size() != 1 || upstreamDataNodes(sinkNodes[0]).size() != 1 ||
                 sinkNodes[0]->engineType().v() != EngineTypeEnum::SDP )
            {
                continue;
            }

            currNode   = *ni;
            prevNode   = ni != allNodes.begin() ? *(ni - 1) : *ni;
            SDPNode* currSDP = NodeFactory::nodeCast<SDPNode*>(currNode);
            SDPNode* nextSDP = NodeFactory::nodeCast<SDPNode*>(sinkNodes[0]);
            if ( nextSDP->engineOpType().v() != EngineOpTypeEnum::SDP_SCALE )
            {
                continue;
            }

            if ( !nextSDP->isUnitScale() )
            {
                continue;
            }

            if ( debugMathOptz() )
            {
                gLogInfo << std::endl;
                gLogInfo << "Try Merging: " << currNode->name() << " & " << nextSDP->name() << std::endl;
            }

            removeNode = currSDP->mergeUnitScaleOp(nextSDP);

            if ( debugMathOptz() )
            {
                if (removeNode)
                    gLogInfo << "Merging: Sucess" << std::endl;
                else
                    gLogInfo << "Merging: Not Feasible" << std::endl;
            }

            if ( removeNode )
            {
                IODirection iod = IODirectionEnum::INPUT;

                PROPAGATE_ERROR_FAIL( removeNodeFromAST(removeNode, iod) );
                break;
            }
        }

        // if the last pass through all nodes didn't change the AST anymore,
        // that means all optimizations are applied; no more scope left
        if ( ni == allNodes.end() )
        {
            maxOptimized = true;
        }
        else
        {
            // rinse and repeat on newly ordered nodes;
            // starting from the node prior to the recently operated one
            allNodes = orderedNodes();
            startNodeIter = std::find(allNodes.begin(), allNodes.end(), prevNode);
            if (startNodeIter == allNodes.end())
            {
                startNodeIter = allNodes.begin();   // just in case
            }
            PROPAGATE_ERROR_FAIL(refreshGraphState());
        }
    } while(!maxOptimized);


     // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*-----------------------Merge Activation Operations-------------------------*/
NvDlaError engine_ast::Graph::mergeActivationOperations()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    NodeSequenceIterator ni = allNodes.begin();
    NodeSequenceIterator startNodeIter = ni;
    bool maxOptimized = false;
    Node* currNode   = NULL;
    Node* prevNode   = NULL;
    Node* removeNode = NULL;

    if ( !profile()->canSDPMergeMathOps() && !profile()->canSDPBustNOPs() )
    {
        // nothing to do
        goto fail;
    }

    PROPAGATE_ERROR_FAIL( mergeUnitScaleOperations() );

    do
    {
        for (ni = startNodeIter; ni != allNodes.end(); ++ni)
        {
            /* Currently only mathematical ops executable on CONV/SDP can be combined;
             * skip the rest
             */
            if ((*ni)->engineType().v() != EngineTypeEnum::SDP &&
                (*ni)->engineType().v() != EngineTypeEnum::CONVOLUTION)
            {
                continue;
            }

            NodeSequence sinkNodes = downstreamDataNodes((*ni));
            /* Attempt to combine only those sdp nodes which are 1:1 connected and
             * not part of a multi-way junction
             */
            if ( sinkNodes.size() != 1 || upstreamDataNodes(sinkNodes[0]).size() != 1 ||
                 sinkNodes[0]->engineType().v() != EngineTypeEnum::SDP )
            {
                continue;
            }

            currNode   = *ni;
            prevNode   = ni != allNodes.begin() ? *(ni - 1) : *ni;
            SDPNode* nextSDP = NodeFactory::nodeCast<SDPNode*>(sinkNodes[0]);

            if ( debugMathOptz() )
            {
                gLogInfo << std::endl;
                gLogInfo << "Try Merging: " << currNode->name() << " & " << nextSDP->name() << std::endl;
            }

            removeNode = currNode->mergeWithSDPOp(nextSDP);

            if ( debugMathOptz() )
            {
                if (removeNode)
                    gLogInfo << "Merging: Sucess" << std::endl;
                else
                    gLogInfo << "Merging: Not Feasible" << std::endl;
            }

            if (removeNode)
            {
                IODirection iod;
                NodeSequence gNodes = orderedNodes();

                /* Before removing, delegate operation mode to the next op iff it exists;
                 * don't bother if it is already removed from graph
                 */
                if ((removeNode == currNode) && (std::find(gNodes.begin(), gNodes.end(), nextSDP) != gNodes.end()))
                {
                    SDPNode* removeSDP = NULL;
                    ASSERT(removeNode->engineType().v() == EngineTypeEnum::SDP);
                    removeSDP = NodeFactory::nodeCast<SDPNode*>(removeNode);
                    nextSDP->params().setConvMode(removeSDP->params().convMode());
                    nextSDP->params().setWinogradParams(removeSDP->params().winogradParams());
                    nextSDP->params().setNumGroups(removeSDP->params().numGroups());
                }
                else if (removeNode == nextSDP)
                {
                    NodeSequence removeSDPSinkNodes = downstreamDataNodes(nextSDP);
                    NodeWithSameEngineType match_next_sdp(EngineTypeEnum::SDP);
                    NodeSequenceIterator dni = std::find_if(removeSDPSinkNodes.begin(),
                                                            removeSDPSinkNodes.end(),
                                                            match_next_sdp);
                    if (dni != removeSDPSinkNodes.end())
                    {
                        SDPNode* removeSDP = NodeFactory::nodeCast<SDPNode*>(removeNode);
                        SDPNode* removeSDPSinkSDP = NodeFactory::nodeCast<SDPNode*>(*dni);
                        removeSDPSinkSDP->params().setConvMode(removeSDP->params().convMode());
                        removeSDPSinkSDP->params().setWinogradParams(removeSDP->params().winogradParams());
                        removeSDPSinkSDP->params().setNumGroups(removeSDP->params().numGroups());
                    }
                }

                /*
                 * If collapsing an SDP node into Conv, detach the SDP from its output side
                 * and retain the stream tensor from Conv
                 * If collapsing an SDP node into another SDP, detach the collapsing SDP from
                 * the side that connects to the prevailing SDP
                 */
                if (currNode->engineType().v() == EngineTypeEnum::CONVOLUTION)
                {
                    iod = IODirectionEnum::OUTPUT;
                }
                else
                {
                    iod = (removeNode == currNode) ? IODirectionEnum::OUTPUT : IODirectionEnum::INPUT;
                }

                PROPAGATE_ERROR_FAIL( removeNodeFromAST(removeNode, iod) );
                break;
            }
        }

        // if the last pass through all nodes didn't change the AST anymore,
        // that means all optimizations are applied; no more scope left
        if ( ni == allNodes.end() )
        {
            maxOptimized = true;
        }
        else
        {
            // rinse and repeat on newly ordered nodes;
            // starting from the node prior to the recently operated one
            allNodes = orderedNodes();
            startNodeIter = std::find(allNodes.begin(), allNodes.end(), prevNode);
            if (startNodeIter == allNodes.end())
            {
                startNodeIter = allNodes.begin();   // just in case
            }
            PROPAGATE_ERROR_FAIL(refreshGraphState());
        }
    } while(!maxOptimized);

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*------------------Update scaling factors for EW op--------------------*/
NvDlaError engine_ast::Graph::updateScalingFactors()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();

    if (profile()->computePrecision().v() != surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8)
    {
        // nop
        goto fail;
    }

    if ( profile()->tensorScalingMode().v() != nvdla::TensorScalingMode::PER_TENSOR )
    {
        // don't support any other scaling mode than PER_TENSOR
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support tensor scaling mode: %s\n",
                                profile()->tensorScalingMode().c_str());
    }

    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        engine_ast::Node* currNode = *ni;

        EngineOpType eng_op_type = currNode->engineOpType();

        EdgeSequence inputEdges;
        EdgeSequence siblingEdges0;
        EdgeSequence siblingEdges1;

        engine_ast::Edge* inputEdge0 = NULL;
        engine_ast::Edge* inputEdge1 = NULL;
        engine_ast::Edge* updateEdge = NULL;

        std::vector<NvF32> inputTensorScales0;
        std::vector<NvF32> inputTensorScales1;
        std::vector<NvF32> updateTensorScales;

        if (eng_op_type.v() != EngineOpTypeEnum::SDP_ELEMENTWISE)
        {
            continue;
        }

        inputEdges = upstreamDataEdges(currNode);

        /* element wise op should have exactly two input edges. */
        if (inputEdges.size() != 2)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Elt wise op has input edges (%d) != 2 ", inputEdges.size());
        }

        inputEdge0 = inputEdges.at(0);
        inputTensorScales0 = inputEdge0->originalTensor()->getChannelScales();

        inputEdge1 = inputEdges.at(1);
        inputTensorScales1 = inputEdge1->originalTensor()->getChannelScales();

        ASSERT (inputTensorScales0.size() == inputTensorScales1.size())
        if (inputTensorScales0.at(0) == inputTensorScales1.at(0))
        {
            // Incoming scale values are same, no need for update.
            continue;
        }

        siblingEdges0 = siblingDataEdges(inputEdge0);
        siblingEdges1 = siblingDataEdges(inputEdge1);

        /**
         * Elementwise fusion has 3 possible cases
         * 1. Both input nodes to elementwise layer has single output edges
         * 2. One input node has multiple output edges while another has single output edge
         * 3. Both input nodes to elementwise layer has multiple output edges
         *
         * #1, any of the input nodes can be rescaled using scaling factor of another
         *     node. it requires selecting correct scaling factor to use.
         * #2, general policy is to rescale node with single output edge as rescaling
         *     node with multiple edges will cause incorrect input to another node
         * #3, in such case, we need to select scaling factor from two input nodes
         *     and use new SDP scaling node to rescaling
         *
         * Current implementation support #1 and #2, but does not support scaling factor
         * selection for #1.
         */

        if (siblingEdges0.size() == 0 && siblingEdges1.size() == 0)
        {
            /* case 1 */
            updateEdge = inputEdge0;
            updateTensorScales = inputTensorScales1;
        }
        else if (siblingEdges0.size() == 0)
        {
            /** case 2:
             *   when no src node corresponding to first input edge exists (or)
             *      when src node corresponding to first input edge exists, having only
             *      one output edge = first input edge
             *   In other words, no siblings to first input edge
             *   Handled in similar way as that of case 1
             **/
            updateEdge = inputEdge0;
            updateTensorScales = inputTensorScales1;
        }
        else if (siblingEdges1.size() == 0)
        {
            /** case 2:
             *   when no src node corresponding to second input edge exists (or)
             *      when src node corresponding to second input edge exists, having only
             *      one output edge = second input edge
             *   In other words, no siblings to second input edge
             *   Handled in similar way as that of case 1
             **/
            updateEdge = inputEdge1;
            updateTensorScales = inputTensorScales0;
        }
        else
        {
            /* TODO: to handle case 3: when both input nodes have multiple outputs */
            ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported,
                                "Both input nodes having multiple output edges is not supported yet!");
        }
        updateEdge->originalTensor()->setChannelScales(updateTensorScales);

    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*----------------------Pre-process Aux Data------------------------------*/
NvDlaError engine_ast::Graph::quantizeAuxData()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();

    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->quantizeAuxData());
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*------------------Low Precision Conversions--------------------------*/
NvDlaError engine_ast::Graph::handleLowPrecisionConversions()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->handleLowPrecisionConversions());
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*--------------------------------Fuse Nodes---------------------------*/
NvDlaError engine_ast::Graph::fuseOnTheFlyNodes()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->fuseOnTheFlyNodes());
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*----------------------- Fuse SDP operations into SDP subengines -------------------------*/
NvDlaError engine_ast::Graph::fuseSDPSubEngineOps()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    NodeSequenceIterator ni = allNodes.begin();
    NodeSequenceIterator startNodeIter = ni;
    bool maxOptimized = false;
    Node* currNode   = NULL;
    Node* prevNode   = NULL;
    Node* removeNode = NULL;

    if (!profile()->canSDPFuseSubEngineOps())
    {
        // nothing to do
        goto fail;
    }

    if (debugFuseSubEngineOps())
    {
        printGraph(this, true, "pree fuseSDPSubEngineOps");
    }

    do
    {
        for (ni = startNodeIter; ni != allNodes.end(); ++ni)
        {
            /* Only SDP ops can be fused. */
            if ((*ni)->engineType().v() != EngineTypeEnum::SDP)
            {
                continue;
            }

            NodeSequence sinkNodes = downstreamDataNodes((*ni));
            /* Attempt to combine only those sdp nodes which are 1:1 connected and
             * not part of a multi-way junction
             */
            if ( sinkNodes.size() != 1 || upstreamDataNodes(sinkNodes[0]).size() > 2 ||
                 sinkNodes[0]->engineType().v() != EngineTypeEnum::SDP )
            {
                continue;
            }

            currNode   = *ni;
            prevNode   = ni != allNodes.begin() ? *(ni - 1) : *ni;
            SDPNode* currSDP = NodeFactory::nodeCast<SDPNode*>(currNode);
            SDPNode* nextSDP = NodeFactory::nodeCast<SDPNode*>(sinkNodes[0]);

            if ( debugFuseSubEngineOps() )
            {
                gLogInfo << std::endl;
                gLogInfo << "Try Fusing: " << currNode->name() << " & " << nextSDP->name() << std::endl;
            }

            removeNode = currSDP->fuseSDPSubEngineOp(nextSDP);

            if ( debugFuseSubEngineOps() )
            {
                if (removeNode)
                    gLogInfo << "Fusing: Sucess" << std::endl;
                else
                    gLogInfo << "Fusing: Not Feasible" << std::endl;
            }

            if (removeNode)
            {
                IODirection iod;
                NodeSequence gNodes = orderedNodes();

                /* Before removing, delegate operation mode to the next op iff it exists;
                 * don't bother if it is already removed from graph
                 */
                if ((removeNode == currNode) && (std::find(gNodes.begin(), gNodes.end(), nextSDP) != gNodes.end()))
                {
                    SDPNode* removeSDP = NULL;
                    ASSERT(removeNode->engineType().v() == EngineTypeEnum::SDP);
                    removeSDP = NodeFactory::nodeCast<SDPNode*>(removeNode);
                    nextSDP->params().setConvMode(removeSDP->params().convMode());
                    nextSDP->params().setWinogradParams(removeSDP->params().winogradParams());
                    nextSDP->params().setNumGroups(removeSDP->params().numGroups());
                }
                iod = (removeNode == currNode) ? IODirectionEnum::OUTPUT : IODirectionEnum::INPUT;
                PROPAGATE_ERROR_FAIL( removeNodeFromAST(removeNode, iod) );
                break;
            }
        }

        // if the last pass through all nodes didn't change the AST anymore,
        // that means all optimizations are applied; no more scope left
        if ( ni == allNodes.end() )
        {
            maxOptimized = true;
        }
        else
        {
            // rinse and repeat on newly ordered nodes;
            // starting from the node prior to the recently operated one
            allNodes = orderedNodes();
            startNodeIter = std::find(allNodes.begin(), allNodes.end(), prevNode);
            if (startNodeIter == allNodes.end())
            {
                startNodeIter = allNodes.begin();   // just in case
            }
            PROPAGATE_ERROR_FAIL(refreshGraphState());
        }
    } while(!maxOptimized);

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

    if (debugFuseSubEngineOps())
    {
        printGraph(this, true, "post fuseSDPSubEngineOps");
    }

fail:
    return e;
}

/*----------------------Weight Translation ----------------------------*/
NvDlaError engine_ast::Graph::translateAuxData(/*some test point criterion */)
{
    NvDlaError e = NvDlaSuccess;

    NodeSequence allNodes = orderedNodes();

    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        engine_ast::Node* curr_node = *ni;
        EngineOpType eng_op_type = curr_node->engineOpType();
        switch(eng_op_type.v())
        {
            case EngineOpTypeEnum::CONVOLUTION_CONV:
            case EngineOpTypeEnum::CONVOLUTION_FC:
            case EngineOpTypeEnum::CONVOLUTION_DECONV:
            case EngineOpTypeEnum::SDP_BIAS:
            case EngineOpTypeEnum::SDP_BATCH_NORM:
            case EngineOpTypeEnum::SDP_SCALE:
            case EngineOpTypeEnum::SDP_SUPER:
            {
                PROPAGATE_ERROR_FAIL(curr_node->translateAuxData());
                break;
            }
            default: break;
        }
    }

    // check dirty and re-determine graph order
    checkDirty();

fail:
    return e;
}

/*--------------------------Buffer Reservation-------------------------*/
NvDlaError engine_ast::Graph::reserveAllBuffers()
{
    NvDlaError e = NvDlaSuccess;
    EdgeSequence allEdges = orderedEdges();

    // update the size requirements of each registered TBD
    for (EdgeSequence::const_iterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->reserveBuffer());
    }

fail:
    return e;
}

/*-------------------------Group Atomic Operations---------------------*/
NvDlaError engine_ast::Graph::groupAtomicOperations()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();

    for ( NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); )
    {
        Node* currNode = (*ni);
        NodeSequence groupableOpNodes;
        do {
            groupableOpNodes.push_back(currNode);
            currNode = currNode->dependencyParams(/*batchId*/0).fusedNode(IODirectionEnum::OUTPUT);
        } while ( currNode );

        if (groupableOpNodes.size() > 1)
        {
            MultiOpsNode* multiOpsSuperNode = engine_ast::NodeFactory::newMultiOpsNode(groupableOpNodes, this);
            NVDLA_UNUSED(multiOpsSuperNode);
        }
        ni += groupableOpNodes.size();
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*-------------------------------Split Nodes---------------------------*/
NvDlaError engine_ast::Graph::splitNodes()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();

    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni) {
        PROPAGATE_ERROR_FAIL((*ni)->splitNodes());
        PROPAGATE_ERROR_FAIL(refreshGraphState());
    }

    // check dirty and re-determine graph order
    checkDirty();

fail:
    return e;
}

/*----------------------------Handle Multi-Batch-----------------------*/
NvDlaError engine_ast::Graph::handleMultiBatch()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    EdgeSequence inEdges = inputEdges();
    EdgeSequence outEdges = outputEdges();
    EdgeSequence allEdges = orderedEdges();
    for (NodeSequenceIterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->handleMultiBatch());
    }

    // first handle multi-batch buffer offsets for bindable tensors
    for (EdgeSequenceIterator ei = inEdges.begin(); ei != inEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->handleMultiBatch());
    }
    for (EdgeSequenceIterator ei = outEdges.begin(); ei != outEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->handleMultiBatch());
    }

    // then handle multi-batch buffer offsets for non-bindable tensors
    for (EdgeSequenceIterator ei = allEdges.begin(); ei != allEdges.end(); ++ei)
    {
        PROPAGATE_ERROR_FAIL((*ei)->handleMultiBatch());
    }

    // check dirty and re-determine graph order
    checkDirty();

fail:
    return e;
}

/*---------------------------------Flatten Graph-----------------------*/
/*
 * Since dependencies calculation needs to see all possible operation nodes in the flat land
 * and cannot efficiently deal with hierarchical nodes, flatten all the superNodes that there
 * may be by plugging in their nested graph contents into the base graph.
 */
NvDlaError engine_ast::Graph::flattenGraph()
{
    NvDlaError e = NvDlaSuccess;
    NodeSequence allNodes = orderedNodes();
    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        if ( (*ni)->engineType() == EngineTypeEnum::MULTI_OPS)
        {
            PROPAGATE_ERROR_FAIL(engine_ast::NodeFactory::nodeCast<MultiOpsNode*>(*ni)->plugNestedGraph());
        }
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*---------------------------------Topologically Sort-----------------------*/
NvDlaError engine_ast::Graph::topologicalSort(NodeSequence& topological_order)
{
   /*
    * Compiler needs to come up with a fixed schedule order for firmware to pick up.
    * Topological order is needed when there are potential diamond/fork/inception situations.
    */

   /* This function comes up with a DF topological order.
    * It pushes nodes that are reachable from the current node onto a stack,
    * and goes down on every thread until it reaches a node whose parents are not all visited
    *
    * E.g.
    *           (op m)                                              (op m)
    *             |___________                                       |___________
    *            / \          \                                                  \
    *           /   \          \                                                  \
    *          /     \          \                                                  \
    *         /       \          \                                                  \
    *     (op a)     (op x)     (op p)                        (op a)    (op x)     (op p)
    *        |         |          |       sorted                |   \\    |    \\     |
    *        |         |          |      ---------------->      |    \\   |     \\    |
    *     (op b)     (op y)     (op q)                        (op b)  ==(op y)   ==(op q)
    *        \         /                                        \
    *         \       /                                          \
    *          \     /                                            \
    *           \   /                                              \
    *          (op n)                                             (op n)
    *
    *             |
    *             |  or
    *            \ /
    *
    *           (op m)
    *             |
    *            /
    *           /
    *          /
    *         /
    *     (op a)     (op x)      (op p)
    *        |    //   |       //   |
    *        |   //    |      //    |
    *     (op b)==   (op y)  //  (op q)
    *                  /    //
    *                 /    //
    *                /    //
    *               /    //
    *          (op n)=====
    *
    * Some edges are hidden in the above graph to show the final order, but are not removed
    */
    NvDlaError e = NvDlaSuccess;
    NodeUnorderedSet visitedNodes;
    std::stack<Node *> S;

    // Scan network input edges
    const EdgeSequence& inEdges = inputEdges();
    if (inEdges.size() > 0)
    {
        EdgeSequence::const_iterator ei = inEdges.end();
        do {
            --ei;
            NodeSequence consumerNodes = downstreamNodes(*ei);

            if (consumerNodes.size() > 0)
            {
                NodeSequence::const_iterator ni = consumerNodes.end();
                do {
                    --ni;
                    // push the node only if its not dependent on any other nodes
                    const NodeSequence& ancestors = upstreamNodes(*ni);
                    if (ancestors.size() == 0)
                    {
                        S.push(*ni);
                    }
                }
                while (ni != consumerNodes.begin());
            }
        }
        while (ei != inEdges.begin());
    }

    while (!S.empty()) {
        Node * currNode = S.top();
        S.pop();
        if (visitedNodes.find(currNode) == visitedNodes.end())
        {
            topological_order.push_back(currNode);
            visitedNodes.insert(currNode);

           /* Push downstream nodes with compute edges first because data edges take precedence
            * In a convolution split diamond,
            *           (split)
            *             |
            *            / \
            *           /   \
            *          /     \
            *         /       \
            *     (op C1)====>(op C2)
            *        |         |
            *        |         |
            *     (op S1)====>(op S2)
            *        \         /
            *         \       /
            *          \     /
            *           \   /
            *          (concat)
            * We need to make sure fused nodes are stick together,
            * S1 needs to be popped off stack before C2
            */

            // push hazard edge children first, then compute edge children, then data edge children
            // so stack popping order will be data > compute > hazard
            NodeSequence consumerHazardNodes = downstreamHazardNodes(currNode);
            if (consumerHazardNodes.size() > 0)
            {
                NodeSequenceIterator ni = consumerHazardNodes.end();
                do {
                    --ni;
                    // for each downstream node, check if its producer nodes are all visited
                    // it can't be pushed to the stack until its producers are visited
                    NodeSequence producerNodes = upstreamNodes(*ni);

                    bool ancestorReady = true;
                    for (NodeSequenceIterator nj = producerNodes.begin(); nj != producerNodes.end(); ++nj)
                    {
                        if (visitedNodes.find(*nj) == visitedNodes.end())
                        {
                            ancestorReady = false;
                            break;
                        }
                    }

                    if (ancestorReady)
                    {
                        S.push(*ni);
                    }
                }
                while (ni != consumerHazardNodes.begin());
            }
            // same logic for downstream nodes connected via compute edge
            NodeSequence consumerComputeNodes = downstreamComputeNodes(currNode);
            if (consumerComputeNodes.size() > 0)
            {
                NodeSequenceIterator ni = consumerComputeNodes.end();
                do {
                    --ni;
                    NodeSequence producerNodes = upstreamNodes(*ni);

                    bool ancestorReady = true;
                    for (NodeSequenceIterator nj = producerNodes.begin(); nj != producerNodes.end(); ++nj)
                    {
                        if (visitedNodes.find(*nj) == visitedNodes.end())
                        {
                            ancestorReady = false;
                            break;
                        }
                    }

                    if (ancestorReady)
                    {
                        S.push(*ni);
                    }
                }
                while (ni != consumerComputeNodes.begin());
            }
            // same logic for downstream nodes connected via data edge
            NodeSequence consumerDataNodes = downstreamDataNodes(currNode);
            if (consumerDataNodes.size() > 0)
            {
                NodeSequenceIterator ni = consumerDataNodes.end();
                do {
                    --ni;
                    NodeSequence producerNodes = upstreamNodes(*ni);

                    bool ancestorReady = true;
                    for (NodeSequenceIterator nj = producerNodes.begin(); nj != producerNodes.end(); ++nj)
                    {
                        if (visitedNodes.find(*nj) == visitedNodes.end())
                        {
                            ancestorReady = false;
                            break;
                        }
                    }

                    if (ancestorReady)
                    {
                        S.push(*ni);
                    }
                }
                while (ni != consumerDataNodes.begin());
            }
        }
        else
        {
            // currNode is already visited, there is a cycle
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Can't resolve a non-acyclic graph ");
        }
    }
fail:
    return e;
}

/*---------------------------Resolve Data Dependencies-----------------*/
NvDlaError engine_ast::Graph::resolveDataDependencies(const NodeSequence& allNodes)
{
    NvDlaError e = NvDlaSuccess;

    for (NodeSequence::const_iterator ni = allNodes.begin(), nj = ni+1; nj != allNodes.end(); ni = nj++)
    {
        PROPAGATE_ERROR_FAIL( (*ni)->resolveDataDependencies(*nj) );
    }

fail:
    return e;
}

/*---------------------------Resolve Compute Dependencies--------------*/
/* The current version of the firmware needs chaining of operations of the same type so
 * that it can schedule them one after the other. Hence connect 2 ops of the same type in the
 * causal/ordered list of nodes as a producer-consumer pair
 */
NvDlaError engine_ast::Graph::resolveComputeDependencies(const NodeSequence& allNodes)
{
    NvDlaError e = NvDlaSuccess;
    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->resolveComputeDependencies(allNodes));
    }

fail:
    return e;
}

/*---------------------------Resolve Software Dependencies-------------*/
NvDlaError engine_ast::Graph::resolveSoftwareDependencies()
{
    NvDlaError e = NvDlaSuccess;
    for (NodeSequence::const_iterator ni = orderedNodes().begin(); ni != orderedNodes().end(); ++ni) {
        PROPAGATE_ERROR_FAIL((*ni)->resolveSoftwareDependencies());
    }

fail:
        return e;
}

/*--------------------Regroup Atomic Operations------------------------*/
NvDlaError engine_ast::Graph::regroupAtomicOperations()
{
    NvDlaError e = NvDlaSuccess;

    Graph::NodeSet allNodes = nodes();
    for ( Graph::NodeSetIterator ni = allNodes.begin(); ni != allNodes.end(); ++ni )
    {
        if ((*ni)->engineType() == MULTI_OPS)
        {
            PROPAGATE_ERROR_FAIL(NodeFactory::nodeCast<MultiOpsNode*>(*ni)->unplugNestedGraph());
        }
    }

    // check dirty and re-determine graph order
    checkDirty();
    PROPAGATE_ERROR_FAIL(refreshGraphState());

fail:
    return e;
}

/*----------------------Determine Task Boundaries----------------------*/
NvDlaError engine_ast::Graph::determineTaskBoundaries(const NodeSequence& allNodes)
{
    NvDlaError e = NvDlaSuccess;
    bool isEMU = false;
    NvS16 taskId;

    Graphlet *graphlet = 0;

    graphlets().clear();

    //
    // figure out how many tasks there are going to be and which nodes are in each (sub)set.
    //
    // detect EMU<->DLA changes and create a new task for each.
    //

    isEMU = (*allNodes.begin())->isEMUEngineType();
    graphlet = new Graphlet();
    graphlets().push_back(graphlet);
    taskId = 0;

    for ( NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni )
    {
        Node *node = *ni;

        if ( node->isEMUEngineType() != isEMU )
        {
            graphlet = new Graphlet();
            // start of next graphlet
            graphlets().push_back(graphlet);
            taskId++;
        }

        node->setTaskId(taskId);
        if (!node->isDLAEngineType() && !node->isEMUEngineType())
        {
            /*
             * Don't bother to add non-DLA and non-CPU ops to any of the graphlets/tasklets
             * However a multi-op destination engine(s) is still not determined; so
             * keep that around in the graphlet/tasklet
             */
            node->dependencyParams(/*batchId*/0).setAnnotationId(-1);
        }
        else
        {
            // Add cpu/dla/multi-ops to graphlet
            graphlet->nodeList().push_back(node);
        }

        isEMU = node->isEMUEngineType();

        // capture per graphlet, the head_ops for each engine
        if ( NULL == graphlet->opHeads()[node->engineType().v()] )
        {
            graphlet->opHeads()[node->engineType().v()] = node;
        }
    }

    return e;
}

/*----------------------------Annotate Nodes---------------------------*/
// the order of traversal after this phase must be baked in.
// don't forget that the graph apps can force a new ordering which is the
// start point for the next... so need a way to specify what the new order
// is.
//
// when a phase re-orders things it needs to be able to hand off the new
// ordering and that should result in a new scoreboard (update) based upon
// that ordering.
//
NvDlaError engine_ast::Graph::annotateNodes(NvS16& lastUsedAnnId)
{
    NvDlaError e = NvDlaSuccess;

    bool isEMUGraphlet = false;

    vector< Graphlet* >::iterator gli;
    for ( gli = graphlets().begin(); gli != graphlets().end(); ++gli )
    {
        isEMUGraphlet = (*gli)->nodeList()[0]->isEMUEngineType();
        NodeSequence& graphletNodes = (*gli)->nodeList();
        for ( NodeSequenceIterator ni = graphletNodes.begin(); ni != graphletNodes.end(); ++ni )
        {
            Node *node = *ni;
            if (isEMUGraphlet != node->isEMUEngineType())
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "All nodes in a graphlet should be either all CPU or all DLA");
            }

            if (!node->isDLAEngineType() && !node->isEMUEngineType())
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Non-DLA and non-CPU nodes shouldn't be part of any graphlet");
            }
            else
            {
                PROPAGATE_ERROR_FAIL(node->selfAnnotate(lastUsedAnnId));
            }
        }
        // reset annotation id for next graphlet/task
        lastUsedAnnId = -1;
    }

    /* Determine annotation node-id's of the producers/consumers for 1st batch(batch_id: 0)
     *  since from now on operations of all batches should possess unique annotation id's
     */
    for ( gli = graphlets().begin(); gli != graphlets().end(); ++gli )
    {
        NodeSequence& graphletNodes = (*gli)->nodeList();
        for ( NodeSequenceIterator ni = graphletNodes.begin(); ni != graphletNodes.end(); ++ni )
        {
            Node* node = *ni;
            for (size_t et = 0; et < EngineType::num_elements(); ++et)
            {
                NvU32 firstBatchId = 0;
                Node* consumer  = node->dependencyParams(firstBatchId).consumer(et).node();
                NvS16 consAnnId = node->dependencyParams(firstBatchId).consumer(et).nodeAnnId();
                /* populate iff not done before */
                if (consAnnId == -1)
                {
                    node->dependencyParams(firstBatchId).consumer(et).setNodeAnnId(consumer ?
                            consumer->dependencyParams(firstBatchId).annotationId() : -1);
                }

                Node* producer  = node->dependencyParams(firstBatchId).producer(et).node();
                NvS16 prodAnnId = node->dependencyParams(firstBatchId).producer(et).nodeAnnId();
                /* populate iff not done before */
                if (prodAnnId == -1)
                {
                    node->dependencyParams(firstBatchId).producer(et).setNodeAnnId(producer ?
                            producer->dependencyParams(firstBatchId).annotationId() : -1);
                }
            }
        }
    }

fail:
    return e;
}

/*-------------------------Resolve Multi-Batch Dependencies------------*/
NvDlaError engine_ast::Graph::resolveMultiBatchDependencies()
{
    NvDlaError e = NvDlaSuccess;

    NodeSequence allNodes = orderedNodes();
    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->resolveMultiBatchDependencies());
    }

    if (debugDepGraph())
    {
        printDependencyGraph(this);
    }

fail:
        return e;
}

/*----------------------------Verify Dependency Graph------------------*/
NvDlaError engine_ast::Graph::verifyDependencyGraph()
{
    NvDlaError e = NvDlaSuccess;

    NodeSequence allNodes = orderedNodes();
    for (NodeSequence::const_iterator ni = allNodes.begin(); ni != allNodes.end(); ++ni)
    {
        PROPAGATE_ERROR_FAIL((*ni)->verifyDependencyParams());
    }

fail:
    return e;
}

// entire graph
NvDlaError engine_ast::Graph::resolveMemory(const engine_ast::NodeSequence &topological_order)
{
    NvDlaError e = NvDlaSuccess;

    if ( debugMemoryLayout() )
    {
        gLogInfo << "beginning resolveMemory phase" << endl;
    }

    if ( !m_memoryResolver )
    {
        m_memoryResolver = new memory::MemoryResolver();
    }
    PROPAGATE_ERROR_FAIL( m_memoryResolver->visitNodes(topological_order) );

fail:
    return e;
}


/*----------------------------Enable BDMA Copy-Out Debug Buffers-------------*/
NvDlaError engine_ast::AddCopyOutDebugBDMA::visitBegin(engine_ast::Graph *g)
{
    m_graph = g;
    m_debugBindId = 0;
    return NvDlaSuccess;
}

NvDlaError engine_ast::AddCopyOutDebugBDMA::visitNode(engine_ast::Node *treatNode)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Edge *dbgBDMAOutEdge  = 0;
    engine_ast::Edge *dbgBDMACompEdge = 0;
    engine_ast::Edge *nodeOutEdge = 0;
    engine_ast::Node *dbgBDMANode = 0;
    EdgeSequence dataOutputs;
    Dims4 dbgTensorDims;
    NodeSequence consumerNodes;
    NvU32 numBatches = treatNode->graph()->profile()->multiBatchSize();

    if ( treatNode->isEMUEngineType() )
    {
        return NvDlaSuccess;
    }

    /**
     *
     * fork the op output into a BDMA op that writes out to memory
     *                    (op-n)
     *                    / | \
     *                   /  |  \
     *                  /   |   \
     *                 /    |    \
     *                /(op-n+1)==(d-bdma)----> DRAM
     *               /              ||
     *          (op-n+2)============''
     */

    dataOutputs = m_graph->downstreamDataEdges(treatNode);
    if ( dataOutputs.size() != 1 )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "This shouldn't happen - I am confused!!");
    }
    nodeOutEdge = dataOutputs[0];

    // stream tensors can't be captured
    if ( nodeOutEdge->originalTensor()->getTensorType() == TensorType::kSTREAM )
    {
        return NvDlaSuccess;
    }

    if ( debugCopyOutDebug() )
    {
        gLogInfo << "enable debug buffers:\n\tadd dbg dma for orig node=" << treatNode->id() <<
            " -> node_out_edge=" << nodeOutEdge->id() << endl;
    }

    dbgTensorDims = nodeOutEdge->originalTensor()->getDimensions();

    dbgBDMANode = engine_ast::NodeFactory::newSingleBDMANode(m_graph);
    PROPAGATE_ERROR_FAIL(dbgBDMANode->populateEdgePorts());

    dbgBDMAOutEdge = m_graph->addDataEdge((canonical_ast::Edge*)0, dbgBDMANode, 0, nodeOutEdge->originalTensor()); // dangling

    // reserve bind id for #batches
    for (NvU32 nn = 0; nn < numBatches; ++nn)
    {
        dbgBDMAOutEdge->setBindId(m_debugBindId++, IOD_Debug);
    }
    // m_graph->addDebugTensor(m_graph->newDebugTensorName(), dbgTensorDims) );

    dbgBDMACompEdge = m_graph->addComputeEdge(dbgBDMANode, NULL); // dangling here

    if ( debugCopyOutDebug() )
    {
        gLogInfo << "\tbdma node=" << dbgBDMANode->id() << " -> bdma comp edge=" << dbgBDMACompEdge->id() << endl;
        gLogInfo << "\tbdma node=" << dbgBDMANode->id() << "-> bdma out edge=" << dbgBDMAOutEdge->id() <<  endl;
    }

    // compute bound each consumer of current node to the new bdma node
    // so that their operation is stalled until the bdma op is finished
    consumerNodes = m_graph->downstreamNodes(nodeOutEdge);
    for (NodeSequence::const_iterator cni = consumerNodes.begin(); cni != consumerNodes.end(); ++cni)
    {
        m_graph->appendNodeToEdge(dbgBDMACompEdge, ast::EdgeSideEnum::SECOND, (*cni));
        if ( debugCopyOutDebug() )
        {
            gLogInfo << "\tbdma comp edge=" << dbgBDMACompEdge->id() << " -> consumer node=" << (*cni)->id() << endl;
        }
    }

    // Finally add the bdma node to current node's output edge after all consumer nodes
    // are bound to the bdma node with compute edges
    m_graph->appendNodeToEdge(nodeOutEdge, ast::EdgeSideEnum::SECOND, dbgBDMANode);
    if ( debugCopyOutDebug() )
    {
        gLogInfo << "\tnode out edge=" << nodeOutEdge->id() << " -> bdma node=" << dbgBDMANode->id() << endl;
    }


 fail:
    return e;
}




//----------------------------------------------------------------------
//                           Code Emission Utils
//----------------------------------------------------------------------
NvDlaError engine_ast::Graph::createTensorDescListEntry(surface::TensorSurfaceDesc *sd, ILoadable::TensorDescListEntry &t, NvU32 memAtomSize)
{
    NvDlaError e = NvDlaSuccess;

    surface::SurfaceFormat sf;

    if ( !sd ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    t.name = sd->name();
    t.size = sd->tensorBufferDesc()->size();
    t.offset = 0; // XXX when appropriate, hook up with tsd+tbd not just tbd info

    sf = sd->surfaceFormat();

    t.dims.n = sd->dimensions().n;
    t.dims.c = sd->dimensions().c;
    t.dims.h = sd->dimensions().h;
    t.dims.w = sd->dimensions().w;

    t.dataFormat = sd->dataFormat();
    t.pixelMapping = NVDLA_PIXEL_MAPPING_PITCH_LINEAR;

    t.stride[0] = sf.bytesPerElement();
    t.stride[1] = sd->lineStride();
    t.stride[2] = sd->surfaceStride();
    t.stride[3] = sd->planeStride();
    t.stride[4] = 0;
    t.stride[5] = 0;
    t.stride[6] = 0;
    t.stride[7] = 0;


    switch ( sf.precision().e() ) {
        case surface::NVDLA_PRECISION_INT8:
            t.dataType = DataType::INT8;
            break;
        case surface::NVDLA_PRECISION_INT16:
            t.dataType = DataType::INT16;
            break;
        case surface::NVDLA_PRECISION_FP16:
            t.dataType = DataType::HALF;
            break;
        case surface::NVDLA_PRECISION_UINT8:
            t.dataType = DataType::UINT8;
            break;
        case surface::NVDLA_PRECISION_UINT16:
            t.dataType = DataType::UINT16;
            break;
        case surface::NVDLA_PRECISION_UNKNOWN:
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }


    switch( sf.category().e() )
    {
        case surface::IMG:
            t.dataCategory = DataCategory::IMAGE;
            switch ( sf.e() )
            {
                case surface::NVDLA_IMG_R8:  t.pixelFormat = PixelFormat::R8;  break;
                case surface::NVDLA_IMG_R10: t.pixelFormat = PixelFormat::R10; break;
                case surface::NVDLA_IMG_R12: t.pixelFormat = PixelFormat::R12; break;
                case surface::NVDLA_IMG_R16: t.pixelFormat = PixelFormat::R16; break;
                case surface::NVDLA_IMG_R16_I: t.pixelFormat = PixelFormat::R16_I; break;
                case surface::NVDLA_IMG_R16_F: t.pixelFormat = PixelFormat::R16_F; break;
                case surface::NVDLA_IMG_A16B16G16R16: t.pixelFormat = PixelFormat::A16B16G16R16; break;
                case surface::NVDLA_IMG_X16B16G16R16: t.pixelFormat = PixelFormat::X16B16G16R16; break;
                case surface::NVDLA_IMG_A16B16G16R16_F: t.pixelFormat = PixelFormat::A16B16G16R16_F; break;
                case surface::NVDLA_IMG_A16Y16U16V16: t.pixelFormat = PixelFormat::A16Y16U16V16; break;
                case surface::NVDLA_IMG_V16U16Y16A16: t.pixelFormat = PixelFormat::V16U16Y16A16; break;
                case surface::NVDLA_IMG_A16Y16U16V16_F: t.pixelFormat = PixelFormat::A16Y16U16V16_F; break;
                case surface::NVDLA_IMG_A8B8G8R8: t.pixelFormat = PixelFormat::A8B8G8R8; break;
                case surface::NVDLA_IMG_A8R8G8B8: t.pixelFormat = PixelFormat::A8R8G8B8; break;
                case surface::NVDLA_IMG_B8G8R8A8: t.pixelFormat = PixelFormat::B8G8R8A8; break;
                case surface::NVDLA_IMG_R8G8B8A8: t.pixelFormat = PixelFormat::R8G8B8A8; break;
                case surface::NVDLA_IMG_X8B8G8R8: t.pixelFormat = PixelFormat::X8B8G8R8; break;
                case surface::NVDLA_IMG_X8R8G8B8: t.pixelFormat = PixelFormat::X8R8G8B8; break;
                case surface::NVDLA_IMG_B8G8R8X8: t.pixelFormat = PixelFormat::B8G8R8X8; break;
                case surface::NVDLA_IMG_R8G8B8X8: t.pixelFormat = PixelFormat::R8G8B8X8; break;
                case surface::NVDLA_IMG_A2B10G10R10: t.pixelFormat = PixelFormat::A2B10G10R10; break;
                case surface::NVDLA_IMG_A2R10G10B10: t.pixelFormat = PixelFormat::A2R10G10B10; break;
                case surface::NVDLA_IMG_B10G10R10A2: t.pixelFormat = PixelFormat::B10G10R10A2; break;
                case surface::NVDLA_IMG_R10G10B10A2: t.pixelFormat = PixelFormat::R10G10B10A2; break;
                case surface::NVDLA_IMG_A2Y10U10V10: t.pixelFormat = PixelFormat::A2Y10U10V10; break;
                case surface::NVDLA_IMG_V10U10Y10A2: t.pixelFormat = PixelFormat::V10U10Y10A2; break;
                case surface::NVDLA_IMG_A8Y8U8V8: t.pixelFormat = PixelFormat::A8Y8U8V8; break;
                case surface::NVDLA_IMG_V8U8Y8A8: t.pixelFormat = PixelFormat::V8U8Y8A8; break;
                case surface::NVDLA_IMG_Y8___U8V8_N444: t.pixelFormat = PixelFormat::Y8___U8V8_N444; break;
                case surface::NVDLA_IMG_Y8___V8U8_N444: t.pixelFormat = PixelFormat::Y8___V8U8_N444; break;
                case surface::NVDLA_IMG_Y10___U10V10_N444: t.pixelFormat = PixelFormat::Y10___U10V10_N444; break;
                case surface::NVDLA_IMG_Y10___V10U10_N444: t.pixelFormat = PixelFormat::Y10___V10U10_N444; break;
                case surface::NVDLA_IMG_Y12___U12V12_N444: t.pixelFormat = PixelFormat::Y12___U12V12_N444; break;
                case surface::NVDLA_IMG_Y12___V12U12_N444: t.pixelFormat = PixelFormat::Y12___V12U12_N444; break;
                case surface::NVDLA_IMG_Y16___U16V16_N444: t.pixelFormat = PixelFormat::Y16___U16V16_N444; break;
                case surface::NVDLA_IMG_Y16___V16U16_N444: t.pixelFormat = PixelFormat::Y16___V16U16_N444; break;

                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
            }
            break;

        case surface::WEIGHT:
            t.dataCategory = DataCategory::WEIGHT;
            switch ( sf.e() )
            {
#if 0
                case surface::NVDLA_WEIGHT_DC_INT8: t.weight_format = WeightFormat::DC_INT8; break;
                case surface::NVDLA_WEIGHT_DC_INT8_COMPRESSED: t.weight_format = WeightFormat::DC_INT8_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_WG_INT8: t.weight_format = WeightFormat::WG_INT8; break;
                case surface::NVDLA_WEIGHT_WG_INT8_COMPRESSED: t.weight_format = WeightFormat::WG_INT8_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_CE_INT8: t.weight_format = WeightFormat::CE_INT8; break;
                case surface::NVDLA_WEIGHT_CE_INT8_COMPRESSED: t.weight_format = WeightFormat::CE_INT8_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_DECONV_INT8: t.weight_format = WeightFormat::DECONV_INT8; break;
                case surface::NVDLA_WEIGHT_DECONV_INT8_COMPRESSED: t.weight_format = WeightFormat::DECONV_INT8_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_DC_INT16: t.weight_format = WeightFormat::DC_INT16; break;
                case surface::NVDLA_WEIGHT_DC_INT16_COMPRESSED: t.weight_format = WeightFormat::DC_INT16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_WG_INT16: t.weight_format = WeightFormat::WG_INT16; break;
                case surface::NVDLA_WEIGHT_WG_INT16_COMPRESSED: t.weight_format = WeightFormat::WG_INT16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_CE_INT16: t.weight_format = WeightFormat::CE_INT16; break;
                case surface::NVDLA_WEIGHT_CE_INT16_COMPRESSED: t.weight_format = WeightFormat::CE_INT16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_DECONV_INT16: t.weight_format = WeightFormat::DECONV_INT16; break;
                case surface::NVDLA_WEIGHT_DECONV_INT16_COMPRESSED: t.weight_format = WeightFormat::DECONV_INT16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_DC_FP16: t.weight_format = WeightFormat::DC_FP16; break;
                case surface::NVDLA_WEIGHT_DC_FP16_COMPRESSED: t.weight_format = WeightFormat::DC_FP16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_WG_FP16: t.weight_format = WeightFormat::WG_FP16; break;
                case surface::NVDLA_WEIGHT_WG_FP16_COMPRESSED: t.weight_format = WeightFormat::WG_FP16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_CE_FP16: t.weight_format = WeightFormat::CE_FP16; break;
                case surface::NVDLA_WEIGHT_CE_FP16_COMPRESSED: t.weight_format = WeightFormat::CE_FP16_COMPRESSED; break;
                case surface::NVDLA_WEIGHT_DECONV_FP16: t.weight_format = WeightFormat::DECONV_FP16; break;
                case surface::NVDLA_WEIGHT_DECONV_FP16_COMPRESSED: t.weight_format = WeightFormat::DECONV_FP16_COMPRESSED; break;
#endif
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
            }
            break;

        case surface::FEATURE_DATA:
            t.dataCategory = DataCategory::FEATURE;
            t.pixelFormat = PixelFormat::FEATURE;
            if (memAtomSize == 8)
                t.pixelFormat = PixelFormat::FEATURE_X8;
            t.dataFormat = NVDLA_DATA_FORMAT_NCxHWx;
            break;

        case surface::M_PLANAR:
            t.dataCategory = DataCategory::PLANAR;
            switch ( sf.e() )
            {
#if 0
                case surface::NVDLA_M_PLANAR_INT8: t.planar_format = PlanarFormat::INT8; break;
                case surface::NVDLA_M_PLANAR_INT16: t.planar_format = PlanarFormat::INT16; break;
                case surface::NVDLA_M_PLANAR_FP16: t.planar_format = PlanarFormat::FP16; break;
#endif
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
            }

            break;
        case surface::BIAS_DATA:
            t.dataCategory = DataCategory::BIAS;
            switch ( sf.e() )
            {
#if 0
                case surface::NVDLA_BIAS_DATA_INT8: t.bias_format = BiasFormat::INT8; break;
                case surface::NVDLA_BIAS_DATA_INT16: t.bias_format = BiasFormat::INT16; break;
                case surface::NVDLA_BIAS_DATA_FP16: t.bias_format = BiasFormat::FP16; break;
#endif
                default:
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
            }
            break;
        default:
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }


    if ( debugSurfaces() )
    {
        gLogInfo << "create tensor desc precision=" << (int)sf.precision().v() <<
            " category=" << (int)sf.category().v() <<
            " sf=" << (int)sf.v() << endl;

        gLogInfo << "\tname         : " << t.name << endl;
        gLogInfo << "\tn,c,h,w      : " << t.dims.n << "," << t.dims.c << "," << t.dims.h << "," << t.dims.w << endl;
        gLogInfo << "\tdata format  : " << int(t.dataFormat) << endl;
        gLogInfo << "\tdata type    : " << int(t.dataType) << endl;
        gLogInfo << "\tdata category: " << int(t.dataCategory) << endl;
        gLogInfo << "\tpixel format : " << int(t.pixelFormat) << endl;
        gLogInfo << "\tpixel mapping: " << int(t.pixelMapping) << endl;

        gLogInfo << "\tstrides  : " <<
            t.stride[0] << " " << t.stride[1] << " " << t.stride[2] << " " << t.stride[3] <<
            t.stride[4] << " " << t.stride[5] << " " << t.stride[6] << " " << t.stride[7] <<
            endl;
    }


 fail:
    return e;
}


static NvDlaError addMemEntriesForPools
(
    engine_ast::Graph* g,
    vector< Loadable::MemoryListEntry >& graphMemObjects,
    NvS16& memId,
    vector< Loadable::AddressListEntry>& graphAddrObjects,
    NvS16& addrId
)
{
    NvDlaError e = NvDlaSuccess;
    vector< memory::Pool > *memPools = g->resourceMgr()->memoryPools();

    for ( size_t pid = 0; pid < memPools->size(); ++pid )
    {
        memory::Pool &pool = (*memPools)[pid];

        memory::LocationEnum location = pool.location().e();
        memory::PoolTypeEnum poolType = pool.type().e();
        NvU8 domain;
        NVDLA_UNUSED(poolType);

        // we only care about actual memory
        switch (location)
        {
            case memory::LocationEnum::lCVSRAM:
                domain = ILoadable::MemoryDomain_SRAM;
                break;

            case memory::LocationEnum::lDRAM:
                domain = ILoadable::MemoryDomain_SYSMEM;
                break;

            default:
                continue;
        }

        // don't bother if it is empty.
        if ( !pool.sizeUsed() )
        {
            continue;
        }

        pool.setMemoryId(memId++);
        pool.setAddressId(addrId++);

        Loadable::MemoryListEntry memEntry;

        memEntry.id        = pool.memoryId();
        memEntry.size      = NvU64(4096) * ((pool.sizeUsed() + NvU64(4095))/NvU64(4096));
        memEntry.alignment = 4096; // page size
        memEntry.domain    = domain;
        memEntry.flags     = Loadable::MemoryListEntry::flags_alloc();


        if ( pool.contents().begin() != pool.contents().end() )
        {
            set<surface::TensorSurfaceDesc *>::iterator ci;
            memEntry.flags |= Loadable::MemoryListEntry::flags_set();

            for (ci = pool.contents().begin(); ci != pool.contents().end(); ++ci )
            {
                // tbd: sort these for repeatability
                memory::TensorBufferDesc *tbd = (*ci)->tensorBufferDesc();
                if ( !tbd )
                {
                    THROW_ERROR(NvDlaError_InvalidState);
                }
                memEntry.contents.push_back(tbd->id());
                memEntry.offsets.push_back((*ci)->bufferOffset() + tbd->poolOffset());
            }
        }

        if ( g->debugMemoryLayout() )
        {
            gLogInfo << "(Pool) Memory list entry=" << memEntry.id << " size=" << memEntry.size <<
                " used=" << pool.sizeUsed() << " domain=" << (int)memEntry.domain << " flags=" << (int)memEntry.flags << endl;
            for ( size_t ci=0; ci < memEntry.contents.size(); ++ci )
            {
                gLogInfo << "\tcontent: " << memEntry.contents[ci] << " @ " << memEntry.offsets[ci] << endl;
            }
            gLogInfo << endl;
        }

        graphMemObjects.push_back(memEntry);
        {
            Loadable::AddressListEntry addrEntry;
            addrEntry.id     = pool.addressId();
            addrEntry.mem_id = memEntry.id;
            addrEntry.offset = 0;
            addrEntry.size   = memEntry.size;
            graphAddrObjects.push_back(addrEntry);
        }
    }

    return e;
}

static NvDlaError addMemEntriesForBuffers
(
    engine_ast::Graph* g,
    vector< Loadable::MemoryListEntry >& graphMemObjects,
    vector< Loadable::TensorDescListEntry >& tensorDescEntries,
    NvS16& memId
)
{
    NvDlaError e = NvDlaSuccess;

    NvU32 numBatches = g->profile()->multiBatchSize();
    NvU32 memAtomSize = g->target_config()->memoryAtomicSize();
    vector< memory::TensorBufferDesc *> allBuffers   = g->resourceMgr()->getBufferDescs();

    for ( vector< memory::TensorBufferDesc *>::iterator it = allBuffers.begin(); it != allBuffers.end(); ++it )
    {

        NvU8 domain;
        memory::TensorBufferDesc *tbd = *it;
        bool isAUX = tbd->content();
        bool isBindable = tbd->bindable();

        if ( !tbd )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState);
        }

        // pooled buffers don't need individual memory entries.
        // their containing pools are allocated and pinned once during runtime and that's it
        if ( tbd->pool() )
        {
            if ( tbd->bindable() )
            {
                string delim("");
                gLogInfo << tbd->id() << " is pooled in " << tbd->pool()->name() << " but bindable?  surfaces=[";
                for ( set<surface::TensorSurfaceDesc*>::iterator si = tbd->surfaces().begin();
                      si != tbd->surfaces().end(); ++si )
                {
                    gLogInfo << delim << (*si)->id();
                    delim = ", ";
                }
                gLogInfo << "]" << endl;
                ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "found bindable yet pooled buffer=%s", tbd->id().c_str());

            }
            continue;
        }

        // prepare memory list entries for the buffer descs of all batches
        for ( NvU32 nn = 0; nn < numBatches; ++nn )
        {
            // we only care about actual memory...
            switch ( tbd->memoryLoc(/*batchId*/nn).e() )
            {
                case memory::LocationEnum::lCVSRAM:
                    domain = ILoadable::MemoryDomain_SRAM;
                    break;
                case memory::LocationEnum::lDRAM:
                    domain = ILoadable::MemoryDomain_SYSMEM;
                    break;
                case memory::LocationEnum::lSTREAM:
                    continue;
                case memory::LocationEnum::lUNKNOWN:
                    if ( tbd->bindable() )
                    {
                        // bindable memory isn't actually determined yet.
                        // but call it sysmem for now?
                        domain = ILoadable::MemoryDomain_SYSMEM;
                    }
                    else
                    {
                        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "discovered non-bindable %s at unknown mem domain", tbd->id().c_str());
                    }
                    break;
                default:
                    continue; /// !!!
            }

            if ( tbd->size() == 0 )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "zero sized buffer %s", tbd->id().c_str());
            }

            // aux buffer is shared by all batches, so allow only 1 memEntry
            if (isAUX && nn > 0)
            {
                tbd->setMemoryId(tbd->memoryId(/*batchId*/0), /*batchId*/nn);
                continue;
            }

            // bindable buffer is a multi-batch NCHW buffer, so allow only 1 memEntry
            if (isBindable && nn > 0)
            {
                tbd->setMemoryId(tbd->memoryId(/*batchId*/0), /*batchId*/nn);
                continue;
            }

            Loadable::MemoryListEntry memEntry;
            // non pooled, need to create a new memory id to match
            tbd->setMemoryId(memId++, nn);
            tbd->setPoolOffset(0, nn); // pool() is 0.  but just in case.

            memEntry.id        = tbd->memoryId(nn);
            memEntry.size      = tbd->size();
            memEntry.alignment = 4096; // overkill?, page size
            memEntry.domain    = domain;
            memEntry.flags     = Loadable::MemoryListEntry::flags_alloc();

            if ( tbd->content() )
            {
                memEntry.flags |= Loadable::MemoryListEntry::flags_set();
                memEntry.contents.push_back(tbd->id());
                memEntry.offsets.push_back(0);
            }


            if ( tbd->bindable() ) // note: bindable are always non-pooled!
            {
                ILoadable::TensorDescListEntry newTensorDesc;
                NvS16 bindId;
                enum IOD bindDomain;

                // this is where loadable-visible tensor desc ids get allocated.
                memEntry.tensor_desc_id = tensorDescEntries.size();
                newTensorDesc.id = memEntry.tensor_desc_id;

                g->createTensorDescListEntry(tbd->boundSurface(0), newTensorDesc, memAtomSize);
                newTensorDesc.memId = tbd->memoryId(/*batchId*/nn);
                tensorDescEntries.push_back(newTensorDesc);

                bindId = tbd->bindId(bindDomain);
                memEntry.bind_id = bindId;
                switch ( bindDomain )
                {
                    case IOD_Input:
                        memEntry.flags |= Loadable::MemoryListEntry::flags_input();
                        break;
                    case IOD_Output:
                        memEntry.flags |= Loadable::MemoryListEntry::flags_output();
                        break;
                    case IOD_Debug:
                        memEntry.flags |= Loadable::MemoryListEntry::flags_debug();
                        break;
                    default:
                        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "n/a bindable mem list entry %d",
                                             memEntry.id);
                }
                if ( g->debugMemoryLayout() )
                {
                    gLogInfo << "(Bindable)";
                }
            }


            if ( g->debugMemoryLayout() )
            {
                gLogInfo << "(Buffer) Memory list entry for tbd=" << tbd->id() << ":" << nn
                         << " : "  << nn
                         << " size=" << memEntry.size
                         << " domain=" << (int)memEntry.domain
                         << " flags=" << (int)memEntry.flags;
                for ( size_t ci=0; ci < memEntry.contents.size(); ++ci )
                {
                    gLogInfo << "\tcontent: " << memEntry.contents[ci] << " @ " << memEntry.offsets[ci] << endl;
                }
                gLogInfo << endl;
            }

            graphMemObjects.push_back(memEntry);
        }
    }

fail:
    return e;
}

static NvDlaError addAddrEntriesForSurfaces
(
    Loadable* l,
    engine_ast::Graph* g,
    vector< Loadable::AddressListEntry >& graphAddrObjects,
    NvS16& addrId
)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 numBatches = g->profile()->multiBatchSize();
    vector< surface::TensorSurfaceDesc *> allSurfaces = g->resourceMgr()->getSurfaceDescs();

    for (vector< surface::TensorSurfaceDesc *>::iterator tsdi = allSurfaces.begin(); tsdi != allSurfaces.end(); ++tsdi)
    {

        surface::TensorSurfaceDesc *tsd = *tsdi;
        memory::TensorBufferDesc *tbd = tsd->tensorBufferDesc();
        bool isAUX = tsd->content();

        if ( !tbd )
        {
            gLogWarning << "no buffer for surface=" << tbd->id() << endl;
            continue;
        }

        if ( tbd->size() == 0 ) {
            continue;
        }

        // prepare address list entries for the surface descs of all batches
        for (NvU32 nn = 0; nn < numBatches; ++nn)
        {
            // aux surface is shared by all batches, so allow only 1 addrEntry
            if (isAUX && nn > 0)
            {
                tsd->setAddressId      (tsd->addressId(0),       nn);
                tsd->setAddressIdOffset(tsd->addressIdOffset(0), nn);
                continue;
            }

            if ( !tbd->pool(nn) )
            {
                Loadable::AddressListEntry addrEntry;
                tsd->setAddressId      (addrId++, nn);
                tsd->setAddressIdOffset(0, nn);

                addrEntry.id     = tsd->addressId(nn);
                addrEntry.mem_id = tbd->memoryId(nn);
                addrEntry.offset = tbd->poolOffset(/*batchId*/nn) + tsd->bufferOffset(/*batchId*/nn);
                addrEntry.size   = tbd->size();
                graphAddrObjects.push_back(addrEntry);
            }
            else
            {
                if ( tbd->pool(nn)->addressId() <= 0 )
                {
                    THROW_ERROR(NvDlaError_InvalidState);
                }
                tsd->setAddressId(tbd->pool(nn)->addressId(), nn);
                tsd->setAddressIdOffset(tbd->poolOffset(nn) + tsd->bufferOffset(nn), nn);
            }

            if ( tsd->content() )
            {

                ILoadable::Blob memEntryBlob;

                // within the loadable the content is associated with the buffer, not the surface.
                memEntryBlob.name = tbd->id();
                memEntryBlob.size = tsd->size();
                memEntryBlob.interface = ILoadable::Interface_NONE;
                memEntryBlob.version   = ILoadable::Version(0, 0, 0);

                l->setSymbolContent(tbd->id(), memEntryBlob, tsd->address<NvU8>(nn));
            }

            if ( g->debugMemoryLayout() )
            {
                gLogInfo << "(Surface) Address list entry for tsd=" << tsd->id() << "/" << tbd->id() << ":" << nn
                         << " -> " << tsd->addressId()
                          << " offset=" << tsd->addressIdOffset()
                         << " size=" << tbd->size() << endl;
            }



        }
    }

    return e;
}

//
// prepares address and memory list entries
//
NvDlaError engine_ast::Graph::prepareMemoryListEntries(Loadable *l)
{
    NvDlaError e = NvDlaSuccess;

    NvS16 memId = 0;
    NvS16 addrId = 0;

    vector< Loadable::MemoryListEntry > graphMemObjects;
    vector< Loadable::AddressListEntry > graphAddrObjects;
    vector< Loadable::TensorDescListEntry > tensorDescEntries;

    //
    // now we start
    //
    memId = 1;
    addrId = 1;

    //
    // each memory pool gets an id.  pools need to come before buffers
    // because buffers can and will use the pools' memory ids as needed.
    //
    PROPAGATE_ERROR_FAIL(addMemEntriesForPools(this, graphMemObjects, memId, graphAddrObjects, addrId));

    //
    // create memory id entries for non-pooled buffers
    //
    PROPAGATE_ERROR_FAIL(addMemEntriesForBuffers(this, graphMemObjects, tensorDescEntries, memId));

    //
    // surfaces generate address id entries
    //
    PROPAGATE_ERROR_FAIL(addAddrEntriesForSurfaces(l, this, graphAddrObjects, addrId));

    l->setMemoryListEntries(graphMemObjects);
    l->setAddressListEntries(graphAddrObjects);
    l->setTensorDescListEntries(tensorDescEntries);

fail:
    return e;
}

void engine_ast::Graph::resetRelocEntries()
{
    m_relocEntries.clear();
}

void engine_ast::Graph::insertRelocEntry(ILoadable::RelocEntry re)
{
    m_relocEntries.push_back(re);
}

NvDlaError engine_ast::Graph::gatherRelocEntries(NvS16 ops,  NvU8 *ops_base,
                                           NvS16 surfs, NvU8 *surfs_base,
                                           NvS16 deps, NvU8 *deps_base)
{
    NvDlaError e = NvDlaSuccess;

    uintptr_t originalOffset;
    uintptr_t useBase;

    vector<ILoadable::RelocEntry>::iterator ri;

    for ( ri = m_relocEntries.begin(); ri != m_relocEntries.end(); ++ri )
    {
        if ( ri->writeId )
        {
            continue;
        }

        if ( ! ( (ri->interface == NVDLA_LOADABLE_INTERFACE_DLA1) ||
                 (ri->interface == NVDLA_LOADABLE_INTERFACE_EMU1)) )
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "bogus interface");
            continue; /*NOT REACHED*/
        }

        originalOffset = ri->offset;

        if ( ri->subInterface == NVDLA_LOADABLE_SUB_INTERFACE_DLA1_OPS ||
             ri->subInterface == NVDLA_LOADABLE_SUB_INTERFACE_EMU1_OPS)
        {
            ri->writeId = ops;
            useBase = uintptr_t(ops_base);
            ri->offset = ri->offset - useBase;
        }
        else if ( ri->subInterface == NVDLA_LOADABLE_SUB_INTERFACE_DLA1_SURFS ||
                  ri->subInterface == NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS )
        {
            ri->writeId = surfs;
            useBase = uintptr_t(surfs_base);
            ri->offset = ri->offset - useBase;
        }
        else if ( ri->subInterface == NVDLA_LOADABLE_SUB_INTERFACE_DLA1_DEPS )
        {
            ri->writeId = deps;
            useBase = uintptr_t(deps_base);
            ri->offset = ri->offset - useBase;
        }
        else
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "bogus sub interface");
        }

        if ( debugRelocs() ) {
            gLogInfo << "gathered reloc entry: address id=" << ri->addressListId <<
                " writeId=" << ri->writeId <<
                " useBase=" <<  std::hex << useBase <<
                " originalOffset=" << std::hex << originalOffset <<
                " offset=" << std::hex << ri->offset << std::dec <<
                " interface=" << (int)ri->interface <<
                " subInterface=" << (int)ri->subInterface <<
                " relocType=" << (int)ri->relocType << endl;
        }
    }

 fail:
    return e;
}


} // nvdla::priv
} // nvdla
