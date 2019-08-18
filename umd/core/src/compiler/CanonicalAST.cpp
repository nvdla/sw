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

#include <iostream>
#include <algorithm>
#include <string>

#include "priv/Check.h"

#include "priv/CanonicalAST.h"

#include "priv/Network.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

using std::map;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::string;
using std::endl;
using std::ostream;
using std::stringstream;

namespace nvdla
{

class ILayer;

namespace priv
{

ENUM_PARAMETER_STATIC(canonical_ast::CanonicalOpType,  CANONICAL_OPERATION_TYPE_ENUMS,  "CanonicalOpTypeEnum")

NvU32 canonical_ast::Node::m_next_id = 0;
NvU32 canonical_ast::Edge::m_next_id = 0;

canonical_ast::Node *canonical_ast::newCanonicalNode(Layer *orig_nw_layer)
{
    LayerType original_type = orig_nw_layer->getType();

    switch (original_type)
    {
        case LayerType::kCONVOLUTION: {
            ConvolutionLayer *conv_layer = LayerFactory::derivedPriv< ConvolutionLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newConvNode(conv_layer);
          }
        case LayerType::kFULLY_CONNECTED: {
            FullyConnectedLayer *fc_layer = LayerFactory::derivedPriv< FullyConnectedLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newFCNode(fc_layer);
          }
        case LayerType::kACTIVATION: {
            ActivationLayer *act_layer = LayerFactory::derivedPriv< ActivationLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newActivationNode(act_layer);
          }
        case LayerType::kPOOLING: {
            PoolingLayer *pool_layer = LayerFactory::derivedPriv< PoolingLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newPoolingNode(pool_layer);
          }
        case LayerType::kLRN: {
            LRNLayer *lrn_layer = LayerFactory::derivedPriv< LRNLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newLRNNode(lrn_layer);
          }
        case LayerType::kSCALE: {
            ScaleLayer *scale_layer = LayerFactory::derivedPriv< ScaleLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newScaleNode(scale_layer);
          }
        case LayerType::kBATCH_NORM: {
            BatchNormLayer *bn_layer = LayerFactory::derivedPriv< BatchNormLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newBatchNormNode(bn_layer);
          }
        case LayerType::kSOFTMAX: {
            SoftMaxLayer *sm_layer = LayerFactory::derivedPriv< SoftMaxLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newSoftMaxNode(sm_layer);
          }
        case LayerType::kCONCATENATION: {
            ConcatenationLayer *concat_layer = LayerFactory::derivedPriv< ConcatenationLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newConcatNode(concat_layer);
          }
        case LayerType::kDECONVOLUTION: {
            DeconvolutionLayer *deconv_layer = LayerFactory::derivedPriv< DeconvolutionLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newDeconvNode(deconv_layer);
          }
        case LayerType::kELEMENTWISE: {
            ElementWiseLayer *ew_layer = LayerFactory::derivedPriv< ElementWiseLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newEWNode(ew_layer);
          }
        case LayerType::kSLICE: {
            SliceLayer *slice_layer = LayerFactory::derivedPriv< SliceLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newSplitNode(slice_layer);
          }
        default:
            return NULL;
    }

    return NULL;
}

canonical_ast::Graph* canonical_ast::Graph::clone()
{
    REPORT_ERROR(NvDlaError_NotSupported, "Graph cloning is not supported for Canonical AST");

    return NULL;
}

//
// the following generates a 1:1 mapping with the Canonical graph input.
//
canonical_ast::Graph *canonical_ast::generateGraph(Network *network)
{
    vector<canonical_ast::Edge *> input_edges;
    vector<canonical_ast::Edge *> output_edges;

    map<canonical_ast::Node *, Layer *, Graph::nodeCompareFn>  node_layer;
    map<canonical_ast::Node *, Layer *, Graph::nodeCompareFn>::iterator lni;

    map<Tensor *, canonical_ast::Edge *>  tensor_edge;
    map<Tensor *, Tensor *>  nw_tensor_to_can_tensor;
    map<Tensor *, canonical_ast::Edge *>::iterator tei;

    Graph *graph = new Graph();

    vector<Tensor *> network_inputs;
    for (int ni = 0; ni < network->getNumInputs(); ++ni)
    {
        network_inputs.push_back(TensorFactory::priv(network->getInput(ni)));
    }
    input_edges.resize(network_inputs.size());


    vector<Tensor *> network_outputs;
    for (int ni = 0; ni < network->getNumOutputs(); ++ni)
    {
        network_outputs.push_back(TensorFactory::priv(network->getOutput(ni)));
    }
    output_edges.resize(network_outputs.size());

    //    gLogInfo << "canonical_ast::" << __func__ << " network shows " << network_inputs.size() << " inputs and " <<
    //        network_outputs.size() << " outputs" << endl;

    for (int li = 0; li < network->getNumLayers(); li++)
    {
        ILayer *ilayer = network->getLayer(li);
        Layer *layer = LayerFactory::priv(ilayer);
        if ( !(ilayer && layer) )
        {
            gLogError << __func__ << " encountered null layer at network layer index=" << li << endl;
            continue;
        }

        canonical_ast::Node *can_node = newCanonicalNode(layer);
        if ( !can_node )
        {
            delete graph; // blow up
            graph = 0;
            goto done;
        }
        can_node->setGraph(graph);
        graph->insertNode(can_node);

        can_node->setId(graph->nextNodeId());
        can_node->setName(layer->getName());

        node_layer[can_node] = layer;
    }

    //
    // Now all the layer nodes are in the graph.
    // For each layer assemble the edges.
    //

    for (lni = node_layer.begin(); lni != node_layer.end(); ++lni)
    {
        canonical_ast::Node *node = lni->first;
        Layer *l = lni->second;

        size_t input_tensors = 0, output_tensors = 0, aux_input_tensors = 0;
        vector<Tensor *> io_tensors, aux_tensors;
        NVDLA_UNUSED(aux_input_tensors);

        for(int ii = 0, II = l->getNumInputs(); ii < II; ++ii)
        {
            Tensor *tensor = TensorFactory::priv(l->getInput(ii));
            if ( !tensor )
            {
                gLogError << __func__ << " 3.<null>.i." << ii << endl;
                continue;
            }
            io_tensors.push_back(tensor);
            input_tensors++;
        }
        for(int oo = 0, OO = l->getNumOutputs(); oo < OO; ++oo)
        {
            Tensor *tensor = TensorFactory::priv(l->getOutput(oo));
            if ( ! tensor )
            {
                gLogError << __func__ << " 3.<null>.o." << oo << endl;
                continue;
            }
            io_tensors.push_back(tensor);
            output_tensors++;
        }

        for(size_t io = 0, IO = io_tensors.size(); io < IO; ++io)
        {
            Tensor *nw_tensor = io_tensors[io];
            bool is_input = io < input_tensors;
            ast::EdgeSide edge_side( is_input ? ast::EdgeSideEnum::SECOND : ast::EdgeSideEnum::FIRST);
            ast::EdgeDirection edge_dir(ast::EdgeDirectionEnum::DIRECTED);

            map<Tensor *, canonical_ast::Edge *>::iterator f = tensor_edge.find(nw_tensor);
            canonical_ast::Edge *can_edge = 0;
            Tensor* can_tensor = 0;
            if ( f == tensor_edge.end() )
            {
                can_edge = new canonical_ast::Edge();
                can_edge->setGraph(graph);

                can_tensor = nw_tensor->clone();
                can_tensor->setNetwork(NULL);   // get rid of any connections back to the network builder
                can_tensor->setTensorType(TensorType::kIO);
                can_edge->setId(graph->nextEdgeId());
                can_edge->setOriginalTensor(can_tensor);
                graph->insertEdge(can_edge);

                tensor_edge[nw_tensor] = can_edge;
                nw_tensor_to_can_tensor[nw_tensor] = can_tensor;
            } else {
                can_edge = f->second;
            }
            graph->appendNodeToEdge(can_edge, edge_side, node);

            // if this is an input node it could be one of the network inputs.
            // if so keep track of it.
            if ( is_input )
            {
                for ( size_t iti = 0; iti < network_inputs.size(); iti++)
                {
                    if ( nw_tensor == network_inputs[iti] )
                    {
                        // gLogInfo << " identified input edge: " << (int)iti << " tensor id " << tensor->getName() << endl;
                        input_edges[iti] = can_edge;
                        can_tensor = nw_tensor_to_can_tensor[nw_tensor];
                        can_tensor->setTensorType(TensorType::kNW_INPUT);
                        break;
                    }
                }
                node->markInputEdge(can_edge);
            }
            else
            {
                for ( size_t oti = 0; oti < network_outputs.size(); oti++)
                {
                    if ( nw_tensor == network_outputs[oti] )
                    {
                        // gLogInfo << " identified output edge: " << (int)oti << " tensor id " << tensor->getName() << endl;
                        output_edges[oti] = can_edge;
                        can_tensor = nw_tensor_to_can_tensor[nw_tensor];
                        can_tensor->setTensorType(TensorType::kNW_OUTPUT);
                        break;
                    }
                }
                node->markOutputEdge(can_edge);
            }
        }
    }

    if ( input_edges.size() )
    {
        graph->setInputEdges(input_edges);
    }
    if ( output_edges.size() )
    {
        graph->setOutputEdges(output_edges);
    }

    graph->scoredOrdering()->generate();
    graph->markClean();

done:
    return graph;
}

ostream &canonical_ast::outputJson(canonical_ast::Graph *graph, ostream &os)
{
    string sep;
    os << "[ {" << " \"app\" : \"\"}  " << endl; // to signal content flavor

    //
    // nodes
    //
    sep = string(",");
    for (Graph::NodeSetIterator ni = graph->nodes().begin(); ni != graph->nodes().end(); ++ni)
    {
        os << sep;
        outputJson(graph, *ni, os);
        sep = string(", ");
    }

    //
    // edges
    //
    for (Graph::EdgeSetIterator ei = graph->edges().begin(); ei != graph->edges().end(); ++ei)
    {
        os << sep;
        outputJson(graph, *ei, os);
        sep = string(",");
    }
    os << "]" << endl;
    return os;
}



ostream &canonical_ast::outputJson(canonical_ast::Graph *graph, canonical_ast::Edge *edge, ostream &os)
{
#if 0
    static int dummy = 0;
    string edge_text_id = edge->originalTensor()->getName();

    if ( edge_text_id == string("") )
    {
        edge->originalTensor()->setName( string("e-"+toString(dummy++)).c_str());
        edge_text_id = edge->originalTensor()->getName();
    }
#endif
    // edge label to gather the fan-in (almost always 1) and fan-out
    os << "{ \"class\":\"edge\", \"id\":\"" << edge->id() <<
        "\", \"is_input\":" << ( (graph->inputEdges().end()==std::find(graph->inputEdges().begin(), graph->inputEdges().end(), edge))?"false":"true") <<
        ", \"is_output\":" << ((graph->outputEdges().end()==std::find(graph->outputEdges().begin(), graph->outputEdges().end(), edge))?"false":"true");


    //  now create "line" elements to represent the fan-in && fan-out of the edge
    const vector<canonical_ast::Node*> source_nodes = graph->upstreamNodes(edge);  // source
    const vector<canonical_ast::Node*> target_nodes = graph->downstreamNodes(edge); // target

    std::string delim0 = "\n\t,";
    os << ", \"sources\":[";
    if ( source_nodes.size() )
    {
        std::string source_delim = "";

        for ( size_t s = 0, S = source_nodes.size(); s < S; ++s )
        {
            os << source_delim << '"' << source_nodes[s]->id() << '"';
            source_delim = ", ";
        }
    }
    os << "], \"targets\":[";
    if ( target_nodes.size() )
    {
        std::string target_delim = "";
        for ( size_t t = 0, T = target_nodes.size(); t < T; ++t )
        {
            string target_node_id = target_nodes[t]->id();
            os << target_delim << '"' << target_nodes[t]->id() << '"';
            target_delim = ", ";
        }
    }
    os << "]}";
    return os;
}

ostream &canonical_ast::outputJson(canonical_ast::Graph *, canonical_ast::Node *node, ostream &os)
{
    os << " { \"class\":\"node\", \"id\":\"" << node->id() << "\" }";
    return os;
}


bool canonical_ast::serializeTo(WisdomContainerEntry *)
{
    return false;
}

bool canonical_ast::deserializeFrom(WisdomContainerEntry *)
{
    return false;
}

//  Canonical operation parameters
void canonical_ast::ConvolutionNode::captureNetworkParams(ConvolutionLayer* origNwLayer)
{
    Dims4 weightDims, biasDims;
    params().setBiasMode(origNwLayer->getBiasMode());
    params().setHasBiasTerm(origNwLayer->getBiasWeights().count > 0 ? true : false);
    params().setTopLeftPadding(origNwLayer->getTopLeftPadding());
    params().setBottomRightPadding(origNwLayer->getBottomRightPadding());
    params().setPaddingValue(origNwLayer->getPaddingValue());
    params().setStride(origNwLayer->getStride());
    params().setDilation(origNwLayer->getDilation());
    params().setWeights(origNwLayer->getKernelWeights());
    params().setBiasData(origNwLayer->getBiasWeights());
    params().setNumGroups(origNwLayer->getNumGroups());
    NvU32 kernChannels  = params().weights().count/
                         (origNwLayer->getNumOutputMaps() *
                          origNwLayer->getKernelSize().h *
                          origNwLayer->getKernelSize().w);
    weightDims.n = origNwLayer->getNumOutputMaps();
    weightDims.c = kernChannels;
    weightDims.h = origNwLayer->getKernelSize().h;
    weightDims.w = origNwLayer->getKernelSize().w;
    params().setWeightDims(weightDims);

    if (params().hasBiasTerm())
    {
        switch(origNwLayer->getBiasMode())
        {
            case BiasMode::bUNIFORM:
                biasDims.c = 1;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bCHANNEL:
                biasDims.c = params().biasData().count;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bm_ELEMENTWISE:
                biasDims.c = origNwLayer->getInput(0)->getDimensions().c;
                biasDims.h = origNwLayer->getInput(0)->getDimensions().h;
                biasDims.w = origNwLayer->getInput(0)->getDimensions().w;
                break;
            default:
                REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)origNwLayer->getBiasMode());
        }
        params().setBiasDims(biasDims);
    }

    return;
}

void canonical_ast::FullyConnectedNode::captureNetworkParams(FullyConnectedLayer* origNwLayer)
{
    Dims4 weightDims, biasDims;
    params().setBiasMode(origNwLayer->getBiasMode());
    params().setHasBiasTerm(origNwLayer->getBiasWeights().count > 0 ? true : false);
    params().setWeights(origNwLayer->getKernelWeights());
    params().setBiasData(origNwLayer->getBiasWeights());
    // the kernel weights of an inner product have the same dimensions as the input
    weightDims.n = origNwLayer->getNumOutputChannels();
    weightDims.c = origNwLayer->getInput(0)->getDimensions().c; // fixme: probably need fix
    weightDims.h = origNwLayer->getInput(0)->getDimensions().h; // fixme: probably need fix
    weightDims.w = origNwLayer->getInput(0)->getDimensions().w; // fixme: probably need fix
    params().setWeightDims(weightDims);

    if (params().hasBiasTerm())
    {
        switch(origNwLayer->getBiasMode())
        {
            case BiasMode::bUNIFORM:
                biasDims.c = 1;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bCHANNEL:
                biasDims.c = params().biasData().count;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bm_ELEMENTWISE:
                biasDims.c = origNwLayer->getInput(0)->getDimensions().c;
                biasDims.h = origNwLayer->getInput(0)->getDimensions().h;
                biasDims.w = origNwLayer->getInput(0)->getDimensions().w;
                break;
            default:
                REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)origNwLayer->getBiasMode());
        }
        params().setBiasDims(biasDims);
    }

    return;
}

void canonical_ast::ActivationNode::captureNetworkParams(ActivationLayer* origNwLayer)
{
    params().setActivationType(origNwLayer->getActivationType());
}

void canonical_ast::PoolingNode::captureNetworkParams(PoolingLayer* origNwLayer)
{
    params().setPoolType(origNwLayer->getPoolingType());
    params().setTopLeftPadding(origNwLayer->getTopLeftPadding());
    params().setBottomRightPadding(origNwLayer->getBottomRightPadding());
    params().setKernelDims(origNwLayer->getWindowSize());
    params().setStride(origNwLayer->getStride());
}

void canonical_ast::LRNNode::captureNetworkParams(LRNLayer* origNwLayer)
{
    params().setLocalSize(origNwLayer->getWindowSize());
    params().setAlpha(origNwLayer->getAlpha());
    params().setBeta(origNwLayer->getBeta());
    params().setK(origNwLayer->getK());
}

void canonical_ast::ScaleNode::captureNetworkParams(ScaleLayer* origNwLayer)
{
    Dims4 scaleDims, shiftDims, powerDims;
    params().setMode(origNwLayer->getMode());
    params().setShift(origNwLayer->getShift());
    params().setScale(origNwLayer->getScale());
    params().setPower(origNwLayer->getPower());
    params().setHasBiasTerm(origNwLayer->getShift().count > 0 ? true : false);
    switch(origNwLayer->getMode())
    {
        case ScaleMode::sUNIFORM:
            scaleDims.c = 1;
            scaleDims.h = 1;
            scaleDims.w = 1;
            break;
        case ScaleMode::sCHANNEL:
            scaleDims.c = params().scale().count;
            scaleDims.h = 1;
            scaleDims.w = 1;
            break;
        case ScaleMode::sm_ELEMENTWISE:
            scaleDims.c = origNwLayer->getInput(0)->getDimensions().c;
            scaleDims.h = origNwLayer->getInput(0)->getDimensions().h;
            scaleDims.w = origNwLayer->getInput(0)->getDimensions().w;
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown scale mode: %d", (int)origNwLayer->getMode());
    }
    params().setScaleDims(scaleDims);

    if (params().hasBiasTerm())
    {
        shiftDims = scaleDims;
        params().setShiftDims(shiftDims);
    }

    if (params().power().count > 0)
    {
        powerDims.c = 1;
        powerDims.h = 1;
        powerDims.w = 1;
        params().setPowerDims(powerDims);
    }
}

void canonical_ast::BatchNormNode::captureNetworkParams(BatchNormLayer* origNwLayer)
{
    Dims4 meanDims;
    Dims4 varianceDims;
    params().setMode(origNwLayer->getMode());
    params().setMean(origNwLayer->getMean());
    params().setVariance(origNwLayer->getVariance());
    params().setEpsilon(origNwLayer->getEpsilon());
    switch (origNwLayer->getParams().mode)
    {
        case BatchNormMode::bnUNIFORM:
            meanDims.c = 1;
            meanDims.h = 1;
            meanDims.w = 1;
            break;
        case BatchNormMode::bnm_CHANNEL:
            meanDims.c = params().mean().count;
            meanDims.h = 1;
            meanDims.w = 1;
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown batch norm mode: %d", (int)origNwLayer->getMode());
    }
    varianceDims = meanDims;
    params().setMeanDims(meanDims);
    params().setVarianceDims(varianceDims);
}

void canonical_ast::SoftMaxNode::captureNetworkParams(SoftMaxLayer* origNwLayer)
{
}

void canonical_ast::ConcatenationNode::captureNetworkParams(ConcatenationLayer* origNwLayer)
{
    params().setNumInputs(origNwLayer->getNumInputs());
}

void canonical_ast::SplitNode::captureNetworkParams(SliceLayer* origNwLayer)
{
    this->getParams().setNumOutputs(origNwLayer->getNumOutputs());
}

void canonical_ast::DeconvolutionNode::captureNetworkParams(DeconvolutionLayer* origNwLayer)
{
    Dims4 weightDims, biasDims;
    params().setBiasMode(origNwLayer->getBiasMode());
    params().setHasBiasTerm(origNwLayer->getBiasWeights().count > 0 ? true : false);
    params().setTopLeftPadding(origNwLayer->getTopLeftPadding());
    params().setBottomRightPadding(origNwLayer->getBottomRightPadding());
    params().setPaddingValue(origNwLayer->getPaddingValue());
    params().setStride(origNwLayer->getStride());
    params().setNumGroups(origNwLayer->getNumGroups());
    params().setDilation(origNwLayer->getDilation());
    params().setWeights(origNwLayer->getKernelWeights());
    params().setBiasData(origNwLayer->getBiasWeights());
    NvU32 kernChannels  = params().weights().count/
                         (origNwLayer->getNumOutputMaps() *
                          origNwLayer->getKernelSize().h *
                          origNwLayer->getKernelSize().w);

    weightDims.n = origNwLayer->getNumOutputMaps();
    weightDims.c = kernChannels;
    weightDims.h = origNwLayer->getKernelSize().h;
    weightDims.w = origNwLayer->getKernelSize().w;
    params().setWeightDims(weightDims);

    if (params().hasBiasTerm())
    {
        switch(origNwLayer->getBiasMode())
        {
            case BiasMode::bUNIFORM:
                biasDims.c = 1;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bCHANNEL:
                biasDims.c = params().biasData().count;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bm_ELEMENTWISE:
                biasDims.c = origNwLayer->getInput(0)->getDimensions().c;
                biasDims.h = origNwLayer->getInput(0)->getDimensions().h;
                biasDims.w = origNwLayer->getInput(0)->getDimensions().w;
                break;
            default:
                REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)origNwLayer->getBiasMode());
        }
        params().setBiasDims(biasDims);
    }

    return;
}

void canonical_ast::ElementWiseNode::captureNetworkParams(ElementWiseLayer* origNwLayer)
{
    params().setType(origNwLayer->getOperation());
}

// explicitly instantiate the priv maps of each node type
map<canonical_ast::Node*, canonical_ast::ConvolutionNode*> canonical_ast::NodeFactory::s_conv_priv =
    map<canonical_ast::Node*, canonical_ast::ConvolutionNode*>();

map<canonical_ast::Node*, canonical_ast::FullyConnectedNode*> canonical_ast::NodeFactory::s_fc_priv =
    map<canonical_ast::Node*, canonical_ast::FullyConnectedNode*>();

map<canonical_ast::Node*, canonical_ast::ActivationNode*> canonical_ast::NodeFactory::s_act_priv =
    map<canonical_ast::Node*, canonical_ast::ActivationNode*>();

map<canonical_ast::Node*, canonical_ast::PoolingNode*> canonical_ast::NodeFactory::s_pool_priv =
    map<canonical_ast::Node*, canonical_ast::PoolingNode*>();

map<canonical_ast::Node*, canonical_ast::LRNNode*> canonical_ast::NodeFactory::s_lrn_priv =
    map<canonical_ast::Node*, canonical_ast::LRNNode*>();

map<canonical_ast::Node*, canonical_ast::ScaleNode*> canonical_ast::NodeFactory::s_scale_priv =
    map<canonical_ast::Node*, canonical_ast::ScaleNode*>();

map<canonical_ast::Node*, canonical_ast::BatchNormNode*> canonical_ast::NodeFactory::s_bn_priv =
    map<canonical_ast::Node*, canonical_ast::BatchNormNode*>();

map<canonical_ast::Node*, canonical_ast::SoftMaxNode*> canonical_ast::NodeFactory::s_sm_priv =
    map<canonical_ast::Node*, canonical_ast::SoftMaxNode*>();

map<canonical_ast::Node*, canonical_ast::ConcatenationNode*> canonical_ast::NodeFactory::s_concat_priv =
    map<canonical_ast::Node*, canonical_ast::ConcatenationNode*>();

map<canonical_ast::Node*, canonical_ast::DeconvolutionNode*> canonical_ast::NodeFactory::s_deconv_priv =
    map<canonical_ast::Node*, canonical_ast::DeconvolutionNode*>();

map<canonical_ast::Node*, canonical_ast::ElementWiseNode*> canonical_ast::NodeFactory::s_ew_priv =
    map<canonical_ast::Node*, canonical_ast::ElementWiseNode*>();

map<canonical_ast::Node*, canonical_ast::SplitNode*> canonical_ast::NodeFactory::s_split_priv =
    map<canonical_ast::Node*, canonical_ast::SplitNode*>();

canonical_ast::ConvolutionNode* canonical_ast::NodeFactory::newConvNode(ConvolutionLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ConvolutionNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ConvolutionNode();
    d->captureNetworkParams(orig_nw_layer);

    s_conv_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::FullyConnectedNode* canonical_ast::NodeFactory::newFCNode(FullyConnectedLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::FullyConnectedNode* D;

    B b;
    D d;

    b = d = new canonical_ast::FullyConnectedNode();
    d->captureNetworkParams(orig_nw_layer);

    s_fc_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ActivationNode* canonical_ast::NodeFactory::newActivationNode(ActivationLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ActivationNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ActivationNode();
    d->captureNetworkParams(orig_nw_layer);

    s_act_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::PoolingNode* canonical_ast::NodeFactory::newPoolingNode(PoolingLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::PoolingNode* D;

    B b;
    D d;

    b = d = new canonical_ast::PoolingNode();
    d->captureNetworkParams(orig_nw_layer);

    s_pool_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::LRNNode* canonical_ast::NodeFactory::newLRNNode(LRNLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::LRNNode* D;

    B b;
    D d;

    b = d = new canonical_ast::LRNNode();
    d->captureNetworkParams(orig_nw_layer);

    s_lrn_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ScaleNode* canonical_ast::NodeFactory::newScaleNode(ScaleLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ScaleNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ScaleNode();
    d->captureNetworkParams(orig_nw_layer);

    s_scale_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::BatchNormNode* canonical_ast::NodeFactory::newBatchNormNode(BatchNormLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::BatchNormNode* D;

    B b;
    D d;

    b = d = new canonical_ast::BatchNormNode();
    d->captureNetworkParams(orig_nw_layer);

    s_bn_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::SoftMaxNode* canonical_ast::NodeFactory::newSoftMaxNode(SoftMaxLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::SoftMaxNode* D;

    B b;
    D d;

    b = d = new canonical_ast::SoftMaxNode();
    d->captureNetworkParams(orig_nw_layer);

    s_sm_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ConcatenationNode* canonical_ast::NodeFactory::newConcatNode(ConcatenationLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ConcatenationNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ConcatenationNode();
    d->captureNetworkParams(orig_nw_layer);

    s_concat_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::SplitNode* canonical_ast::NodeFactory::newSplitNode(SliceLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::SplitNode* D;

    B b;
    D d;

    b = d = new canonical_ast::SplitNode();
    d->captureNetworkParams(orig_nw_layer);

    s_split_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::DeconvolutionNode* canonical_ast::NodeFactory::newDeconvNode(DeconvolutionLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::DeconvolutionNode* D;

    B b ;
    D d;

    b = d = new canonical_ast::DeconvolutionNode();
    d->captureNetworkParams(orig_nw_layer);

    s_deconv_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ElementWiseNode* canonical_ast::NodeFactory::newEWNode(ElementWiseLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ElementWiseNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ElementWiseNode();
    d->captureNetworkParams(orig_nw_layer);

    s_ew_priv.insert(std::pair<B, D>(b, d));
    return d;
}

namespace canonical_ast
{

template <> canonical_ast::ConvolutionNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ConvolutionNode*>::iterator i = s_conv_priv.find(base);
    if ( i == s_conv_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::FullyConnectedNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::FullyConnectedNode*>::iterator i = s_fc_priv.find(base);
    if ( i == s_fc_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ActivationNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ActivationNode*>::iterator i = s_act_priv.find(base);
    if ( i == s_act_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::PoolingNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::PoolingNode*>::iterator i = s_pool_priv.find(base);
    if ( i == s_pool_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::LRNNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::LRNNode*>::iterator i = s_lrn_priv.find(base);
    if ( i == s_lrn_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ScaleNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ScaleNode*>::iterator i = s_scale_priv.find(base);
    if ( i == s_scale_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::BatchNormNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::BatchNormNode*>::iterator i = s_bn_priv.find(base);
    if ( i == s_bn_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::SoftMaxNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::SoftMaxNode*>::iterator i = s_sm_priv.find(base);
    if ( i == s_sm_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ConcatenationNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ConcatenationNode*>::iterator i = s_concat_priv.find(base);
    if ( i == s_concat_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::SplitNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::SplitNode*>::iterator i = s_split_priv.find(base);
    if ( i == s_split_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::DeconvolutionNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::DeconvolutionNode*>::iterator i = s_deconv_priv.find(base);
    if ( i == s_deconv_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ElementWiseNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ElementWiseNode*>::iterator i = s_ew_priv.find(base);
    if ( i == s_ew_priv.end() )
        return NULL;
    return i->second;
}


bool CanonicalParams::hasBiasTerm() const        { return false; }
void CanonicalParams::setHasBiasTerm(bool /*b*/) { }

}; // nvdla::priv::canonical_ast_interface

}; // nvdla::priv

}; // nvdla::
