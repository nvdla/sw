/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "priv/WeightTranslationUnit.h"

using std::map;
using std::string;
using std::endl;

namespace nvdla
{

namespace priv
{

// explicitly instantiate the priv maps of each node type
map<engine_ast::Node*, engine_ast::ConvCoreConvolutionOpNode*> engine_ast::NodeFactory::s_conv_conv_priv =
    map<engine_ast::Node*, engine_ast::ConvCoreConvolutionOpNode*>();

map<engine_ast::Node*, engine_ast::ConvCoreFullyConnectedOpNode*> engine_ast::NodeFactory::s_conv_fc_priv =
    map<engine_ast::Node*, engine_ast::ConvCoreFullyConnectedOpNode*>();

map<engine_ast::Node*, engine_ast::ConvCoreDeconvolutionOpNode*> engine_ast::NodeFactory::s_conv_deconv_priv =
    map<engine_ast::Node*, engine_ast::ConvCoreDeconvolutionOpNode*>();

map<engine_ast::Node*, engine_ast::SDPScaleOpNode*> engine_ast::NodeFactory::s_sdp_scale_priv =
    map<engine_ast::Node*, engine_ast::SDPScaleOpNode*>();

map<engine_ast::Node*, engine_ast::SDPBatchNormOpNode*> engine_ast::NodeFactory::s_sdp_bn_priv =
    map<engine_ast::Node*, engine_ast::SDPBatchNormOpNode*>();

map<engine_ast::Node*, engine_ast::SDPActivationOpNode*> engine_ast::NodeFactory::s_sdp_act_priv =
    map<engine_ast::Node*, engine_ast::SDPActivationOpNode*>();

map<engine_ast::Node*, engine_ast::SDPElementWiseOpNode*> engine_ast::NodeFactory::s_sdp_ew_priv =
    map<engine_ast::Node*, engine_ast::SDPElementWiseOpNode*>();

map<engine_ast::Node*, engine_ast::SDPBiasOpNode*> engine_ast::NodeFactory::s_sdp_bias_priv =
    map<engine_ast::Node*, engine_ast::SDPBiasOpNode*>();

map<engine_ast::Node*, engine_ast::SDPNOPNode*> engine_ast::NodeFactory::s_sdp_nop_priv =
    map<engine_ast::Node*, engine_ast::SDPNOPNode*>();

map<engine_ast::Node*, engine_ast::SDPSuperOpNode*> engine_ast::NodeFactory::s_sdp_super_priv =
    map<engine_ast::Node*, engine_ast::SDPSuperOpNode*>();

map<engine_ast::Node*, engine_ast::PDPNode*> engine_ast::NodeFactory::s_pdp_priv =
    map<engine_ast::Node*, engine_ast::PDPNode*>();

map<engine_ast::Node*, engine_ast::CDPLRNOpNode*> engine_ast::NodeFactory::s_cdp_lrn_priv =
    map<engine_ast::Node*, engine_ast::CDPLRNOpNode*>();

map<engine_ast::Node*, engine_ast::CPUScaleOpNode*> engine_ast::NodeFactory::s_cpu_scale_priv =
    map<engine_ast::Node*, engine_ast::CPUScaleOpNode*>();

map<engine_ast::Node*, engine_ast::CPUSoftMaxOpNode*> engine_ast::NodeFactory::s_cpu_sm_priv =
    map<engine_ast::Node*, engine_ast::CPUSoftMaxOpNode*>();

map<engine_ast::Node*, engine_ast::RubikNode*> engine_ast::NodeFactory::s_rubik_priv =
    map<engine_ast::Node*, engine_ast::RubikNode*>();

map<engine_ast::Node*, engine_ast::ConcatenationNode*> engine_ast::NodeFactory::s_concat_priv =
    map<engine_ast::Node*, engine_ast::ConcatenationNode*>();

map<engine_ast::Node*, engine_ast::SplitNode*> engine_ast::NodeFactory::s_split_priv =
    map<engine_ast::Node*, engine_ast::SplitNode*>();

map<engine_ast::Node*, engine_ast::BDMASingleDMAOpNode*> engine_ast::NodeFactory::s_single_bdma_priv =
    map<engine_ast::Node*, engine_ast::BDMASingleDMAOpNode*>();

map<engine_ast::Node*, engine_ast::BDMAGroupDMAOpNode*> engine_ast::NodeFactory::s_group_bdma_priv =
    map<engine_ast::Node*, engine_ast::BDMAGroupDMAOpNode*>();

map<engine_ast::Node*, engine_ast::MultiOpsNode*> engine_ast::NodeFactory::s_multi_ops_priv =
    map<engine_ast::Node*, engine_ast::MultiOpsNode*>();

void engine_ast::NodeFactory::clearMaps(void)
{
    s_conv_conv_priv.clear();
    s_conv_fc_priv.clear();
    s_conv_deconv_priv.clear();
    s_sdp_scale_priv.clear();
    s_sdp_bn_priv.clear();
    s_sdp_act_priv.clear();
    s_sdp_ew_priv.clear();
    s_sdp_bias_priv.clear();
    s_sdp_nop_priv.clear();
    s_sdp_super_priv.clear();
    s_pdp_priv.clear();
    s_cdp_lrn_priv.clear();
    s_cpu_scale_priv.clear();
    s_cpu_sm_priv.clear();
    s_rubik_priv.clear();
    s_concat_priv.clear();
    s_split_priv.clear();
    s_single_bdma_priv.clear();
    s_group_bdma_priv.clear();
    s_multi_ops_priv.clear();
}

//
//  All the variants of these sup-op node creators are
//  needed because the s_xx_yy_priv maps to lock them into
//  are different. can't templatize them
//
engine_ast::ConvCoreConvolutionOpNode* engine_ast::NodeFactory::newConvCoreConvolutionOpNode
(
    canonical_ast::ConvolutionNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::ConvCoreConvolutionOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::ConvCoreConvolutionOpNode(origCanNode, numBatches);
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    // determine op mode for the conv op: DC / WINOGRAD
    WeightTrns::WeightDims weightDims (dd->params().rawWeights().count,
                                       dd->params().weightDims().n,
                                       dd->params().weightDims().c,
                                       dd->params().weightDims().w,
                                       dd->params().weightDims().h,
                                       dd->params().stride().w,
                                       dd->params().stride().h);
    // fixme: disable winograd with group conv since group conv tends to bloat weight size
    // by a factor of 'inputC / auxC' which can be arbitrarily large to fit in CBUFF
    bool canWG          = engGraph->profile()->canWinograd();
    bool isWGPossible   = WeightTrns::isWGPossible(weightDims);
    bool isGroupConv    = dd->params().numGroups() > 1;
    bool isDilation     = dd->params().dilation() != Dims2(1,1);
    bool isInt8         = engGraph->profile()->computePrecision().v() ==
                          surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8;
    if ( canWG && isWGPossible && !isGroupConv && !isDilation && !isInt8 )
    {
        dd->setName(std::string("wg-conv-") + toString(s_conv_conv_priv.size()));
        dd->params().setConvMode(engine_ast::ConvolutionModeEnum::CONV_WINOGRAD);
    }
    else
    {
        dd->setName(std::string("dc-conv-") + toString(s_conv_conv_priv.size()));
        dd->params().setConvMode(engine_ast::ConvolutionModeEnum::CONV_DIRECT);
    }

    s_conv_conv_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::ConvCoreFullyConnectedOpNode* engine_ast::NodeFactory::newConvCoreFullyConnectedOpNode
(
    canonical_ast::FullyConnectedNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::ConvCoreFullyConnectedOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::ConvCoreFullyConnectedOpNode(origCanNode, numBatches);
    dd->setName(std::string("fc-") + toString(s_conv_fc_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    dd->params().setConvMode(engine_ast::ConvolutionModeEnum::CONV_DIRECT);

    s_conv_fc_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::ConvCoreDeconvolutionOpNode* engine_ast::NodeFactory::newConvCoreDeconvolutionOpNode
(
    canonical_ast::DeconvolutionNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::ConvCoreDeconvolutionOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::ConvCoreDeconvolutionOpNode(origCanNode, numBatches);
    dd->setName(std::string("deconv-") + toString(s_conv_deconv_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    dd->params().setConvMode(engine_ast::ConvolutionModeEnum::CONV_DIRECT);

    s_conv_deconv_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::SDPActivationOpNode* engine_ast::NodeFactory::newSDPActivationOpNode
(
    canonical_ast::ActivationNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPActivationOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::SDPActivationOpNode(origCanNode, numBatches);
    dd->setName(std::string("act-") + toString(s_sdp_act_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    engGraph->insertNode(b);

    if (origCanNode)
    {
        // those SDP-Act ops added in engine-land to circumvent DLA limitations or
        // to improve DLA performance may not have a canonical node equivalent
        dd->captureCanonicalParams();
    }

    s_sdp_act_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::SDPScaleOpNode* engine_ast::NodeFactory::newSDPScaleOpNode
(
    canonical_ast::ScaleNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPScaleOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::SDPScaleOpNode(origCanNode, numBatches);
    dd->setName(std::string("sdp-scale-") + toString(s_sdp_scale_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    engGraph->insertNode(b);

    if (origCanNode)
    {
        // those SDP-SCALE ops added in engine-land to circumvent DLA limitations or
        // to improve DLA performance may not have a canonical node equivalent
        dd->captureCanonicalParams();
    }

    s_sdp_scale_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::SDPBatchNormOpNode* engine_ast::NodeFactory::newSDPBatchNormOpNode
(
    canonical_ast::BatchNormNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPBatchNormOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::SDPBatchNormOpNode(origCanNode, numBatches);
    dd->setName(std::string("sdp-bn-") + toString(s_sdp_bn_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    if (origCanNode)
    {
        // those SDP-BN ops added in engine-land to circumvent DLA limitations or
        // to improve DLA performance may not have a canonical node equivalent
        dd->captureCanonicalParams();
    }
    engGraph->insertNode(b);

    s_sdp_bn_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::SDPElementWiseOpNode* engine_ast::NodeFactory::newSDPElementWiseOpNode
(
    canonical_ast::ElementWiseNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPElementWiseOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::SDPElementWiseOpNode(origCanNode, numBatches);
    dd->setName(std::string("ew-") + toString(s_sdp_ew_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    s_sdp_ew_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::SDPBiasOpNode* engine_ast::NodeFactory::newSDPBiasOpNode
(
    canonical_ast::Node* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPBiasOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();


    b = dd = new engine_ast::SDPBiasOpNode(origCanNode, numBatches);
    dd->setName(std::string("bias-") + toString(s_sdp_bias_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    engGraph->insertNode(b);

    if (origCanNode)
    {
        // those SDP-Bias ops added in engine-land to circumvent DLA limitations or
        // to improve DLA performance may not have a canonical node equivalent
        dd->captureCanonicalParams();
    }

    s_sdp_bias_priv.insert(std::pair<B, DD>(b, dd));

    return dd;
}

engine_ast::SDPNOPNode* engine_ast::NodeFactory::newSDPNOPNode
(
    canonical_ast::Node* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPNOPNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::SDPNOPNode(origCanNode, numBatches);
    dd->setName(std::string("sdp-nop-") + toString(s_sdp_nop_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    if (origCanNode)
    {
        dd->captureCanonicalParams();
    }

    s_sdp_nop_priv.insert(std::pair<B, DD>(b, dd));

    return dd;
}

engine_ast::SDPSuperOpNode* engine_ast::NodeFactory::newSDPSuperOpNode
(
    canonical_ast::Node* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SDPSuperOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::SDPSuperOpNode(origCanNode, numBatches);
    dd->setName(std::string("sdp-super-") + toString(s_sdp_super_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    engGraph->insertNode(b);

    if (origCanNode)
    {
        // There may not be a cannonical equivalent
        dd->captureCanonicalParams();
    }

    s_sdp_super_priv.insert(std::pair<B, DD>(b, dd));

    return dd;
}

engine_ast::PDPNode* engine_ast::NodeFactory::newPDPNode
(
    canonical_ast::PoolingNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::PDPNode* D;

    B b;
    D d;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = d = new engine_ast::PDPNode(origCanNode, numBatches);
    d->setName(std::string("pdp-") + toString(s_pdp_priv.size()));
    d->setId(engGraph->nextNodeId());
    d->setGraph(engGraph);
    d->captureCanonicalParams();
    engGraph->insertNode(b);

    s_pdp_priv.insert(std::pair<B, D>(b, d));
    return d;
}

engine_ast::CDPLRNOpNode* engine_ast::NodeFactory::newCDPLRNOpNode
(
    canonical_ast::LRNNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::CDPLRNOpNode* D;

    B b;
    D d;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = d = new engine_ast::CDPLRNOpNode(origCanNode, numBatches);
    d->setName(std::string("cdp-lrn-") + toString(s_cdp_lrn_priv.size()));
    d->setId(engGraph->nextNodeId());
    d->setGraph(engGraph);
    d->captureCanonicalParams();
    engGraph->insertNode(b);

    s_cdp_lrn_priv.insert(std::pair<B, D>(b, d));
    return d;
}

engine_ast::CPUScaleOpNode* engine_ast::NodeFactory::newCPUScaleOpNode
(
    canonical_ast::ScaleNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::CPUScaleOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::CPUScaleOpNode(origCanNode, numBatches);
    dd->setName(std::string("cpu-scale-") + toString(s_cpu_scale_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    s_cpu_scale_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::CPUSoftMaxOpNode* engine_ast::NodeFactory::newCPUSoftMaxOpNode
(
    canonical_ast::SoftMaxNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::CPUSoftMaxOpNode* DD;

    B b;
    DD dd;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = dd = new engine_ast::CPUSoftMaxOpNode(origCanNode, numBatches);
    dd->setName(std::string("cpu-sm-") + toString(s_cpu_sm_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    dd->captureCanonicalParams();
    engGraph->insertNode(b);

    s_cpu_sm_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::RubikNode* engine_ast::NodeFactory::newRubikNode
(
    canonical_ast::DeconvolutionNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::RubikNode* D;

    B b;
    D d;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = d = new engine_ast::RubikNode(origCanNode, numBatches);
    d->setName(std::string("rubik-") + toString(s_rubik_priv.size()));
    d->setId(engGraph->nextNodeId());
    d->setGraph(engGraph);
    d->captureCanonicalParams();
    engGraph->insertNode(b);

    s_rubik_priv.insert(std::pair<B, D>(b, d));
    return d;
}

engine_ast::ConcatenationNode* engine_ast::NodeFactory::newConcatNode
(
    canonical_ast::ConcatenationNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::ConcatenationNode* D;

    B b;
    D d;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = d = new engine_ast::ConcatenationNode(origCanNode, numBatches);
    d->setName(std::string("concat-") + toString(s_concat_priv.size()));
    d->setId(engGraph->nextNodeId());
    d->setGraph(engGraph);
    d->captureCanonicalParams();
    engGraph->insertNode(b);

    s_concat_priv.insert(std::pair<B, D>(b, d));
    return d;
}

engine_ast::SplitNode* engine_ast::NodeFactory::newSplitNode
(
    canonical_ast::SplitNode* origCanNode,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::SplitNode* D;

    B b;
    D d;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = d = new engine_ast::SplitNode(origCanNode, numBatches);
    d->setName(std::string("split-") + toString(s_split_priv.size()));
    d->setId(engGraph->nextNodeId());
    d->setGraph(engGraph);
    d->captureCanonicalParams();
    engGraph->insertNode(b);

    s_split_priv.insert(std::pair<B, D>(b, d));
    return d;
}

engine_ast::BDMASingleDMAOpNode* engine_ast::NodeFactory::newSingleBDMANode
(
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::BDMASingleDMAOpNode* DD;

    B b ;
    DD dd;

    b = dd = new engine_ast::BDMASingleDMAOpNode();
    dd->setName(std::string("single-bdma-") + toString(s_single_bdma_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    engGraph->insertNode(b);

    s_single_bdma_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::BDMAGroupDMAOpNode* engine_ast::NodeFactory::newGroupBDMANode
(
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::BDMAGroupDMAOpNode* DD;

    B b ;
    DD dd;

    b = dd = new engine_ast::BDMAGroupDMAOpNode();
    dd->setName(std::string("group-bdma-") + toString(s_group_bdma_priv.size()));
    dd->setId(engGraph->nextNodeId());
    dd->setGraph(engGraph);
    engGraph->insertNode(b);

    s_group_bdma_priv.insert(std::pair<B, DD>(b, dd));
    return dd;
}

engine_ast::MultiOpsNode* engine_ast::NodeFactory::newMultiOpsNode
(
    engine_ast::Graph::NodeSequence& groupedOps,
    engine_ast::Graph* engGraph
)
{
    typedef typename engine_ast::Node* B;
    typedef typename engine_ast::MultiOpsNode* D;

    B b;
    D d;
    NvU16 numBatches = engGraph->profile()->multiBatchSize();

    b = d = new engine_ast::MultiOpsNode(numBatches);
    d->setName(std::string("multi-ops-") + toString(s_multi_ops_priv.size()));
    d->setId(engGraph->nextNodeId());
    d->setGraph(engGraph);
    engGraph->insertNode(b);

    s_multi_ops_priv.insert(std::pair<B, D>(b,d));

    // embed nested graph inside the super node
    d->setNestedGraph(new NestedGraph());
    d->nestedGraph()->setContainingSuperNode(b);
    d->nestedGraph()->populateNestedGraph(groupedOps);
    d->setIsOnline(true);

    // mark input/output edges of the multi-op super node
    d->populateEdgePorts();

    return d;
}

namespace engine_ast
{

template <> engine_ast::ConvCoreConvolutionOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::ConvCoreConvolutionOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::ConvCoreConvolutionOpNode*>::iterator i = s_conv_conv_priv.find(base);
    if ( i == s_conv_conv_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::ConvCoreFullyConnectedOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::ConvCoreFullyConnectedOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::ConvCoreFullyConnectedOpNode*>::iterator i = s_conv_fc_priv.find(base);
    if ( i == s_conv_fc_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::ConvCoreDeconvolutionOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::ConvCoreDeconvolutionOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::ConvCoreDeconvolutionOpNode*>::iterator i = s_conv_deconv_priv.find(base);
    if ( i == s_conv_deconv_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}


template <> engine_ast::ConvCoreNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::ConvCoreNode* nc = NULL;
    if (!base)
        return nc;
    switch(base->engineOpType().v()) {
        case EngineOpTypeEnum::CONVOLUTION_CONV:   nc = nodeCast<engine_ast::ConvCoreConvolutionOpNode*>(base); break;
        case EngineOpTypeEnum::CONVOLUTION_DECONV: nc = nodeCast<engine_ast::ConvCoreDeconvolutionOpNode*>(base); break;
        case EngineOpTypeEnum::CONVOLUTION_FC:     nc = nodeCast<engine_ast::ConvCoreFullyConnectedOpNode*>(base); break;
        default: goto done;
    }

done:
    return nc;
}


template <> engine_ast::SDPActivationOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPActivationOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SDPActivationOpNode*>::iterator i = s_sdp_act_priv.find(base);
    if ( i == s_sdp_act_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SDPBiasOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPBiasOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SDPBiasOpNode*>::iterator i = s_sdp_bias_priv.find(base);
    if ( i == s_sdp_bias_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SDPNOPNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPNOPNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SDPNOPNode*>::iterator i = s_sdp_nop_priv.find(base);
    if ( i == s_sdp_nop_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SDPSuperOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPSuperOpNode* nc = NULL;
    map<engine_ast::Node*, engine_ast::SDPSuperOpNode*>::iterator i = s_sdp_super_priv.find(base);
    if ( i == s_sdp_super_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SDPScaleOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPScaleOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SDPScaleOpNode*>::iterator i = s_sdp_scale_priv.find(base);
    if ( i == s_sdp_scale_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SDPBatchNormOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPBatchNormOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SDPBatchNormOpNode*>::iterator i = s_sdp_bn_priv.find(base);
    if ( i == s_sdp_bn_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SDPElementWiseOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPElementWiseOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SDPElementWiseOpNode*>::iterator i = s_sdp_ew_priv.find(base);
    if ( i == s_sdp_ew_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}


template <> engine_ast::SDPNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SDPNode* nc = NULL;
    if (!base)
        return nc;
    switch(base->engineOpType().v()) {
        case EngineOpTypeEnum::SDP_ACTIVATION:  nc = nodeCast<engine_ast::SDPActivationOpNode*>(base); break;
        case EngineOpTypeEnum::SDP_BIAS:        nc = nodeCast<engine_ast::SDPBiasOpNode*>(base); break;
        case EngineOpTypeEnum::SDP_ELEMENTWISE: nc = nodeCast<engine_ast::SDPElementWiseOpNode*>(base); break;
        case EngineOpTypeEnum::SDP_NOP:         nc = nodeCast<engine_ast::SDPNOPNode*>(base); break;
        case EngineOpTypeEnum::SDP_SCALE:       nc = nodeCast<engine_ast::SDPScaleOpNode*>(base); break;
        case EngineOpTypeEnum::SDP_BATCH_NORM:  nc = nodeCast<engine_ast::SDPBatchNormOpNode*>(base); break;
        case EngineOpTypeEnum::SDP_SUPER:       nc = nodeCast<engine_ast::SDPSuperOpNode*>(base); break;
        default: goto done;
    }

done:
    return nc;
}

template <> engine_ast::PDPNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::PDPNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::PDPNode*>::iterator i = s_pdp_priv.find(base);
    if ( i == s_pdp_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::CDPLRNOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::CDPLRNOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::CDPLRNOpNode*>::iterator i = s_cdp_lrn_priv.find(base);
    if ( i == s_cdp_lrn_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::CPUScaleOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::CPUScaleOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::CPUScaleOpNode*>::iterator i = s_cpu_scale_priv.find(base);
    if ( i == s_cpu_scale_priv.end() )
        goto done;
    else
    nc = i->second;
done:
    return nc;
}

template <> engine_ast::CPUSoftMaxOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::CPUSoftMaxOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::CPUSoftMaxOpNode*>::iterator i = s_cpu_sm_priv.find(base);
    if ( i == s_cpu_sm_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::RubikNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::RubikNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::RubikNode*>::iterator i = s_rubik_priv.find(base);
    if ( i == s_rubik_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::ConcatenationNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::ConcatenationNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::ConcatenationNode*>::iterator i = s_concat_priv.find(base);
    if ( i == s_concat_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::SplitNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::SplitNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::SplitNode*>::iterator i = s_split_priv.find(base);
    if ( i == s_split_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::BDMASingleDMAOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::BDMASingleDMAOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::BDMASingleDMAOpNode*>::iterator i = s_single_bdma_priv.find(base);
    if ( i == s_single_bdma_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::BDMAGroupDMAOpNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::BDMAGroupDMAOpNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::BDMAGroupDMAOpNode*>::iterator i = s_group_bdma_priv.find(base);
    if ( i == s_group_bdma_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

template <> engine_ast::MultiOpsNode* NodeFactory::nodeCast(engine_ast::Node* base)
{
    engine_ast::MultiOpsNode* nc = NULL;
    if (!base)
        return nc;
    map<engine_ast::Node*, engine_ast::MultiOpsNode*>::iterator i = s_multi_ops_priv.find(base);
    if ( i == s_multi_ops_priv.end() )
        goto done;
    else
        nc = i->second;
done:
    return nc;
}

}; // nvdla::priv::engine_ast

}; // nvdla::priv
}; // nvdla::
