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

#include <memory>
#include <iostream>
#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
#include <fstream>
#endif
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>

#include "ErrorMacros.h"

#include "priv/Check.h"
#include "priv/Network.h"
#include "priv/Profiler.h"
#include "priv/Profile.h"
#include "priv/Compiler.h"
#include "priv/Loadable.h"
#include "priv/Wisdom.h"
#include "priv/TargetConfig.h"

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
#include "priv/DlaPrototestInterface.pb.h"
#include <google/protobuf/text_format.h>
#endif

#include "priv/DLAInterface.h"


#include "priv/loadable_generated.h"


using std::string;
using std::endl;
using std::set;
using std::vector;
using std::list;
using std::map;
using std::endl;

namespace nvdla
{

ICompiler::ICompiler() { }
ICompiler::~ICompiler() { }

#if 0
ICompiler *createCompiler()
{
    priv::CompilerFactory::CompilerPrivPair p = priv::CompilerFactory::newCompiler();
    return p.i();
}
#endif


namespace priv
{


class DumpCanonicalGraphJson : public DumpGraphBase, public ast::ScoredGraphVisitor<canonical_ast::Graph>
{
public:
    DumpCanonicalGraphJson() : DumpGraphBase("canonical_graph.json", "canonical_graph"),
                             ast::ScoredGraphVisitor<canonical_ast::Graph>() { }

    virtual ~DumpCanonicalGraphJson() { }
    virtual NvDlaError visitBegin(canonical_ast::Graph *, ast::GraphScoreboard<canonical_ast::Graph> &)
    {
        out().open(_m_filename, std::ios::out | std::ios::trunc);
        out() << "{ \"classname\":\"graph\", \"id\":\"" << _m_graph_id <<
            "\", \"filename\":\"" << _m_filename <<
            "\", \"elements\": [";
        _m_delim = "\n";
        return NvDlaSuccess;
    }
    virtual NvDlaError visitElem(canonical_ast::Graph::Elem elem, ast::GraphScoreboard<canonical_ast::Graph>::Score score)
    {
        canonical_ast::Node *node = elem.first;
        canonical_ast::Edge *edge = elem.second;
        if ( node )
        {
            out() << _m_delim; _m_delim = "\n,";
            out() << "{\"class\":\"node\", \"id\" : \"" << node->id() <<
                "\",\"className\":\"" << node->className() <<

                "\",\"score\":{ \"state\": \"" << int(score.state())  << "\"" <<
                ", \"dis_time\": " << score.discoveryTime() << " " <<
                ", \"fab_time\": " << score.discoveryTime() << " " <<
                ", \"fin_time\": " << score.discoveryTime() << " " <<

                " }"; // close score
            out () << "}";

        }
        else if ( edge )
        {
            canonical_ast::Graph *g = edge->graph();
            out() << _m_delim; _m_delim = "\n,";
            string delim("");
            canonical_ast::NodeSequence srcs = g->upstreamNodes(edge);
            canonical_ast::NodeSequence tgts = g->downstreamNodes(edge);
            // note: the (void*) cast hack is to be certain the ids given are unique.
            // nodes already had a property like that.  but edges didn't.
            //out() << "{\"class\":\"edge\", \"id\" : \"e-" << std::hex << (void*)edge << std::dec << "\"" <<
            out() << "{\"class\":\"edge\"" <<
                ", \"id\":\"" << edge->id()<< "\""<<
                // ", \"type\":\"" << edge->edgeType().c_str() << "\"" <<
                ", ";
            out() << "\"sources\":[";

            for ( canonical_ast::NodeSequence::const_iterator si = srcs.begin(); si != srcs.end(); ++si)
            {
                out() << delim << "\"" << (*si)->id() << "\""; delim = ", ";
            }
            delim="";
            out() << "], \"targets\":[";

            for ( canonical_ast::NodeSequence::const_iterator ti = tgts.begin(); ti != tgts.end(); ++ti)
            {
                out() << delim << "\"" << (*ti)->id() << "\""; delim = ", ";
            }
            out() << "], ";

            out () << "\"score\":{ \"state\": \"" << int(score.state())  << "\"" <<
                ", \"dis_time\": " << score.discoveryTime() << " " <<
                ", \"fab_time\": " << score.discoveryTime() << " " <<
                ", \"fin_time\": " << score.discoveryTime() << " " <<
                " }"; // close score

            out() << "}"; // close edge
        }

        return NvDlaSuccess;
    }
    virtual NvDlaError visitNode(canonical_ast::Node *, ast::GraphScoreboard<canonical_ast::Graph>::Score)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitEdge(canonical_ast::Edge *, ast::GraphScoreboard<canonical_ast::Graph>::Score)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitEnd(canonical_ast::Graph *, ast::GraphScoreboard<canonical_ast::Graph> &, NvDlaError ve)
    {
        out() << "]\n}\n";
        out().close();
        return ve;
    }
};


class DumpEngineGraphJson : public DumpGraphBase, public ast::ScoredGraphVisitor<engine_ast::Graph>
{
public:
    DumpEngineGraphJson() : DumpGraphBase("engine_graph.json", "engine_graph"),
                        ast::ScoredGraphVisitor<engine_ast::Graph>() { }

    virtual ~DumpEngineGraphJson() { }

    virtual NvDlaError visitBegin(engine_ast::Graph *, ast::GraphScoreboard<engine_ast::Graph> &)
    {
        out().open(_m_filename, std::ios::out | std::ios::trunc);
        out() << "{ \"classname\":\"graph\", \"id\":\"" << _m_graph_id <<
            "\", \"filename\":\"" << _m_filename <<
            "\", \"elements\": [";
        _m_delim = "\n";
        return NvDlaSuccess;
    }
    virtual NvDlaError visitElem(engine_ast::Graph::Elem elem, ast::GraphScoreboard<engine_ast::Graph>::Score score)
    {
        engine_ast::Node *node = elem.first;
        engine_ast::Edge *edge = elem.second;
        if ( node )
        {
            out() << _m_delim; _m_delim = "\n,";
            out() << "{\"class\":\"node\", \"id\" : \"" << node->name() <<
                "\",\"name\":\"" << node->id() <<
                "\",\"className\":\"" << node->className() <<

                "\",\"score\":{ \"state\": \"" << int(score.state())  << "\"" <<
                ", \"dis_time\": " << score.discoveryTime() << " " <<
                ", \"fab_time\": " << score.discoveryTime() << " " <<
                ", \"fin_time\": " << score.discoveryTime() << " " <<

                " }"; // close score
            out () << "}";

        }
        else if ( edge )
        {
            engine_ast::Graph *g = edge->graph();
            out() << _m_delim; _m_delim = "\n,";
            string delim("");
            engine_ast::NodeSequence srcs = g->upstreamNodes(edge);
            engine_ast::NodeSequence tgts = g->downstreamNodes(edge);
            // note: the (void*) cast hack is to be certain the ids given are unique.
            // nodes already had a property like that.  but edges didn't.
            //out() << "{\"class\":\"edge\", \"id\" : \"e-" << std::hex << (void*)edge << std::dec <<
            out() << "{\"class\":\"edge\"" <<
                ", \"id\":\"" << edge->id()<<
                "\", \"type\":\"" << edge->edgeType().c_str() << "\", ";
            out() << "\"sources\":[";

            for ( engine_ast::NodeSequence::const_iterator si = srcs.begin(); si != srcs.end(); ++si)
            {
                out() << delim << "\"" << (*si)->name() << "\""; delim = ", ";
            }
            delim="";
            out() << "], \"targets\":[";

            for ( engine_ast::NodeSequence::const_iterator ti = tgts.begin(); ti != tgts.end(); ++ti)
            {
                out() << delim << "\"" << (*ti)->name() << "\""; delim = ", ";
            }
            out() << "], ";

            out () << "\"score\":{ \"state\": \"" << int(score.state())  << "\"" <<
                ", \"dis_time\": " << score.discoveryTime() << " " <<
                ", \"fab_time\": " << score.discoveryTime() << " " <<
                ", \"fin_time\": " << score.discoveryTime() << " " <<
                " }"; // close score

            out() << "}"; // close edge
        }
        return NvDlaSuccess;
    }
    virtual NvDlaError visitNode(Node*, ast::GraphScoreboard<engine_ast::Graph>::Score)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitEdge(Edge*, ast::GraphScoreboard<engine_ast::Graph>::Score)
    {
        return NvDlaError_InvalidState;
    }
    virtual NvDlaError visitEnd(engine_ast::Graph *, ast::GraphScoreboard<engine_ast::Graph> &, NvDlaError ve)
    {
        out() << "]\n}\n";
        out().close();
        return ve;
    }
};



CompilerFactory::CompilerPrivPair CompilerFactory::newCompiler()
{
    ICompiler *compiler;
    Compiler *compiler_priv;
    compiler = compiler_priv = new priv::Compiler();
    if (compiler) {
        s_priv.insert(compiler, compiler_priv);
        s_self.insert(compiler, compiler);
    }
    return CompilerPrivPair(compiler, compiler_priv);
}

NvDlaError CompilerFactory::deleteCompiler(ICompiler *compiler)
{
    if (compiler != NULL) {
        Compiler *compiler_priv = priv(compiler);
        if (compiler_priv != NULL) {
            delete compiler_priv;
        }

        s_priv.remove(compiler);
        s_self.remove(compiler);
    }

    return NvDlaSuccess;
}

Compiler *CompilerFactory::priv(ICompiler *compiler)
{
    BiMap<ICompiler *, Compiler *>::left_iterator f = s_priv.find_left(compiler);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ICompiler *CompilerFactory::i(Compiler *compiler)
{
    BiMap<ICompiler *, Compiler *>::right_iterator f = s_priv.find_right(compiler);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

ICompiler *CompilerFactory::self(void *s)
{
    BiMap<void *, ICompiler *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}


BiMap<ICompiler *, Compiler*> CompilerFactory::s_priv;
BiMap<void *, ICompiler*> CompilerFactory::s_self;


Compiler::Compiler() : ICompiler(), m_wisdom(0)
{
}

Compiler::~Compiler()
{
    engine_ast::NodeFactory::clearMaps();

    // Clear all mallocs made, so far but not freed
    engine_ast::MemoryCollector::getInstance()->freeRemainingMemories();
}


IWisdom *Compiler::wisdom() const
{
    return (IWisdom*)m_wisdom; // Wisdom->IWisdom ok, but tbd: hook up WisdomFactory...
}

NvU16 Compiler::getFactoryType() const
{
    return 0; // only one kind so far
}

NvDlaError Compiler::compileCheck(const char *tp_name, const char *target_config_name)
{
    NvDlaError e = NvDlaSuccess;
    CATCH_PROPAGATE_ERROR_FAIL(
        compileInternal(tp_name, target_config_name, nullptr, false /*check compile only*/)
    );

fail:
    return e;
}

NvDlaError Compiler::compile(const char *tp_name, const char *target_config_name, ILoadable **peli)
{
    NvDlaError e = NvDlaSuccess;
    CATCH_PROPAGATE_ERROR_FAIL(
        compileInternal(tp_name, target_config_name, peli, true /*full compile*/)
    );

fail:
    return e;
}

NvDlaError Compiler::compileInternal(const char *tp_name, const char *target_config_name, ILoadable **peli, bool fullCompile)
{
    NvDlaError e = NvDlaSuccess;
    DLAInterface *target_dla = 0;
    DLAInterface *dla_if = 0;
    Profiler *profiler = 0;
    ProfileFactory::ProfilePrivPair p_profile;
    Profile *profile = 0;
    TargetConfig *target_config = 0;
    vector<engine_ast::Graph *> g;
    NVDLA_UNUSED(target_dla);
    NVDLA_UNUSED(dla_if);


    LoadableFactory::LoadablePrivPair l(0, 0);

    // && == with fullCompile or otherwise ok during compileCheck?

    DumpCanonicalGraphJson dump_can;
    DumpEngineGraphJson dump_eng;

    if ( !m_wisdom )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No wisdom available.");
    }

    profiler = ProfilerFactory::priv(m_wisdom->getProfiler());
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No profiler available.");
    }

    profile = ProfileFactory::priv(profiler->getProfile(tp_name));
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find profile to compile.");
    }

    target_config = TargetConfigFactory::priv(profiler->getTargetConfig(target_config_name));
    if ( !target_config )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find target config to compile.");
    }


    PROPAGATE_ERROR_FAIL( compileInternal(profile, target_config, peli, fullCompile) );
fail:
    return e;

}


NvDlaError Compiler::compileInternal(Profile *profile, TargetConfig *target_config, ILoadable **peli, bool fullCompile)
{
    NvDlaError e = NvDlaSuccess;
    DLAInterface *target_dla = 0;
    DLAInterface *dla_if = 0;
    ProfileFactory::ProfilePrivPair p_profile;
    Network *net = 0;
    vector<engine_ast::Graph *> g;
    engine_ast::Graph *final_g = 0;
    bool ok = true, done = false;
    NVDLA_UNUSED(target_dla);
    NVDLA_UNUSED(dla_if);

    canonical_ast::Graph *can_g = NULL;

    LoadableFactory::LoadablePrivPair l(0, 0);

    // && == with fullCompile or otherwise ok during compileCheck?
    bool dumpCanonicalGraph = debugGraphs();
    bool dumpPreCopyOutGraph = debugGraphs();
    bool dumpEmittedGraph = debugGraphs();
    bool dumpPassGraph = debugGraphs();

    DumpCanonicalGraphJson dump_can;
    DumpEngineGraphJson dump_eng;

    net = NetworkFactory::priv(m_wisdom->getNetwork());
    if ( !net )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "No network to compile.");
    }

    if ( debugProfile() )
    {
        gLogInfo << "Compiler profile: " << profile->getName() << endl;
        gLogInfo << "\tcompute precision " << profile->computePrecision().c_str() << endl;
        gLogInfo << "\tnetwork input data format " << profile->networkInputDataFormat().c_str() << endl;
        gLogInfo << "\tnetwork input surface format " << profile->networkInputSurfaceFormat().c_str() << endl;
        gLogInfo << "\tnetwork output data format " << profile->networkOutputDataFormat().c_str() << endl;
        gLogInfo << "\tnetwork output surface format " << profile->networkOutputSurfaceFormat().c_str() << endl;
        gLogInfo << "\tbatch size " << profile->multiBatchSize() << endl;
        gLogInfo << "\ttensor scaling mode " << profile->tensorScalingMode().c_str() << endl;
        gLogInfo << "\tquantization mode " << profile->quantizationMode().c_str() << endl;
    }

    can_g = canonical_ast::generateGraph(net);
    if ( !can_g )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "failed to create canonical graph");
    }

    if ( dumpCanonicalGraph )
    {
        dump_can.setGraphId("canonical_graph");
        dump_can.setFilename("can_g.json");
        PROPAGATE_ERROR_FAIL( dump_can.visitElems( can_g->scoredOrdering()) );
    }


    g.push_back( engine_ast::generateGraph(profile, target_config, can_g) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: generateGraph");
    }

    if ( dumpEmittedGraph )
    {
        engine_ast::Graph::printGraph(g.back(), true, "engine_ast::generateGraph");
        dump_eng.setGraphId("engine_graph");
        dump_eng.setFilename("engine_g.json");
        PROPAGATE_ERROR_FAIL( dump_eng.visitElems( g.back()->scoredOrdering()) );
    }

    g.push_back( registerBuffers(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: registerBuffers");
    }

    g.push_back( preProcessAuxData(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compulation phase: preProcessAuxData");
    }

    g.push_back( mergeActivationOperations(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: mergeActivationOperations");
    }

    g.push_back( updateScalingFactors(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: updateScalingFactors");
    }

    g.push_back( quantizeAuxData(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: quantizeAuxData");
    }

    g.push_back( fuseOnTheFlyNodes(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: fuseOnTheFlyNodes");
    }

    g.push_back( handleLowPrecisionConversions(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: handleLowPrecisionConversions");
    }

    g.push_back( translateAuxData(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: translateAuxData");
    }

    g.push_back( reserveBuffers(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: reserveBuffers");
    }

    /*
    g.push_back( groupAtomicOperations(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: groupAtomicOperations");
    }
    */

    g.push_back( splitNodes(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: splitNodes");
    }

    g.push_back( fuseSubEngineOps(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: fuseSubEngineOps");
    }

    g.push_back( boundGraph(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: boundGraph");
    }

    g.push_back( handleMultiBatch(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: handleMultiBatch");
    }

    /*
    g.push_back( flattenGraph(g.back()) );
    if ( !g.back() )
    {
        PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: flattenGraph");
    }
    */

    if ( dumpEmittedGraph )
    {
        dump_eng.setGraphId("engine_1");
        dump_eng.setFilename("engine_1.json");
        PROPAGATE_ERROR_FAIL( dump_eng.visitElems(g.back()->scoredOrdering()));
    }

    if ( profile->copyOutDebugSurfaces() )
    {
        if ( dumpPreCopyOutGraph )
        {
            // PROPAGATE_ERROR_FAIL( dumpPre.visitElems( g.back()->scoredOrdering()) );
            // g.back()->dumpGraphJson("eng_pre_debug_copy.json", "eng_pre_debug_copy");
        }

        g.push_back( enableCopyOutDebugSurfaces( g.back() ) );

        if ( !g.back() )
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: enableCopyOutDebugSurfaces");
        }

        if ( dumpPreCopyOutGraph )
        {
            dump_eng.setGraphId("engine_2");
            dump_eng.setFilename("engine_2.json");
            PROPAGATE_ERROR_FAIL( dump_eng.visitElems( g.back()->scoredOrdering()) );
        }
    }

    //
    // this is where operation order and memory placement are finalized.
    // make passes until convergence or failure.
    //
    for ( int pass = 0; !done; ++pass )
    {
        if ( dumpPassGraph )
        {
            std::stringstream ss; ss << "eng_pre_pass_" << pass;
            //            g.back()->dumpGraphJson(ss.str() + ".json", ss.str());
        }

        engine_ast::NodeSequence topological_order;

        g.push_back( generateDependencyParams(g.back(), topological_order) );
        if ( !g.back() )
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: generateDependencyGraphState pass=%d", pass);
        }

        g.push_back( resolveMemory(g.back(), topological_order) );
        if ( !g.back() )
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_InvalidState, "failed compilation phase: resolveMemory pass=%d", pass);
        }


        if ( dumpPassGraph )
        {
            std::stringstream ss; ss << "eng_post_pass_" << pass;
            //            g.back()->dumpGraphJson(ss.str() + ".json", ss.str());
        }

        done = !g.back()->dirty();
        ASSERT( done ); // shouldn't have hazards
    }

    final_g = g.back();


    if ( fullCompile )
    {

        if ( dumpEmittedGraph )
        {
        engine_ast::Graph::printGraph(final_g, true, "Final");
            dump_eng.setGraphId("emit");
            dump_eng.setFilename("emit.json");
            PROPAGATE_ERROR_FAIL( dump_eng.visitElems(final_g->scoredOrdering()) );
        }

        PROPAGATE_ERROR_FAIL( emit(final_g, l) );
        ok = l.priv() != 0;

        if ( ok )
        {
            //
            // this version hands back to the active profile with only the name of the profile
            // for look up later.  this creates the "same name as the profile" loadable.
            //
            m_wisdom->insertProfileSymbol( ProfileFactory::i(profile), profile->getName());
            profile->insertLoadable( std::string(profile->getName()), -1, l.i() );

            // build flatbuffer and save it internally
            (void)l.priv()->serialize();

            if ( peli )
            {
                *peli = l.i();
            }
        }
    }

fail:
    //
    // if we start actually cloning graphs then
    // there will be many more here than final_g.
    //
    if ( final_g )
    {
        delete final_g;
    }
    if ( can_g )
    {
        delete can_g;
    }
    return e;
}


NvDlaError Compiler::getLoadableFromWisdom(const char *tp_name,
                                ILoadable **i_loadable)
{
    NvDlaError e = NvDlaSuccess;
    Profiler *profiler = NULL;
    Profile *profile = NULL;

    if (i_loadable == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL loadable");
    }

    if ( !m_wisdom )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No wisdom available.");
    }

    profiler = ProfilerFactory::priv(m_wisdom->getProfiler());
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No profiler available.");
    }

    profile = ProfileFactory::priv(profiler->getProfile(tp_name));
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                            "Couldn't find profile to compile.");
    }

    PROPAGATE_ERROR_FAIL(profile->getLoadable(tp_name, -1, i_loadable));
    if (i_loadable == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                            "Invalid loadable.");
    }


fail:
    return e;
}

NvDlaError Compiler::getLoadableImageSize(const char *tp_name,
                                    NvU64 *size)
{
    NvDlaError e = NvDlaSuccess;
    CATCH_PROPAGATE_ERROR_FAIL(getLoadableImageSizeInternal(tp_name, size));

fail:
    return e;
}

NvDlaError Compiler::getLoadableImageSizeInternal(const char *tp_name,
                                    NvU64 *size)
{
    NvDlaError e = NvDlaSuccess;
    ILoadable *i_loadable = NULL;
    Loadable *loadable_priv = NULL;

    if (size == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL size");
    }

    PROPAGATE_ERROR_FAIL(getLoadableFromWisdom(tp_name, &i_loadable));
    loadable_priv = LoadableFactory::priv(i_loadable);

    PROPAGATE_ERROR_FAIL(loadable_priv->getSerializedDataSize(size));

fail:
    return e;
}


NvDlaError Compiler::getLoadableImage(const char *tp_name,
                                    NvU8 *flatbuf)
{
    NvDlaError e = NvDlaSuccess;
    CATCH_PROPAGATE_ERROR_FAIL(getLoadableImageInternal(tp_name, flatbuf));

fail:
    return e;
}

NvDlaError Compiler::getLoadableImageInternal(const char *tp_name,
                                    NvU8 *flatbuf)
{
    NvDlaError e = NvDlaSuccess;
    ILoadable *i_loadable = NULL;
    Loadable *loadable_priv = NULL;

    if (flatbuf == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No buffer allocated");
    }

    PROPAGATE_ERROR_FAIL(getLoadableFromWisdom(tp_name, &i_loadable));
    loadable_priv = LoadableFactory::priv(i_loadable);

    PROPAGATE_ERROR_FAIL(loadable_priv->getSerializedData(flatbuf));

fail:
    return e;
}

/*

  each DLA task's address list contains two sorts of entries:
      . those used by emitted instructions
      . those which define the task's context (addr0, buffer descriptors, etc).

  task address id 0 is special and must be the network descriptor/addr0.

  we force instructions use address ids 1 -> Ni (whatever we determine 'Ni' is).

  name the task's context entries addr0, addr1, addr2, etc...
  then arrange such that tasks' address lists have the following form:

      taskAddrList[0]          -> task network desc/addr0
      taskAddrList[1]          -> globalAddrList[1]
      taskAddrList[Ni]         -> globalAddrList[Ni]
      taskAddrList[(Ni+1) + 0] -> dep list/addr1
      taskAddrList[(Ni+1) + 1] -> op desc/addr2
      taskAddrList[(Ni+1) + 2] -> surf list/addr3
      taskAddrList[(Ni+1) + 3] -> lut/addr4
      taskAddrList[(Ni+1) + 4] -> dummy/addr5

  read these like:
      . "the task's address list entry 0 refers to the global address list entry for it's addr0."
      . "the task's address list entry 1 refers to global address list entry 1."
      . "the task's address list entry at (Ni+1)+2 refers to the global address list entry for the task's surface list."


  rewriting for a general task 'i', parmeterizing naming and simplifying indexing:

      taskAddrList(i)[0]       -> globalAddrList[ taskContextAddr(i, 0) ]  ; addr0
      taskAddrList(i)[1]       -> globalAddrList[ 1 ];                     ; 1
      taskAddrList(i)[Ni]      -> globalAddrList[ Ni ];                    ; In
      taskAddrList(i)[Ni + 1 ] -> globalAddrList[ taskContextAddr(i, 1) ]  ; addr1
      taskAddrList(i)[Ni + 2 ] -> globalAddrList[ taskContextAddr(i, 2) ]  ; addr2
      taskAddrList(i)[Ni + 3 ] -> globalAddrList[ taskContextAddr(i, 3) ]  ; addr3
      taskAddrList(i)[Ni + 4 ] -> globalAddrList[ taskContextAddr(i, 4) ]  ; addr4
      taskAddrList(i)[Ni + 5 ] -> globalAddrList[ taskContextAddr(i, 5) ]  ; addr5

   the left hand side refers to the entries for any given task 'i' and the right hand side makes use of
   'taskContextAddr' which hasn't been defined yet but returns the relevant index into the global address list.


   the "global address list" from which the tasks' address lists are derived is of the following form:

      globalAddrList[0] -> dummy page/zero hole

      globalAddrList[1]  -> addr id 1 for all emitted instructions
      globalAddrList[Ni] -> last instruction addr id

      ; task-0 context items (assume DLA here and below, with EMU similar)
      globalAddrList[(Ni+1) +0 ]  -> task_0 addr0
      globalAddrList[(Ni+1) +1 ]  -> task_0 dep list
      globalAddrList[(Ni+1) +2 ]  -> task_0 surf desc
      globalAddrList[(Ni+1) +3 ]  -> task_0 lut desc
      globalAddrList[(Ni+1) +4 ]  -> task_0 last context item
      globaladdrList[(Ni+1) +5 ]  -> task_0 dummy
      ...

      ; task-i context
      global_addr_list[(n+1)+(i*6)+0] ; -> task_i addr0
                             ...
      global_addr_list[(n+1)+(i*6)+5] ; -> task_+i last item


   note: the per-task context items get separated (non-contig) into addr0, addrs and then
   everything else when stuffed into the individual task address lists.  because of that the
   offsets to the context items jump around between global and task lists.

   there's potential for many more address list entries than memory list entries.
   depending upon how the address list entries are coalesced (or not) during instruction emit
   they can also be almost 1:1 (if heavily pooled, say).

 */


class TaskAddressList
{
public:
    // static const int16_t addr0_global_task_offs        = 0;
    // static const int16_t dep_graph_global_task_offs    = 1;
    // static const int16_t op_list_global_task_offs      = 2;
    // static const int16_t surf_list_global_task_offs    = 3;
    // static const int16_t lut_list_global_task_offs     = 4;
    // static const int16_t dummy_global_task_offs        = 5;

    static int16_t numContextAddrs() { return 6; } // num_global_task_offs          = 6;
};

class GlobalAddressList
{
public:
    GlobalAddressList() : m_numInstrMemEntries(0), m_numInstrAddrEntries(0), m_numTasks(0) { }

    GlobalAddressList(const GlobalAddressList &o) :
        m_numInstrMemEntries(o.m_numInstrMemEntries), m_numInstrAddrEntries(o.m_numInstrAddrEntries),
        m_memListEntries(o.m_memListEntries), m_addrListEntries(o.m_addrListEntries),
        m_numTasks(o.m_numTasks) {  }

    GlobalAddressList(size_t numTasks,
                      const std::vector<Loadable::MemoryListEntry> &instrMemEntries,
                      const std::vector<Loadable::AddressListEntry> &instrAddrEntries)
    {
        //
        // create mem id and address id entries for the dead page at address id == 0.
        //
        Loadable::MemoryListEntry  memZeroEntry(0, 4096, 4096, Loadable::MemoryListEntry::domain_sysmem(), Loadable::MemoryListEntry::flags_alloc());
        Loadable::AddressListEntry addrZeroEntry(0, memZeroEntry.id, memZeroEntry.size);

        m_memListEntries.push_back(memZeroEntry);
        m_addrListEntries.push_back(addrZeroEntry);

        m_memListEntries.insert (m_memListEntries.end(),  instrMemEntries.begin(),  instrMemEntries.end());
        m_addrListEntries.insert(m_addrListEntries.end(), instrAddrEntries.begin(), instrAddrEntries.end());

        m_numInstrMemEntries  = instrMemEntries.size();
        m_numInstrAddrEntries = instrAddrEntries.size();

        m_numTasks = numTasks;

        m_addrListEntries.resize( 1 /*the zero entry*/ + m_numInstrAddrEntries + (TaskAddressList::numContextAddrs() * m_numTasks),
                                  addrZeroEntry /* fill with references to the zero mem entry */);

    }

    NvS16 taskContextBegin(NvS16 taskId) const
    {
        // skip 0, then 1..n is 1..numInstrAddrEntries
        return (m_numInstrAddrEntries + 1) + (taskId * TaskAddressList::numContextAddrs());
    }

    NvS16 taskContextEnd(NvS16 taskId) const // ala iterator, points to one-beyond
    {
        return taskContextBegin(taskId) + TaskAddressList::numContextAddrs();
    }

#if 0
    size_t numAddrEntries() const
    {
        return  m_addrListEntries.size();
    }
#endif
    size_t numInstrMemEntries() const { return m_numInstrMemEntries; }
    size_t numInstrAddrEntries() const { return m_numInstrAddrEntries; }

    vector<ILoadable::MemoryListEntry> &memList() { return m_memListEntries; }
    vector<ILoadable::AddressListEntry> &addrList() { return m_addrListEntries; }

protected:
    size_t m_numInstrMemEntries;
    size_t m_numInstrAddrEntries;
    vector<ILoadable::MemoryListEntry> m_memListEntries;
    vector<ILoadable::AddressListEntry> m_addrListEntries;
    size_t m_numTasks;

};


NvDlaError Compiler::emit(engine_ast::Graph * g, LoadableFactory::LoadablePrivPair &l)
{
    NvDlaError e = NvDlaSuccess;
    DLAInterface *dla_if = 0;
    EMUInterface *emu_if = 0;

    Loadable *loadable = 0;

    vector< engine_ast::Node* >::iterator anni;

    bool isEMU;
    vector< vector<engine_ast::Node* >::iterator > task_starting_points;
    vector< size_t > task_slot_counts;
    vector< NvS16  > task_ids;
    NVDLA_UNUSED(isEMU);

    GlobalAddressList gal;
    size_t Ni;

    vector<ILoadable::TaskListEntry> task_list_entries;
    size_t num_tasks;

    vector<ILoadable::SubmitListEntry> submit_list_entries;

    vector<ILoadable::EventListEntry> event_list_entries;

    ILoadable::EventListEntry event;
    NvU16 event_id;
    NVDLA_UNUSED(event_id);

    vector<ILoadable::RelocEntry> reloc_entries;

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::fstream protobufFile ("output.protobuf", std::ios::out | std::ios::trunc);
    nvdla_prototest_interface::Test protoTest;
    nvdla_prototest_interface::TestInfo* protoTestInfo = protoTest.mutable_test();
    protoTestInfo->set_allocated_event_list(0);
#endif

    l = LoadableFactory::LoadablePrivPair(0, 0);

    // if nothing else make sure the right version
    // is being targeted, most of the struct references will be
    // offsets and the like of where to put things... might be
    // inline definitions and such to bit ops, etc.

    dla_if = getTargetDLAInterface(NULL);
    emu_if = new EMUInterfaceA();

    if ( debugVersions() )
    {
        gLogInfo << "compiler targeting dla (fw) interface " <<
            (int)dla_if->firmwareTargetVersionMajor() << "." <<
            (int)dla_if->firmwareTargetVersionMinor() << endl;

        gLogInfo << "compiler targeting emu (cpu) interface " <<
            (int)emu_if->emulatorTargetVersionMajor() << "." <<
            (int)emu_if->emulatorTargetVersionMinor() << endl;
    }

    l = LoadableFactory::newLoadable();
    if ( !l )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Error allocating new loadable");
    }
    loadable = l.priv();

    //
    // begin building execution context and placing into the loadable
    //
    g->resetRelocEntries();

    PROPAGATE_ERROR_FAIL( g->prepareMemoryListEntries(loadable) );

    task_slot_counts.resize(g->graphlets().size());

    for (vector<engine_ast::Graph::Graphlet *>::iterator gli = g->graphlets().begin(); gli != g->graphlets().end(); ++gli)
    {
        engine_ast::Graph::Graphlet *graphlet = *gli;
        NvS16 taskId;
        engine_ast::Node *first_node;
        NVDLA_UNUSED(taskId);

        first_node = *graphlet->nodeList().begin();
        task_starting_points.push_back(graphlet->nodeList().begin());
        task_ids.push_back(first_node->taskId());
        task_slot_counts[task_starting_points.size() - 1] = graphlet->nodeList().size();
    }
    num_tasks = task_starting_points.size();

    gal = GlobalAddressList(num_tasks, loadable->getMemoryListEntries(), loadable->getAddressListEntries());
    Ni = gal.numInstrAddrEntries();

    if ( debugTasks() || debugMemoryLayout() )
    {
        gLogInfo << __func__ << " discovered " << num_tasks << " tasks" << endl;
        gLogInfo << "the initial mem list size is  " << gal.numInstrMemEntries() << " entries" << endl;
        gLogInfo << "the initial addr list size is " << gal.numInstrAddrEntries() << " entries" << endl;
    }


#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    protoTestInfo->set_num_tasks(num_tasks);
    protoTestInfo->set_num_buffers(gal.numInstrMemEntries());
#endif

    task_list_entries.resize(num_tasks);


    //
    // scan the set of tasks and assign to submit list entries
    //
    for ( size_t ti = 0; ti < num_tasks; ++ti)
    {
        ILoadable::SubmitListEntry sle;
        sle.id = task_ids.at(ti);
        sle.tasks.push_back(sle.id);
        submit_list_entries.push_back(sle);
    }

    //
    // one chain (target 0) exists to provide inter-task synchronization.
    // this is the chain that keeps cpu(emu) and hw(dla) tasks synchronized.
    // at the end of that chain is an output-bindable event that the caller
    // can use to wait for completion.
    //

    event_list_entries.clear();

    event.id     = 0;
    event.val    = 0;
    event.target = 0;

    //
    // for each task...
    //
    for ( size_t ti = 0; ti < num_tasks; ++ti)
    {
        size_t num_op_slots = task_slot_counts.at(ti);
        NvU32 num_batches   = g->profile()->multiBatchSize();
        NvS16 taskId = task_ids.at(ti);
        NvS16 taskContextBegin = gal.taskContextBegin(ti);

        NvU16 preaction = 0;
        NvU16 postaction = 0;

        if ( ti != 0 )
        {
            preaction = event.id;
            event.op = nvdla::ILoadable::EventListEntry::op_wait();
            event_list_entries.push_back(event);
            event.id++;
        }

        postaction = event.id;
        event.val++;
        event.op = nvdla::ILoadable::EventListEntry::op_signal();
        event_list_entries.push_back(event);
        event.id++;

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
        nvdla_prototest_interface::TasksData* protoTaskData = protoTestInfo->add_task();
        protoTaskData->set_task_id(taskId);

        /* post action */
        protoTaskData->mutable_schedule()->mutable_post_actions()->add_event_id(postaction);

        /* event list */
        nvdla_prototest_interface::Event* protoTestEvent = protoTestInfo->mutable_event_list()->add_event();
        protoTestEvent->set_event_id(0);
        protoTestEvent->set_event_type(nvdla_prototest_interface::Event_EventType::Event_EventType_SYNCPOINT);
        protoTestEvent->set_event_flags(0);

        /* task slots */
        nvdla_prototest_interface::SubmitSlot* protoTestInfoSubmitSlot = protoTestInfo->add_slots();
        protoTestInfoSubmitSlot->add_task_id(taskId);

        /* task data */
        nvdla_prototest_interface::Network* protoNw               = protoTaskData->mutable_network();
        nvdla_prototest_interface::NetworkDesc* protoNwDesc       = protoNw->mutable_param();
        nvdla_prototest_interface::NetworkLayer* protoNwLayers    = protoNw->mutable_layers();
        nvdla_prototest_interface::LUTParamList* protoNwLUTList   = protoNw->mutable_lut_list();
        nvdla_prototest_interface::ROIDescription* protoNwROIDesc = protoNw->mutable_roi_list();
        NVDLA_UNUSED(protoNwLUTList);
#endif

        if ( debugTasks() )
        {
            gLogInfo << "task_id=" << taskId << " has " << num_op_slots << " op slots and " << num_batches << " batches "<< endl;

            gLogInfo << "\taddress list task context at [" << gal.taskContextBegin(ti) << ", " << gal.taskContextEnd(ti) << ")" << endl;
        }

        anni = task_starting_points[ti];

        //        g->resetRelocEntries(); // reloc entries really only matter for one task at a time.

        if ( ! (*anni)->isEMUEngineType() )
        {

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
            protoTaskData->set_engine_id(nvdla_prototest_interface::EngineID::DLA_0);
#endif

            size_t network_desc_size = dla_if->networkDescAccessor(0).struct_size();
            NvU8 *network_desc_mem = new NvU8[ network_desc_size ];
            memset(network_desc_mem, 0, network_desc_size);
            DLANetworkDescAccessor network_desc =  dla_if->networkDescAccessor( network_desc_mem );

            size_t dep_container_entry_size = dla_if->commonOpDescAccessor(0).struct_size();
            size_t dep_container_size       = dep_container_entry_size * num_op_slots * num_batches;
            NvU8 *dep_container_mem         = new NvU8[dep_container_size];
            memset(dep_container_mem, 0, dep_container_size);

            size_t op_container_entry_size = dla_if->operationContainerAccessor(0).struct_size();
            size_t op_container_size       = op_container_entry_size * num_op_slots * num_batches;
            NvU8 *op_container_mem         = new NvU8[op_container_size];
            memset(op_container_mem, 0, op_container_size);

            size_t surf_container_entry_size = dla_if->surfaceContainerAccessor(0).struct_size();
            size_t surf_container_size       = surf_container_entry_size * num_op_slots * num_batches;
            NvU8 *surf_container_mem         = new NvU8[surf_container_size];
            memset(surf_container_mem, 0, surf_container_size);

            // Reserve space for at least one LUT, even if unused
            NvU16  num_LUTS                 = g->lutManager()->getNumRegisteredLuts() <= 0 ? 0 : (NvU16)g->lutManager()->getNumRegisteredLuts();
            size_t lut_container_entry_size = dla_if->lutParamAccessor(0).struct_size();
            size_t lut_container_size       = lut_container_entry_size * ((g->lutManager()->getNumRegisteredLuts() <= 0) ? 1 : g->lutManager()->getNumRegisteredLuts());
            NvU8 *lut_container_mem         = new NvU8[lut_container_size];
            memset(lut_container_mem, 0, lut_container_size);

            DLALUTParamAccessor lut_container = dla_if->lutParamAccessor(lut_container_mem);
            NVDLA_UNUSED(lut_container);

            for ( NvU32 op_slot = 0; op_slot < num_op_slots; ++op_slot )
            {
                for (NvU32 batch_id = 0; batch_id < num_batches; ++batch_id)
                {
                    NvS16 batch_slot = op_slot*num_batches + batch_id;
                    if ( batch_slot != (*anni)->dependencyParams(batch_id).annotationId() )
                    {
                        gLogError << "Inconsistencies between slot_id and annotation id "
                                  << batch_slot
                                  << " != "
                                  << (*anni)->dependencyParams(batch_id).annotationId() << std::endl;
                        goto fail;
                    }

                    DLACommonOpDescAccessor       dep_acc  = dla_if->commonOpDescAccessor(dep_container_mem + (batch_slot * dep_container_entry_size));
                    DLAOperationContainerAccessor op_acc   = dla_if->operationContainerAccessor(op_container_mem + (batch_slot * op_container_entry_size));
                    DLASurfaceContainerAccessor   surf_acc = dla_if->surfaceContainerAccessor(surf_container_mem + (batch_slot * surf_container_entry_size));

                    e = (*anni)->emitOp(g, dla_if, (*anni)->dependencyParams(batch_id).annotationId(), batch_id, dep_acc, op_acc, surf_acc);
#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
                    nvdla_prototest_interface::Layer* protoNewLayer = protoNwLayers->add_layer();
                    e = (*anni)->emitOp((*anni)->dependencyParams(batch_id).annotationId(), batch_id, dla_if, dep_acc, op_acc, surf_acc, protoNewLayer);
#endif
                }
                ++anni;
            }

            std::stringstream task_id_ss;
            task_id_ss << "task-" << taskId;

            string network_desc_symbol   = task_id_ss.str() + "-addr0";
            string dep_container_symbol  = task_id_ss.str() + "-dep_graph";
            string op_container_symbol   = task_id_ss.str() + "-op_list";
            string surf_container_symbol = task_id_ss.str() + "-surf_list";
            string lut_container_symbol  = task_id_ss.str() + "-lut_list";

            ILoadable::Version fwVersion(dla_if->firmwareTargetVersionMajor(),
                                         dla_if->firmwareTargetVersionMinor(),
                                         dla_if->firmwareTargetVersionSubminor());

            ILoadable::Blob network_desc_blob(network_desc_symbol,     network_desc_size,   ILoadable::Interface_DLA1, NVDLA_LOADABLE_SUB_INTERFACE_DLA1_ADDR0, fwVersion);
            ILoadable::Blob dep_container_blob(dep_container_symbol,   dep_container_size,  ILoadable::Interface_DLA1, NVDLA_LOADABLE_SUB_INTERFACE_DLA1_DEPS,  fwVersion);
            ILoadable::Blob op_container_blob(op_container_symbol,     op_container_size,   ILoadable::Interface_DLA1, NVDLA_LOADABLE_SUB_INTERFACE_DLA1_OPS,   fwVersion);
            ILoadable::Blob surf_container_blob(surf_container_symbol, surf_container_size, ILoadable::Interface_DLA1, NVDLA_LOADABLE_SUB_INTERFACE_DLA1_SURFS, fwVersion);
            ILoadable::Blob lut_container_blob(lut_container_symbol,   lut_container_size,  ILoadable::Interface_DLA1, NVDLA_LOADABLE_SUB_INTERFACE_DLA1_LUTS,  fwVersion);

            loadable->setSymbolContent(network_desc_symbol,   network_desc_blob,   (NvU8*)network_desc_mem);
            loadable->setSymbolContent(dep_container_symbol,  dep_container_blob,  (NvU8*)dep_container_mem);
            loadable->setSymbolContent(op_container_symbol,   op_container_blob,   (NvU8*)op_container_mem);
            loadable->setSymbolContent(surf_container_symbol, surf_container_blob, (NvU8*)surf_container_mem);
            loadable->setSymbolContent(lut_container_symbol,  lut_container_blob,  (NvU8*)lut_container_mem);

            NvU8 set_content = ILoadable::MemoryListEntry::flags_alloc() | ILoadable::MemoryListEntry::flags_set();
            NvU8 domain_sysmem = ILoadable::MemoryListEntry::domain_sysmem();

            ILoadable::MemoryListEntry addr0_mle(gal.memList().size(), network_desc_size, 4096, domain_sysmem, set_content, network_desc_symbol);
            ILoadable::MemoryListEntry dep_mle(addr0_mle.id + 1, dep_container_size,  4096, domain_sysmem, set_content, dep_container_symbol);
            ILoadable::MemoryListEntry op_mle( dep_mle.id + 1,   op_container_size,   4096, domain_sysmem, set_content, op_container_symbol);
            ILoadable::MemoryListEntry surf_mle(op_mle.id + 1,   surf_container_size, 4096, domain_sysmem, set_content, surf_container_symbol);
            ILoadable::MemoryListEntry lut_mle(surf_mle.id + 1,  lut_container_size,  4096, domain_sysmem, set_content, lut_container_symbol);
            ILoadable::MemoryListEntry dummy_mle(lut_mle.id + 1, 4096, 4096, domain_sysmem, ILoadable::MemoryListEntry::flags_alloc());

            gal.memList().push_back(addr0_mle);
            gal.memList().push_back(dep_mle);
            gal.memList().push_back(op_mle);
            gal.memList().push_back(surf_mle);
            gal.memList().push_back(lut_mle);
            gal.memList().push_back(dummy_mle);


            task_list_entries.at(ti).id           = taskId;
            task_list_entries.at(ti).interface = ILoadable::TaskListEntry::interface_DLA1();
            task_list_entries.at(ti).instance  = ILoadable::TaskListEntry::instance_ANY();
            task_list_entries.at(ti).preactions.clear();
            task_list_entries.at(ti).postactions.clear();
            if ( ti != 0 )
            {
                task_list_entries.at(ti).preactions.push_back(preaction);
            }
            task_list_entries.at(ti).postactions.push_back(postaction);

            vector<NvU16> taskAddrList;

            gal.addrList().at(taskContextBegin + 0)  = Loadable::AddressListEntry(taskContextBegin + 0, addr0_mle.id, addr0_mle.size);
            gal.addrList().at(taskContextBegin + 1)  = Loadable::AddressListEntry(taskContextBegin + 1, dep_mle.id,   dep_mle.size);
            gal.addrList().at(taskContextBegin + 2)  = Loadable::AddressListEntry(taskContextBegin + 2, op_mle.id,    op_mle.size);
            gal.addrList().at(taskContextBegin + 3)  = Loadable::AddressListEntry(taskContextBegin + 3, surf_mle.id,  surf_mle.size);
            gal.addrList().at(taskContextBegin + 4)  = Loadable::AddressListEntry(taskContextBegin + 4, lut_mle.id,   lut_mle.size);
            gal.addrList().at(taskContextBegin + 5)  = Loadable::AddressListEntry(taskContextBegin + 5, dummy_mle.id, dummy_mle.size);


            taskAddrList.resize( Ni + TaskAddressList::numContextAddrs());

            // taskAddr[0] := addr0
            taskAddrList.at(0) = taskContextBegin + 0;

            // now instruction entries
            for (size_t ii = 1; ii <= Ni; ++ii)
            {
                taskAddrList.at(ii) = ii;
            }

            // then (the rest of) the task context.  note that addr0 is separated.  start at the next (one) instead.
            for (size_t ai = 1, AI = TaskAddressList::numContextAddrs(); ai < AI; ++ai)
            {
                taskAddrList.at(Ni + ai) = taskContextBegin + ai;
            }

            if ( debugTasks() )
            {
                gLogInfo << "\ttask address list (indices into global address list): " << endl;
                for ( size_t ii = 0; ii < taskAddrList.size(); ++ii)
                {
                    gLogInfo << "\t\t" << taskAddrList.at(ii) << endl;
                }
                gLogInfo << "\t\t<>" << endl;
            }

            // Now that we know where dep_graph, op_list, and surf_list live, we can build the network_desc
            *network_desc.dependencyGraphIndex() = Ni + 1;
            *network_desc.operationDescIndex()   = Ni + 2;
            *network_desc.surfaceDescIndex()     = Ni + 3;
            *network_desc.LUTDataIndex()         = Ni + 4;
            *network_desc.ROIArrayIndex() = -1; // Bogus
            *network_desc.surfaceIndex() = -1; // Bogus
            *network_desc.statListIndex() = -1; // Bogus
            *network_desc.numROIs() = 1;
            *network_desc.numOperations() = num_op_slots * num_batches;
            *network_desc.numLUTs() = g->lutManager()->getNumRegisteredLuts();
            *network_desc.numAddresses() = taskAddrList.size();
            *network_desc.dynamicROI() = 0; // Bogus
            *network_desc.inputLayer() = 0; // Bogus

            // Map opHeads
            vector<engine_ast::Graph::Graphlet *>& graphlet = g->graphlets();
            vector<engine_ast::Node*>& opHeads = graphlet[ti]->opHeads();

            *network_desc.opHead(network_desc.op_BDMA())  = opHeads[engine_ast::EngineTypeEnum::BDMA] ? opHeads[engine_ast::EngineTypeEnum::BDMA]->dependencyParams(/*batchId*/0).annotationId() : -1;
            *network_desc.opHead(network_desc.op_CONV())  = opHeads[engine_ast::EngineTypeEnum::CONVOLUTION] ? opHeads[engine_ast::EngineTypeEnum::CONVOLUTION]->dependencyParams(/*batchId*/0).annotationId() : -1;
            *network_desc.opHead(network_desc.op_SDP())   = opHeads[engine_ast::EngineTypeEnum::SDP] ? opHeads[engine_ast::EngineTypeEnum::SDP]->dependencyParams(/*batchId*/0).annotationId() : -1;
            *network_desc.opHead(network_desc.op_PDP())   = opHeads[engine_ast::EngineTypeEnum::PDP] ? opHeads[engine_ast::EngineTypeEnum::PDP]->dependencyParams(/*batchId*/0).annotationId() : -1;
            *network_desc.opHead(network_desc.op_CDP())   = opHeads[engine_ast::EngineTypeEnum::CDP] ? opHeads[engine_ast::EngineTypeEnum::CDP]->dependencyParams(/*batchId*/0).annotationId() : -1;
            *network_desc.opHead(network_desc.op_RUBIK()) = opHeads[engine_ast::EngineTypeEnum::RUBIK] ? opHeads[engine_ast::EngineTypeEnum::RUBIK]->dependencyParams(/*batchId*/0).annotationId() : -1;

            // Write LUTs
            for ( NvU16 lut_slot = 0; lut_slot < num_LUTS; ++lut_slot)
            {
                DLALUTParamAccessor lut_acc = dla_if->lutParamAccessor(lut_container_mem + (lut_slot * lut_container_entry_size));
                PROPAGATE_ERROR_FAIL(g->lutManager()->writeLutData(lut_slot, lut_acc));
            }

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
            protoNwDesc->set_operation_desc_index(*network_desc.operationDescIndex());
            protoNwDesc->set_surface_desc_index(*network_desc.surfaceDescIndex());
            protoNwDesc->set_dependency_graph_index(*network_desc.dependencyGraphIndex());
            protoNwDesc->set_lut_data_index(*network_desc.LUTDataIndex());
            protoNwDesc->set_roi_array_index(*network_desc.ROIArrayIndex());
            protoNwDesc->set_surface_index(*network_desc.surfaceIndex());
            protoNwDesc->add_op_head(*network_desc.opHead(network_desc.op_BDMA()));
            protoNwDesc->add_op_head(*network_desc.opHead(network_desc.op_CONV()));
            protoNwDesc->add_op_head(*network_desc.opHead(network_desc.op_SDP()));
            protoNwDesc->add_op_head(*network_desc.opHead(network_desc.op_PDP()));
            protoNwDesc->add_op_head(*network_desc.opHead(network_desc.op_CDP()));
            protoNwDesc->add_op_head(*network_desc.opHead(network_desc.op_RUBIK()));
            protoNwDesc->set_stat_list_index(*network_desc.statListIndex());
            protoNwDesc->set_num_rois(*network_desc.numROIs());
            protoNwDesc->set_num_operations(*network_desc.numOperations());
            protoNwDesc->set_num_luts(*network_desc.numLUTs());
            protoNwDesc->set_num_addresses(*network_desc.numAddresses());
            protoNwDesc->set_dynamic_roi(*network_desc.dynamicROI());
            protoNwDesc->set_input_layer(*network_desc.inputLayer());

            /* NOP */
            protoNwROIDesc->mutable_roi_arr()->set_array_length(0);
            protoNwROIDesc->mutable_roi_arr()->set_array_reserved(0);
#endif

            task_list_entries.at(ti).address_list = taskAddrList;

            g->gatherRelocEntries(op_mle.id, op_container_mem,
                                  surf_mle.id, surf_container_mem,
                                  dep_mle.id, dep_container_mem);
        }
        else
        {
            task_list_entries.at(ti).id        = taskId;
            task_list_entries.at(ti).interface = ILoadable::TaskListEntry::interface_EMU1();
            task_list_entries.at(ti).instance  = ILoadable::TaskListEntry::instance_ANY();
            task_list_entries.at(ti).preactions.clear();
            task_list_entries.at(ti).postactions.clear();
            if ( ti != 0 )
            {
                task_list_entries.at(ti).preactions.push_back(preaction);
            }
            task_list_entries.at(ti).postactions.push_back(postaction);


            // EMU currently only supports 1 op slot
            size_t num_ops = 1;
            NvU32 num_batches = g->profile()->multiBatchSize();

            // Allocate the network descriptor
            size_t network_desc_size = emu_if->networkDescAccessor(0).struct_size();
            NvU8* network_desc_mem = new NvU8[ network_desc_size ];
            memset(network_desc_mem, 0, network_desc_size);
            EMUNetworkDescAccessor network_desc =  emu_if->networkDescAccessor( network_desc_mem );

            // Allocate the operation descriptor list
            size_t op_container_entry_size = emu_if->operationContainerAccessor(0).struct_size();
            size_t op_container_size       = op_container_entry_size * num_ops * num_batches;
            NvU8 *op_container_mem         = new NvU8[op_container_size];
            memset(op_container_mem, 0, op_container_size);

            // Allocate the operation buffer descriptor list
            size_t op_buffer_container_entry_size = emu_if->operationBufferContainerAccessor(0).struct_size();
            size_t op_buffer_container_size       = op_buffer_container_entry_size * num_ops * num_batches;
            NvU8 *op_buffer_container_mem         = new NvU8[op_buffer_container_size];
            memset(op_buffer_container_mem, 0, op_buffer_container_size);

            for (NvU32 op_slot=0; op_slot < num_ops; ++op_slot)
            {
                for (NvU32 batch_id = 0; batch_id < num_batches; ++batch_id)
                {
                    NvS16 batch_slot = op_slot*num_batches + batch_id;
                    if ( batch_slot != (*anni)->dependencyParams(batch_id).annotationId() )
                    {
                        gLogError << "Inconsistencies between slot_id and annotation id "
                                  << batch_slot
                                  << " != "
                                  << (*anni)->dependencyParams(batch_id).annotationId() << std::endl;
                        goto fail;
                    }
                    EMUOperationContainerAccessor       op_acc  = emu_if->operationContainerAccessor(op_container_mem + (batch_slot * op_container_entry_size));
                    EMUOperationBufferContainerAccessor buf_acc = emu_if->operationBufferContainerAccessor(op_buffer_container_mem + (batch_slot * op_buffer_container_entry_size));

                    e = (*anni)->emitOp(g, emu_if, (*anni)->dependencyParams(batch_id).annotationId(), batch_id, op_acc, buf_acc);
                }
                ++anni;
            }

            std::stringstream task_id_ss;
            task_id_ss << "task-" << taskId;

            string network_desc_symbol   = task_id_ss.str() + "-addr0";
            string op_container_symbol   = task_id_ss.str() + "-op_list";
            string op_buffer_container_symbol = task_id_ss.str() + "-op_buf_list";

            ILoadable::Version emuVersion(emu_if->emulatorTargetVersionMajor(),
                                          emu_if->emulatorTargetVersionMinor(),
                                          emu_if->emulatorTargetVersionSubminor());

            ILoadable::Blob network_desc_blob(network_desc_symbol, network_desc_size, ILoadable::Interface_EMU1, NVDLA_LOADABLE_SUB_INTERFACE_EMU1_ADDR0, emuVersion);
            ILoadable::Blob op_container_blob(op_container_symbol, op_container_size, ILoadable::Interface_EMU1, NVDLA_LOADABLE_SUB_INTERFACE_EMU1_OPS, emuVersion);
            ILoadable::Blob op_buffer_container_blob(op_buffer_container_symbol, op_buffer_container_size, ILoadable::Interface_EMU1, NVDLA_LOADABLE_SUB_INTERFACE_EMU1_SURFS, emuVersion);

            loadable->setSymbolContent(network_desc_symbol, network_desc_blob, (NvU8*)network_desc_mem);
            loadable->setSymbolContent(op_container_symbol, op_container_blob, (NvU8*)op_container_mem);
            loadable->setSymbolContent(op_buffer_container_symbol, op_buffer_container_blob, (NvU8*)op_buffer_container_mem);

            NvU8 alloc = ILoadable::MemoryListEntry::flags_alloc();
            NvU8 set_content = ILoadable::MemoryListEntry::flags_alloc() | ILoadable::MemoryListEntry::flags_set();
            NvU8 domain_sysmem = ILoadable::MemoryListEntry::domain_sysmem();

            ILoadable::MemoryListEntry addr0_mle (gal.memList().size(), network_desc_size,        4096, domain_sysmem, set_content, network_desc_symbol);
            ILoadable::MemoryListEntry op_mle    (addr0_mle.id + 1,     op_container_size,        4096, domain_sysmem, set_content, op_container_symbol);
            ILoadable::MemoryListEntry op_buf_mle(op_mle.id + 1,        op_buffer_container_size, 4096, domain_sysmem, set_content, op_buffer_container_symbol);

            gal.memList().push_back(addr0_mle);
            gal.memList().push_back(op_mle);
            gal.memList().push_back(op_buf_mle);

            ILoadable::MemoryListEntry dummy1_mle(op_buf_mle.id + 1, 4096, 4096, domain_sysmem, alloc);
            ILoadable::MemoryListEntry dummy2_mle(dummy1_mle.id + 1, 4096, 4096, domain_sysmem, alloc);
            ILoadable::MemoryListEntry dummy3_mle(dummy2_mle.id + 1, 4096, 4096, domain_sysmem, alloc);

            gal.memList().push_back(dummy1_mle);
            gal.memList().push_back(dummy2_mle);
            gal.memList().push_back(dummy3_mle);

            vector<NvU16> taskAddrList;

            gal.addrList().at(taskContextBegin + 0) = ILoadable::AddressListEntry(taskContextBegin + 0, addr0_mle.id, addr0_mle.size);
            gal.addrList().at(taskContextBegin + 1) = ILoadable::AddressListEntry(taskContextBegin + 1, op_mle.id, op_mle.size);
            gal.addrList().at(taskContextBegin + 2) = ILoadable::AddressListEntry(taskContextBegin + 2, op_buf_mle.id, op_buf_mle.size);
            gal.addrList().at(taskContextBegin + 3) = ILoadable::AddressListEntry(taskContextBegin + 3, dummy1_mle.id, dummy1_mle.size);
            gal.addrList().at(taskContextBegin + 4) = ILoadable::AddressListEntry(taskContextBegin + 4, dummy2_mle.id, dummy2_mle.size);
            gal.addrList().at(taskContextBegin + 5) = ILoadable::AddressListEntry(taskContextBegin + 5, dummy3_mle.id, dummy3_mle.size);

            taskAddrList.resize(Ni + TaskAddressList::numContextAddrs() );

            // addr0
            taskAddrList.at(0) = taskContextBegin + 0;

            // now instruction entries
            for (size_t ii = 1; ii <= Ni; ++ii)
            {
                taskAddrList.at(ii) = ii;
            }

            // then (the rest of) the task context.
            for (size_t ai = 1, AI = TaskAddressList::numContextAddrs(); ai < AI; ++ai)
            {
                taskAddrList.at(Ni + ai) = taskContextBegin + ai;
            }

            if ( debugTasks() )
            {
                gLogInfo << "\ttask address list (indices into global address list): " << endl;
                for ( size_t ii = 0; ii < taskAddrList.size(); ++ii)
                {
                    gLogInfo << "\t\t" << taskAddrList.at(ii) << endl;
                }
                gLogInfo << "\t\t<>" << endl;
            }

            // Now that we know where dep_graph, op_list, and surf_list live, we can build the network_desc
            *network_desc.operationDescIndex()       = Ni + 1; // op_mle.id
            *network_desc.operationBufferDescIndex() = Ni + 2; // op_buf_mle.id
            *network_desc.numOperations()            = num_ops * num_batches;

            task_list_entries.at(ti).address_list = taskAddrList;

            g->gatherRelocEntries(op_mle.id, op_container_mem,
                                  op_buf_mle.id, op_buffer_container_mem,
                                  -1, 0);
        }
    }

    //
    // now that all the tasks have set up their context state
    // elements the memory and address lists are viable.
    //
    loadable->setMemoryListEntries(gal.memList());

    loadable->setAddressListEntries(gal.addrList());

    loadable->setTaskListEntries(task_list_entries);

    loadable->setSubmitListEntries(submit_list_entries);

    loadable->setEventListEntries(event_list_entries);

    loadable->setRelocEntries(g->getRelocEntries());

#ifdef NVDLA_COMPILER_OUTPUT_FOR_PROTOTEST
    if (!protoTest.SerializePartialToOstream(&protobufFile)) {
        PROPAGATE_ERROR_FAIL(NvDlaError_FileWriteFailed, "Serialize to ostream failed for nvdla_interfacce.Test");
    }
    protobufFile.close();
    google::protobuf::ShutdownProtobufLibrary();
#endif

fail:
    if ( e != NvDlaSuccess)
    {
        l = LoadableFactory::LoadablePrivPair(0, 0);
    }
    return e;
}

engine_ast::Graph *Compiler::registerBuffers(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;

    //
    // before registering buffers, we should populate the edges of the eng_ast with
    // tensor surface descriptors and then reserve buffers for suitable edges
    //
    PROPAGATE_ERROR_FAIL(graph->registerAllSurfaces());
    PROPAGATE_ERROR_FAIL(graph->registerAllBuffers());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::preProcessAuxData(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->preProcessAuxData());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::mergeActivationOperations(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->mergeActivationOperations());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::updateScalingFactors(engine_ast::Graph *input_graph)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->updateScalingFactors());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::quantizeAuxData(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->quantizeAuxData());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::handleLowPrecisionConversions(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->handleLowPrecisionConversions());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::translateAuxData(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->translateAuxData());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::reserveBuffers(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->reserveAllBuffers());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::fuseOnTheFlyNodes(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;
    PROPAGATE_ERROR_FAIL(new_graph->fuseOnTheFlyNodes());

    return new_graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::fuseSubEngineOps(engine_ast::Graph *input_graph)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = input_graph;
    PROPAGATE_ERROR_FAIL(graph->fuseSDPSubEngineOps());

    return graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::groupAtomicOperations(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;
    PROPAGATE_ERROR_FAIL(new_graph->groupAtomicOperations());

    return new_graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::splitNodes(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;
    PROPAGATE_ERROR_FAIL(new_graph->splitNodes());

    return new_graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::boundGraph(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    engine_ast::Graph *new_graph = input_graph;

    return new_graph;
}

engine_ast::Graph *Compiler::handleMultiBatch(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;
    PROPAGATE_ERROR_FAIL( new_graph->handleMultiBatch() );

    return new_graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::flattenGraph(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;
    PROPAGATE_ERROR_FAIL( new_graph->flattenGraph() );

    return new_graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::generateDependencyParams
(
    engine_ast::Graph *input_graph,
    engine_ast::NodeSequence &topological_order
)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    NvS16 lastUsedAnnId = -1;
    engine_ast::Graph *new_graph = input_graph;
    PROPAGATE_ERROR_FAIL( new_graph->topologicalSort(topological_order) );

    // dependencies are resolved in a flattened graph
    PROPAGATE_ERROR_FAIL( new_graph->resolveDataDependencies(topological_order) );
    PROPAGATE_ERROR_FAIL( new_graph->resolveComputeDependencies(topological_order) );
    PROPAGATE_ERROR_FAIL( new_graph->resolveSoftwareDependencies() );

    // determine DLA/EMU/DLA/etc task boundaries
    PROPAGATE_ERROR_FAIL( new_graph->determineTaskBoundaries(topological_order) );

    PROPAGATE_ERROR_FAIL( new_graph->annotateNodes(lastUsedAnnId) );

    PROPAGATE_ERROR_FAIL( new_graph->resolveMultiBatchDependencies() );

    // validate dependency params of each node in the graph
    PROPAGATE_ERROR_FAIL( new_graph->verifyDependencyGraph() );

    return new_graph;

fail:
    return NULL;
}

engine_ast::Graph *Compiler::resolveMemory
(
    engine_ast::Graph *input_graph,
    const engine_ast::NodeSequence &topological_order
)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;

    //PROPAGATE_ERROR_FAIL( new_graph->flattenGraph() );
    PROPAGATE_ERROR_FAIL( new_graph->resolveMemory(topological_order) );

    return new_graph;

fail:
    return NULL;
}


engine_ast::Graph *Compiler::enableCopyOutDebugSurfaces(engine_ast::Graph *input_graph)
{
    //engine_ast::Graph *new_graph = input_graph->clone();
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *new_graph = input_graph;

    engine_ast::AddCopyOutDebugBDMA addBDMAs;

    if ( !input_graph->target_config()->isBDMACapable() )
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "BDMA engine is not supported with this target config");

    PROPAGATE_ERROR_FAIL( addBDMAs.visitNodes( new_graph->ordering() ) );

    return new_graph;

fail:
    return NULL;
}


DLAInterface *Compiler::getTargetDLAInterface(Profile *)
{
    // dummy, we know exactly which for now.
    DLAInterface *dla_if = new DLAInterfaceA();
    return dla_if;
}

EMUInterface *Compiler::getTargetEMUInterface(Profile *)
{
    // dummy, we know exactly which for now.
    EMUInterface *emu_if = new EMUInterfaceA();
    return emu_if;
}

//
// set/get compilation target attributes.  to what do we bind them?
// until compile() is called maybe the active profile?
//
NvDlaError Compiler::getDataType(DataType::UnderlyingType *data_type) const
{
    NvDlaError e = NvDlaSuccess;
    if ( !data_type || wisdom() == NULL)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    PROPAGATE_ERROR_FAIL( wisdom()->getDataType(data_type) );
 fail:
    return e;
}



} // nvdla::priv


} // nvdla
