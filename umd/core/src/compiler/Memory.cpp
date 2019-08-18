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

#include <math.h>   // floor, log
#include <vector>
#include <list>

#include "priv/EngineAST.h"
#include "priv/Memory.h"
#include "priv/Profile.h"
#include "priv/Surface.h"
#include "BuddyAlloc.h"
#include "ErrorMacros.h"

using std::endl;
using std::vector;
using std::set;
using std::unordered_set;
using std::list;
static NvU32 timestamp = 0;
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string reset("\033[0m");


namespace nvdla
{
namespace priv
{

SEQUENCE_ENUM_STATIC_MEMBERS(memory::LocationEnum,           NvU8,  MEMORY_LOCATION_ENUMS,     "MemoryLocationEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(memory::MemoryBufferTypeEnum,   NvU8,  MEMORY_BUFFER_TYPE_ENUMS,  "MemoryBufferTypeEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(memory::TensorCategoryEnum,     NvU8, TENSOR_CATEGORY_ENUMS,      "TensorCategoryEnum")


namespace memory
{
Pool::~Pool ()
{
    if ( m_addr_mgr )
    {
        NvDlaBuddyAlloc.destruct(m_addr_mgr);
        delete m_addr_mgr;
        m_addr_mgr = 0;
    }
    if ( m_base_addr )
    {
        free(m_base_addr);
        m_base_addr = 0;
    }
}
TensorBufferDesc::~TensorBufferDesc() { }

NvDlaError Pool::init(PoolType pt, NvU64 poolSize, NvU32 minBufferSize)
{
    NvDlaError e = NvDlaSuccess;
    NvU32 minElementSizeLog2 = 0;

    m_name            = std::string(pt.c_str());
    m_type            = pt;
    m_size            = poolSize;
    m_min_buffer_size = minBufferSize;
    // xxx: try to remove this alloc.  surfaces can do this.
    m_base_addr        = malloc(poolSize);
    minElementSizeLog2 = floor(log(minBufferSize)/log(2));
    m_addr_mgr         = new NvDlaBuddyAllocInst;
    PROPAGATE_ERROR_FAIL( NvDlaBuddyAlloc.construct(m_addr_mgr, static_cast<const void*>(m_base_addr), (NvU32)poolSize, minElementSizeLog2) );

fail:
    return e;
}

Location Pool::location()
{
    Location ret;
    switch(m_type.v())
    {
        case memory::PoolTypeEnum::GLOBAL_DRAM_POOL:
        case memory::PoolTypeEnum::LOCAL_DRAM_POOL:
            ret = memory::LocationEnum::lDRAM;
            break;

        case memory::PoolTypeEnum::LOCAL_CVSRAM_POOL:
            ret = memory::LocationEnum::lCVSRAM;
            break;

        default:
            ret = memory::LocationEnum::lUNKNOWN;
    }
    return ret;
}

NvDlaError Pool::allocate(TensorBufferDesc* bufferDesc, NvU16 batchId)
{
    NvDlaError e = NvDlaSuccess;
    void* allocAddr = NULL;
    NvU64 allocSize = bufferDesc->size();
    NvU64 poolOffset = 0;

    if ( allocSize == 0 )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidSize, "zero sized buffer desc");
    }

    allocAddr = NvDlaBuddyAlloc.allocate(m_addr_mgr, allocSize);

    if ( allocAddr == NULL )
    {
        // this spews a message which might be too scary for the uninitiated.
        // we hit this in many/typical  situations...
        if ( debug() )
        {
            gLogInfo << name() << " mem pool exhausted, req. size=" << allocSize << std::endl;
        }
        e = NvDlaError_InsufficientMemory;
        goto fail;
    }

    poolOffset = reinterpret_cast<NvU64>(allocAddr) - reinterpret_cast<NvU64>(m_base_addr);

    bufferDesc->setAddress(allocAddr, batchId);
    bufferDesc->setPool(this, batchId);
    bufferDesc->setMemoryLoc(location(), batchId);
    bufferDesc->setPoolOffset(poolOffset, batchId);
    bufferDesc->setAllocated(batchId);

    m_size_used = std::max<NvU64>(m_size_used, poolOffset + allocSize);

    if ( debug() )
    {
        gLogInfo << "\t\t" << name() << " alloc " << bufferDesc->id() << " @" << poolOffset << " +" << allocSize << " loc=" << (int)location().v() << std::endl;
    }
fail:
    return e;
}

//
// we need to keep track of the original allocation for memlist
// entry creation. just tell the allocator to release the address
// range and don't forget about the original...
//
NvDlaError Pool::deallocate(TensorBufferDesc* bufferDesc, NvU16 batchId)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 buffAddr = 0;
    if ( !bufferDesc->allocated(batchId) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "Attempt to deallocate an un-allocated buffer");
    }
    buffAddr = reinterpret_cast<NvU64>(m_base_addr) + bufferDesc->poolOffset(batchId);

    if ( debug() )
    {
        gLogInfo << "\t\t" << name() << " dealloc " << bufferDesc->id() << " @ " << bufferDesc->poolOffset(batchId)
                 << " @ " << buffAddr << std::endl;
    }

    PROPAGATE_ERROR_FAIL( NvDlaBuddyAlloc.deallocate(m_addr_mgr, reinterpret_cast<void*>(buffAddr)) );

#if 0
    // NO, can't do this.  We need a record of the alloc for
    // the memory/address list generator.
    bufferDesc->setAddress(NULL);
    bufferDesc->setPool(NULL);
    bufferDesc->setMemoryLoc(memory::LocationEnum::lUNKNOWN);
    bufferDesc->setPoolOffset(0);
    bufferDesc->clearAllocated();
#endif

fail:
    return e;
}


NvDlaError TensorBufferDesc::addSurface(surface::TensorSurfaceDesc *tsd)
{
    NvDlaError e = NvDlaSuccess;

    if ( !tsd )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "null surface");
    }

    m_surfaces.insert(tsd);

fail:
    return e;
}

std::set<surface::TensorSurfaceDesc *>& TensorBufferDesc::surfaces()
{
    return m_surfaces;
}


bool TensorBufferDesc::content() const
{
    for ( std::set<surface::TensorSurfaceDesc *>::iterator tsdi = m_surfaces.begin();
          tsdi != m_surfaces.end();
          ++tsdi )
    {
        if ( (*tsdi)->content() )
        {
            return true;
        }
    }
    return false;
}

bool TensorBufferDesc::bindable() const
{
    int r = false;
    for ( std::set<surface::TensorSurfaceDesc *>::iterator tsdi = m_surfaces.begin();
          tsdi != m_surfaces.end();
          ++tsdi )
    {
        if ( debugBinding() )
        {
            gLogInfo << "\t\t\t\t::Buffer buffer=" << id() << " surface=" << (*tsdi)->id() << " bindable=" << (*tsdi)->bindable() << endl;
        }
        if ( (*tsdi)->bindable() )
        {
            r = true;
        }
    }
    return r;
}

NvS16 TensorBufferDesc::bindId(enum IOD &bindDomain) const
{
    NvS16 r = -1;
    std::string delim("");

    if (debugBinding() )
    {
        gLogInfo << "\t\t\t\t::Buffer bindId(buffer=" << id() << ") [";
    }

    for ( std::set<surface::TensorSurfaceDesc *>::iterator tsdi = m_surfaces.begin();
          tsdi != m_surfaces.end();
          ++tsdi )
    {
        if ( debugBinding() )
        {
            gLogInfo << delim << (*tsdi)->id() << " bind_id=" << (*tsdi)->bindable() << endl;
            delim = ", ";
        }
        if ( (*tsdi)->bindable() )
        {
            r = (*tsdi)->bindId(bindDomain);
        }
    }

    if (debugBinding())
    {
        gLogInfo << "]" << endl;
    }

    return r;
}

surface::TensorSurfaceDesc *TensorBufferDesc::boundSurface(size_t i) const
{
    size_t n = 0;
    for ( std::set<surface::TensorSurfaceDesc *>::iterator tsdi = m_surfaces.begin();
          tsdi != m_surfaces.end();
          ++tsdi )
    {
        if ( (*tsdi)->bindable() )
        {
            if ( i == n )
            {
                if ( debugBinding() )
                {
                    gLogInfo << "\t\t\t\t::Buffer boundSurface(i=" << i << ") -> " << (*tsdi)->id() << endl;
                }
                return *tsdi;
            }
            n++;
        }
    }
    return 0;
}


/*----------------------------Resolve Memory---------------------------*/

//
// regarding memory-reuse:  the only place it makes sense to worry
// about re-using memory is within the pools.  outside of that we
// must leave the buffers in place; they can't be manipulated
// while the dla engine is working.
// further, it doesn't make much sense to bother with releasing
// global memory pooled items.  those are almost always
// content-bearing and so must be initalized at 0-time and
// thereafter no new buffers are created within it (transients
// are created in "local" memory).
// so, we really only need to worry about alloc/dealloc within
// the local sdram and cvsram pools.
//
//


memory::MemoryResolver::MemoryResolver() :
    ast::GraphVisitor<engine_ast::Graph>(),
    m_useMemPool(false),
    m_useReusePooledMemory(false),
    m_useGreedyEviction(false),
    m_useCVSRAM(false),
    m_pools(NULL),
    m_localCVSRAM(NULL),
    m_localSDRAM(NULL),
    m_globalSDRAM(NULL),
    m_debug(false),
    m_inLocalPool()
{

}

typedef PtrPrintIdList<surface::TensorSurfaceDesc> PrintSurfaceIds;

NvDlaError memory::MemoryResolver::visitBegin(engine_ast::Graph *graph)
{
    m_debug = graph->debugMemoryLayout();

    m_useMemPool = graph->profile()->useMemPool();
    m_useReusePooledMemory = graph->profile()->useReusePooledMemory();
    m_useGreedyEviction    = graph->profile()->useGreedyEviction();
    m_useCVSRAM = graph->profile()->useCVSRAMAllocate();

    m_pools = graph->resourceMgr()->memoryPools();

    m_localCVSRAM = &(*m_pools)[memory::PoolTypeEnum::LOCAL_CVSRAM_POOL];
    m_localSDRAM  = &(*m_pools)[memory::PoolTypeEnum::LOCAL_DRAM_POOL];
    m_globalSDRAM = &(*m_pools)[memory::PoolTypeEnum::GLOBAL_DRAM_POOL];

    if ( m_debug )
    {
        gLogInfo << "begin memory resolver pooling=" << m_useMemPool <<
            " reuse=" << m_useReusePooledMemory << " greedy_eviction=" << m_useGreedyEviction << endl;
        gLogInfo << "\tlocal cvsram size=" << m_localCVSRAM->size() << " local sdram size=" << m_localSDRAM->size() <<
            " global sdram size=" << m_globalSDRAM->size() << endl;
    }

    return NvDlaSuccess;
}

NvDlaError memory::MemoryResolver::tryAllocInsidePool(engine_ast::Node* node,
                                                   surface::TensorSurfaceDesc* tsd,
                                                   memory::TensorBufferDesc* tbd,
                                                   vector<memory::Pool *>& tryPools,
                                                   bool isAUX, bool& retry)
{
    NvDlaError e = NvDlaSuccess;
    memory::Pool *batchCommonPool = 0;
    memory::Pool *selectedPool = 0;
    NvU32 numBatches = node->graph()->profile()->multiBatchSize();

    //
    // try allocating buffers for all batches, however aux buffers are
    // allocated only once and shared among all batches
    //
    for (NvU32 nn = 0; nn < numBatches; ++nn)
    {
        selectedPool = 0;
        for ( size_t pi = 0; pi < tryPools.size(); ++pi )
        {
            // aux buffers are allocated once and shared among all batches
            if (isAUX && nn > 0)
            {
                selectedPool = batchCommonPool;
                tbd->setAddress(tbd->address<void*>(/*batchId*/0), /*batchId*/nn);
                tbd->setPoolOffset(tbd->poolOffset(/*batchId*/0));
                tbd->setAllocated(/*batchId*/nn);
                break;
            }

            // for non-aux buffers, attempt to allocate
            e = tryPools[pi]->allocate(tbd, /*batchId*/nn);
            if ( m_debug )
            {
                std::string tensorType = isAUX ? "AUX" :
                        (tsd->producers().find(node) != tsd->producers().end() ? "OUTPUT" : "INPUT");
                gLogInfo << "[MEMTOOL] t = " << timestamp << "\t"
                         << node->name() << "'s\t" << tensorType << "\t"
                         << tbd->id() << "-B" << nn << red << "\tALLOC " << reset << endl;
                timestamp++;
            }

            // if alloc succeeded in a pool, move on to next batch
            if ( e == NvDlaSuccess )
            {
                selectedPool = tryPools[pi];
                if (nn == 0)
                {
                    batchCommonPool = selectedPool;
                }
                else if ((selectedPool != batchCommonPool) && m_debug)
                {
                    // FIXME: In future allow buffers for different batches to live in different pools
                    //ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Pool for batch %d : %s is not "
                    /*ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Pool for batch %d : %s is not "
                                                           "the same as that for batch 0(%s)",
                                                           nn, selectedPool->name().c_str(),
                                                           batchCommonPool->name().c_str());*/
                    gLogInfo << "Pool for batch " << nn << ":" << selectedPool->name()
                             << " is not the same as that for batch 0:" << batchCommonPool->name() << endl;
                }
                break;
            }
            // if this is the last try pool and if that failed too, return error
            else if ( pi == (tryPools.size() - 1) )
            {
                goto fail;
            }
            // else try next pool

        }   // finish try pools

        if ( selectedPool )
        {
            if ( selectedPool == m_localSDRAM || selectedPool == m_localCVSRAM )
            {
                m_inLocalPool.insert(tsd);
            }
        }
        else
        {
            //
            // if any of the batches fail to fit, deallocate pool spaces for all of them and error.
            // something might free-up and a retry is useful for the buffer for all batches.
            // don't spew a scary message yet as it might be ok.
            //
            retry = true;
            for (NvU32 kk = 0; kk <= nn; ++kk)
            {
                ORIGINATE_ERROR_FAIL(selectedPool->deallocate(tbd, kk));
                tbd->setAddress(NULL, /*batchId*/kk);
                tbd->setPoolOffset(0, /*batchId*/kk);
                tbd->clearAllocated(/*batchId*/kk);
            }   // finish dealloc for allocated batches
            e = NvDlaError_InsufficientMemory;
            goto fail;
        }

        if ( m_debug )
        {
            gLogInfo << "\t\t\tplaced " << tbd->id() << " batch-" << nn << " inside " << selectedPool->name()
                     << "@" << tbd->poolOffset(nn) << endl;
        }

    }   // finish alloc for all batches

fail:
    return e;
}

NvDlaError memory::MemoryResolver::tryAllocOutsidePool(engine_ast::Node* node,
                                                    surface::TensorSurfaceDesc* tsd,
                                                    memory::TensorBufferDesc* tbd,
                                                    bool isAUX)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 allocSize = tbd->size();
    void* allocAddr = 0;
    NvU32 numBatches = node->graph()->profile()->multiBatchSize();

    if ( allocSize == 0 )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidSize, "zero sized buffer desc");
    }

    if ( isAUX )
    {
        allocAddr = new NvU8[allocSize];
        if ( allocAddr == NULL )
        {
            // note: cannot be retried.  compile-time memory size problem.
            ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory,
                                 "mem exhausted, req. size: %lld", allocSize);
        }
    }

    //
    // aux tensors are shared among all batches; whereas
    // actual memory allocation for non-pooled non-aux buffers
    // happen someplace else (in runtime::loadMemory()) - so no mem alloc for them
    //
    for (NvU32 nn = 0; nn < numBatches; ++nn)
    {
        if ( isAUX )
        {
            tbd->setAddress(allocAddr, nn);
        }
        tbd->setPool(0, nn);
        tbd->setPoolOffset(0, nn);
        tbd->setMemoryLoc(memory::LocationEnum::lDRAM, nn);
        tbd->setAllocated(nn);

        if ( m_debug )
        {
            gLogInfo << "\t\t\tplaced " << tsd->id() << "/" << tbd->id() << " batch-" << nn << " inside DRAM "
                     << "@" << tbd->address<void*>(nn) << endl;
        }
    }

fail:
    return e;
}
NvDlaError memory::MemoryResolver::resolveSurfacePlacement(engine_ast::Node *node,
                                                        surface::TensorSurfaceDesc *tsd,
                                                        memory::TensorBufferDesc *tbd,
                                                        bool isAUX, bool &retry,
                                                        bool allowFallback)
{
    NvDlaError e = NvDlaSuccess;
    retry = false;

    //
    // dont allow pooling on buffers that are bindable by the app,
    // even if it may have been split by compiler
    //
    bool pooling = m_useMemPool && !(tbd->bindable() || tsd->copyOutDebugSurface());

    if ( m_debug )
    {
        gLogInfo << "\t\tresolve placement/alloc for tsd=" << tsd->id() << "/" << tbd->id() <<
            " aux=" << isAUX << " pooling=" << pooling << endl;
    }

    if ( pooling )
    {
        //
        // pooling enabled.  determine which pool(s) to try.
        // first/best choice is tryPools[0] and any fallbacks
        // are after.
        //
        vector<memory::Pool *> tryPools;

        if ( isAUX )
        {
            tryPools.push_back(m_globalSDRAM);
        }
        else
        {
            bool emuDetected = tsd->referencedByEMU();

            if ( m_useCVSRAM && !emuDetected )
            {
                // can't allow cvsram if EMU is involved.
                tryPools.push_back(m_localCVSRAM);
            }
            tryPools.push_back(m_localSDRAM);
        }

        //
        // cull fallbacks if we don't want to allow them
        //
        if ( (tryPools.size() > 1) && !allowFallback )
        {
            tryPools.erase( tryPools.begin()+1, tryPools.end() );
        }

        //
        // now try allocating buffers for all batches
        //
        PROPAGATE_ERROR_FAIL(tryAllocInsidePool(node, tsd, tbd, tryPools, isAUX, retry));
    }
    else
    {
        PROPAGATE_ERROR_FAIL(tryAllocOutsidePool(node, tsd, tbd, isAUX));
    }

    // if ( m_debug ) { gLogInfo << "\t\tend place tsd=" << tsd->id() << " e=" << int(e) << endl; }



fail:
    return e;
}

// records earliest annotationId a deallocation becomes possible for a surface.
//typedef std::pair<int, surface::TensorSurfaceDesc *> DeallocableSurface;

#if 0
// if deallocables is a vector use this to sort using std::sort
struct DeallocableSorter
{
    bool operator() (memory::MemoryResolver::DeallocableSurface i, memory::MemoryResolver::DeallocableSurface j)
    {
        bool i_lt_j = i.first < j.first;
        return i_lt_j;
    }

};
#else

// if deallocables is a list then use this to sort using list::sort
static bool CompareDeallocable(const memory::MemoryResolver::DeallocableSurface &i,
                               const memory::MemoryResolver::DeallocableSurface &j)
{
    return i.first < j.first;
}
#endif


//
// this functor is called over the set of allocations
// which could possibly be released.  it runs in the context of a
// specific node's annotationId.  it's purpose is to determine
// which, if any of those surfaces could be deallocated at that point.
//

class CheckSurfaceRelease
{
public:
    CheckSurfaceRelease(engine_ast::Node *atNode,
                        list<memory::MemoryResolver::DeallocableSurface> &deallocable) :
        m_atNode(atNode),
        m_deallocable(deallocable) {}

    ~CheckSurfaceRelease() { }

    void operator()(surface::TensorSurfaceDesc * surface)
    {
        NvDlaError e = NvDlaSuccess;
        int atId = m_atNode->dependencyParams().annotationId();
        if ( atId < 0 )
        {
            THROW_ERROR(NvDlaError_InvalidState, "check surface release on bogus op?");
        }

        // gLogInfo << "\t\t\tcheck surface release: " << surface->id() << " at node " << m_atNode->id() << endl;

        const engine_ast::Graph::NodeUnorderedSet &consumers = surface->consumers();
        engine_ast::Graph::NodeUnorderedSet::const_iterator
            cbegin = consumers.begin(),
            cend = consumers.end(),
            ci;

        //
        // last scheduled reference is max of these.
        //
        int lastRefId = -1;
        for ( ci = cbegin; ci != cend; ++ci )
        {
            lastRefId = std::max<int>(lastRefId, (*ci)->dependencyParams().annotationId());
        }

        if ( lastRefId == -1 )
        {
            // gLogWarning << "detected a surface without any consumers?" << endl;
            // don't free it, it's bogus but worst case it jams things up?
            return; // THROW_ERROR(NvDlaError_InvalidState, "detected a surface w/o any consumers?");
        }

        //
        // if lastRefId is in the past, and
        // if its not that of a fused upstream node,
        // then release is possible.
        //
        int fusedUpstreamNodeId = m_atNode->dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT) ?
                m_atNode->dependencyParams().fusedNode(engine_ast::IODirectionEnum::INPUT)->dependencyParams().annotationId() : -1;
        if ( lastRefId < atId && lastRefId != fusedUpstreamNodeId)
        {
            memory::TensorBufferDesc *tbd = surface->tensorBufferDesc();
            if ( !tbd )
            {
                THROW_ERROR(NvDlaError_InvalidState, "no buffer for surface %s", surface->id().c_str());
            }
            memory::Pool *pool = tbd->pool();
            if ( !pool )
            {
                THROW_ERROR(NvDlaError_InvalidState, "no pool assigned to surface/buffer %s/%s",
                            surface->id().c_str(), tbd->id().c_str() );
            }

            m_deallocable.push_back( memory::MemoryResolver::DeallocableSurface(lastRefId, surface) );
        }

    }

protected:
    engine_ast::Node *m_atNode;
    list< memory::MemoryResolver::DeallocableSurface > &m_deallocable;
};






NvDlaError memory::MemoryResolver::placeSurfaceContent(engine_ast::Node *node,
                                                        surface::TensorSurfaceDesc *tsd)
{
    NvDlaError e = NvDlaSuccess;
    memory::TensorBufferDesc *tbd;

    if ( !tsd )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "no suface");
    }
    tbd = tsd->tensorBufferDesc();

    if ( !tbd )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "no buffer for surface");
    }
    if ( tsd->bufferOffset() != 0 )
    {
        ORIGINATE_ERROR_FAIL( NvDlaError_InvalidState, "not expecting non-zero buffer offset for aux data");
    }
    if ( tsd->size() != tbd->size() )
    {
        ORIGINATE_ERROR_FAIL( NvDlaError_InvalidState, "not expecting differing sizes on aux data buffer vs. surface");
    }

    if ( ! tsd->content() )
    {
        if ( m_debug )
        {
            gLogInfo << "\t\t\ttsd=" << tsd->id() << " set content pooled=" << (tbd->pool() ? true : false) << endl;
        }

        tsd->setContent( node->getAuxData(tsd->parentEdge()) );

        if ( tbd->pool() )
        {
            tbd->pool()->insertContent( tsd );
        }
    }

fail:
    return e;
}


//
// resolveHazards can be called before annotation.
//
NvDlaError memory::MemoryResolver::resolveHazards(engine_ast::Node *op, vector< SurfacePair > &collisions)
{
    NvDlaError e = NvDlaSuccess;
    engine_ast::Graph *graph = op->graph();

    for ( size_t ci = 0, CI = collisions.size(); ci != CI; ++ci )
    {
        surface::TensorSurfaceDesc *newer = collisions[ci].first;
        surface::TensorSurfaceDesc *older = collisions[ci].second;

        // create a hazard dependency between the nodes at which
        // the operations occur. such that the op(s?) involved in
        // the newer use of the surface wait for the older ones
        // to complete.  i.e: older +++> newer

        for ( unordered_set<Node *>::const_iterator readi = older->consumers().begin();
              readi != older->consumers().end(); ++readi )
        {
            for ( unordered_set<Node *>::const_iterator writej = newer->producers().begin();
                  writej != newer->producers().end(); ++writej )
            {
                if ( readi == writej )
                {
                    // continue as it makes no sense to wait on yourself.
                    // shouldn't happen but at least don't freak anything out.
                    gLogInfo << "warning: hazard waiting on ourselves? " << (*readi)->id() << " vs. " << (*writej)->id() << endl;
                    continue;
                }

                if ( (*writej)->dependsOn(*readi, engine_ast::viaComputeHazard, engine_ast::allowDataCompute ) )
                {
                    // already there.
                    if ( graph->debugMemHazards() )
                    {
                        gLogInfo << "info: hazard avoided: " << (*writej)->id() << " already depends on " << (*readi)->id() << endl;
                    }
                    continue;
                }
                else if ( (*readi)->dependsOn(*writej, engine_ast::viaComputeHazard, engine_ast::allowDataCompute) )
                {
                    // adding a hazard here would cause a cycle.  uh?
                    //ORIGINATE_ERROR_FAIL( NvDlaError_InvalidState, "failed due to memory hazard induced compute dependency cycle");
                    if ( graph->debugMemHazards() )
                    {
                        gLogInfo << "warning: hazard cycle?" << (*readi)->id() << " depends on " << (*writej)->id() << " ???" << endl;
                    }
                }
                else
                {
                    Edge *hazardEdge = graph->addHazardEdge(*readi, *writej);
                    if ( graph->debugMemHazards() )
                    {
                        gLogInfo << "\tinserted hazard edge=" << hazardEdge->id() << " between nodes " <<
                            (*readi)->name() << " and " << (*writej)->name() <<
                            " to resolve memory reuse hazard" << endl;
                    }
                }
            }
        }
    }

    // fail:
    return e;
}

typedef list<surface::TensorSurfaceDesc *> SurfaceList;
typedef vector<surface::TensorSurfaceDesc *> SurfaceVector;

struct AllocInfo
{
    AllocInfo(surface::TensorSurfaceDesc * s, memory::TensorBufferDesc *b, bool a) :
        m_surface(s), m_buffer(b), m_isAUX(a) { }

    AllocInfo(const AllocInfo &o) :
        m_surface(o.m_surface), m_buffer(o.m_buffer), m_isAUX(o.m_isAUX) { }

    AllocInfo() : m_surface(NULL), m_buffer(NULL), m_isAUX(false) { }

    surface::TensorSurfaceDesc * m_surface;
    memory::TensorBufferDesc *m_buffer;
    bool m_isAUX;
};

//
// called during resolveMemory()
//
NvDlaError memory::MemoryResolver::visitNode(Node *node)
{
    NvDlaError e = NvDlaSuccess;

    SurfaceList auxSurfaces, inputSurfaces, outputSurfaces, ioSurfaces;

    SurfaceList allocated;
    SurfaceList::iterator allocatedi;

    list<DeallocableSurface> deallocable;
    CheckSurfaceRelease checkSurfaceRelease(node, deallocable);

    vector<SurfacePair> collisions;

    vector<AllocInfo> tryAllocs, retryAllocs;

    vector< SurfaceList *> surfaceLists;

    // set up some references for later.
    surfaceLists.push_back(&inputSurfaces);
    surfaceLists.push_back(&auxSurfaces);
    surfaceLists.push_back(&ioSurfaces);
    surfaceLists.push_back(&outputSurfaces);

    //
    // unless we're doing greedy eviction we won't try to free
    // until after the first attempt to allocate.
    //
    bool okToFree = false || m_useGreedyEviction;
    int numFreed = 0;
    bool allowFallback = false;

    if ( !node )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "invalid element");
    }
    if ( node->dependencyParams().annotationId() == -1 )
    {
        // skip
        goto fail;
    }

    //
    // setup orthogonal lists for input, output, in/out and aux surfaces
    //
    {
        SurfaceVector vin  = node->inputSurfaces(), vaux = node->auxSurfaces(), vout = node->outputSurfaces();

        inputSurfaces.insert (inputSurfaces.end(),  vin.begin(),  vin.end());
        auxSurfaces.insert   (auxSurfaces.end(),    vaux.begin(), vaux.end());
        outputSurfaces.insert(outputSurfaces.end(), vout.begin(), vout.end());
        //
        // pull any surfaces which are on both sides to the ioSurface set
        // this is likely to be stream tensors if at all.
        //
        for ( SurfaceList::iterator isi = inputSurfaces.begin(); isi != inputSurfaces.end(); ++isi )
        {
            if ( std::find(outputSurfaces.begin(), outputSurfaces.end(), *isi) != outputSurfaces.end() )
            {
                ioSurfaces.push_back(*isi);
            }
        }
        for ( SurfaceList::iterator iosi = ioSurfaces.begin(); iosi != ioSurfaces.end(); ++iosi)
        {
            inputSurfaces.remove(*iosi);
            outputSurfaces.remove(*iosi);
        }
    }

    if ( m_debug )
    {
        gLogInfo << "\tnode=" << node->name() << " anno=" << node->taskId() << "." <<
            node->dependencyParams().annotationId();
        gLogInfo << " in=[";   std::for_each(inputSurfaces.begin(),  inputSurfaces.end(),  PrintSurfaceIds());
        gLogInfo << "] aux=["; std::for_each(auxSurfaces.begin(),    auxSurfaces.end(),    PrintSurfaceIds());
        gLogInfo << "] io=["; std::for_each(ioSurfaces.begin(), ioSurfaces.end(), PrintSurfaceIds());
        gLogInfo << "] out=["; std::for_each(outputSurfaces.begin(), outputSurfaces.end(), PrintSurfaceIds());
        gLogInfo << "]" << endl;

#if 0
        if ( m_inLocalPool.size() )
        {
            gLogInfo << "\t\tsurfaces in local pools=" << m_inLocalPool.size() << " [";
            std::for_each(m_inLocalPool.begin(), m_inLocalPool.end(), PrintSurfaceIds());
            gLogInfo << "]" << endl;

        }
#endif
    }


    //
    // run consistency checks and/or culling applicable to all surfaces
    //
    for ( size_t sli = 0, SLI = surfaceLists.size(); sli != SLI; ++sli )
    {
        SurfaceVector toRemove;

        for ( SurfaceList::iterator si = surfaceLists[sli]->begin(); si != surfaceLists[sli]->end(); ++si )
        {
            surface::TensorSurfaceDesc *tsd = *si;
            memory::TensorBufferDesc *tbd = tsd->tensorBufferDesc();

            if ( tsd->tensorCategory().e() == memory::TensorCategoryEnum::UNKNOWN_TENSOR )
            {
                ORIGINATE_ERROR_FAIL( NvDlaError_InvalidState, "unknown tensor type");
            }

            if ( tsd->tensorCategory().e() == memory::TensorCategoryEnum::STREAM_TENSOR )
            {
                if ( m_debug ) { gLogInfo << "\t\ttsd=" << tsd->id() << " (stream tensor)" << endl; }
                toRemove.push_back(tsd);
                continue;
            }

            if ( !tbd )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "no buffer for tsd=%s", tsd->id().c_str());
            }

            if ( tbd->allocated() )
            {
                toRemove.push_back(tsd);
            }
        }

        for ( size_t ri = 0, RI = toRemove.size(); ri != RI; ++ri )
        {
            surfaceLists[sli]->remove(toRemove[ri]);
        }
    }


    //
    // prime the set of buffers which we (still) need to place.
    //
    for ( SurfaceList::iterator si = inputSurfaces.begin(); si != inputSurfaces.end(); ++si)
    {
        tryAllocs.push_back(AllocInfo(*si, (*si)->tensorBufferDesc(), false /*not aux*/));
    }
    for ( SurfaceList::iterator si = auxSurfaces.begin(); si != auxSurfaces.end(); ++si)
    {
        tryAllocs.push_back(AllocInfo(*si, (*si)->tensorBufferDesc(), true /*aux*/));
    }
    for ( SurfaceList::iterator si = ioSurfaces.begin(); si != ioSurfaces.end(); ++si)
    {
        tryAllocs.push_back(AllocInfo(*si, (*si)->tensorBufferDesc(), false /*not aux*/));
    }
    for ( SurfaceList::iterator si = outputSurfaces.begin(); si != outputSurfaces.end(); ++si)
    {
        tryAllocs.push_back(AllocInfo(*si, (*si)->tensorBufferDesc(), false /*not aux*/));
    }

    //
    // find surfaces which can be freed.  note that the surfaces checked
    // are only those which were allocated out of a local pool.
    //
    CATCH_ERROR_FAIL( std::for_each(m_inLocalPool.begin(), m_inLocalPool.end(), checkSurfaceRelease) );

    deallocable.sort( CompareDeallocable ); // sorted on annotationId

    do
    {
        //
        // greedy -> free everything which can be (m_useGreedyEviction)
        // !greedy-> free one buffer (numFreed) per pass but only after the first pass (okToFree)
        //
        numFreed = 0;

        while ( (deallocable.begin() != deallocable.end()) && ( m_useGreedyEviction || (okToFree && !numFreed)) )
        {
            surface::TensorSurfaceDesc *tsd = deallocable.begin()->second;
            memory::TensorBufferDesc *tbd = tsd->tensorBufferDesc();
            memory::Pool *pool;

            if ( !tbd )
            {
                THROW_ERROR(NvDlaError_InvalidState, "no buffer for surface %s", tsd->id().c_str());
            }

            pool = tbd->pool();
            if ( !pool )
            {
                THROW_ERROR(NvDlaError_InvalidState, "no pool assigned to surface/buffer %s/%s",
                            tsd->id().c_str(), tbd->id().c_str() );
            }

            //
            // the address free happens here.
            //
            if ( m_debug )
            {
                gLogInfo << "\t\t\tdeallocating " << tsd->id() << "/" << tbd->id()
                         << "@" << tbd->address<void*>(0) << endl;
            }

            PROPAGATE_ERROR_THROW( pool->deallocate(tbd) );
            if ( m_debug )
            {
                gLogInfo << "[MEMTOOL] t = " << timestamp << "\t"
                         << tbd->id() << "-B0" << green << "\tDEALLOC " << reset << endl;
                timestamp++;
            }

            m_inLocalPool.erase( tsd );

            m_deallocated.push_back( DeallocableSurface(node->dependencyParams().annotationId(), deallocable.begin()->second) );

            deallocable.pop_front();

            numFreed++;
        }

        allowFallback = (deallocable.begin() == deallocable.end());

        //
        // try allocs and record any need to retry
        //
        for ( size_t ti = 0, TI = tryAllocs.size(); ti != TI; ++ti )
        {
            bool retry;

            AllocInfo &ai = tryAllocs[ti];
            e = resolveSurfacePlacement(node,
                                        ai.m_surface, ai.m_buffer,
                                        ai.m_isAUX, retry, allowFallback);
            //
            // surface placement is successful if it happens for all batches successfully
            //
            if ( e == NvDlaSuccess )
            {
                allocated.push_back(ai.m_surface);
                if ( ai.m_isAUX )
                {
                    PROPAGATE_ERROR_FAIL( placeSurfaceContent(node, ai.m_surface) );
                }
            }
            else
            {
                if ( e == NvDlaError_InsufficientMemory && retry )
                {
                    retryAllocs.push_back(ai);
                }
                else
                {
                    PROPAGATE_ERROR_FAIL(e);
                }
            }
        }

        //
        // setup for next pass
        //
        okToFree = true; // at least one pass happened, ok to free now if needed

        if ( m_debug )
        {
            if ( retryAllocs.size() )
            {
                gLogInfo << "\t\tretry surface placement after deallocating: num surface retries:" << retryAllocs.size() <<
                    " any deallocables? " << !(deallocable.begin() == deallocable.end()) << endl;
            }
        }

        tryAllocs = retryAllocs;
        retryAllocs.clear();

        //
        // keep going as long as we need mem and there's something which can be freed
        //

    } while ( tryAllocs.size() && (deallocable.begin() != deallocable.end()) );


    if ( tryAllocs.size() )
    {
        ORIGINATE_ERROR_FAIL( NvDlaError_InsufficientMemory, "couldn't place all buffers for annotationId %d.%d",
                              node->taskId(), node->dependencyParams().annotationId());
    }

    //
    // all necessary allocations succeeded.  check for overlaps with surfaces which
    // were previously deallocated. anything allocated at a prior node/annotationId
    // would have been resolved there... xxx is that accurate?
    //

    for ( allocatedi = allocated.begin(); allocatedi != allocated.end(); ++allocatedi )
    {
        PROPAGATE_ERROR_FAIL( findReuseHazards(node, *allocatedi, collisions) );
    }

    if ( collisions.size() )
    {
        PROPAGATE_ERROR_FAIL( resolveHazards(node, collisions) );
    }

 fail:

    if ( m_debug )
    {
        gLogInfo << "\tdone node=" << node->name() << " rc=" << (int)e << endl;
    }
    return e;
}

NvDlaError memory::MemoryResolver::findReuseHazards(engine_ast::Node *, surface::TensorSurfaceDesc *tsd,
                                                     vector< memory::MemoryResolver::SurfacePair > &collisions)
{
    NvDlaError e = NvDlaSuccess;

    memory::TensorBufferDesc *tbd = tsd->tensorBufferDesc();
    memory::Pool *pool;

    pool = tbd->pool();
    if ( !pool )
    {
        // nothing to do... can't possibly be re-using if not in a pool.
        goto fail;
    }

    for ( list< DeallocableSurface >::iterator i = m_deallocated.begin(); i != m_deallocated.end(); ++i )
    {
        surface::TensorSurfaceDesc *vs_tsd = 0;
        memory::TensorBufferDesc *vs_tbd = 0;
        vs_tsd = i->second;
        vs_tbd = vs_tsd->tensorBufferDesc();

        if ( vs_tbd->pool() != pool )
        {
            // not in the same pool...  no collision.
            continue;
        }

        NvU64 vsSurfaceBegin = vs_tbd->poolOffset();
        NvU64 vsSurfaceEnd   = vsSurfaceBegin + vs_tbd->size();
        NvU64 surfaceBegin = tbd->poolOffset();
        // NvU64 surfaceEnd   = surfaceBegin + tbd->size();

        if ( (surfaceBegin > vsSurfaceBegin) && (surfaceBegin < vsSurfaceEnd) )
        {
            // there's an overlap here.
            collisions.push_back(SurfacePair(tsd, vs_tsd));
        }
    }

    if ( m_debug )
    {
        if ( collisions.size() )
        {
            std::string delim("");
            gLogInfo << "detected memory reuse hazard(s) in pool " << pool->name() << ": allocated tsd=" << tsd->id() <<
                " vs. deallocated tsd(s)=[";

            for ( size_t ci = 0, CI = collisions.size(); ci != CI; ++ci )
            {
                // there's an overlap here.
                gLogInfo << delim << collisions[ci].second->id();
                delim = ", ";
            }
            gLogInfo << "]" << endl;
        }

    }

 fail:
    return e;
}

//
// note: at the first non-NvDlaSuccess return from any of these... visitGraphEnd will be
// called with that error code as 've'.
//
NvDlaError memory::MemoryResolver::visitEnd(engine_ast::Graph *, NvDlaError ve)
{

    if ( m_debug ) { gLogInfo << "end memory resolver" << endl; }
    return ve;
}

}; // nvdla::priv::memory


}; // nvdla::priv
}; // nvdla
