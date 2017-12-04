/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <sstream>

#include "dlatypes.h"
#include "dlaerror.h"

#include "nvdla_inf.h"

#include "priv/Loadable.h"
#include "priv/Runtime.h"

#include "priv/loadable_generated.h"

#include "ErrorMacros.h"

using std::vector;
using std::stringstream;
using std::string;
using std::endl;


namespace nvdla
{

IRuntime::IRuntime() { }
IRuntime::~IRuntime() { }

IRuntime *createRuntime()
{
    priv::RuntimeFactory::RuntimePrivPair p = priv::RuntimeFactory::newRuntime();
    return p.i();
}

namespace priv
{

RuntimeFactory::RuntimePrivPair RuntimeFactory::newRuntime()
{
    IRuntime *runtime;
    Runtime *runtime_priv;
    runtime = runtime_priv = new priv::Runtime();
    if (runtime) {
        s_priv.insert(runtime, runtime_priv);
        s_self.insert(runtime, runtime);
    }
    return RuntimePrivPair(runtime, runtime_priv);
}

Runtime *RuntimeFactory::priv(IRuntime *runtime)
{
    BiMap<IRuntime *, Runtime *>::left_iterator f = s_priv.find_left(runtime);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

IRuntime *RuntimeFactory::i(Runtime *runtime)
{
    BiMap<IRuntime *, Runtime *>::right_iterator f = s_priv.find_right(runtime);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

IRuntime *RuntimeFactory::self(void *s)
{
    BiMap<void *, IRuntime *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}

BiMap<IRuntime *, Runtime*> RuntimeFactory::s_priv;
BiMap<void *, IRuntime*> RuntimeFactory::s_self;


Runtime::Runtime() :
    IRuntime(),
    m_dla_handle(0),
    h_network_desc_mem(0),
    h_op_desc_mem(0),
    h_surf_desc_mem(0),
    h_dependency_list_mem(0)
{
    m_dla_device_handles[0] = 0;
    m_dla_device_handles[1] = 0;
    m_loaded = 0;
}

Runtime::~Runtime()
{

}

NvU16 Runtime::getFactoryType() const
{
    return 0;
}

void *Runtime::getDLADeviceContext(size_t sel_i)
{
    bool ok = true;
    NvDlaError err;

    if (sel_i > 0) {
        ok = false;
        goto done;
    }

    if ( !m_dla_device_handles[sel_i] ) {
        err = NvDlaInitialize(&m_dla_handle);
        ok = err == NvDlaSuccess;
        if ( !ok ) {
            goto done;
        }

        err = NvDlaOpen((void *)m_dla_handle, sel_i, (void **)&m_dla_device_handles[sel_i]);
        ok = err == NvDlaSuccess;
        if ( !ok ) {
            gLogError << "failed to open dla device" << endl;
            m_dla_device_handles[sel_i] = 0;
        }
    }

 done:
    if ( ok ) {
        return m_dla_device_handles[sel_i];
    }
    return 0;
}


NvU16 Runtime::getMaxDevices()
{
    return getMaxDLADevices();
}

NvU16 Runtime::getNumDevices()
{
    NvU16 num_devs = 0;

    for ( size_t di = 0; di < getMaxDLADevices(); ++di ) {
        if ( getDLADeviceContext(di) ) {
            num_devs++;
        }
    }

    return num_devs;
}

bool Runtime::versionsCompatible(const ILoadable::Version &a, const ILoadable::Version &b)
{
    return (a.major == b.major) && (a.minor == b.minor);
}

bool Runtime::load(NvU8 *buf, int instance)
{
    NvDlaError e = NvDlaSuccess;
    ILoadable *i_loadable;
    Loadable *loadable;

    bool ok = true;

    i_loadable = LoadableFactory::deserializeLoadable(buf);
    if ( !i_loadable )
    {
        ok = false;
        goto done;
    }

    loadable = LoadableFactory::priv(i_loadable);

    if ( instance >= 0 )
    {
        if ( instance >= getNumDevices() )
        {
            gLogError << "Out of bounds DLA instance " << instance << " requested." << endl;
            ok = false;
            goto done;
        }

        m_loaded_instance = size_t(instance);
    }
    else
    {
        m_loaded_instance = 0;
    }

    m_task_entries   = loadable->getTaskListEntries();
    m_submit_entries = loadable->getSubmitListEntries();
    m_memory_entries = loadable->getMemoryListEntries();
    m_address_entries = loadable->getAddressListEntries();
    m_tensor_desc_entries = loadable->getTensorDescListEntries();

    if ( m_submit_entries.size() < 1 || m_task_entries.size() < 1 || m_memory_entries.size() < 1 ) {
        gLogError << "need at least one submit task and memory entry to load" << endl;
        ok = false;
        goto done;
    }

    m_memory.resize(m_memory_entries.size());
    if ( debugMemoryLayout() )
    {
        gLogInfo << "load memory list entries=" << m_memory.size() << endl;
    }

    for ( size_t mi = 0, MI = m_memory_entries.size(); mi != MI; ++mi ) {
        m_memory[mi] = Memory(m_memory_entries[mi]);
    }

    PROPAGATE_ERROR( initBindableMemory() );

    //
    // for all entries hit their load methods.
    // some might not require work yet (io entries, etc).
    // but some may trigger allocation and filling of
    // items/ events.

    for ( size_t mi = 0, MI = m_memory_entries.size(); mi != MI; ++mi ) {
        PROPAGATE_ERROR_FAIL( loadMemory(loadable, &m_memory[mi]) );
    }

    m_address.resize(m_address_entries.size());
    if ( debugMemoryLayout() )
    {
        gLogInfo << "load address list entries=" << m_address.size() << endl;
    }

    for ( size_t ai = 0, AI = m_address_entries.size(); ai != AI; ++ai ) {
        m_address[ai] = Address(m_address_entries[ai]);
        if ( debugMemoryLayout() )
        {
            gLogInfo << "load\t id=" << m_address[ai].id() <<
                " mem_id=" << m_address[ai].mem_id() <<
                " offset=" << m_address[ai].offset() << endl;
        }
    }

    m_task.resize(m_task_entries.size());
    if ( debugTasks() )
    {
        gLogInfo << "load num tasks=" << m_task.size() << endl;
    }

    m_numDLATasks = 0;

    for ( size_t ti = 0, TI = m_task_entries.size(); ti != TI; ++ti )
    {
        m_task[ti] = Task(m_task_entries[ti]);

        //
        // check task entries for explicit dla instance assignments
        // and complain about non-zero requests.  the production
        // runtime doesn't/won't support explicit assignments of
        // dla instances in the loadable.
        //

        if ( m_task[ti].interface() == ILoadable::Interface_DLA1 )
        {
            if ( m_task[ti].instance() != ILoadable::TaskListEntry::instance_ANY() )
            {
                gLogWarning << "the loadable specified dla instance " <<
                    m_task[ti].instance() << " which will be ignored.";
            }
            m_numDLATasks++;
        }

        if ( debugTasks() )
        {
            gLogInfo << "load\ttask id=" << ti << " address list entries=" <<
                m_task[ti].address_list().size() << endl;
        }
    }

    m_submit.resize(m_submit_entries.size());

    for ( size_t si = 0, SI = m_submit_entries.size(); si != SI; ++si ) {
        m_submit[si] = Submit(m_submit_entries[si]);
        if ( debugTasks() )
        {
            gLogInfo << "load\tsubmit id" << si << " tasks=" << m_submit[si].tasks().size() << endl;
        }
    }

    ok = true;
    m_loaded = loadable;

 done:
    return ok;

 fail:
    return false;
}

//
// probably need a bit more after rebind...
//
bool Runtime::bindInputTensor(int index, void *hMem)
{
    std::vector<Memory *>bind_to_mem;
    bool ok = true;
    if ( index < 0 ) {
        ok = false;
        goto done;
    }

    // determine which mem needs to be rebound
    for ( size_t mi = 0, MI = m_memory.size(); mi != MI; ++mi ) {
        if ( m_memory[mi].inputBindId() == index ) {
            bind_to_mem.push_back( &m_memory[mi] );
        }
    }

    // unlikely to be > size 1, but...
    for (size_t bmi = 0, BMI = bind_to_mem.size(); bmi != BMI; ++bmi ) {
        bind_to_mem[bmi]->setHandle(hMem);
    }

 done:
    return ok;
}

bool Runtime::bindOutputTensor(int index, void *hMem)
{
    bool ok = true;
    std::vector<Memory *> bind_to_mem;
    if ( index < 0 ) {
        ok = false;
        goto done;
    }

    // determine which mem needs to be rebound
    for ( size_t mi = 0, MI = m_memory.size(); mi != MI; ++mi ) {
        if ( m_memory[mi].outputBindId() == index ) {
            bind_to_mem.push_back( &m_memory[mi] );
        }
    }

    // unlikely to be > size 1, but...
    for (size_t bmi = 0, BMI = bind_to_mem.size(); bmi != BMI; ++bmi ) {
        bind_to_mem[bmi]->setHandle(hMem);
    }

done:
    return ok;
}

bool Runtime::fillTaskAddressList(Task *task, NvDlaTask *dla_task)
{
    size_t num_memory_ids = m_memory.size();
    size_t num_task_addr_list_entries = task->mEntry.address_list.size();

    if ( num_task_addr_list_entries > NVDLA_MAX_BUFFERS_PER_TASK )
    {
        gLogError << "too many address list entries." << endl;
        return false;
    }

    dla_task->num_addresses = num_task_addr_list_entries;

    for ( size_t ali = 0, ALI = num_task_addr_list_entries; ali != ALI; ++ali )
    {

        NvS16 address_list_entry_id = task->mEntry.address_list[ali];
        if ( ! ( (address_list_entry_id >= 0) && (size_t(address_list_entry_id) < m_address.size() )) )
        {
            gLogError << "dla address list entry=" << ali << " id=" << address_list_entry_id << " is bogus" << endl;
            return false;
        }

        NvS16 memory_id = m_address[address_list_entry_id].mem_id();
        // there should be valid handles at every ali/mem_id for a dla task.
        if ( ! ((memory_id >= 0) && (size_t(memory_id) < num_memory_ids) ))
        {
            gLogError << "mem_id=" << memory_id << " out of bounds." << endl;
            return false;
        }

        if (m_memory[memory_id].hMem == 0)
        {
            gLogError << __func__ << " ali=" << ali << " -> mem_id=" << memory_id << " has a null memory handle." << endl;
            return false;
        }

        Memory *mem = &m_memory[memory_id];
        void *hMem   = mem->getHandle();

        dla_task->address_list[ali].handle = hMem;
        dla_task->address_list[ali].offset = m_address[address_list_entry_id].mEntry.offset;
    }

    return true;
}

bool Runtime::submit()
{
    NvDlaError e = NvDlaSuccess;
    e = submitInternal();
    return e == NvDlaSuccess;
}

NvDlaError Runtime::submitInternal()
{
    NvDlaError e = NvDlaSuccess;
    Task *task;

    size_t ii;

    bool ok = true;
    NVDLA_UNUSED(ok);
    if ( !m_loaded ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "exec requires a successful load first");
    }

    if ( !m_task.size() ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "no tasks to exec");
    }

    if ( !m_submit.size() ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "no submission sets to exec");
    }

    for ( size_t ss=0; ss < m_submit.size(); ss++ ) {

        for ( ii=0; ii < m_submit[ss].tasks().size(); ii++ )
        {
            size_t task_id = m_submit[ss].tasks()[ii];

            if ( task_id >= m_task.size() ) {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "task id out of range");
            }

            task = &m_task[ task_id ];

            switch ( task->interface() ) {

                case ILoadable::Interface_DLA1:
                {
                    void *dev;
                    NvDlaTask dla_task;

                    dev = getDLADeviceContext(m_loaded_instance);

                    std::memset(&dla_task, 0, sizeof(dla_task));

                    dla_task.task_id = task->id();

                    fillTaskAddressList(task, &dla_task);

                    PROPAGATE_ERROR_FAIL( NvDlaSubmit(NULL, dev, &dla_task, 1) );
                }
                break;
                default:
                    ok = false;
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "unrecognized interface %d", task->interface());
                    break;

            } // switch on engine type

        } // each task in a submission

    } // each submission set

fail:
    return e;
}

NvDlaError Runtime::allocateSystemMemory(void **phMem, NvU64 size, void **pData)
{
    NvDlaError e = NvDlaSuccess;
    void *hDla = getDLADeviceContext(m_loaded_instance);

    /* Allocate memory for network */
    PROPAGATE_ERROR_FAIL( NvDlaAllocMem(NULL, hDla, phMem, pData, size, NvDlaHeap_System) );

fail:
    return e;
}

NvDlaError Runtime::loadMemory(Loadable *l, Memory *memory)
{
    NvDlaError e = NvDlaSuccess;
    void *hMem;
    NvU8 *mem;
    NVDLA_UNUSED(mem);

    bool ok = false;
    if ( ! (memory->flags() & ILoadable::MemoryListEntry::flags_alloc()) ) {
        return NvDlaSuccess;
    }

    // skip top-level, bindable buffers.  they are dealt with explicitly elsewhere
    if (  memory->bindable() ) {
        return NvDlaSuccess;
    }

    if ( memory->domain() == ILoadable::MemoryListEntry::domain_sysmem() )
    {

        void *mapped_mem;

        NvU64 size = memory->size();
        void *hDla = getDLADeviceContext(m_loaded_instance);

        /* Allocate memory for network */
        PROPAGATE_ERROR_FAIL( NvDlaAllocMem(m_dla_handle, hDla, &hMem, (void **)(&mapped_mem), size, NvDlaHeap_System) );

        memory->setHandle(hMem);

        if ( memory->flags() & ILoadable::MemoryListEntry::flags_set() )
        {

            if ( memory->contents().size() != memory->offsets().size() ) {
                ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState,
                                     "mismatch on num content blobs vs. num offsets in memory id");
            }
            vector<string> &contents = memory->contents();
            vector<uint64_t> &offsets  = memory->offsets();

            for ( size_t ci = 0, CI = contents.size(); ci != CI; ++ci )
            {
                ILoadable::Blob content_blob;
                NvU8 *data;

                const string &content_symbol = contents[ci];

                ok = l->getSymbolContent(content_symbol, content_blob, data);
                if ( !ok ) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState,
                                         "failed to find buffer content symbol %s",
                                         content_symbol.c_str());
                }

                if ( memory->size() >= (NvU64)(offsets[ci] + content_blob.size) )
                {
                    NvU8 *src = data;
                    NvU8 *dst = (NvU8*)mapped_mem + offsets[ci];

                    for ( size_t byte = 0; byte < content_blob.size; byte++ ) {
                        dst[byte] = src[byte];
                    }
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "content blob too large for pool size");
                }
            }
        }
    }

 fail:
    return e;
}

NvDlaError Runtime::getNetworkDataType(DataType::UnderlyingType * /*data_type*/) const
{
    NvDlaError e = NvDlaSuccess;
    return e;
}

//
// create arrays of {input, output} X {id-ordered bind ids} -> Memory objects
// check for duplicates and other sorts of malformed-ness.
//
NvDlaError Runtime::initBindableMemory()
{
    NvDlaError e = NvDlaSuccess;

    m_tensor_desc.resize(m_tensor_desc_entries.size());

    for ( size_t tdi = 0, TDI = m_tensor_desc_entries.size(); tdi != TDI; ++tdi ) {
        m_tensor_desc[tdi] = TensorDesc(m_tensor_desc_entries[tdi]);
    }

    m_bindable_memory.resize(IOD_Max);

    for ( size_t mi = 0, MI = m_memory.size(); mi != MI; ++mi )
    {
        IOD which_iod;
        int bind_id;

        bind_id = m_memory[mi].bindId(which_iod);
        if ( bind_id == -1 ) {
            continue;
        }

        // insert and detect any duplicates
        {
            std::vector<Memory *> &which_mem = m_bindable_memory[which_iod];
            MemoryId_BindId_Is check_for_dup_id(bind_id);

            if ( which_mem.end() == std::find_if(which_mem.begin(), which_mem.end(), check_for_dup_id) ) {
                which_mem.push_back(&m_memory[mi]);
            } else {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Duplicate bind ids on separate memory objects in runtime.");
            }
        }
    }

    // it's possible that these we're given out of order.
    // sort now based upon bind id in each category.
    {
        Memory_BindId_LT_Compare less_than;
        IOD w;
        NVDLA_UNUSED(w);
        for ( size_t w = 0; w < size_t(IOD_Max); w++ )
        {
            int num_ids = int(m_bindable_memory[w].size());

            if ( ! m_bindable_memory[w].size() ) {
                continue;
            }

            std::sort(m_bindable_memory[w].begin(),  m_bindable_memory[w].end(), less_than);

            // now check to be sure there are no gaps/out-of-bounds ids
            IOD na;
            int first_id = m_bindable_memory[w][0]->bindId(na);
            int last_id  = m_bindable_memory[w][num_ids-1]->bindId(na);

            if ( (first_id != 0) || (last_id != (num_ids-1)) ) {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Out of bounds bind id on memory object");
            }
        }
    }

 fail:
    return e;
}

NvDlaError Runtime::getNumInputTensors(int *inputs)
{
    NvDlaError e = NvDlaSuccess;
    int input_id = 0;
    NVDLA_UNUSED(input_id);
    if ( !inputs )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    *inputs = m_bindable_memory[IOD_Input].size();

 fail:
    return e;
}

NvDlaError Runtime::getNumOutputTensors(int *outputs)
{
    NvDlaError e = NvDlaSuccess;
    if ( !outputs )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    *outputs = m_bindable_memory[IOD_Output].size();
 fail:
    return e;
}

NvDlaError Runtime::getMemoryFromBindId(IOD w, int id, Memory * &bound_mem)
{
    NvDlaError e = NvDlaSuccess;
    if ( (id < 0) || (size_t(id) >= m_bindable_memory[w].size()) ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Bind id out of range:%d", id);
    }

    bound_mem = m_bindable_memory[w][id];
 fail:
    return e;
}

NvDlaError Runtime::getInputTensorDesc(int id, NvDlaTensor *td)
{
    NvDlaError e = NvDlaSuccess;
    Memory *bound_mem = 0;
    int tensor_desc_id = -1;
    ILoadable::TensorDescListEntry t;

    if ( !td )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    PROPAGATE_ERROR_FAIL( getMemoryFromBindId(IOD_Input, id, bound_mem) );

    tensor_desc_id = bound_mem->tensorDescId();
    if ( (tensor_desc_id < 0) || (size_t(tensor_desc_id) >= m_tensor_desc.size()) ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Tensor desc id out of range:%d", tensor_desc_id);
    }

    t = m_tensor_desc[tensor_desc_id].mEntry;

    td->bufferSize = t.size;
    td->n = t.dims.n;
    td->c = t.dims.c;
    td->h = t.dims.h;
    td->w = t.dims.w;
    td->dataFormat = t.data_format;
    td->dataType = t.data_type;
    td->pixelFormat = t.pixel_format;
    td->pixelMapping = t.pixel_mapping;

    td->pitchLinear.lineStride = t.line_stride;
    td->pitchLinear.surfStride = t.surf_stride;
    td->pitchLinear.planeStride = t.plane_stride;

 fail:
    return e;
}

NvDlaError Runtime::getOutputTensorDesc(int id, NvDlaTensor *td)
{
    NvDlaError e = NvDlaSuccess;
    Memory *bound_mem = 0;
    int tensor_desc_id = -1;
    ILoadable::TensorDescListEntry t;

    if ( !td )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    PROPAGATE_ERROR_FAIL( getMemoryFromBindId(IOD_Output, id, bound_mem) );

    tensor_desc_id = bound_mem->tensorDescId();
    if ( (tensor_desc_id < 0) || (size_t(tensor_desc_id) >= m_tensor_desc.size()) ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Tensor desc id out of range:%d", tensor_desc_id);
    }

    t = m_tensor_desc[tensor_desc_id].mEntry;

    td->bufferSize = t.size;
    td->n = t.dims.n;
    td->c = t.dims.c;
    td->h = t.dims.h;
    td->w = t.dims.w;
    td->dataFormat = t.data_format;
    td->dataType = t.data_type;
    td->pixelFormat = t.pixel_format;
    td->pixelMapping = t.pixel_mapping;

    td->pitchLinear.lineStride = t.line_stride;
    td->pitchLinear.surfStride = t.surf_stride;
    td->pitchLinear.planeStride = t.plane_stride;

 fail:
    return e;
}

//
// take a tensor descriptor from the user-facing API and inspect the
// changed elements.  react to those which can be legitimately tweaked
// (.e.g line_stride) and complain about any which cannot.
//
NvDlaError Runtime::mergeSetTensorDesc(IOD /*w*/, int /*bindId*/, int tensorDescId, const ILoadable::TensorDescListEntry *tdl)
{
    NvDlaError e = NvDlaSuccess;

    ILoadable::TensorDescListEntry * origEntry = &m_tensor_desc[tensorDescId].mEntry;
    // Runtime::TensorDesc *origDesc = &m_tensor_desc[tensorDescId];

    bool sizeDiff   = origEntry->size   != tdl->size;
    bool offsetDiff = origEntry->offset != tdl->offset;

    bool dimsDiff = ( (origEntry->dims.n != tdl->dims.n) ||
                      (origEntry->dims.c != tdl->dims.c) ||
                      (origEntry->dims.h != tdl->dims.h) ||
                      (origEntry->dims.w != tdl->dims.w) );

    bool lineStrideDiff  = origEntry->line_stride  != tdl->line_stride;
    bool surfStrideDiff  = origEntry->surf_stride  != tdl->surf_stride;
    bool planeStrideDiff = origEntry->plane_stride != tdl->plane_stride;

    bool dataFormatDiff   = origEntry->data_format   != tdl->data_format;
    bool dataTypeDiff     = origEntry->data_type     != tdl->data_type;
    bool dataCategoryDiff = origEntry->data_category != tdl->data_category;

    bool pixelFormatDiff  = origEntry->pixel_format  != tdl->pixel_format;
    bool pixelMappingDiff = origEntry->pixel_mapping != tdl->pixel_mapping;

    // at the moment no formatting changes are allowed.  eventually we may be
    // in a position to do inline or hand-off formatting operations.
    if ( dataFormatDiff || dataTypeDiff || dataCategoryDiff )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotImplemented, "data format/type/category change requested");
    }

    // pixel format and mapping changes are unlikely to be allowed
    if ( pixelFormatDiff || pixelMappingDiff )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "pixel format/mapping change requested");
    }

    // do we want to be able to set different offsets?  size might need to follow if so.
    if ( sizeDiff || offsetDiff )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotImplemented, "size/offset change requested");
    }

    // allow changing dimensions?  unlikely, but maybe?
    if ( dimsDiff )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "dimensions change requested");
    }

    if ( lineStrideDiff || surfStrideDiff || planeStrideDiff )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotImplemented, "stride changes not implemented yet");
    }

 fail:
    return e;
}

NvDlaError Runtime::setInputTensorDesc(int bindId, const NvDlaTensor *td)
{
    NvDlaError e = NvDlaSuccess;
    Memory *boundMem = 0;
    int tensorDescId = -1;
    ILoadable::TensorDescListEntry tdl;

    if ( !td )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    PROPAGATE_ERROR_FAIL( getMemoryFromBindId(IOD_Input, bindId, boundMem) );

    tensorDescId = boundMem->tensorDescId();
    if ( (tensorDescId < 0) || (size_t(tensorDescId) > m_tensor_desc.size()) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Tensor desc id out of range:%d", tensorDescId);
    }

    //     m_tensor_desc[tensorDescId].mEntry = *tdl;
    PROPAGATE_ERROR_FAIL( mergeSetTensorDesc(IOD_Input, bindId, tensorDescId, &tdl) );

 fail:
    return e;
}

NvDlaError Runtime::setOutputTensorDesc(int bindId, const NvDlaTensor *td)
{
    NvDlaError e = NvDlaSuccess;
    Memory *boundMem = 0;
    int tensorDescId = -1;
    ILoadable::TensorDescListEntry tdl;

    if ( !td )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    PROPAGATE_ERROR_FAIL( getMemoryFromBindId(IOD_Output, bindId, boundMem) );

    tensorDescId = boundMem->tensorDescId();
    if ( (tensorDescId < 0) || (size_t(tensorDescId) > m_tensor_desc.size()) )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Tensor desc id out of range:%d", tensorDescId);
    }

    //    m_tensor_desc[tensorDescId].mEntry = *tdl;
    PROPAGATE_ERROR_FAIL( mergeSetTensorDesc(IOD_Output, bindId, tensorDescId, &tdl) );

 fail:
    return e;

}

} // nvdla::priv

} // nvdla
