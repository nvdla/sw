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

#include "nvdla_os_inf.h"

#include "priv/Check.h"
#include "priv/Loadable.h"

#include "priv/loadable_generated.h"

#include "ErrorMacros.h"

using std::endl;


namespace nvdla
{

ILoadable::ILoadable()  { }
ILoadable::~ILoadable() { }

namespace priv
{

LoadableFactory::LoadablePrivPair LoadableFactory::newLoadable()
{
    ILoadable *loadable;
    Loadable *loadable_priv;
    loadable = loadable_priv = new priv::Loadable();
    if (loadable) {
        s_priv.insert(loadable, loadable_priv);
        s_self.insert(loadable, loadable);
    }
    return LoadablePrivPair(loadable, loadable_priv);
}

Loadable *LoadableFactory::priv(ILoadable *loadable)
{
    BiMap<ILoadable *, Loadable *>::left_iterator f = s_priv.find_left(loadable);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

ILoadable *LoadableFactory::i(Loadable *loadable)
{
    BiMap<ILoadable *, Loadable *>::right_iterator f = s_priv.find_right(loadable);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

ILoadable *LoadableFactory::self(void *s)
{
    BiMap<void *, ILoadable *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}


ILoadable *LoadableFactory::deserializeFrom(const std::string &fb_filename)
{
    NvDlaFileHandle file;
    NvDlaStatType finfo;
    size_t file_size;
    NvU8 *buf = 0;

    NvDlaError rc = NvDlaFopen(fb_filename.c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open " << fb_filename << endl;
        return 0;
    }

    rc = NvDlaFstat(file, &finfo);
    if ( rc != NvDlaSuccess)
    {
        gLogError << "couldn't get file stats for " << fb_filename << endl;
        return 0;
    }

    file_size = finfo.size;
    if ( !file_size ) {
        gLogError << "zero-length for " << fb_filename << endl;
        return 0;
    }

    buf = new NvU8[file_size];

    NvDlaFseek(file, 0, NvDlaSeek_Set);

    size_t actually_read = 0;

    rc = NvDlaFread(file, buf, file_size, &actually_read);
    if ( rc != NvDlaSuccess )
    {
        gLogError << "read error for " << fb_filename << endl;
        return 0;
    }
    NvDlaFclose(file);
    if ( actually_read != file_size ) {
        gLogError << "read wrong size for buffer? " << actually_read << endl;
        return 0;
    }


    return deserializeLoadable(buf);


}

BiMap<ILoadable *, Loadable*> LoadableFactory::s_priv;
BiMap<void *, ILoadable*> LoadableFactory::s_self;

// there's only one type of loadable for now. so only one of these... so it looks
// silly.  see the same paths in "LayerFactory::deserialize*" for why it makes sense
// to organize this way preemptively.
ILoadable *LoadableFactory::deserializeLoadable(NvU8 *buf)
{
    //    gLogError << __func__ << endl;
    LoadableFactory::LoadablePrivPair n = LoadableFactory::newLoadable();
    if ( !n ) {
        gLogError << __func__ << " error allocating new loadable" << endl;
        return NULL;
    }
    n.priv()->deserializeFrom(buf);
    return n.i();
}

Loadable::Loadable()
{

}

NvU16 Loadable::getFactoryType() const
{
    return 0; // only one type of loadable so far, not complicated by factory splits
}


std::string Loadable::getName() const
{
    return mName;
}

int Loadable::getNumMemoryListEntries() const
{
    return (int)mMemoryListEntries.size();
}

ILoadable::MemoryListEntry Loadable::getMemoryListEntry(NvU16 mem_id) const
{
    return mMemoryListEntries[mem_id];
}
const std::vector<ILoadable::MemoryListEntry> &Loadable::getMemoryListEntries() const
{
    return mMemoryListEntries;
}


int Loadable::getNumEventListEntries() const
{
    return (int)mEventListEntries.size();
}

ILoadable::EventListEntry Loadable::getEventListEntry(NvU16 event_id) const
{
    return mEventListEntries[event_id];
}
const std::vector<ILoadable::EventListEntry> &Loadable::getEventListEntries() const
{
    return mEventListEntries;
}


int Loadable::getNumTaskListEntries() const
{
    return mTaskListEntries.size();
}

ILoadable::TaskListEntry Loadable::getTaskListEntry(NvU16 task_id) const
{
    return mTaskListEntries[task_id];
}
const std::vector<ILoadable::TaskListEntry> &Loadable::getTaskListEntries() const
{
    return mTaskListEntries;
}


int Loadable::getNumSubmitListEntries() const
{
    return mSubmitListEntries.size();
}
ILoadable::SubmitListEntry Loadable::getSubmitListEntry(NvU16 submit_id) const
{
    return mSubmitListEntries[submit_id];
}
const std::vector<ILoadable::SubmitListEntry> &Loadable::getSubmitListEntries() const
{
    return mSubmitListEntries;
}


int Loadable::getNumAddressListEntries() const
{
    return mAddressListEntries.size();
}

ILoadable::AddressListEntry Loadable::getAddressListEntry(NvU16 address_list_index) const
{
    return mAddressListEntries[address_list_index];
}
const std::vector<ILoadable::AddressListEntry> &Loadable::getAddressListEntries() const
{
    return mAddressListEntries;
}


int Loadable::getNumTensorDescListEntries() const
{
    return mTensorDescListEntries.size();
}
ILoadable::TensorDescListEntry Loadable::getTensorDescListEntry(NvU16 tensor_desc_list_index) const
{
    return mTensorDescListEntries[tensor_desc_list_index];
}
const std::vector<ILoadable::TensorDescListEntry> &Loadable::getTensorDescListEntries() const
{
    return mTensorDescListEntries;
}


//
// internally facing
//
void Loadable::setMemoryListEntries(const std::vector<ILoadable::MemoryListEntry> &m)
{
    mMemoryListEntries = m;
}
void Loadable::setEventListEntries(const std::vector<ILoadable::EventListEntry> &m)
{
    mEventListEntries = m;
}
void Loadable::setTaskListEntries(const std::vector<ILoadable::TaskListEntry> &m)
{
    mTaskListEntries = m;
}
void Loadable::setSubmitListEntries(const std::vector<ILoadable::SubmitListEntry> &m)
{
    mSubmitListEntries = m;
}
void Loadable::setAddressListEntries(const std::vector<ILoadable::AddressListEntry> &e)
{
    mAddressListEntries = e;
}
void Loadable::setTensorDescListEntries(const std::vector<ILoadable::TensorDescListEntry> &e)
{
    mTensorDescListEntries = e;
}


int  Loadable::setSymbolContent(std::string name, const ILoadable::Blob &b, NvU8 *data)
{
    if ( debugSymbolContent() )
    {
        gLogInfo <<  "set symbol content name=" << name << " size=" << b.size << endl;
    }

    mSymbols[name].name = b.name;
    mSymbols[name].interface = b.interface;
    mSymbols[name].version   = b.version;
    mSymbols[name].size = b.size;
    mSymbols[name].data = data;

    return 0;
}

bool Loadable::getSymbolContent(std::string name, ILoadable::Blob &blob, NvU8 * &data)
{
    std::map<std::string, Symbol>::iterator f = mSymbols.find(name);

    if ( f == mSymbols.end() ) {

        if ( debugSymbolContent() )
        {
            gLogInfo <<  "missing symbol content for name=" << name << endl;
        }
        return false;
    }

    blob.name = f->second.name;
    blob.size = f->second.size;
    blob.version = f->second.version;
    blob.interface = f->second.interface;
    data = f->second.data;
    return true;

}

NvDlaError Loadable::getNetworkDataType(DataType::UnderlyingType *d) const
{
    NvDlaError e = NvDlaSuccess;
    if ( !d )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    *d = DataType::UnderlyingType(DataType::HALF);

 fail:
    return e;
}

NvDlaError Loadable::getNumInputTensors(int *inputs) const
{
    NvDlaError e = NvDlaSuccess;
    if ( !inputs )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    *inputs = 0;
    for (size_t mi = 0, MI = mMemoryListEntries.size(); mi != MI; ++mi ) {
        if ( mMemoryListEntries[mi].flags & ILoadable::MemoryListEntry::flags_input() ) {
            (*inputs)++;
        }
    }
 fail:
    return e;
}

NvDlaError Loadable::getNumOutputTensors(int *outputs) const
{
    NvDlaError e = NvDlaSuccess;
    if ( !outputs )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    *outputs = 0;
    for (size_t mi = 0, MI = mMemoryListEntries.size(); mi != MI; ++mi ) {
        if ( mMemoryListEntries[mi].flags & ILoadable::MemoryListEntry::flags_output() ) {
            (*outputs)++;
        }
    }
 fail:
    return e;
}

class MemoryListEntry_Input_BindId_Is
{
public:
    MemoryListEntry_Input_BindId_Is(NvU16 id) : m_find_id(id) { }
    bool operator()(const ILoadable::MemoryListEntry &mle) {
        return (mle.flags & mle.flags_input()) && (mle.bind_id == m_find_id);
    }
protected:
    NvU16 m_find_id;
};

class MemoryListEntry_Output_BindId_Is
{
public:
    MemoryListEntry_Output_BindId_Is(NvU16 id) : m_find_id(id) { }
    bool operator()(const ILoadable::MemoryListEntry &mle) {
        return (mle.flags & mle.flags_output()) && (mle.bind_id == m_find_id);
    }
protected:
    NvU16 m_find_id;
};

NvDlaError Loadable::getInputTensorDesc(NvU16 id, TensorDescListEntry *t) const
{
    NvDlaError e = NvDlaSuccess;
    std::vector<ILoadable::MemoryListEntry>::const_iterator f_mem;
    MemoryListEntry_Input_BindId_Is find_mle_with_input_id(id);

    if ( !t )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    f_mem = std::find_if(mMemoryListEntries.begin(), mMemoryListEntries.end(), find_mle_with_input_id);

    if ( f_mem == mMemoryListEntries.end() ) {
        // doesn't exist
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    if (f_mem->tensor_desc_id >= mTensorDescListEntries.size() ) {
        // exists but is bogus
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    *t = mTensorDescListEntries[f_mem->tensor_desc_id];

 fail:
    return e;
}

NvDlaError Loadable::getOutputTensorDesc(NvU16 id, TensorDescListEntry *t) const
{
    NvDlaError e = NvDlaSuccess;
    std::vector<ILoadable::MemoryListEntry>::const_iterator f_mem;
    MemoryListEntry_Output_BindId_Is find_mle_with_output_id(id);

    if ( !t )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    f_mem = std::find_if(mMemoryListEntries.begin(), mMemoryListEntries.end(), find_mle_with_output_id);

    if ( f_mem == mMemoryListEntries.end() ) {
        // doesn't exist
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    if (f_mem->tensor_desc_id >= mTensorDescListEntries.size() ) {
        // exists but is bogus
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    *t = mTensorDescListEntries[f_mem->tensor_desc_id];

 fail:
    return e;

}

Loadable::~Loadable()
{

}

bool Loadable::serializeToFlatBufferFile(const std::string &filename) const
{
    NvDlaError e = NvDlaSuccess;
    flatbuffers::FlatBufferBuilder fbb;

    std::vector<flatbuffers::Offset<nvdla::loadable::SubmitListEntry>>  submit_list;
    std::vector<flatbuffers::Offset<nvdla::loadable::TaskListEntry>>    task_list;
    std::vector<flatbuffers::Offset<nvdla::loadable::MemoryListEntry>>  memory_list;
    std::vector<flatbuffers::Offset<nvdla::loadable::AddressListEntry>> address_list;
    std::vector<flatbuffers::Offset<nvdla::loadable::EventListEntry>>   event_list;
    std::vector<flatbuffers::Offset<nvdla::loadable::Blob>>             blobs;
    std::vector<flatbuffers::Offset<nvdla::loadable::TensorDescListEntry>> tensor_desc_list;

    nvdla::loadable::Version loadable_version(nvdla::loadable::LoadableVersionMajor_VAL,
                                              nvdla::loadable::LoadableVersionMinor_VAL,
                                              nvdla::loadable::LoadableVersionSubMinor_VAL);

    for ( size_t ti = 0, TI = mTaskListEntries.size(); ti != TI; ++ti ) {
        const ILoadable::TaskListEntry & tle = mTaskListEntries[ti];
        auto addr_list_v = fbb.CreateVector<uint16_t>(tle.address_list);
        auto pre_actions_v = fbb.CreateVector<uint16_t>(tle.preactions);
        auto post_actions_v = fbb.CreateVector<uint16_t>(tle.postactions);

        nvdla::loadable::TaskListEntryBuilder tleb(fbb);
        tleb.add_address_list(addr_list_v);
        tleb.add_pre_actions (pre_actions_v);
        tleb.add_post_actions(post_actions_v);
        tleb.add_id(tle.id);
        nvdla::loadable::Interface if_id;
        if ( tle.interface == nvdla::ILoadable::TaskListEntry::interface_DLA1() ) {
            if_id = nvdla::loadable::Interface_DLA1;
        } else {
            if_id = nvdla::loadable::Interface_NONE;
        }
        tleb.add_interface(if_id);
        tleb.add_instance(tle.instance);
        task_list.push_back(tleb.Finish());
    }

    for ( size_t si = 0, SI = mSubmitListEntries.size(); si != SI; ++si ) {
        const ILoadable::SubmitListEntry & sle = mSubmitListEntries[si];
        auto tasks_v = fbb.CreateVector<uint16_t>(sle.tasks);
        nvdla::loadable::SubmitListEntryBuilder sleb(fbb);
        sleb.add_id(sle.id);
        sleb.add_task_id(tasks_v);
        submit_list.push_back(sleb.Finish());
    }

    for ( size_t mi = 0, MI = mMemoryListEntries.size(); mi != MI; ++mi) {

        const ILoadable::MemoryListEntry & mle = mMemoryListEntries[mi];
        auto contents_v = fbb.CreateVectorOfStrings(mle.contents);
        auto offsets_v = fbb.CreateVector<uint64_t>(mle.offsets);
        nvdla::loadable::MemoryListEntryBuilder mleb(fbb);
        mleb.add_contents(contents_v);
        mleb.add_offsets(offsets_v);
        mleb.add_size(mle.size);
        mleb.add_alignment(mle.alignment);
        mleb.add_bind_id(mle.bind_id);
        mleb.add_tensor_desc_id(mle.tensor_desc_id);
        mleb.add_flags( (nvdla::loadable::MemoryFlags) mle.flags);
        mleb.add_id(mle.id);
        mleb.add_domain((nvdla::loadable::MemoryDomain) mle.domain);
        memory_list.push_back(mleb.Finish());
    }


    for ( size_t ai = 0, AI = mAddressListEntries.size(); ai != AI; ++ai) {
        const ILoadable::AddressListEntry & ale = mAddressListEntries[ai];
        nvdla::loadable::AddressListEntryBuilder aleb(fbb);
        aleb.add_size(ale.size);
        aleb.add_offset(ale.offset);
        aleb.add_mem_id(ale.mem_id);
        aleb.add_id(ale.id);
        address_list.push_back(aleb.Finish());
    }


    for ( size_t ei = 0, EI = mEventListEntries.size(); ei != EI; ++ei) {
        const ILoadable::EventListEntry & ele = mEventListEntries[ei];
        nvdla::loadable::EventListEntryBuilder eleb(fbb);
        //        eleb.add_flags( (nvdla::loadable::EventFlags) ele.flags );
        eleb.add_id(ele.id);
        //        eleb.add_type( (nvdla::loadable::EventType) ele.type);
        eleb.add_op( (nvdla::loadable::EventOp) ele.op);
        eleb.add_target( ele.target );
        eleb.add_val( ele.val );
        event_list.push_back(eleb.Finish());
    }

    for ( std::map<std::string, Symbol>::const_iterator si = mSymbols.begin(); si != mSymbols.end(); ++si) {
        const Symbol &sym = si->second;
        auto data_v = fbb.CreateVector<uint8_t>(sym.data, sym.size);
        auto name_s = fbb.CreateString(sym.name.c_str());
        nvdla::loadable::BlobBuilder bb(fbb);
        bb.add_data(data_v);
        bb.add_size(sym.size);
        nvdla::loadable::Version v(sym.version.major, sym.version.minor, sym.version.sub_minor);
        bb.add_version(&v);
        bb.add_interface( (nvdla::loadable::Interface) sym.interface);
        bb.add_name(name_s);
        blobs.push_back(bb.Finish());
    }

    for ( size_t tdi = 0, TDI = mTensorDescListEntries.size(); tdi != TDI; ++tdi) {
        const ILoadable::TensorDescListEntry & ele = mTensorDescListEntries[tdi];
        nvdla::loadable::TensorDescListEntryBuilder tdleb(fbb);

        tdleb.add_id(ele.id);
        tdleb.add_mem_id(ele.mem_id);
        tdleb.add_size(ele.size);
        tdleb.add_offset(ele.offset);

        tdleb.add_data_format(  nvdla::loadable::DataFormat(ele.data_format) );
        tdleb.add_data_type(  nvdla::loadable::DataType(ele.data_type) );
        tdleb.add_data_category(  nvdla::loadable::DataCategory(ele.data_category) );
        tdleb.add_pixel_format(  nvdla::loadable::PixelFormat(ele.pixel_format) );
        tdleb.add_pixel_mapping(  nvdla::loadable::PixelMapping(ele.pixel_mapping) );
        tdleb.add_n( ele.dims.n );
        tdleb.add_c( ele.dims.c );
        tdleb.add_h( ele.dims.h );
        tdleb.add_w( ele.dims.w );

        tdleb.add_line_stride( ele.line_stride );
        tdleb.add_surf_stride( ele.surf_stride );
        tdleb.add_plane_stride( ele.plane_stride );

        tdleb.add_reserved0( 0 );
        tdleb.add_reserved1( 0 );
        tdleb.add_reserved2( 0 );
        tdleb.add_reserved3( 0 );

        tensor_desc_list.push_back(tdleb.Finish());
    }


    flatbuffers::Offset<nvdla::loadable::Loadable> l =
        CreateLoadableDirect(fbb, &loadable_version, &task_list, &memory_list, &address_list, &event_list, &blobs, &tensor_desc_list, &submit_list);

    //fbb.Finish(l, filename.c_str());
    fbb.Finish(l, "NVDA");

    size_t file_size = fbb.GetSize();
    NvU8 * file_data = (NvU8*) fbb.GetBufferPointer();

    NvDlaFileHandle file;
    PROPAGATE_ERROR_FAIL(NvDlaFopen(filename.c_str(), NVDLA_OPEN_WRITE, &file));
    PROPAGATE_ERROR_FAIL(NvDlaFwrite(file, file_data, file_size));
    NvDlaFclose(file);

fail:
    return true;
}


bool Loadable::deserializeFrom(NvU8 *flatbuf)
{
    const nvdla::loadable::Loadable *loadable = nvdla::loadable::GetLoadable(flatbuf);

    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::TaskListEntry>> *task_list    = loadable->task_list();
    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::SubmitListEntry>>  *submit_list  = loadable->submit_list();
    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::MemoryListEntry>>  *memory_list  = loadable->memory_list();
    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::AddressListEntry>> *address_list = loadable->address_list();
    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::EventListEntry>>   *event_list   = loadable->event_list();
    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::Blob>>             *blobs        = loadable->blobs();
    const flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::TensorDescListEntry>> *tensor_desc_list        = loadable->tensor_desc_list();

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::TaskListEntry>>::const_iterator tli = task_list->begin();

    for (; tli != task_list->end(); ++tli) {

        ILoadable::TaskListEntry tle;

        tle.id = tli->id();
        tle.interface = tli->interface();
        tle.instance = tli->instance();

        if ( tli->address_list() ) {
            flatbuffers::Vector<short unsigned int>::const_iterator ali = tli->address_list()->begin();
            for (; ali != tli->address_list()->end(); ++ali) {
                tle.address_list.push_back(*ali);
            }
        }

        if ( tli->pre_actions() ) {
            flatbuffers::Vector<short unsigned int>::const_iterator preli = tli->pre_actions()->begin();
            for (; preli != tli->pre_actions()->end(); ++preli) {
                tle.preactions.push_back(*preli);
            }
        }

        if ( tli->post_actions() ) {
            flatbuffers::Vector<short unsigned int>::const_iterator postli = tli->post_actions()->begin();
            for (; postli != tli->post_actions()->end(); ++postli) {
                tle.postactions.push_back(*postli);
            }
        }
        mTaskListEntries.push_back(tle);
    }

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::SubmitListEntry>>::const_iterator sli = submit_list->begin();

    for (; sli != submit_list->end(); ++sli) {

        ILoadable::SubmitListEntry sle;

        sle.id = sli->id();

        if ( sli->task_id() ) {
            flatbuffers::Vector<short unsigned int>::const_iterator tli = sli->task_id()->begin();
            for (; tli != sli->task_id()->end(); ++tli) {
                sle.tasks.push_back(*tli);
            }
        }
        mSubmitListEntries.push_back(sle);
    }

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::MemoryListEntry>>::const_iterator li = memory_list->begin();

    for (; li != memory_list->end(); ++li) {

        ILoadable::MemoryListEntry mle;

        mle.id = li->id();
        mle.size = li->size();
        mle.alignment = li->alignment();
        mle.flags = li->flags();
        mle.domain = li->domain();
        mle.bind_id = li->bind_id();
        mle.tensor_desc_id = li->tensor_desc_id();

        if ( li->contents() ) {
            flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String> >::const_iterator mli = li->contents()->begin();
            for (; mli != li->contents()->end(); ++mli) {
                mle.contents.push_back(mli->str());
            }
        }

        if ( li->offsets() ) {
            flatbuffers::Vector<uint64_t>::const_iterator mli = li->offsets()->begin();
            for (; mli != li->offsets()->end(); ++mli) {
                mle.offsets.push_back(*mli);
            }
        }
        mMemoryListEntries.push_back(mle);
    }

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::AddressListEntry>>::const_iterator adli = address_list->begin();

    for (; adli != address_list->end(); ++adli ) {

        ILoadable::AddressListEntry ale;

        ale.size = adli->size();
        ale.offset = adli->offset();
        ale.mem_id = adli->mem_id();
        ale.id = adli->id();

        mAddressListEntries.push_back(ale);
    }

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::EventListEntry>>::const_iterator eli = event_list->begin();

    for (; eli != event_list->end(); ++eli ) {

        ILoadable::EventListEntry ele;

        ele.id = eli->id();
        ele.op = eli->op();
        ele.target = eli->target();
        ele.val = eli->val();

        mEventListEntries.push_back(ele);
    }

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::Blob>>::const_iterator bi = blobs->begin();

    for (; bi != blobs->end(); ++bi ) {

        std::string blob_name = bi->name()->str();

        mSymbols[blob_name].name = blob_name;
        mSymbols[blob_name].size = bi->size();
        mSymbols[blob_name].version.major = bi->version()->major();
        mSymbols[blob_name].version.minor = bi->version()->minor();
        mSymbols[blob_name].version.sub_minor = bi->version()->sub_minor();
        mSymbols[blob_name].interface = (nvdla::ILoadable::Interface)bi->interface();

        NvU8 *blob_data = new NvU8[mSymbols[blob_name].size];
        memset(blob_data, 0, mSymbols[blob_name].size);

        NvU8 *binblob = (NvU8 *)bi->data()->Data();
        memcpy((void*)blob_data, (void *)binblob, mSymbols[blob_name].size);

        mSymbols[blob_name].data = blob_data;

    }

    flatbuffers::Vector<flatbuffers::Offset<nvdla::loadable::TensorDescListEntry>>::const_iterator tdle = tensor_desc_list->begin();

    for (; tdle != tensor_desc_list->end(); ++tdle ) {

        ILoadable::TensorDescListEntry ele;

        ele.id = tdle->id();
        ele.mem_id = tdle->mem_id();
        ele.size = tdle->size();
        ele.offset = tdle->offset();

        ele.data_format = tdle->data_format();
        ele.data_type = tdle->data_type();
        ele.data_category = tdle->data_category();
        ele.pixel_format = tdle->pixel_format();
        ele.pixel_mapping = tdle->pixel_mapping();
        ele.dims.n = tdle->n();
        ele.dims.c = tdle->c();
        ele.dims.h = tdle->h();
        ele.dims.w = tdle->w();

        ele.line_stride = tdle->line_stride();
        ele.surf_stride = tdle->surf_stride();
        ele.plane_stride = tdle->plane_stride();

        mTensorDescListEntries.push_back(ele);
    }

    return true;
}



} // nvdla::priv

} // nvdla
