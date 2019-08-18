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

#include <sstream>
#include <vector>
#include <string>

#include "priv/Check.h"

#include "priv/Wisdom.h"
#include "priv/WisdomContainer.h"

#include "nvdla_os_inf.h"

using std::string;
using std::endl;
using std::vector;
using std::stringstream;

namespace nvdla
{

IWisdomContainer::IWisdomContainer(IWisdom *) { }
IWisdomContainer::~IWisdomContainer() { }

IWisdomContainerEntry::IWisdomContainerEntry() { }
IWisdomContainerEntry::~IWisdomContainerEntry() { }

namespace priv
{

//----------------------------------------------------------------------
// WisdomContainerEntry
//----------------------------------------------------------------------

WisdomContainerEntry::WisdomContainerEntry() : m_container(0) { }


WisdomContainerEntry::WisdomContainerEntry(WisdomContainer *c,
                                           const std::string &path,
                                           const std::string &name,
                                           IWisdomContainerEntry::EntryType type) :
    m_container(c),
    m_path(path),
    m_name(name),
    m_type(type)
{
    //    gLogError << "this=[" << this << "] new entry c=[" << c << "] path=[" << path <<
    //        "] name=[" << name << "] type=[" << int(type) << "]\n";
}
WisdomContainerEntry::~WisdomContainerEntry()
{

}


IWisdomContainer *WisdomContainerEntry::container() const
{
    return m_container;
}

const std::string WisdomContainerEntry::path() const
{
    return m_path;
}

const std::string WisdomContainerEntry::name() const
{
    return m_name;
}

IWisdomContainerEntry::EntryType WisdomContainerEntry::type() const
{
    return m_type;
}

bool WisdomContainerEntry::writeUInt8(NvU8 v)
{
    if ( ENTRY_TYPE_UINT8 != m_type )
    {
        return false;
    }
    return m_container->writeUInt8(pathName(), v);
}

bool WisdomContainerEntry::writeString(const std::string &v)
{
    if ( ENTRY_TYPE_STRING != m_type )
    {
        return false;
    }
    return m_container->writeString(pathName(), v);
}

bool WisdomContainerEntry::writeUInt8Vector(const NvU8 *v, size_t size)
{
    if ( ENTRY_TYPE_UINT8_VECTOR != m_type )
    {
        return false;
    }
    return m_container->writeUInt8Vector(pathName(), v, size);
}

bool WisdomContainerEntry::writeUInt8Vector(const std::vector<NvU8> &v)
{
    if ( ENTRY_TYPE_UINT8_VECTOR != m_type )
    {
        return false;
    }
    return m_container->writeUInt8Vector(pathName(), v);
}


bool WisdomContainerEntry::writeUInt32(NvU32 v)
{
    if ( ENTRY_TYPE_UINT32 != m_type )
    {
        return false;
    }
    return m_container->writeUInt32(pathName(), v);
}

bool WisdomContainerEntry::writeUInt64(NvU64 v)
{
    if ( ENTRY_TYPE_UINT64 != m_type )
    {
        return false;
    }
    return m_container->writeUInt64(pathName(), v);
}

bool WisdomContainerEntry::writeInt32(NvS32 v)
{
    if ( ENTRY_TYPE_INT32 != m_type )
    {
        return false;
    }
    return m_container->writeInt32(pathName(), v);
}

bool WisdomContainerEntry::writeFloat32(NvF32 v)
{
    if ( ENTRY_TYPE_FLOAT32 != m_type )
    {
        return false;
    }
    return m_container->writeFloat32(pathName(), v);
}

bool WisdomContainerEntry::readUInt8(NvU8 &v) const
{
    if ( ENTRY_TYPE_UINT8 != m_type )
    {
        return false;
    }
    return m_container->readUInt8(pathName(), v);
}

bool WisdomContainerEntry::readString(std::string &v) const
{
    if ( ENTRY_TYPE_STRING != m_type )
    {
        return false;
    }
    return m_container->readString(pathName(), v);
}

bool WisdomContainerEntry::readUInt8Vector(NvU8 **v, size_t *size) const
{
    if ( ENTRY_TYPE_UINT8_VECTOR != m_type )
    {
        return false;
    }
    return m_container->readUInt8Vector(pathName(), v, size);
}

bool WisdomContainerEntry::readUInt8Vector(std::vector<NvU8> &v) const
{
    if ( ENTRY_TYPE_UINT8_VECTOR != m_type )
    {
        return false;
    }
    return m_container->readUInt8Vector(pathName(), v);
}

bool WisdomContainerEntry::readUInt32(NvU32 &v) const
{
    if ( ENTRY_TYPE_UINT32 != m_type )
    {
        return false;
    }
    return m_container->readUInt32(pathName(), v);
}

bool WisdomContainerEntry::readUInt64(NvU64 &v) const
{
    if ( ENTRY_TYPE_UINT64 != m_type )
    {
        return false;
    }
    return m_container->readUInt64(pathName(), v);
}

bool WisdomContainerEntry::readInt32(NvS32 &v) const
{
    if ( ENTRY_TYPE_INT32 != m_type )
    {
        return false;
    }
    return m_container->readInt32(pathName(), v);
}

bool WisdomContainerEntry::readFloat32(NvF32 &v) const
{
    if ( ENTRY_TYPE_FLOAT32 != m_type )
    {
        return false;
    }
    return m_container->readFloat32(pathName(), v);
}


// externally facing
bool WisdomContainerEntry::insertEntry(const std::string &name,
                                       IWisdomContainerEntry::EntryType type,
                                       IWisdomContainerEntry *&entry)
{
    if ( ENTRY_TYPE_OBJECT != m_type )
    {
        return false;
    }
    return m_container->insertEntry(pathName(), name, type, entry);
}


// internally facing
bool WisdomContainerEntry::insertEntry(const std::string &name,
                                       IWisdomContainerEntry::EntryType type,
                                       WisdomContainerEntry *entry)
{
    if ( ENTRY_TYPE_OBJECT != m_type )
    {
        return false;
    }
    return m_container->insertEntry(pathName(), name, type, entry);
}



// externally visible version which allocs the entry
bool WisdomContainerEntry::getEntry(const std::string &name,
                                    IWisdomContainerEntry::EntryType type,
                                    IWisdomContainerEntry *&entry)
{
    if ( ENTRY_TYPE_OBJECT != m_type )
    {
        return false;
    }
    return m_container->getEntry(m_path, name, type, entry);
}

// internal version which uses the stack...
bool WisdomContainerEntry::getEntry(const std::string &name,
                                    IWisdomContainerEntry::EntryType type,
                                    WisdomContainerEntry *entry) const
{
    if ( ENTRY_TYPE_OBJECT != m_type )
    {
        return false;
    }
    return m_container->getEntry(m_path, name, type, entry);
}


bool WisdomContainerEntry::getEntryNames(std::vector<std::string> *names)
{
    if ( ENTRY_TYPE_OBJECT != m_type )
    {
        return false;
    }
    return m_container->getEntryNames(pathName(), names);
}

bool WisdomContainerEntry::removeEntry(const std::string &name)
{
    if ( ENTRY_TYPE_OBJECT != m_type )
    {
        return false;
    }
    return m_container->removeEntry(pathName() + "/" + name);
}

bool WisdomContainerEntry::insertEntryIfNotPresent(const std::string &name,
                                                   EntryType type,
                                                   WisdomContainerEntry *entry)
{
    bool ok;
    ok = getEntry(name, type, entry);
    if ( !ok ) {
        // try inserting it
        ok = insertEntry(name, type, entry);
    }
    return ok;
}

//----------------------------------------------------------------------
// serialize and deserialize some useful types by name into a sub-object
// of an exsting OBJECT type wisdom container entry.
//----------------------------------------------------------------------
bool WisdomContainerEntry::readUInt8Enum(const std::string &name, NvU8 &e) const
{
    bool ok = true;
    WisdomContainerEntry b_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_UINT8, &b_entry);
    ok = ok && b_entry.readUInt8(e);
    return ok;
}

bool WisdomContainerEntry::readString(const std::string &name, std::string &s) const
{
    bool ok = true;
    WisdomContainerEntry s_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_STRING, &s_entry);
    ok = ok && s_entry.readString(s);
    return ok;
}

bool WisdomContainerEntry::readUInt8Vector(const std::string &name, std::vector<NvU8> &v) const
{
    bool ok = true;
    WisdomContainerEntry v_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_UINT8_VECTOR, &v_entry);
    ok = ok && v_entry.readUInt8Vector(v);
    return ok;
}

bool WisdomContainerEntry::readUInt32(const std::string &name, NvU32 &v) const
{
    bool ok = true;
    WisdomContainerEntry v_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_UINT32, &v_entry);
    ok = ok && v_entry.readUInt32(v);
    return ok;
}

bool WisdomContainerEntry::readUInt64(const std::string &name, NvU64 &v) const
{
    bool ok = true;
    WisdomContainerEntry v_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_UINT64, &v_entry);
    ok = ok && v_entry.readUInt64(v);
    return ok;
}

bool WisdomContainerEntry::readInt32(const std::string &name, NvS32 &v) const
{
    bool ok = true;
    WisdomContainerEntry v_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_INT32, &v_entry);
    ok = ok && v_entry.readInt32(v);
    return ok;
}

bool WisdomContainerEntry::readFloat32(const std::string &name, NvF32 &v) const
{
    bool ok = true;
    WisdomContainerEntry v_entry;
    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_FLOAT32, &v_entry);
    ok = ok && v_entry.readFloat32(v);
    return ok;
}

bool WisdomContainerEntry::readObject(const std::string &name) const
{
    WisdomContainerEntry obj_entry;
    bool ok;
    ok = getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &obj_entry);
    return ok;
}

bool WisdomContainerEntry::readDims2(const std::string &name, Dims2 &dims) const
{
    bool ok = true;
    WisdomContainerEntry dims_entry, dims_w_entry, dims_h_entry;
    NvS32 w, h;

    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &dims_entry);
    ok = ok && dims_entry.getEntry("w", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_w_entry);
    ok = ok && dims_entry.getEntry("h", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_h_entry);

    ok = ok && dims_w_entry.readInt32(w);
    ok = ok && dims_h_entry.readInt32(h);

    if ( ok ) {
        dims.w = (int)w;
        dims.h = (int)h;
    }
    return ok;
}

bool WisdomContainerEntry::readDims3(const std::string &name, Dims3 &dims) const
{
    bool ok = true;
    WisdomContainerEntry dims_entry, dims_w_entry, dims_h_entry, dims_c_entry;
    NvS32 w, h, c;

    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &dims_entry);
    ok = ok && dims_entry.getEntry("w", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_w_entry);
    ok = ok && dims_entry.getEntry("h", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_h_entry);
    ok = ok && dims_entry.getEntry("c", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_c_entry);

    ok = ok && dims_w_entry.readInt32(w);
    ok = ok && dims_h_entry.readInt32(h);
    ok = ok && dims_c_entry.readInt32(c);

    if ( ok ) {
        dims.w = (int)w;
        dims.h = (int)h;
        dims.c = (int)c;
    }
    return ok;
}

bool WisdomContainerEntry::readDims4(const std::string &name, Dims4 &dims) const
{
    bool ok = true;
    WisdomContainerEntry dims_entry, dims_w_entry, dims_h_entry, dims_c_entry, dims_n_entry;
    NvS32 w, h, c, n;

    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &dims_entry);
    ok = ok && dims_entry.getEntry("w", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_w_entry);
    ok = ok && dims_entry.getEntry("h", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_h_entry);
    ok = ok && dims_entry.getEntry("c", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_c_entry);
    ok = ok && dims_entry.getEntry("n", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_n_entry);

    ok = ok && dims_w_entry.readInt32(w);
    ok = ok && dims_h_entry.readInt32(h);
    ok = ok && dims_c_entry.readInt32(c);
    ok = ok && dims_n_entry.readInt32(n);

    if ( ok ) {
        dims.w = (int)w;
        dims.h = (int)h;
        dims.c = (int)c;
        dims.n = (int)n;
    }
    return ok;
}

bool WisdomContainerEntry::readWeights(const std::string &name, Weights &w) const
{
    bool ok = true;
    WisdomContainerEntry weights_entry, type_entry, count_entry, values_entry;
    NvS32 t = 0, c = 0;
    NvU8 *v = 0;
    size_t s = 0;

    ok = ok && getEntry(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &weights_entry);
    ok = ok && weights_entry.getEntry("type",   IWisdomContainerEntry::ENTRY_TYPE_INT32, &type_entry);
    ok = ok && weights_entry.getEntry("count",  IWisdomContainerEntry::ENTRY_TYPE_INT32, &count_entry);
    ok = ok && weights_entry.getEntry("values", IWisdomContainerEntry::ENTRY_TYPE_UINT8_VECTOR, &values_entry);

    ok = ok && type_entry.readInt32(t);
    w.type = (DataType)t;
    ok = ok && count_entry.readInt32(c);
    w.count = c;
    w.values = NULL;
    if ( w.count )
    {
        ok = ok && values_entry.readUInt8Vector(&v, &s);
        if ( ok )
        {
            w.values = v;
            w.count = (int)s; // why set this again?
            // if it is different it is an error?
            // at least warn?
        }
    }
    return ok;
}

bool WisdomContainerEntry::writeUInt8Enum(const std::string &name, NvU8 v)
{
    WisdomContainerEntry b_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_UINT8, &b_entry);
    ok = ok && b_entry.writeUInt8(v);
    return ok;
}

bool WisdomContainerEntry::writeString(const std::string &name, const std::string &s)
{
    WisdomContainerEntry s_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_STRING, &s_entry);
    ok = ok && s_entry.writeString(s);
    return ok;
}

bool WisdomContainerEntry::writeUInt8Vector(const std::string &name, const std::vector<NvU8> &v)
{
    WisdomContainerEntry v_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_UINT8_VECTOR, &v_entry);
    ok = ok && v_entry.writeUInt8Vector(v);
    return ok;
}

bool WisdomContainerEntry::writeUInt32(const std::string &name, NvU32 v)
{
    WisdomContainerEntry v_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_UINT32, &v_entry);
    ok = ok && v_entry.writeUInt32(v);
    return ok;
}

bool WisdomContainerEntry::writeUInt64(const std::string &name, NvU64 v)
{
    WisdomContainerEntry v_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_UINT64, &v_entry);
    ok = ok && v_entry.writeUInt64(v);
    return ok;
}

bool WisdomContainerEntry::writeInt32(const std::string &name, NvS32 v)
{
    WisdomContainerEntry v_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_INT32, &v_entry);
    ok = ok && v_entry.writeInt32(v);
    return ok;
}

bool WisdomContainerEntry::writeFloat32(const std::string &name, NvF32 v)
{
    WisdomContainerEntry v_entry;
    bool ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_FLOAT32, &v_entry);
    ok = ok && v_entry.writeFloat32(v);
    return ok;
}

bool WisdomContainerEntry::writeObject(const std::string &name)
{
    WisdomContainerEntry obj_entry;
    bool ok;
    ok = insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &obj_entry);
    return ok;
}

bool WisdomContainerEntry::writeDims2(const std::string &name, const Dims2 &dims)
{
    bool ok = true;
    WisdomContainerEntry dims_entry, dims_w_entry, dims_h_entry;
    NvS32 w, h;
    NVDLA_UNUSED(w);
    NVDLA_UNUSED(h);
    ok = ok && insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &dims_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("w", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_w_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("h", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_h_entry);

    ok = ok && dims_w_entry.writeInt32(dims.w);
    ok = ok && dims_h_entry.writeInt32(dims.h);

    return ok;
}

bool WisdomContainerEntry::writeDims3(const std::string &name, const Dims3 &dims)
{
    bool ok = true;
    WisdomContainerEntry dims_entry, dims_w_entry, dims_h_entry, dims_c_entry;
    NvS32 w, h, c;
    NVDLA_UNUSED(w);
    NVDLA_UNUSED(h);
    NVDLA_UNUSED(c);
    ok = ok && insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &dims_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("w", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_w_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("h", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_h_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("c", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_c_entry);

    ok = ok && dims_w_entry.writeInt32(dims.w);
    ok = ok && dims_h_entry.writeInt32(dims.h);
    ok = ok && dims_c_entry.writeInt32(dims.c);
    return ok;
}

bool WisdomContainerEntry::writeDims4(const std::string &name, const Dims4 &dims)
{
    bool ok = true;
    WisdomContainerEntry dims_entry, dims_w_entry, dims_h_entry, dims_c_entry, dims_n_entry;
    NvS32 w, h, c;
    NVDLA_UNUSED(w);
    NVDLA_UNUSED(h);
    NVDLA_UNUSED(c);
    ok = ok && insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &dims_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("w", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_w_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("h", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_h_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("c", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_c_entry);
    ok = ok && dims_entry.insertEntryIfNotPresent("n", IWisdomContainerEntry::ENTRY_TYPE_INT32, &dims_n_entry);

    ok = ok && dims_w_entry.writeInt32(dims.w);
    ok = ok && dims_h_entry.writeInt32(dims.h);
    ok = ok && dims_c_entry.writeInt32(dims.c);
    ok = ok && dims_n_entry.writeInt32(dims.n);
    return ok;
}

bool WisdomContainerEntry::writeWeights(const std::string &name, const Weights &w)
{
    bool ok = true;
    WisdomContainerEntry weights_entry, type_entry, values_entry, count_entry;
    ok = ok && insertEntryIfNotPresent(name, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &weights_entry);
    ok = ok && weights_entry.insertEntryIfNotPresent("type",   IWisdomContainerEntry::ENTRY_TYPE_INT32, &type_entry);
    ok = ok && weights_entry.insertEntryIfNotPresent("count",  IWisdomContainerEntry::ENTRY_TYPE_INT32, &count_entry);
    ok = ok && weights_entry.insertEntryIfNotPresent("values", IWisdomContainerEntry::ENTRY_TYPE_UINT8_VECTOR, &values_entry);

    ok = ok && type_entry.writeInt32(w.type);
    ok = ok && count_entry.writeInt32(w.count);
    if ( w.count )
    {
        ok = ok && values_entry.writeUInt8Vector((const NvU8*)w.values, size_t(w.count));
    }
    return ok;
}


//----------------------------------------------------------------------
// WisdomContainer
//----------------------------------------------------------------------

//!
//! Note: The default wisdom container uses NvDla to build up an attribute
//! store in the filesystem.  This is being used in the early phases of
//! wisdom development for ease of debug and flexibility. We will likely
//! move to something else later (protobuf or some other serialization).
//!

WisdomContainer::WisdomContainer(Wisdom *wisdom) :
    IWisdomContainer(wisdom),
    m_wisdom(wisdom),
    m_wisdom_priv(wisdom),
    m_root(""),
    m_root_entry(this, string(""), string(""), IWisdomContainerEntry::ENTRY_TYPE_OBJECT)
{

}

WisdomContainer::~WisdomContainer()
{

}

IWisdom *WisdomContainer::wisdom()
{
    return m_wisdom;
}

Wisdom *WisdomContainer::wisdom_priv()
{
    return m_wisdom_priv;
}

IWisdomContainerEntry *WisdomContainer::root()
{
    return &m_root_entry;
}

WisdomContainerEntry *WisdomContainer::root_priv()
{
    return &m_root_entry;
}

bool WisdomContainer::open(const std::string &dir_path)
{
    if (m_root != string(""))
    {
        close();
    }
    if ( 0 == dir_path.size() )
    {
        gLogError << "a path is needed to open a wisdom container" << endl;
        return false;
    }

    NvDlaDirHandle dir;
    NvDlaError err = NvDlaOpendir(dir_path.c_str(), &dir);
    if ( err != NvDlaSuccess )
    {
        // gLogError << "couldn't open wisdom directory " << dir << endl;
        return false;
    }
    m_root = std::string(dir_path.c_str());

    // note "" is not a valid dirpath, so you can't pass "" and get "/" as root
    if ( m_root.substr(m_root.size()-1,1) != std::string("/") ) {
        m_root += string("/");
    }


    NvDlaClosedir(dir);

    return true;
}

bool WisdomContainer::isOpen()
{
    return m_root.size();
}

void WisdomContainer::close()
{
    m_root = string("");

}

//!
//! The container never knows the EntryType of the objects... so it must be
//! told how to interpret them.  The user of the container is responsible
//! for maintaining that information... the Wisdom fills that role by keeping
//! its own dictionary/symbol table aside...
//!

//
// this version is publicly visble and allocates the entry.
//
bool WisdomContainer::getEntry(const std::string &path,
                               const std::string &name,
                               IWisdomContainerEntry::EntryType type,
                               IWisdomContainerEntry *&entry)
{
    if ( !isOpen() )
    {
        return false;
    }

    entry = 0;

    const string fname(entryFilename(path, name));
    // gLogDebug << "getEntry for root=[" << m_root << "] path=[" << path <<
    //        "] name=[" << name << "] type=" << int(type) << " fname=[" << fname << "]" << endl;


    bool ok;
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + fname).c_str(), NVDLA_OPEN_READ, &file);
    ok = rc == NvDlaSuccess;
    if (!ok)
    {
        // gLogError << "couldn't open (read) entry file " << fname << endl;
    }
    else
    {
        NvDlaFclose(file);
        if ( type == IWisdomContainerEntry::ENTRY_TYPE_OBJECT ) {
            // gLogError << "open ok, instance new (dir) entry... " << fname << endl;
            std::string dirname = path;
            if ( name.size() && (name != string(".")) ) {
                dirname += "/" + name;
            }
            entry = new WisdomContainerEntry(this, dirname, string(""), type);

        } else {
            // gLogError << "open ok, instance new entry... " << fname << endl;
            entry = new WisdomContainerEntry(this, path, name, type);
        }
    }

    return ok && entry;
}

//
// this version of getEntry is internal-only and makes use of an on-the-stack entry
//
bool WisdomContainer::getEntry(const std::string &path,
                               const std::string &name,
                               IWisdomContainerEntry::EntryType type,
                               WisdomContainerEntry *entry)
{
    if ( !isOpen() )
    {
        return false;
    }

    const string fname(entryFilename(path, name));
    // gLogDebug << "getEntry for root=[" << m_root << "] path=[" << path <<
    //        "] name=[" << name << "] type=" << int(type) << " fname=[" << fname << "]" << endl;


    bool ok;
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + fname).c_str(), NVDLA_OPEN_READ, &file);
    ok = rc == NvDlaSuccess;
    if (!ok)
    {
        // gLogError << "couldn't open (read) entry file " << fname << endl;
    }
    else
    {
        NvDlaFclose(file);
        if ( type == IWisdomContainerEntry::ENTRY_TYPE_OBJECT ) {
            // gLogError << "open ok, instance new (dir) entry... " << fname << endl;
            std::string dirname = path;
            if ( name.size() && (name != string(".")) ) {
                dirname += "/" + name;
            }
            *entry = WisdomContainerEntry(this, dirname, string(""), type);
        } else {
            // gLogError << "open ok, instance new entry... " << fname << endl;
            *entry = WisdomContainerEntry(this, path, name, type);
        }
    }

    return ok;
}


//
// TBD: double-check/test that readdir here is returning
// dirs as well as file names ("." and ".." should show up).
//
bool WisdomContainer::getEntryNames(const std::string &path, std::vector<std::string> *names)
{
    if ( !isOpen() )
    {
        return false;
    }

    NvDlaDirHandle dir;
    string dir_path;
    if ( path.size() && (path != string(".") && path != string("./"))) {
        dir_path = m_root + "/" + path;
    } else {
        dir_path = m_root;
    }


    NvDlaError err = NvDlaOpendir(dir_path.c_str(), &dir);
    if ( err != NvDlaSuccess ) {
        gLogError << "couldn't open wisdom attribute directory " << dir_path << endl;
        return false;
    }

    char name_buf[255+1]; // doc or otherwise parameterize ???
    do {
        err = NvDlaReaddir(dir, name_buf, 255);
        if ( err == NvDlaSuccess ) { // || == NvDlaError_EndOfDirList dups the last entry...
            // skip zero and special/dot files
            if ( (name_buf[0] != '\0') && (name_buf[0] != '.') ) {
                names->push_back(string(name_buf)); // prefix with current path?
            }
        }
    } while (err == NvDlaSuccess);

    NvDlaClosedir(dir);

    return true;
}

bool WisdomContainer::removeEntry(const std::string &path)
{
    // TBD, complicated by missing rmdir in NvDla*
    gLogError << "removeEntry TBD " << path << endl;
    return true;
}

// internally facing
bool WisdomContainer::insertEntry(const std::string &path,
                                  const std::string &name,
                                  IWisdomContainerEntry::EntryType type,
                                  WisdomContainerEntry *entry)
{
    bool ok = true;
    NvDlaError rc = NvDlaSuccess;


    switch ( type )
    {
    case IWisdomContainerEntry::ENTRY_TYPE_OBJECT:
        {
            int dir_depth = std::count(name.begin(), name.end(), '/');
            int start_pos = 0;
            while(dir_depth)
            {
                std::string parent_dir = name.substr(0, name.find('/', start_pos));
                std::string parent_dir_path(entryDirname(path, parent_dir));
                rc = NvDlaMkdir(const_cast<char*>((m_root + parent_dir_path).c_str()));
                start_pos = name.find('/', start_pos) + 1;
                dir_depth--;
            }
            const std::string dirname(entryDirname(path, name));

            // gLogError << "insertEntry root=[" << m_root << "] path=[" << path <<
            //             "] name=[" << name << "] dirname=[" << dirname << "] type=" << int(type) << endl;

            rc = NvDlaMkdir( const_cast<char *>((m_root + dirname).c_str()) );
            ok =  ((rc == NvDlaSuccess) || (rc == NvDlaError_PathAlreadyExists));

            if ( ok )
            {
                *entry = WisdomContainerEntry(this, dirname, string(""), type);
            }
        }
        break;

    case IWisdomContainerEntry::ENTRY_TYPE_STRING:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT8_VECTOR:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT32:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT64:
    case IWisdomContainerEntry::ENTRY_TYPE_INT32:
    case IWisdomContainerEntry::ENTRY_TYPE_FLOAT32:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT8:
        {
            const std::string fname(entryFilename(path, name));

            //        gLogError << "insertEntry root=[" << m_root << "] path=[" << path <<
            //            "] name=[" << name << "] filename=[" << fname << "] type=" << int(type) << endl;

            NvDlaFileHandle file;
            NvDlaError rc = NvDlaFopen((m_root + fname).c_str(), NVDLA_OPEN_WRITE, &file);
            if ( rc != NvDlaSuccess )
            {
                gLogError << "couldn't open (write) entry file " << fname << endl;
                return false;
            }
            if ( ok )
            {
                NvDlaFclose(file);
                // gLogDebug << "instance new entry for " << fname << endl;
                *entry = WisdomContainerEntry(this, path, name, type);
            }
        }
        break;

    default:
        gLogError << "invalid type (" << (int)type << ") given to create entry path=[" <<
            path << "] name=[" << name << "]" << endl;
        ok = false;
    }

    return ok;
}

// externally facing
bool WisdomContainer::insertEntry(const std::string &path,
                                  const std::string &name,
                                  IWisdomContainerEntry::EntryType type,
                                  IWisdomContainerEntry *&entry)
{

    bool ok = true;
    NvDlaError rc = NvDlaSuccess;
    entry = 0;

    switch ( type )
    {

    case IWisdomContainerEntry::ENTRY_TYPE_OBJECT:
        {
            const std::string dirname(entryDirname(path, name));
            rc = NvDlaMkdir( const_cast<char *>((m_root + dirname).c_str()) );
            ok =  ((rc == NvDlaSuccess) || (rc == NvDlaError_PathAlreadyExists));
            if ( ok )
            {
                entry = new WisdomContainerEntry(this, dirname, string(""), type);
            }
        }
        break;

    case IWisdomContainerEntry::ENTRY_TYPE_STRING:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT8_VECTOR:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT32:
    case IWisdomContainerEntry::ENTRY_TYPE_UINT64:
    case IWisdomContainerEntry::ENTRY_TYPE_INT32:
    case IWisdomContainerEntry::ENTRY_TYPE_FLOAT32:
        {
            const std::string fname(entryFilename(path, name));
            NvDlaFileHandle file;
            NvDlaError rc = NvDlaFopen((m_root + fname).c_str(), NVDLA_OPEN_WRITE, &file);
            if ( rc != NvDlaSuccess )
            {
                gLogError << "couldn't open (write) entry file " << fname << endl;
                return false;
            }
            if ( ok )
            {
                NvDlaFclose(file);
                entry = new WisdomContainerEntry(this, path, name, type);
            }
        }
        break;

    default:
        gLogError << "invalid type (" << (int)type << ") given to create entry path=[" <<
            path << "] name=[" << name << "]" << endl;
        ok = false;
    }

    return ok;
}

//
// return a + b w/o introducing superfluous slashes
//
static const std::string catPaths(const std::string a, const std::string b)
{
    if ( !a.size() ) {
        return b;
    }
    if ( a[a.size()-1] != '/' ) {
        return a + "/" + b;
    }
    return a + b;
}

const std::string WisdomContainer::entryDirname(const std::string &path, const std::string &name) const
{
    if ( name.size() && (name != std::string(".") && name != std::string("./")) ) {
        return ( catPaths(path, name) );
    }
    return path;
}

const std::string WisdomContainer::entryFilename(const std::string &path, const std::string &name) const
{

    return catPaths(path, name);
}

const std::string WisdomContainer::entryFilename(IWisdomContainerEntry *entry) const
{
    return catPaths(entry->path(),entry->name());
}

const std::string WisdomContainer::entryFilename(const std::string &name) const
{
    return name;
}

bool WisdomContainer::writeUInt8(const std::string &path, NvU8 v)
{
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_WRITE, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }

    rc = NvDlaFwrite(file, &v, sizeof(NvU8));
    NvDlaFclose(file);

    return rc == NvDlaSuccess;
}

bool WisdomContainer::writeString(const std::string &path, const std::string &v)
{
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_WRITE, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }

    rc = NvDlaFwrite(file, v.c_str(), v.size());
    NvDlaFclose(file);

    return rc == NvDlaSuccess;
}

bool WisdomContainer::writeUInt8Vector(const std::string &path, const NvU8 *v, size_t size)
{
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_WRITE, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }
    rc = NvDlaFwrite(file, v, size);
    NvDlaFclose(file);
    return rc == NvDlaSuccess;
}

bool WisdomContainer::writeUInt8Vector(const std::string &path, const std::vector<NvU8> &v)
{
    return writeUInt8Vector(path, (NvU8*)&(*v.begin()), size_t(v.size()));
}

bool WisdomContainer::writeUInt32(const std::string &path, NvU32 v)
{
    stringstream ss; ss << v;
    return writeString(path, ss.str());
}

bool WisdomContainer::writeUInt64(const std::string &path, NvU64 v)
{
    stringstream ss; ss << v;
    return writeString(path, ss.str());
}

bool WisdomContainer::writeInt32(const std::string &path, NvS32 v)
{
    stringstream ss; ss << v;
    return writeString(path, ss.str());
}

bool WisdomContainer::writeFloat32(const std::string &path, NvF32 v)
{
    stringstream ss; ss << v;
    return writeString(path, ss.str());
}

bool WisdomContainer::readUInt8(const std::string &path, NvU8 &v)
{
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }

    v = 0;

    rc = NvDlaFgetc(file, &v);
    NvDlaFclose(file);

    return true;
}

bool WisdomContainer::readString(const std::string &path, std::string &v)
{
    NvDlaFileHandle file;
    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }
    v.clear();

    NvU8 c;
    for (rc = NvDlaFgetc(file, &c); rc == NvDlaSuccess; rc = NvDlaFgetc(file, &c) ) {
        v.push_back(char(c));
    }
    NvDlaFclose(file);

    return true;
}

//
// careful with the args here.  the intent is for the user to call in once to get the size
// and perform its own allocation if necessary by calling back in with ret_vals pointing to
// an appropriately sized buffer.  and, it's all or nothing...   so any garbage value being
// pointed to (the value of the pointer itself) will be assumed to be a valid storage location.
// if not, zero it and a new buffer will be allocated using the default new allocator.
// either way the caller then owns the memory and must manage its eventual clean-up.
//
bool WisdomContainer::readUInt8Vector(const std::string &path, NvU8 **ret_vals, size_t *ret_size)
{
    NvDlaFileHandle file;
    NvDlaStatType finfo;
    size_t file_size;

    if ( !ret_size ) {
        return false;
    }

    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }

    rc = NvDlaFstat(file, &finfo);
    if ( rc != NvDlaSuccess)
    {
        gLogError << "couldn't get file stats for " << entryFilename(path) << endl;
        return false;
    }
    file_size = finfo.size;

    *ret_size = file_size;

    if ( NULL == ret_vals ) {
        return true;
    }
    if ( !file_size )  {
        return true;
    }
    if ( NULL == (*ret_vals) ) {
        *ret_vals = new NvU8[file_size];
        if ( NULL == (*ret_vals) ) {
            gLogError << " oom" << entryFilename(path) << endl;
            return false;
        }
    }
    NvDlaFseek(file, 0, NvDlaSeek_Set);

    size_t actually_read = 0;
    NVDLA_UNUSED(actually_read);

    rc = NvDlaFread(file, *ret_vals, file_size, ret_size);
    if ( rc != NvDlaSuccess )
    {
        return false;
    }
    if ( *ret_size != file_size )
    {
        return false;
    }

    NvDlaFclose(file);

    return true;
}

bool WisdomContainer::readUInt8Vector(const std::string &path, std::vector<NvU8> &v)
{
    NvDlaFileHandle file;
    NvDlaStatType finfo;
    size_t file_size;
    NvDlaError rc = NvDlaFopen((m_root + entryFilename(path)).c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        gLogError << "couldn't open entry " << entryFilename(path) << endl;
        return false;
    }

    rc = NvDlaFstat(file, &finfo);
    if ( rc != NvDlaSuccess)
    {
        gLogError << "couldn't get file stats for " << entryFilename(path) << endl;
        return false;
    }
    file_size = finfo.size;

    //
    // NOTE:
    //
    // there's an underlying (std::vector, or other) allocator here. if it is already holding onto some
    // chunk of memory for the vector, it will be making a choice about ditching it or re-using, etc.
    // but, do not attempt to be clever about it here!  allow those things to happen on their own.
    // for scenarios where we're making use of special allocators to track/meter memory requirements
    // doing anything else would be counter-productive.
    //
    //
    v.resize(file_size);

    if ( !file_size )  {
        return true;
    }

    NvDlaFseek(file, 0, NvDlaSeek_Set);

    size_t actually_read = 0;

    rc = NvDlaFread(file, &(*v.begin()), file_size, &actually_read);
    if ( rc != NvDlaSuccess )
    {
        return false;
    }
    if ( actually_read != file_size )
    {
        return false;
    }

    NvDlaFclose(file);

    return true;
}


bool WisdomContainer::readUInt32(const std::string &path, NvU32 &v)
{
    string sv;
    bool ok = readString(path, sv);
    if ( ok ) {
        stringstream ss(sv);
        ss >> v;
    }
    return ok;

}

bool WisdomContainer::readUInt64(const std::string &path, NvU64 &v)
{
    string sv;
    bool ok = readString(path, sv);
    if ( ok ) {
        stringstream ss(sv);
        ss >> v;
    }
    return ok;

}

bool WisdomContainer::readInt32(const std::string &path, NvS32 &v)
{
    string sv;
    bool ok = readString(path, sv);
    if ( ok ) {
        stringstream ss(sv);
        ss >> v;
    }
    return ok;
}

bool WisdomContainer::readFloat32(const std::string &path, NvF32 &v)
{
    string sv;
    bool ok = readString(path, sv);
    if ( ok ) {
        stringstream ss(sv);
        ss >> v;
    }
    return ok;
}


} // nvdla::priv
} // nvdla
