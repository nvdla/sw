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

#include <string>
#include <sstream>

#include "priv/Network.h"
#include "priv/Check.h"

#include "priv/Wisdom.h"
#include "priv/WisdomContainer.h"

using std::vector;
using std::string;
using std::endl;

namespace nvdla
{

IWisdom::IWisdom() { }
IWisdom::~IWisdom() { }

IWisdom *createWisdom()
{
    priv::WisdomFactory::WisdomPrivPair wp_pair = priv::WisdomFactory::newWisdom();
    return wp_pair.i();
}

NvDlaError destroyWisdom(IWisdom *wisdom)
{
    NvDlaError e = NvDlaSuccess;
    PROPAGATE_ERROR_FAIL(priv::WisdomFactory::deleteWisdom(wisdom));

fail:
    return e;
}

namespace priv
{


WisdomFactory::WisdomPrivPair WisdomFactory::newWisdom()
{
    IWisdom *wisdom;
    Wisdom *wisdom_priv;
    wisdom = wisdom_priv = new priv::Wisdom();
    if (wisdom) {
        s_priv.insert(wisdom, wisdom_priv);
        s_self.insert(wisdom, wisdom);
    }
    return WisdomPrivPair(wisdom, wisdom_priv);
}

NvDlaError WisdomFactory::deleteWisdom(IWisdom *wisdom)
{
    if (wisdom != NULL) {
        Wisdom *wisdom_priv = priv(wisdom);
        if (wisdom_priv != NULL) {
            delete wisdom_priv;
        }

        s_priv.remove(wisdom);
        s_self.remove(wisdom);
    }

    return NvDlaSuccess;
}

Wisdom *WisdomFactory::priv(IWisdom *wisdom)
{
    BiMap<IWisdom *, Wisdom *>::left_iterator f = s_priv.find_left(wisdom);

    if ( f == s_priv.end_left() ) {
        return NULL;
    }
    return f->second;
}

IWisdom *WisdomFactory::i(Wisdom *wisdom)
{
    BiMap<IWisdom *, Wisdom *>::right_iterator f = s_priv.find_right(wisdom);
    if ( f == s_priv.end_right() ) {
        return NULL;
    }
    return f->second;
}

IWisdom *WisdomFactory::self(void *s)
{
    BiMap<void *, IWisdom *>::left_iterator f = s_self.find_left(s);
    if ( f == s_self.end_left() ) {
        return NULL;
    }
    return f->second;
}


BiMap<IWisdom *, Wisdom*> WisdomFactory::s_priv;
BiMap<void *, IWisdom*> WisdomFactory::s_self;


Wisdom::Wisdom() : m_container(0), m_network(0), m_compiler(0), m_profiler(0)
{

}

Wisdom::~Wisdom()
{
    if (m_compiler != NULL) {
        CompilerFactory::deleteCompiler(m_compiler);
        m_compiler = NULL;
    }
    if (m_profiler != NULL) {
        ProfilerFactory::deleteProfiler(m_profiler);
        m_profiler = NULL;
    }
    if ( m_container ) {
        close();
        m_container = NULL;
    }
}

bool Wisdom::open(const std::string &uri)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);

    bool status = false;
    CATCH_ERROR_FAIL(status = openInternal(uri));

fail:
    return status;
}

bool Wisdom::openInternal(const std::string &uri)
{
    m_container = new WisdomContainer(this);
    if ( !m_container ) {
        return false;
    }

    bool ok = m_container->open(uri);
    if ( !ok ) {
        return false;
    }

    IWisdomContainerEntry *root = m_container->root();
    WisdomContainerEntry *root_priv = m_container->root_priv();
    NVDLA_UNUSED(root);

    NvU32 data_type;
    if ( root_priv->readUInt32("data_type", data_type) ) {
        m_data_type = data_type;
    } else {
        m_data_type = DataType::INT16;
    }
    return ok;
}

void Wisdom::close()
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    CATCH_ERROR_FAIL(closeInternal());

fail:
    return;
}

void Wisdom::closeInternal()
{
    m_container->close();
    m_container = 0;
}

IWisdomContainerEntry *Wisdom::getRootEntry()
{
    if ( !m_container ) {
        gLogError << "no root?!?!" << endl;
        return 0;
    }
    return m_container->root();
}


bool Wisdom::setNetworkTransient(INetwork *inetwork)
{
    Network *network = NetworkFactory::priv(inetwork);
    if ( !network ) {
        gLogError << "unrecognized network presented to Wisdom" << endl;
        return false;
    }
    //
    // prior to serialization all objects need entries in the symbol table.
    // since the network ultimately refers to them all, just triggering
    // it's symbols to be resolved is sufficient to be able to deserialize
    // it later.
    //
    bool ok = network->assignSymbols(m_container->wisdom_priv());
    m_network = network;
    return ok;
}

bool Wisdom::setNetwork(INetwork *inetwork)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    bool status = false;
    CATCH_ERROR_FAIL(status = setNetworkInternal(inetwork));

fail:
    return status;
}

bool Wisdom::setNetworkInternal(INetwork *inetwork)
{

    bool ok = setNetworkTransient(inetwork);
    if ( !ok )
    {
        return false;
    }

    Network *network = NetworkFactory::priv(inetwork);
    if ( !network ) {
        gLogError << "unrecognized network presented to Wisdom" << endl;
        return false;
    }
    if ( !m_container ) {
        gLogError << "can't Wisdom::setNetwork unless a container is available" << endl;
        return false;
    }
    WisdomContainerEntry network_entry;

    ok = m_container->root_priv()->getEntry("network",
                                            IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                            &network_entry);
    if ( ok ) {
        gLogError << "can't Wisdom::setNetwork if 'network' is already present in the container" << endl;
        return false;
    }

    ok = m_container->root_priv()->insertEntry("network",
                                               IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                               &network_entry);
    if ( !ok ) {
        gLogError << "Wisdom::setNetwork couldn't insert network into a container entry" << endl;
        return false;
    }
    //
    // this will cause all the referenced tensors (inputs, outputs, weights, etc)
    // and layers to be serialized as well.
    //
    ok = network->serializeTo(&network_entry);
    if ( !ok  ) {
        gLogError << "Wisdom::setNetwork couldn't serialize network into a container entry" << endl;
        return false;
    }

    return ok;

}

INetwork *Wisdom::getNetwork()
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    INetwork* inetwork = NULL;
    CATCH_ERROR_FAIL(inetwork = getNetworkInternal());

fail:
    return inetwork;
}

INetwork *Wisdom::getNetworkInternal()
{
    //     gLogError << __func__ << " getting network." << endl;
    if ( m_network ) {
        return m_network;
    }
    if ( !m_container ) {
        gLogError << "no container to get network from" << endl;
        return NULL;
    }

    //     gLogError << __func__ << " getting network entry" << endl;
    // deserialize from container
    WisdomContainerEntry network_entry;
    bool ok = m_container->root_priv()->getEntry("network",
                                                 IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                                 &network_entry);
    if ( !ok ) {
        gLogError << "no network present" << endl;
        return NULL;
    }

    m_network = NetworkFactory::deserializeFrom(&network_entry);

    return m_network;
}


bool Wisdom::insertNetworkSymbol(INetwork *net, const std::string &sym)
{
    return m_symbol_table.insertNetwork(net, sym);
}

INetwork *Wisdom::findNetworkSymbol(const std::string &sym)
{
    return m_symbol_table.findNetwork(sym);
}

bool Wisdom::insertLayerSymbol(ILayer *layer, const std::string &sym)
{
    // gLogError << "this=" << this << " layer=" << layer << " sym=[" << sym << "]" << endl;
    return m_symbol_table.insertLayer(layer, sym);
}

ILayer *Wisdom::findLayerSymbol(const std::string &sym)
{
    return m_symbol_table.findLayer(sym);
}

bool Wisdom::findLayerSymbol(Layer *find_this, std::string &found_name)
{
    return m_symbol_table.findLayer(find_this, found_name);
}

bool Wisdom::findILayerSymbol(ILayer *find_this, std::string &found_name)
{
    Layer *layer;
    bool ok = NULL != find_this;
    if ( ok ) {
        layer = LayerFactory::priv(find_this);
    }
    ok = ok && (NULL != layer);
    ok = ok && findLayerSymbol(layer, found_name);
    return ok;
}

bool Wisdom::assignLayerSymbol(Layer *assign_layer, std::string &assigned_sym)
{
    std::stringstream ss;
    bool ok = false;
    bool assigned = false;
    Layer *found_layer;
    std::string found_sym;
    std::string try_sym;
    NVDLA_UNUSED(assigned);

    // check to see if it is already assigned.
    if ( findLayerSymbol(assign_layer, found_sym) ) {
        assigned_sym = found_sym;
        return true;
    }

    //
    // first choice is to have symbol name == layer name.
    // assuming it is a valid symbol name to start with...
    //
    assigned_sym.clear();
    try_sym = assign_layer->getName();
    if ( try_sym.size() /*tbd: isValidSymbolName(try_sym) */ ) {
        found_layer = LayerFactory::priv(findLayerSymbol(try_sym));
        if ( found_layer ) {
            // by construction, found_layer cannot be == assign_layer here.
            // if it had been, the check above for idempotency would have taken care of it.
            // so we found layer a layer already using this layer's name as a symbol name.
            // someone dup'd layer names.  so assign a new one...
        }
        else {
            assigned_sym = try_sym;
        }
    }

    // assign something dumb, but almost certain to be ok.
    // XXX: after round round of serialization and subsequent
    // modification and reserialization this could be WRONG!!!
    if ( assigned_sym.empty() ) {
        ss << assign_layer; // note this is a pointer!
        assigned_sym = "layer-" + ss.str();
    }

    ok = insertLayerSymbol( LayerFactory::i(assign_layer), assigned_sym );

    return ok;
}

// root/layers/<symbol>/<layer>
bool Wisdom::setLayer(Layer *layer)
{
    bool ok = true;
    string sym;
    WisdomContainerEntry *root = NULL;
    WisdomContainerEntry layers_entry;
    WisdomContainerEntry layer_entry;

    ok = NULL != m_container;
    if (ok) {
        root = m_container->root_priv();
    }
    if ( !ok ) {
        goto done;
    }
    ok = root->insertEntryIfNotPresent("layers",
                                       IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                       &layers_entry);
    ok = ok && findLayerSymbol(layer, sym);
    if ( !ok ) {
        gLogError << __func__ << " couldn't find create or find root symbol for layer name=" <<
            layer->getName() << " sym=" << sym << endl;
        goto done;
    }
    ok = layers_entry.insertEntryIfNotPresent(sym,
                                              IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                              &layer_entry);
    ok = ok && layer->serializeTo(&layer_entry);
    if ( !ok ) {
        gLogError << __func__ << " couldn't find entry for or serialize layer name=" <<
            layer->getName() << " sym=" << sym << endl;
        goto done;
    }

    // gLogError << __func__ << " serialized layer name=" << layer->getName() << " sym=" << sym << endl;

 done:
    return ok;
}

// root/layers/<symbol>/<layer>
ILayer *Wisdom::getLayerFromSymbol(const std::string &sym)
{
    bool ok = true;
    ILayer *layer = findLayerSymbol(sym);
    if ( layer ) {
        return layer;
    }
    // gLogError << __func__ << " sym=[" << sym << "]" << endl;
    // request to instantiate is implied.
    // find the symbol and deserialize from it.
    WisdomContainerEntry layers_entry;
    WisdomContainerEntry layer_entry;

    ok = m_container->root_priv()->getEntry("layers",
                                            IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                            &layers_entry);
    if ( !ok ) {
        gLogError << __func__ << " no layers entry for the wisdom" << endl;
        return NULL;
    }
    ok = layers_entry.getEntry(sym, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &layer_entry);
    if ( !ok ) {
        gLogError << __func__ << " no entry for sym=[" << sym << "]" << endl;
        return NULL;
    }

    layer = LayerFactory::deserializeFrom(&layer_entry);
    if ( !layer ) {
        gLogError << __func__ << " uh, deserialize freaked out" << endl;
        return NULL;
    }
    // gLogError << __func__ << " deserialized layer=" << layer << endl;

    ok = insertLayerSymbol(layer, sym);

    if ( !ok ) {
        gLogError << __func__ << "couldn't insert layer symbol sym=[" << sym << "]" << endl;
        return NULL;
    }
    return layer;
}

bool Wisdom::insertTensorSymbol(ITensor *tensor, const std::string &sym)
{
    return m_symbol_table.insertTensor(tensor, sym);
}

ITensor *Wisdom::findTensorSymbol(const std::string &sym)
{
    return m_symbol_table.findTensor(sym);
}

bool Wisdom::findTensorSymbol(Tensor *find_this, std::string &found_name)
{
    return m_symbol_table.findTensor(find_this, found_name);
}

bool Wisdom::findITensorSymbol(ITensor *find_this, std::string &found_name)
{
    Tensor *tensor;
    bool ok = NULL != find_this;
    if (ok ) {
        tensor = TensorFactory::priv(find_this);
    }
    ok = ok && (NULL != tensor);
    ok = ok && findTensorSymbol(tensor, found_name);
    return ok;
}

bool Wisdom::assignTensorSymbol(Tensor *assign_tensor, std::string &assigned_sym)
{
    std::stringstream ss;
    bool ok = false;
    bool assigned = false;
    Tensor *found_tensor;
    std::string found_sym;
    std::string try_sym;
    NVDLA_UNUSED(assigned);

    // check to see if it is already assigned.
    if ( findTensorSymbol(assign_tensor, found_sym) ) {
        assigned_sym = found_sym;
        return true;
    }

    //
    // first choice is to have symbol name == tensor name.
    // assuming it is a valid symbol name to start with...
    //
    assigned_sym.clear();
    try_sym = assign_tensor->getName();
    if ( try_sym.size() /*tbd: isValidSymbolName(try_sym) */ ) {
        found_tensor = TensorFactory::priv(findTensorSymbol(try_sym));
        if ( found_tensor ) {
            // by construction, found_tensor cannot be == assign_tensor here.
            // if it had been, the check above for idempotency would have taken care of it.
            // so we found tensor a tensor already using this tensor's name as a symbol name.
            // someone dup'd tensor names.  so assign a new one...
        }
        else {
            assigned_sym = try_sym;
        }
    }

    // assign something dumb, but almost certain to be ok.
    // XXX: after round round of serialization and subsequent
    // modification and reserialization this could be WRONG!!!
    if ( assigned_sym.empty() ) {
        ss << assign_tensor; // note this is a pointer!
        assigned_sym = "tensor-" + ss.str();
    }

    ok = insertTensorSymbol( TensorFactory::i(assign_tensor), assigned_sym );


    return ok;
}


// root/tensors/<symbol>/<tensor>
bool Wisdom::setTensor(Tensor *tensor)
{
    bool ok = true;
    string sym;
    WisdomContainerEntry *root = NULL;
    WisdomContainerEntry tensors_entry;
    WisdomContainerEntry tensor_entry;

    ok = NULL != m_container;
    if (ok) {
        root = m_container->root_priv();
    }
    if ( !ok ) {
        goto done;
    }
    ok = root->insertEntryIfNotPresent("tensors",
                                       IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                       &tensors_entry);
    ok = ok && findTensorSymbol(tensor, sym);
    if ( !ok ) {
        gLogError << __func__ << " couldn't find create or find root symbol for tensor name=" <<
            tensor->getName() << " sym=" << sym << endl;
        goto done;
    }
    ok = tensors_entry.insertEntryIfNotPresent(sym,
                                               IWisdomContainerEntry::ENTRY_TYPE_OBJECT,
                                               &tensor_entry);
    ok = ok && tensor->serializeTo(&tensor_entry);
    if ( !ok ) {
        gLogError << __func__ << " couldn't find entry for or serialize tensor name=" <<
            tensor->getName() << " sym=" << sym << endl;
        goto done;
    }

    // gLogError << __func__ << " serialized tensor name=" << tensor->getName() << " sym=" << sym << endl;

 done:
    return ok;
}


ITensor *Wisdom::getTensorFromSymbol(const std::string &sym)
{
    bool ok = true;

    ITensor *tensor = findTensorSymbol(sym);
    if ( tensor ) {
        return tensor;
    }
    // request to instantiate is implied.
    // find the symbol and deserialize from it.
    WisdomContainerEntry tensors_entry;
    WisdomContainerEntry tensor_entry;

    ok = m_container->root_priv()->getEntry("tensors", IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &tensors_entry);
    if ( !ok ) {
        gLogError << __func__ << " no tensors root entry" << endl;
        goto done;
    }
    ok = tensors_entry.getEntry(sym, IWisdomContainerEntry::ENTRY_TYPE_OBJECT, &tensor_entry);
    if ( !ok ) {
        gLogError << __func__ << " no tensor entry for sym=[" << sym << "]" << endl;
        goto done;
    }
    tensor = TensorFactory::deserializeFrom(&tensor_entry);
    if ( !tensor ) {
        gLogError << __func__ << " error attempting to deserialize tensor sym=[" << sym << "]" << endl;
        ok = false;
        goto done;
    }
    ok = insertTensorSymbol(tensor, sym);

 done:
    if (!ok)
        return NULL;
    return tensor;
}


bool SymbolTable::insertNetwork(INetwork *net, const std::string &sym)
{
    if ( findNetwork(sym) ) {
        return false;
    }
    m_sym_net.insert(sym, net);
    return true;
}

bool SymbolTable::insertLayer(ILayer *layer, const std::string &sym)
{
    if ( findLayer(sym) ) {
        return false;
    }
    m_sym_layer[sym] = layer;
    m_layer_sym[layer] = sym;
    return true;
}

bool SymbolTable::insertTensor(ITensor *tensor, const std::string &sym)
{
    if ( findTensor(sym) ) {
        return false;
    }
    m_sym_tensor[sym] = tensor;
    m_tensor_sym[tensor] = sym;
    return true;
}


bool SymbolTable::insertLoadable(ILoadable *loadable, const std::string &sym)
{
    if ( findLoadable(sym) ) {
        return false;
    }
    m_sym_loadable[sym] = loadable;
    m_loadable_sym[loadable] = sym;
    return true;
}

bool SymbolTable::insertProfile(IProfile *profile, const std::string &sym)
{
    if ( findProfile(sym) ) {
        return false;
    }
    //     gLogInfo << "wisdom insertProfile with symbol name " << sym << endl;
    m_sym_profile[sym] = profile;
    m_profile_sym[profile] = sym;
    return true;
}



INetwork *SymbolTable::findNetwork(const std::string &sym)
{
    SymNetIter f = m_sym_net.find_left(sym);
    if ( f == m_sym_net.end_left() ) {
        return NULL;
    }
    return f->second;
}

bool SymbolTable::findNetwork(Network *find_network, std::string &sym)
{
    INetwork *inetwork = NetworkFactory::i(find_network);
    if ( !inetwork ) {
        return false;
    }
    SymNetIter f = m_sym_net.find_left(sym);
    if ( f == m_sym_net.end_left() ) {
        return false;
    }
    sym = f->first;
    return true;
}


ILayer *SymbolTable::findLayer(const std::string &sym)
{
    SymLayerIter f = m_sym_layer.find(sym);
    if ( f == m_sym_layer.end() ) {
        return NULL;
    }
    return f->second;
}

bool SymbolTable::findLayer(Layer *find_this, std::string &found_name)
{
    // get the ILayer which goes with it, then look that up.
    ILayer *find_this_ilayer = LayerFactory::i(find_this);
    LayerSymIter f;

    if ( !find_this_ilayer ) {
        gLogError << __func__ << " unexpected zero interface pointer looking up " << find_this << endl;
        return false;
    }

    f = m_layer_sym.find(find_this_ilayer);
    if ( f == m_layer_sym.end() ) {
        // gLogError << __func__ << " no ilayer " << find_this_ilayer << endl;
        return false;
    }

    found_name = f->second;
    return true;
}

ITensor *SymbolTable::findTensor(const std::string &sym)
{
    SymTensorIter f = m_sym_tensor.find(sym);
    if ( f == m_sym_tensor.end() ) {
        return NULL;
    }
    return f->second;
}

bool SymbolTable::findTensor(Tensor *find_this, std::string &found_name)
{
    // get the ITensor which goes with it, then look that up.
    ITensor *find_this_itensor = TensorFactory::i(find_this);
    TensorSymIter f;

    if ( !find_this_itensor ) {
        return false;
    }

    f = m_tensor_sym.find(find_this_itensor);
    if ( f == m_tensor_sym.end() ) {
        return false;
    }

    found_name = f->second;
    return true;
}





IProfiler *Wisdom::getProfiler()
{
    if (!m_profiler) {
        ProfilerFactory::ProfilerPrivPair profiler = ProfilerFactory::newProfiler();
        m_profiler = profiler.priv();
        profiler.priv()->setWisdom(this);
    }
    return m_profiler; // Profiler->IProfiler ok here for now
}

ICompiler *Wisdom::getCompiler()
{
    if (!m_compiler) {
        CompilerFactory::CompilerPrivPair compiler = CompilerFactory::newCompiler();
        m_compiler = compiler.priv();
        compiler.priv()->setWisdom(this);
    }
    return m_compiler;
}

ILoadable *SymbolTable::findLoadable(const std::string &sym)
{
    SymLoadableIter f = m_sym_loadable.find(sym);
    if ( f == m_sym_loadable.end() ) {
        return NULL;
    }
    return f->second;
}

bool Wisdom::insertLoadableSymbol(ILoadable *loadable, const std::string &sym)
{
    // gLogError << "this=" << this << " loadable=" << loadable << " sym=[" << sym << "]" << endl;
    return m_symbol_table.insertLoadable(loadable, sym);
}

ILoadable *Wisdom::findLoadableSymbol(const std::string &sym)
{
    return m_symbol_table.findLoadable(sym);
}

bool Wisdom::findLoadableSymbol(Loadable *find_this, std::string &found_name)
{
    return m_symbol_table.findLoadable(find_this, found_name);
}

bool Wisdom::findILoadableSymbol(ILoadable *find_this, std::string &found_name)
{
    Loadable *loadable;
    bool ok = NULL != find_this;
    if ( ok ) {
        loadable = LoadableFactory::priv(find_this);
    }
    ok = ok && (NULL != loadable);
    ok = ok && findLoadableSymbol(loadable, found_name);
    return ok;
}

bool SymbolTable::findLoadable(Loadable *find_this, std::string &found_name)
{
    // get the ILoadable which goes with it, then look that up.
    ILoadable *find_this_iloadable = LoadableFactory::i(find_this);
    LoadableSymIter f;

    if ( !find_this_iloadable ) {
        gLogError << __func__ << " unexpected zero interface pointer looking up " << find_this << endl;
        return false;
    }

    f = m_loadable_sym.find(find_this_iloadable);
    if ( f == m_loadable_sym.end() ) {
        // gLogError << __func__ << " no iloadable " << find_this_iloadable << endl;
        return false;
    }

    found_name = f->second;
    return true;
}


//
//
//
IProfile *SymbolTable::findProfile(const std::string &sym)
{
    SymProfileIter f = m_sym_profile.find(sym);
    if ( f == m_sym_profile.end() )
    {
        return NULL;
    }
    return f->second;
}

bool Wisdom::insertProfileSymbol(IProfile *profile, const std::string &sym)
{
    // gLogError << "this=" << this << " profile=" << profile << " sym=[" << sym << "]" << endl;
    return m_symbol_table.insertProfile(profile, sym);
}

IProfile *Wisdom::findProfileSymbol(const std::string &sym)
{
    return m_symbol_table.findProfile(sym);
}

bool Wisdom::findProfileSymbol(Profile *find_this, std::string &found_name)
{
    return m_symbol_table.findProfile(find_this, found_name);
}

bool Wisdom::findIProfileSymbol(IProfile *find_this, std::string &found_name)
{
    Profile *profile;
    bool ok = NULL != find_this;
    if ( ok )
    {
        profile = ProfileFactory::priv(find_this);
    }
    ok = ok && (NULL != profile);
    ok = ok && findProfileSymbol(profile, found_name);
    return ok;
}

bool SymbolTable::findProfile(Profile *find_this, std::string &found_name)
{
    // get the IProfile which goes with it, then look that up.
    IProfile *find_this_iprofile = ProfileFactory::i(find_this);
    ProfileSymIter f;

    if ( !find_this_iprofile )
    {
        gLogError << __func__ << " unexpected zero interface pointer looking up " << find_this << endl;
        return false;
    }

    f = m_profile_sym.find(find_this_iprofile);
    if ( f == m_profile_sym.end() )
    {
        gLogError << __func__ << " no iprofile " << find_this_iprofile << endl;
        return false;
    }

    found_name = f->second;
    return true;
}

//
//
//
NvDlaError Wisdom::setDataType(DataType::UnderlyingType d)
{
    NvDlaError e = NvDlaSuccess;
    CATCH_PROPAGATE_ERROR_FAIL(setDataTypeInternal(d));

fail:
    return e;
}

NvDlaError Wisdom::setDataTypeInternal(DataType::UnderlyingType d)
{
    NvDlaError e = NvDlaError_Success;

    if ( m_container ) {
        WisdomContainerEntry data_type_entry;
        if ( ! m_container->root_priv()->writeUInt32("data_type", d) ) {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't write data type into a wisdom container.");
        }
    }

    m_data_type = d;

 fail:
    return e;
}

NvDlaError Wisdom::getDataType(DataType::UnderlyingType *data_type) const
{
    NvDlaError e = NvDlaSuccess;
    if ( !data_type )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }
    *data_type = m_data_type;
 fail:
    return e;
}



} // nvdla::priv
} // nvdla
