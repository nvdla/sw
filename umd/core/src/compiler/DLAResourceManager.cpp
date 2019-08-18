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

#include "nvdla/IType.h"
#include "priv/DLAResourceManager.h"
#include "priv/Memory.h"

using std::set;
using std::endl;

namespace nvdla
{

namespace priv
{

SEQUENCE_ENUM_STATIC_MEMBERS(memory::PoolTypeEnum, NvU8,  POOL_TYPE_ENUMS, "PoolTypeEnum")

namespace memory
{

//
// DLAResourceManager copy constructor for clone
//
DLAResourceManager::DLAResourceManager(const DLAResourceManager &o) :
    m_name(o.m_name),
    m_next_buffer_id(o.m_next_buffer_id),
    m_next_surface_desc_id(o.m_next_surface_desc_id),
    m_buffer_desc_directory(o.m_buffer_desc_directory),
    m_surface_desc_directory(o.m_surface_desc_directory)
{
    //FIXME: add logic to suitably clone
}

DLAResourceManager::~DLAResourceManager()
{


}

surface::TensorSurfaceDesc* DLAResourceManager::regTensorSurfaceDesc
(
    TensorType type,
    NvU16 numBatches
)
{
    surface::TensorSurfaceDesc* sd = new surface::TensorSurfaceDesc(numBatches);
    TensorCategory tc;
    switch(type){
        case TensorType::kDEBUG:
        case TensorType::kNW_INPUT:
        case TensorType::kNW_OUTPUT:  tc  = TensorCategoryEnum::EXTERNAL_TENSOR; break;
        case TensorType::kWEIGHT:
        case TensorType::kBIAS:
        case TensorType::kBATCH_NORM:
        case TensorType::kSCALE:      tc  = TensorCategoryEnum::GLOBAL_TENSOR; break;
        case TensorType::kIO:         tc  = TensorCategoryEnum::LOCAL_TENSOR; break;
        case TensorType::kSTREAM:     tc  = TensorCategoryEnum::STREAM_TENSOR; break;
        default:
        {
            REPORT_ERROR(NvDlaError_BadParameter, "Unable to categorize tensor type: %d", type);
            return NULL; // something wrong
        }
    }

    std::string sdID = nextSurfaceDescId();
    sd->setId(sdID);
    sd->setTensorCategory(tc);
    m_surface_desc_directory.insert(sd);

    return sd;
}

bool DLAResourceManager::unregTensorSurfaceDesc(surface::TensorSurfaceDesc *tsd)
{
    if (tsd)
        m_surface_desc_directory.erase(tsd);
    return true;
}

std::vector< TensorBufferDesc *> DLAResourceManager::getBufferDescs()
{
    std::vector< TensorBufferDesc *> allBDs;

    allBDs.reserve(m_buffer_desc_directory.size());

    for (TensorBufferDirectoryIter it = m_buffer_desc_directory.begin();
         it != m_buffer_desc_directory.end(); ++it)
    {
        allBDs.push_back(*it);
    }

    return allBDs;
}

std::vector< surface::TensorSurfaceDesc *> DLAResourceManager::getSurfaceDescs()
{
    std::vector< surface::TensorSurfaceDesc *> allSDs;

    allSDs.reserve(m_surface_desc_directory.size());

    for (TensorSurfaceDirectoryIter it = m_surface_desc_directory.begin();
         it != m_surface_desc_directory.end(); ++it)
    {
        allSDs.push_back(*it);
    }

    return allSDs;
}


TensorBufferDesc* DLAResourceManager::regTensorBufferDesc(NvU16 numBatches)
{
    TensorBufferDesc* tbd = new TensorBufferDesc(numBatches);

    std::string bID = nextBufferId();
    tbd->setId(bID);
    m_buffer_desc_directory.insert(tbd);

    return tbd;
}

bool DLAResourceManager::unregTensorBufferDesc(TensorBufferDesc *tbd)
{
    if (tbd)
        m_buffer_desc_directory.erase(tbd);
    return true;
}

} // nvdla::priv::memory
} // nvdla::priv
} // nvdla
