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

#include <set>
#include <string>

#include "priv/Check.h"
#include "priv/Surface.h"
#include "priv/Memory.h"
#include "priv/WeightTranslationUnit.h"

#include "ErrorMacros.h"

#include "priv/EngineAST.h"

using std::set;
using std::unordered_set;
using std::string;
using std::endl;

namespace nvdla
{
namespace priv
{

SEQUENCE_ENUM_STATIC_MEMBERS(surface::SurfaceCategoryEnum,       NvU8, SURFACE_CATEGORY_ENUMS,         "SurfaceCategoryEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(surface::SurfacePrecisionEnum,      NvU8, SURFACE_PRECISION_ENUMS,        "SurfacePrecisionEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(surface::BiasDataCategoryEnum,      NvU8, BIAS_DATA_CATEGORY_ENUMS,       "BiasDataCategoryEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(surface::BatchNormDataCategoryEnum, NvU8, BATCH_NORM_DATA_CATEGORY_ENUMS, "BatchNormDataCategoryEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(surface::ScaleDataCategoryEnum,     NvU8, SCALE_DATA_CATEGORY_ENUMS,      "ScaleDataCategoryEnum")
SEQUENCE_ENUM_STATIC_MEMBERS(surface::PixelMappingEnum,          NvU8, PIXEL_MAPPING_ENUMS,            "PixelMappingEnum")

SURFACE_ENUM_STATIC_MEMBERS(surface::SurfaceFormatEnum,          NvU8,  SURFACE_FORMAT_ENUMS,          "SurfaceFormatEnum")

namespace surface
{

TensorSurfaceDesc::~TensorSurfaceDesc() { }

/*----------------------------TENSOR SURFACE DESC APIs-----------------------*/
NvU64 TensorSurfaceDesc::size()
{
    NvU64 size = 0ULL;
    if (m_size != 0)
    {
        size = m_size;
        goto done;
    }

    switch(m_surface_format.category().v())
    {
        case SurfaceCategoryEnum::IMG:              size = IMGDesc::size(this); break;
        case SurfaceCategoryEnum::FEATURE_DATA:     size = FeatureDataDesc::size(this); break;
        case SurfaceCategoryEnum::WEIGHT:           size = WeightDesc::size(this); break;
        case SurfaceCategoryEnum::BIAS_DATA:        size = BiasDataDesc::size(this); break;
        case SurfaceCategoryEnum::BATCH_NORM_DATA:  size = BatchNormDataDesc::size(this); break;
        case SurfaceCategoryEnum::SCALE_DATA:       size = ScaleDataDesc::size(this); break;
        default:
        {
            size = 0;
            REPORT_ERROR(NvDlaError_NotSupported, "Unsupported surface category: %s", m_surface_format.category().c_str());
        }
    }
    m_size = size;
done:
    return size;
}

NvU32 TensorSurfaceDesc::lineStride()
{
    NvU32 ls = 0;
    if (m_line_stride != 0)
    {
        ls = m_line_stride;
        goto done;
    }

    switch(m_surface_format.category().v())
    {
        case SurfaceCategoryEnum::IMG:              ls = IMGDesc::lineStride(this); break;
        case SurfaceCategoryEnum::FEATURE_DATA:     ls = FeatureDataDesc::lineStride(this); break;
        case SurfaceCategoryEnum::BIAS_DATA:        ls = BiasDataDesc::lineStride(this); break;
        case SurfaceCategoryEnum::BATCH_NORM_DATA:  ls = BatchNormDataDesc::lineStride(this); break;
        case SurfaceCategoryEnum::SCALE_DATA:       ls = ScaleDataDesc::lineStride(this); break;
        case SurfaceCategoryEnum::WEIGHT:           ls = 0; break;
        default:
            REPORT_ERROR(NvDlaError_NotSupported, "Unsupported surface category: %s", m_surface_format.category().c_str());
    }
    m_line_stride = ls;
done:
    return ls;
}

NvU32 TensorSurfaceDesc::surfaceStride()
{
    NvU32 ss = 0;
    if (m_surface_stride != 0)
    {
        ss = m_surface_stride;
        goto done;
    }

    switch(m_surface_format.category().v())
    {
        case SurfaceCategoryEnum::IMG:              ss = 0; break;
        case SurfaceCategoryEnum::FEATURE_DATA:     ss = FeatureDataDesc::surfaceStride(this); break;
        case SurfaceCategoryEnum::BIAS_DATA:        ss = BiasDataDesc::surfaceStride(this); break;
        case SurfaceCategoryEnum::BATCH_NORM_DATA:  ss = BatchNormDataDesc::surfaceStride(this); break;
        case SurfaceCategoryEnum::SCALE_DATA:       ss = ScaleDataDesc::surfaceStride(this); break;
        case SurfaceCategoryEnum::WEIGHT:           ss = 0; break;
        default:
            REPORT_ERROR(NvDlaError_NotSupported, "Unsupported surface category: %s", m_surface_format.category().c_str());
    }
    m_surface_stride = ss;
done:
    return ss;
}

NvU32 TensorSurfaceDesc::planeStride() const
{
    NvU32 ps = 0;
    switch(m_surface_format.category().v())
    {
        case SurfaceCategoryEnum::BIAS_DATA:
        case SurfaceCategoryEnum::BATCH_NORM_DATA:
        case SurfaceCategoryEnum::SCALE_DATA:
        case SurfaceCategoryEnum::WEIGHT:
        case SurfaceCategoryEnum::FEATURE_DATA:
        case SurfaceCategoryEnum::IMG:
            ps = 0; break;
        default:
            REPORT_ERROR(NvDlaError_NotSupported, "Unsupported surface category: %s", m_surface_format.category().c_str());
    }

    return ps;
}

bool TensorSurfaceDesc::referencedByEMU() const
{
    bool emuDetected = false;
    for( unordered_set<engine_ast::Node *>::const_iterator pi = producers().begin();
         pi != producers().end(); ++pi )
    {
        emuDetected |= (*pi)->isEMUEngineType();
    }
    for ( unordered_set<engine_ast::Node *>::const_iterator ci = consumers().begin();
          ci != consumers().end(); ++ci )
    {
        emuDetected |= (*ci)->isEMUEngineType();
    }
    return emuDetected;
}

bool TensorSurfaceDesc::isSurfaceSymmetricTo(TensorSurfaceDesc* other)
{
    bool dims3Diff = (m_dims.c != other->m_dims.c || m_dims.h != other->m_dims.h || m_dims.w != other->m_dims.w);
    bool sizeDiff = m_size != other->m_size;
    bool lsDiff   = m_line_stride != other->m_line_stride;
    bool ssDiff   = m_surface_stride != other->m_surface_stride;

    return (!dims3Diff && !sizeDiff && !lsDiff && !ssDiff);
}

/*--------------------------------IMG DESC APIs------------------------------*/
PixelMapping IMGDesc::pixelMapping(const TensorSurfaceDesc* tsd)
{
    // FIXME: currently treating all images as pitch linear.
    //      in future, PL or BL would be determined from img dims
    PixelMapping pm = PixelMappingEnum::PITCH_LINEAR;
    NVDLA_UNUSED(tsd);
    return pm;
}

NvU32 IMGDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    NvU32 ls = 0ULL;
    PixelMapping pm = pixelMapping(tsd);
    switch(pm.v())
    {
        case PixelMappingEnum::PITCH_LINEAR:
            ls = PitchLinearSurfaceDesc::lineStride(tsd); break;
        default:
            REPORT_ERROR(NvDlaError_NotSupported, "Unsupported pixel mapping: %s", pm.c_str());
    }
    return ls;
}

NvU32 IMGDesc::surfaceStride(const TensorSurfaceDesc*)
{
    return 0;
}

NvU64 IMGDesc::size(const TensorSurfaceDesc* tsd)
{
    NvU64 size = 0ULL;
    PixelMapping pm = pixelMapping(tsd);
    switch(pm.v())
    {
        case PixelMappingEnum::PITCH_LINEAR:
            size = PitchLinearSurfaceDesc::size(tsd); break;
        default:
            size = 0;
    }
    return size;
}
/*-------------------IMG PITCH LINEAR SURFACE DESC APIs----------------------*/
NvU32 PitchLinearSurfaceDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU32 lineStride   = 0;
    NvU8 bpe = tsd->surfaceFormat().bytesPerElement();
    NvS8  cpa  = tsd->surfaceFormat().channelsPerAtom();

    if (tsd->surfaceFormat().v() > SurfaceFormatEnum::NVDLA_IMG_Y16___V16U16_N444)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Surface Format %s is not Pitch Linear", tsd->surfaceFormat().c_str());
    }

    if (tsd->surfaceFormat().v() < SurfaceFormatEnum::NVDLA_IMG_Y8___U8V8_N444)
    {
        // interleave format, single plannar
        lineStride   = ROUNDUP_AND_ALIGN(tsd->dimensions().w * bpe * cpa, 32);
    }
    else
    {
        lineStride   = ROUNDUP_AND_ALIGN(tsd->dimensions().w * bpe, 32);
    }

fail:
    return lineStride;
}

NvU32 PitchLinearSurfaceDesc::lineUVStride(const TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU32 lineUVStride = 0;
    NvU8 bpe = tsd->surfaceFormat().bytesPerElement();
    NvS8 cpa = tsd->surfaceFormat().channelsPerAtom();

    if (tsd->surfaceFormat().v() > SurfaceFormatEnum::NVDLA_IMG_Y16___V16U16_N444)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Surface Format %s is not Pitch Linear", tsd->surfaceFormat().c_str());
    }

    if (tsd->surfaceFormat().v() < SurfaceFormatEnum::NVDLA_IMG_Y8___U8V8_N444)
    {
        // interleave format, single plannar
        lineUVStride = 0;
    }
    else
    {
        lineUVStride = ROUNDUP_AND_ALIGN(tsd->dimensions().w * bpe * (cpa - 1), 32);
    }

fail:
    return lineUVStride;
}

NvU64 PitchLinearSurfaceDesc::size(const TensorSurfaceDesc* tsd)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(e);
    NvU64 size = 0;

    if (tsd->surfaceFormat().v() > SurfaceFormatEnum::NVDLA_IMG_Y16___V16U16_N444)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Surface Format %s is not Pitch Linear", tsd->surfaceFormat().c_str());
    }

    ///// FIXME: treat pitch linear image as FD format for now - conservative size (~x32) (remove comment safely?)
    //size = tsd->dimensions().c * lineStride(tsd) * tsd->dimensions().h;
    size = tsd->dimensions().h * lineStride(tsd);

fail:
    return size;
}

/*-----------------------FEATURE DATA DESC APIs------------------------------*/
NvU32 FeatureDataDesc::numBatches(const TensorSurfaceDesc* tsd)
{
    //FIXME: currently single batch mode
    NVDLA_UNUSED(tsd);
    return 1;
}

NvU32 FeatureDataDesc::channelsPerGroup(const TensorSurfaceDesc* tsd)
{
    nvdla::priv::engine_ast::Edge *edge_par =  tsd->parentEdge();
    NvU32 atom_k_size = edge_par->graph()->target_config()->atomicKSize();

    return tsd->surfaceFormat().precision().v() == SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? atom_k_size : (atom_k_size / 2);
}

NvU32 FeatureDataDesc::channelGroups(const TensorSurfaceDesc* tsd)
{
    return (NvU32)ceilf((NvF32)tsd->dimensions().c / (NvF32)channelsPerGroup(tsd));
}

NvU32 FeatureDataDesc::height(const TensorSurfaceDesc* tsd)
{
    return tsd->dimensions().h;
}

NvU32 FeatureDataDesc::width(const TensorSurfaceDesc* tsd)
{
    return tsd->dimensions().w;
}

NvU32 FeatureDataDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    return tsd->dimensions().w * channelsPerGroup(tsd) * tsd->surfaceFormat().bytesPerElement();
}

NvU32 FeatureDataDesc::surfaceStride(const TensorSurfaceDesc* tsd)
{
    return lineStride(tsd) * tsd->dimensions().h;
}

NvU64 FeatureDataDesc::size(const TensorSurfaceDesc* tsd)
{
    return surfaceStride(tsd) * channelGroups(tsd);
}

/*-----------------------ELTWISE DATA DESC APIs------------------------------*/
NvU32 EltwiseDataDesc::channelsPerGroup(const TensorSurfaceDesc* tsd)
{
    nvdla::priv::engine_ast::Edge *edge_par =  tsd->parentEdge();
    NvU32 atom_k_size = edge_par->graph()->target_config()->atomicKSize();

    return tsd->surfaceFormat().precision().v() == SurfacePrecisionEnum::NVDLA_PRECISION_FP16 ? (atom_k_size / 2) : atom_k_size;
}

NvU32 EltwiseDataDesc::channelGroups(const TensorSurfaceDesc* tsd)
{
    return (NvU32)ceilf((NvF32)tsd->dimensions().c / (NvF32)channelsPerGroup(tsd));
}

NvU32 EltwiseDataDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    return tsd->dimensions().w * channelsPerGroup(tsd) * tsd->surfaceFormat().bytesPerElement();
}

NvU32 EltwiseDataDesc::surfaceStride(const TensorSurfaceDesc* tsd)
{
    return lineStride(tsd) * tsd->dimensions().h;
}

NvU64 EltwiseDataDesc::size(const TensorSurfaceDesc* tsd)
{
    return surfaceStride(tsd) * channelGroups(tsd);
}

/*--------------------------WEIGHT DESC APIs---------------------------------*/
bool WeightDesc::iscompressed(const TensorSurfaceDesc* tsd)
{
    return string(tsd->surfaceFormat().c_str()).find("COMPRESSED") != string::npos ? true : false;
}

NvU32 WeightDesc::fullChnlsPerGrp(const TensorSurfaceDesc* tsd)
{
    NvU32 fullChnlsPerGrp = 0;

    if (string(tsd->surfaceFormat().c_str()).find("DC") != string::npos)
    {
        nvdla::priv::engine_ast::Edge *edge_par =  tsd->parentEdge();
        NvU32 atom_k_size = edge_par->graph()->target_config()->atomicCSize();
        fullChnlsPerGrp = atom_k_size;
    }
    else if (string(tsd->surfaceFormat().c_str()).find("WG") != string::npos)
    {
        fullChnlsPerGrp = WG_FULL_CHANNELS_PER_ATOM;
    }

    return fullChnlsPerGrp;
}

NvU32 WeightDesc::fullChnlGroups(const TensorSurfaceDesc* tsd)
{
    NvU32 fullChnlGroups = 0;
    if (string(tsd->surfaceFormat().c_str()).find("DC") != string::npos)
    {
        fullChnlGroups = tsd->dimensions().c / fullChnlsPerGrp(tsd);
    }
    else if (string(tsd->surfaceFormat().c_str()).find("WG") != string::npos)
    {
        fullChnlGroups = tsd->dimensions().c / fullChnlsPerGrp(tsd);
    }

    return fullChnlGroups;
}

NvU32 WeightDesc::partialChnlsPerGrp(const TensorSurfaceDesc* tsd)
{
    NvU32 partialChnlsPerGrp  = 0;
    if (string(tsd->surfaceFormat().c_str()).find("DC") != string::npos)
    {
        partialChnlsPerGrp = tsd->dimensions().c % fullChnlsPerGrp(tsd);
    }
    else if (string(tsd->surfaceFormat().c_str()).find("WG") != string::npos)
    {
        partialChnlsPerGrp = 0;     // In WG, #channels are rounded upto nearest multiple of 32
    }

    return partialChnlsPerGrp;
}

NvU32 WeightDesc::fullKrnlsPerGrp(const TensorSurfaceDesc* tsd)
{
    NvU32 fullKrnlsPerGrp = 0;

    nvdla::priv::engine_ast::Edge *edge_par =  tsd->parentEdge();
    NvU32 atom_k_size = edge_par->graph()->target_config()->atomicKSize();

    if (string(tsd->surfaceFormat().c_str()).find("DC") != string::npos)
    {
        fullKrnlsPerGrp = tsd->surfaceFormat().precision().v() == SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? atom_k_size : (atom_k_size / 2);
    }
    else if (string(tsd->surfaceFormat().c_str()).find("WG") != string::npos)
    {
        fullKrnlsPerGrp = tsd->surfaceFormat().precision().v() == SurfacePrecisionEnum::NVDLA_PRECISION_INT8 ? atom_k_size : (atom_k_size / 2);
    }

    return fullKrnlsPerGrp;
}

NvU32 WeightDesc::fullKrnlGroups(const TensorSurfaceDesc* tsd)
{
    NvU32 fullKrnlGroups = 0;
    if (string(tsd->surfaceFormat().c_str()).find("DC") != string::npos)
    {
        fullKrnlGroups = tsd->dimensions().n / fullKrnlsPerGrp(tsd);
    }
    else if (string(tsd->surfaceFormat().c_str()).find("WG") != string::npos)
    {
        fullKrnlGroups = tsd->dimensions().n / fullKrnlsPerGrp(tsd);
    }

    return fullKrnlGroups;
}

NvU32 WeightDesc::partialKrnlsPerGrp(const TensorSurfaceDesc* tsd)
{
    NvU32 partialKrnlsPerGrp = 0;
    if (string(tsd->surfaceFormat().c_str()).find("DC") != string::npos)
    {
        partialKrnlsPerGrp = tsd->dimensions().n / fullKrnlsPerGrp(tsd);
    }
    else if (string(tsd->surfaceFormat().c_str()).find("WG") != string::npos)
    {
        partialKrnlsPerGrp = tsd->dimensions().n / fullKrnlsPerGrp(tsd);
    }

    return partialKrnlsPerGrp;
}

NvU32 WeightDesc::wgs(const TensorSurfaceDesc* tsd)
{
    gLogError << __func__ << "Not Yet Supported" << endl;
    NVDLA_UNUSED(tsd);
    return 0;
}

NvU32 WeightDesc::wmb(const TensorSurfaceDesc* tsd)
{
    gLogError << __func__ << "Not Yet Supported" << endl;
    NVDLA_UNUSED(tsd);
    return 0;
}

NvU64 WeightDesc::size(const TensorSurfaceDesc* tsd)
{
    nvdla::priv::engine_ast::Edge *edge_par =  tsd->parentEdge();
    NvU32 cbuf_bank_width = edge_par->graph()->target_config()->bufEntryWidth();

    return ROUNDUP_AND_ALIGN(tsd->dimensions().n *
                             tsd->dimensions().c *
                             tsd->dimensions().h *
                             tsd->dimensions().w *
                             tsd->surfaceFormat().bytesPerElement(), cbuf_bank_width);
}

NvU32 WeightDesc::bytesPerKernel(const TensorSurfaceDesc* tsd)
{
    return (tsd->dimensions().c *
            tsd->dimensions().h *
            tsd->dimensions().w *
            tsd->surfaceFormat().bytesPerElement());
}

NvU64 WeightDesc::rawSize(const TensorSurfaceDesc* tsd)
{
    return (tsd->dimensions().n *
            tsd->dimensions().c *
            tsd->dimensions().h *
            tsd->dimensions().w *
            tsd->surfaceFormat().bytesPerElement());
}

/*--------------------------BIAS DATA DESC APIs------------------------------*/
BiasDataCategory BiasDataDesc::biasDataCategory(const TensorSurfaceDesc* tsd)
{
    BiasDataCategory bdc = BiasDataCategoryEnum::BIAS_DATA_CATEGORY_UNKNOWN;
    /* Bias Data could be any of the 3 types:
     *   per-Layer:     1 x 1 x 1
     *   per-Channel:   1 x 1 x C
     *   per-Element:   W x H x C
     */
    if (tsd->dimensions().c == 1 && tsd->dimensions().h == 1 && tsd->dimensions().w == 1)
    {
        bdc = BiasDataCategoryEnum::PER_LAYER_BIAS_DATA;
    }
    else
    {
        if (tsd->dimensions().h == 1 && tsd->dimensions().w == 1)
        {
            bdc = BiasDataCategoryEnum::PER_CHANNEL_BIAS_DATA;
        }
        else
        {
            bdc = BiasDataCategoryEnum::PER_ELEMENT_BIAS_DATA;
        }
    }
    return bdc;
}

NvU32 BiasDataDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    NvU32 lineStride = 0;

    lineStride = EltwiseDataDesc::lineStride(tsd);
    if (tsd->alignLineStride())
    {
        lineStride = lineStride * 2;
    }

    return lineStride;
}

NvU32 BiasDataDesc::surfaceStride(const TensorSurfaceDesc* tsd)
{
    return lineStride(tsd) * tsd->dimensions().h;
}

NvU64 BiasDataDesc::size(const TensorSurfaceDesc* tsd)
{
    NvU32 size = 0;

    switch(biasDataCategory(tsd).e()) {
        case BiasDataCategoryEnum::PER_LAYER_BIAS_DATA:
        case BiasDataCategoryEnum::PER_CHANNEL_BIAS_DATA:
            size = tsd->dimensions().c * tsd->dimensions().h
                * tsd->dimensions().w * tsd->surfaceFormat().bytesPerElement();
            break;
        case BiasDataCategoryEnum::PER_ELEMENT_BIAS_DATA:
            size = EltwiseDataDesc::size(tsd);
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter,
                        "Unable to compute size: unknown SDP data category %s",
                        biasDataCategory(tsd).c_str());
    }

    return size;
}

/*--------------------------BATCH NORM DATA DESC APIs------------------------*/
BatchNormDataCategory BatchNormDataDesc::batchNormDataCategory(const TensorSurfaceDesc* tsd)
{
    BatchNormDataCategory bndc = BatchNormDataCategoryEnum::BATCH_NORM_DATA_CATEGORY_UNKNOWN;
    /* Batch Norm Data can be any of the 2 types:
     *   per-Layer:     1 x 1 x 1
     *   per-Channel:   1 x 1 x C
     */
    if (tsd->dimensions().c == 1)
    {
        bndc = BatchNormDataCategoryEnum::PER_LAYER_BATCH_NORM_DATA;
    }
    else if (tsd->dimensions().h == 1 && tsd->dimensions().w == 1)
    {
        bndc = BatchNormDataCategoryEnum::PER_CHANNEL_BATCH_NORM_DATA;
    }
    return bndc;
}

NvU32 BatchNormDataDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    return EltwiseDataDesc::lineStride(tsd);
}

NvU32 BatchNormDataDesc::surfaceStride(const TensorSurfaceDesc* tsd)
{
    return lineStride(tsd) * tsd->dimensions().h;
}

NvU64 BatchNormDataDesc::size(const TensorSurfaceDesc* tsd)
{
    return (2 *                                         /* 1 for mean and 1 for variance */
            tsd->surfaceFormat().bytesPerElement() *
            tsd->dimensions().c);
}

/*--------------------------SCALE DATA DESC APIs-----------------------------*/
ScaleDataCategory ScaleDataDesc::scaleDataCategory(const TensorSurfaceDesc* tsd)
{
    ScaleDataCategory sdc = ScaleDataCategoryEnum::SCALE_DATA_CATEGORY_UNKNOWN;
    /* Scale Data can be any of the 3 types:
     *   per-Layer:     1 x 1 x 1
     *   per-Channel:   1 x 1 x C
     *   per-Element:   W x H x C
     */
    if (tsd->dimensions().c == 1 && tsd->dimensions().h == 1 && tsd->dimensions().w == 1)
    {
        sdc = ScaleDataCategoryEnum::PER_LAYER_SCALE_DATA;
    }
    else
    {
        if (tsd->dimensions().h == 1 && tsd->dimensions().w == 1)
        {
            sdc = ScaleDataCategoryEnum::PER_CHANNEL_SCALE_DATA;
        }
        else
        {
            sdc = ScaleDataCategoryEnum::PER_ELEMENT_SCALE_DATA;
        }
    }
    return sdc;
}

NvU32 ScaleDataDesc::lineStride(const TensorSurfaceDesc* tsd)
{
    NvU32 lineStride = EltwiseDataDesc::lineStride(tsd);

    return lineStride;
}

NvU32 ScaleDataDesc::surfaceStride(const TensorSurfaceDesc* tsd)
{
    return lineStride(tsd) * tsd->dimensions().h;
}

NvU64 ScaleDataDesc::size(const TensorSurfaceDesc* tsd)
{
    NvU32 size = 0;

    switch(scaleDataCategory(tsd).e()) {
        case ScaleDataCategoryEnum::PER_LAYER_SCALE_DATA:
        case ScaleDataCategoryEnum::PER_CHANNEL_SCALE_DATA:
            size = tsd->dimensions().c * tsd->dimensions().h
                * tsd->dimensions().w * tsd->surfaceFormat().bytesPerElement();
            break;
        case ScaleDataCategoryEnum::PER_ELEMENT_SCALE_DATA:
            size = EltwiseDataDesc::size(tsd);
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter,
                        "Unable to compute size: unknown SDP data category %s",
                        scaleDataCategory(tsd).c_str());
    }

    return size;
}

};  // nvdla::priv::surface::
};  // nvdla::priv::
};  // nvdla::
