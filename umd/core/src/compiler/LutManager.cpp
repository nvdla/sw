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

#include "half.h"
#include "priv/Compiler.h"
#include "priv/LutManager.h"

#include <sstream>

namespace nvdla
{
namespace priv
{

LutManager::LutManager() :
    m_lutHandles(),
    m_lutParams(),
    m_hNextFree(0)
{

}

LutManager::~LutManager()
{

}

NvDlaError LutManager::registerLRN(surface::SurfacePrecisionEnum precision, NvU32 localSize, NvF32 alpha, NvF32 beta, NvF32 k, LutHandle* hLut)
{
    NvDlaError e = NvDlaSuccess;
    LutParams lutParams;

    lutParams.type = LutManager::LUT_TYPE_LRN;
    lutParams.precision = precision;
    lutParams.lrnParams.localSize = localSize;
    lutParams.lrnParams.alpha = alpha;
    lutParams.lrnParams.beta = beta;
    lutParams.lrnParams.k = k;

    PROPAGATE_ERROR_FAIL(getHandle(&lutParams, hLut));

fail:
    return e;
}

NvDlaError LutManager::registerSigmoid(surface::SurfacePrecisionEnum precision, LutHandle* hLut)
{
    NvDlaError e = NvDlaSuccess;
    LutParams lutParams;

    lutParams.type = LutManager::LUT_TYPE_SIGMOID;
    lutParams.precision = precision;

    PROPAGATE_ERROR_FAIL(getHandle(&lutParams, hLut));

fail:
    return e;
}

NvDlaError LutManager::registerTanh(surface::SurfacePrecisionEnum precision, LutHandle* hLut)
{
    NvDlaError e = NvDlaSuccess;
    LutParams lutParams;

    lutParams.type = LutManager::LUT_TYPE_TANH;
    lutParams.precision = precision;

    PROPAGATE_ERROR_FAIL(getHandle(&lutParams, hLut));

fail:
    return e;
}

NvS16 LutManager::getIndex(LutHandle hLut) const
{
    return static_cast<NvS16>(hLut);
}

NvS16 LutManager::getNumRegisteredLuts()
{
    return static_cast<NvS16>(m_hNextFree);
}

NvDlaError LutManager::writeLutData(NvU16 lutSlot, DLALUTParamAccessor lutAcc)
{
    NvDlaError e = NvDlaSuccess;

    LutHandle hLut = static_cast<LutHandle>(lutSlot);
    LutParams lutParams = m_lutParams[hLut];

    switch(lutParams.type)
    {
    case LUT_TYPE_LRN:
        PROPAGATE_ERROR_FAIL(writeLRNData(&lutParams, lutAcc));
        break;
    case LUT_TYPE_SIGMOID:
        PROPAGATE_ERROR_FAIL(writeSigmoidData(&lutParams, lutAcc));
        break;
    case LUT_TYPE_TANH:
        PROPAGATE_ERROR_FAIL(writeTanhData(&lutParams, lutAcc));
        break;
    default:
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unsupported LutParam type %d", lutParams.type);
    }

fail:
    return e;
}

NvDlaError LutManager::writeLRNData(const LutParams* lutParams, DLALUTParamAccessor lutAcc)
{
    NvDlaError e = NvDlaSuccess;

    if (!lutParams)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    if (lutParams->precision == surface::NVDLA_PRECISION_FP16)
    {
        // LRN, FP16, N=any, alpha=any, beta=any, k=any
        // mode=EXP, start_index=0
        NvU32 startIndex = 0;
        NvS16 lrnFP16Table[lutAcc.numLinearExpTable()];

        for (NvU32 ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvF64 input = std::pow(2, (ii + startIndex));
            NvF64 output = 1.0 / std::pow(lutParams->lrnParams.k + (lutParams->lrnParams.alpha / lutParams->lrnParams.localSize) * input, lutParams->lrnParams.beta);

            half_float::half outputFP16 = static_cast<half_float::half>(static_cast<NvF32>(output));
            half_float::half* p = &outputFP16;
            lrnFP16Table[ii] = *(reinterpret_cast<NvS16*>(p)); // Better to cast via union types instead
        }

        // Program LE Lut
        for (size_t ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvS16* expTable = lutAcc.linearExpTable(ii);
            *expTable = lrnFP16Table[ii];
        }

        // Program LO Lut
        for (size_t ii=0; ii<lutAcc.numLinearOnlyTable(); ii++)
        {
            NvS16* linearTable = lutAcc.linearOnlyTable(ii);
            *linearTable = 0x0000;
        }

        // Program the LE method
        *lutAcc.method() = lutAcc.method_Exponential();

        // Program LE/LO offsets
        DLALUTOffsetAccessor lutLEOffsetAcc = lutAcc.linearExpOffsetAccessor();
        *lutLEOffsetAcc.expOffset() = startIndex; // (2^start_index)
        DLALUTOffsetAccessor lutLOOffsetAcc = lutAcc.linearOnlyOffsetAccessor();
        *lutLOOffsetAcc.fracBits() = -128; // disabled

        // Program LE/LO offsets, valid only for LINEAR method
        *lutAcc.linearExpStart() = 0;
        *lutAcc.linearExpEnd() = 0;
        *lutAcc.linearOnlyStart() = 1;
        *lutAcc.linearOnlyEnd() = 1;

        // Program LE/LO slopes, 0.0f
        DLASlopeAccessor leuSlopeAcc = lutAcc.linearExpUnderflowSlopeAccessor();
        *leuSlopeAcc.dataF() = 0.0f;
        DLASlopeAccessor leoSlopeAcc = lutAcc.linearExpOverflowSlopeAccessor();
        *leoSlopeAcc.dataF() = 0.0f;
        DLASlopeAccessor louSlopeAcc = lutAcc.linearOnlyUnderflowSlopeAccessor();
        *louSlopeAcc.dataF() = 0.0f;
        DLASlopeAccessor looSlopeAcc = lutAcc.linearOnlyOverflowSlopeAccessor();
        *looSlopeAcc.dataF() = 0.0f;

        // Program priority bits
        *lutAcc.hybridPriority() = lutAcc.priority_LinearExp();
        *lutAcc.underflowPriority() = lutAcc.priority_LinearExp();
        *lutAcc.overflowPriority() = lutAcc.priority_LinearExp();
    }
    else if (lutParams->precision == surface::NVDLA_PRECISION_INT8)
    {
        REPORT_ERROR(NvDlaError_BadValue, "Empty INT8 LRN LUT");

        // LRN, INT8, N=any, alpha=any, beta=any, k=any
        // mode=EXP, start_index=0
        NvU32 startIndex = 0;

        // Program LE Lut
        for (size_t ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvS16* expTable = lutAcc.linearExpTable(ii);
            *expTable = 0x0000;
        }

        // Program LO Lut
        for (size_t ii=0; ii<lutAcc.numLinearOnlyTable(); ii++)
        {
            NvS16* linearTable = lutAcc.linearOnlyTable(ii);
            *linearTable = 0x0000;
        }

        // Program the LE method
        *lutAcc.method() = lutAcc.method_Exponential();

        // Program LE/LO offsets
        DLALUTOffsetAccessor lutLEOffsetAcc = lutAcc.linearExpOffsetAccessor();
        *lutLEOffsetAcc.expOffset() = startIndex; // (2^start_index)
        DLALUTOffsetAccessor lutLOOffsetAcc = lutAcc.linearOnlyOffsetAccessor();
        *lutLOOffsetAcc.fracBits() = -128; // disabled

        // Program LE/LO offsets, valid only for LINEAR method
        *lutAcc.linearExpStart() = 0;
        *lutAcc.linearExpEnd() = 0;
        *lutAcc.linearOnlyStart() = 1;
        *lutAcc.linearOnlyEnd() = 1;

        // Program LE/LO slopes, {0 >> 0}
        DLAFloatDataAccessor leuSlopeAcc = lutAcc.linearExpUnderflowSlopeAccessor().dataIAccessor();
        *leuSlopeAcc.scale() = 0x0000;
        *leuSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor leoSlopeAcc = lutAcc.linearExpOverflowSlopeAccessor().dataIAccessor();
        *leoSlopeAcc.scale() = 0x0000;
        *leoSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor louSlopeAcc = lutAcc.linearOnlyUnderflowSlopeAccessor().dataIAccessor();
        *louSlopeAcc.scale() = 0x0000;
        *louSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor looSlopeAcc = lutAcc.linearOnlyOverflowSlopeAccessor().dataIAccessor();
        *looSlopeAcc.scale() = 0x0000;
        *looSlopeAcc.shifter() = 0x00;

        // Program priority bits
        *lutAcc.hybridPriority() = lutAcc.priority_LinearExp();
        *lutAcc.underflowPriority() = lutAcc.priority_LinearExp();
        *lutAcc.overflowPriority() = lutAcc.priority_LinearExp();
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unsupported LRN Precision: %d", lutParams->precision);
    }

fail:
    return e;
}

NvDlaError LutManager::writeSigmoidData(const LutParams* lutParams, DLALUTParamAccessor lutAcc)
{
    NvDlaError e = NvDlaSuccess;

    if (!lutParams)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    if (lutParams->precision == surface::NVDLA_PRECISION_FP16)
    {
        // SIGMOID, FP16
        // mode=LINEAR, num_elements=257, index_offset=-16, index_sel=-3
        NvF32 indexOffset = -16.0f;
        NvS16 indexSel = -3;
        NvU16 sigmoidFP16Lut[] = {
            0x0002, 0x0002, 0x0002, 0x0003, 0x0003, 0x0004, 0x0004, 0x0005,
            0x0005, 0x0006, 0x0007, 0x0007, 0x0008, 0x000a, 0x000b, 0x000c,
            0x000e, 0x0010, 0x0012, 0x0014, 0x0017, 0x001a, 0x001e, 0x0021,
            0x0026, 0x002b, 0x0031, 0x0037, 0x003f, 0x0047, 0x0050, 0x005b,
            0x0067, 0x0075, 0x0084, 0x0096, 0x00aa, 0x00c1, 0x00da, 0x00f7,
            0x0118, 0x013e, 0x0168, 0x0198, 0x01ce, 0x020b, 0x0251, 0x02a0,
            0x02fa, 0x035f, 0x03d2, 0x0454, 0x04e8, 0x058f, 0x064c, 0x0723,
            0x080b, 0x0895, 0x0931, 0x09e2, 0x0aaa, 0x0b8e, 0x0c48, 0x0cd9,
            0x0d7f, 0x0e3a, 0x0f0e, 0x0ffe, 0x1087, 0x1122, 0x11d0, 0x1296,
            0x1377, 0x143a, 0x14ca, 0x156d, 0x1626, 0x16f7, 0x17e4, 0x1878,
            0x1910, 0x19bc, 0x1a7f, 0x1b5c, 0x1c2b, 0x1cb8, 0x1d58, 0x1e0e,
            0x1eda, 0x1fc2, 0x2064, 0x20f9, 0x21a0, 0x225d, 0x2333, 0x2412,
            0x249b, 0x2535, 0x25e2, 0x26a5, 0x2781, 0x283c, 0x28c7, 0x2963,
            0x2a12, 0x2ad6, 0x2bb1, 0x2c53, 0x2cdb, 0x2d72, 0x2e1a, 0x2ed4,
            0x2fa1, 0x3041, 0x30bd, 0x3144, 0x31d6, 0x3275, 0x3320, 0x33d8,
            0x344e, 0x34b5, 0x3522, 0x3594, 0x360a, 0x3684, 0x3701, 0x3780,
            0x3800, 0x3840, 0x387f, 0x38be, 0x38fb, 0x3936, 0x396f, 0x39a5,
            0x39d9, 0x3a0a, 0x3a38, 0x3a63, 0x3a8a, 0x3aaf, 0x3ad1, 0x3af0,
            0x3b0c, 0x3b25, 0x3b3d, 0x3b52, 0x3b65, 0x3b76, 0x3b85, 0x3b93,
            0x3b9f, 0x3baa, 0x3bb4, 0x3bbc, 0x3bc4, 0x3bcb, 0x3bd1, 0x3bd6,
            0x3bdb, 0x3bdf, 0x3be3, 0x3be7, 0x3be9, 0x3bec, 0x3bee, 0x3bf0,
            0x3bf2, 0x3bf4, 0x3bf5, 0x3bf7, 0x3bf8, 0x3bf9, 0x3bfa, 0x3bfa,
            0x3bfb, 0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfe, 0x3bfe,
            0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
            0x3bff, 0x3bff, 0x3bff, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
            0x3c00,
        };

        // Program LE Lut
        for (size_t ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvS16* expTable = lutAcc.linearExpTable(ii);
            *expTable = 0x0000;
        }

        // Program LO Lut
        for (size_t ii=0; ii<lutAcc.numLinearOnlyTable(); ii++)
        {
            NvU16* linearTable = reinterpret_cast<NvU16*>(lutAcc.linearOnlyTable(ii));
            *linearTable = sigmoidFP16Lut[ii];
        }

        // Program the LE method
        *lutAcc.method() = lutAcc.method_Linear();

        // Program LE/LO offsets
        DLALUTOffsetAccessor lutLEOffsetAcc = lutAcc.linearExpOffsetAccessor();
        *lutLEOffsetAcc.fracBits() = -128; // disables lut
        DLALUTOffsetAccessor lutLOOffsetAcc = lutAcc.linearOnlyOffsetAccessor();
        *lutLOOffsetAcc.fracBits() = indexSel; // index_sel

        // Program LE/LO offsets, valid only for LINEAR method
        *lutAcc.linearExpStart() = 0; // disabled
        *lutAcc.linearExpEnd() = 0; // unused
        uint32_t* pIndexOffset = reinterpret_cast<uint32_t*>(&indexOffset);
        *lutAcc.linearOnlyStart() = static_cast<uint64_t>(*pIndexOffset); // index_offset
        *lutAcc.linearOnlyEnd() = 0; // unused

        // Program LE/LO slopes, 0.0f
        DLASlopeAccessor leuSlopeAcc = lutAcc.linearExpUnderflowSlopeAccessor();
        *leuSlopeAcc.dataF() = 0x0000;
        DLASlopeAccessor leoSlopeAcc = lutAcc.linearExpOverflowSlopeAccessor();
        *leoSlopeAcc.dataF() = 0x0000;
        DLASlopeAccessor louSlopeAcc = lutAcc.linearOnlyUnderflowSlopeAccessor();
        *louSlopeAcc.dataF() = 0x0000;
        DLASlopeAccessor looSlopeAcc = lutAcc.linearOnlyOverflowSlopeAccessor();
        *looSlopeAcc.dataF() = 0x0000;

        // Program priority bits
        *lutAcc.hybridPriority() = lutAcc.priority_LinearOnly();
        *lutAcc.underflowPriority() = lutAcc.priority_LinearOnly();
        *lutAcc.overflowPriority() = lutAcc.priority_LinearOnly();
    }
    else if (lutParams->precision == surface::NVDLA_PRECISION_INT8)
    {
        REPORT_ERROR(NvDlaError_BadValue, "Empty INT8 Sigmoid LUT");

        // SIGMOID, INT8, N=any, alpha=any, beta=any, k=any
        // mode=EXP, start_index=0
        NvU32 startIndex = 0;

        // Program LE Lut
        for (size_t ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvS16* expTable = lutAcc.linearExpTable(ii);
            *expTable = 0x0000;
        }

        // Program LO Lut
        for (size_t ii=0; ii<lutAcc.numLinearOnlyTable(); ii++)
        {
            NvS16* linearTable = lutAcc.linearOnlyTable(ii);
            *linearTable = 0x0000;
        }

        // Program the LE method
        *lutAcc.method() = lutAcc.method_Exponential();

        // Program LE/LO offsets
        DLALUTOffsetAccessor lutLEOffsetAcc = lutAcc.linearExpOffsetAccessor();
        *lutLEOffsetAcc.expOffset() = startIndex; // (2^start_index)
        DLALUTOffsetAccessor lutLOOffsetAcc = lutAcc.linearOnlyOffsetAccessor();
        *lutLOOffsetAcc.fracBits() = -128; // disabled

        // Program LE/LO offsets, valid only for LINEAR method
        *lutAcc.linearExpStart() = 0;
        *lutAcc.linearExpEnd() = 0;
        *lutAcc.linearOnlyStart() = 1;
        *lutAcc.linearOnlyEnd() = 1;

        // Program LE/LO slopes, {0 >> 0}
        DLAFloatDataAccessor leuSlopeAcc = lutAcc.linearExpUnderflowSlopeAccessor().dataIAccessor();
        *leuSlopeAcc.scale() = 0x0000;
        *leuSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor leoSlopeAcc = lutAcc.linearExpOverflowSlopeAccessor().dataIAccessor();
        *leoSlopeAcc.scale() = 0x0000;
        *leoSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor louSlopeAcc = lutAcc.linearOnlyUnderflowSlopeAccessor().dataIAccessor();
        *louSlopeAcc.scale() = 0x0000;
        *louSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor looSlopeAcc = lutAcc.linearOnlyOverflowSlopeAccessor().dataIAccessor();
        *looSlopeAcc.scale() = 0x0000;
        *looSlopeAcc.shifter() = 0x00;

        // Program priority bits
        *lutAcc.hybridPriority() = lutAcc.priority_LinearExp();
        *lutAcc.underflowPriority() = lutAcc.priority_LinearExp();
        *lutAcc.overflowPriority() = lutAcc.priority_LinearExp();
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unsupported Sigmoid Precision: %d", lutParams->precision);
    }

fail:
    return e;
}

NvDlaError LutManager::writeTanhData(const LutParams* lutParams, DLALUTParamAccessor lutAcc)
{
    NvDlaError e = NvDlaSuccess;

    if (!lutParams)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    if (lutParams->precision == surface::NVDLA_PRECISION_FP16)
    {
        // TANH, FP16
        // mode=LINEAR, num_elements=257, index_offset=-4, index_sel=-5
        NvF32 indexOffset = -4.0f;
        NvS16 indexSel = -5;
        NvU16 tanhFP16Lut[] = {
            0xbbff, 0xbbff, 0xbbfe, 0xbbfe, 0xbbfe, 0xbbfe, 0xbbfe, 0xbbfe,
            0xbbfe, 0xbbfe, 0xbbfd, 0xbbfd, 0xbbfd, 0xbbfd, 0xbbfd, 0xbbfc,
            0xbbfc, 0xbbfc, 0xbbfc, 0xbbfb, 0xbbfb, 0xbbfb, 0xbbfb, 0xbbfa,
            0xbbfa, 0xbbf9, 0xbbf9, 0xbbf9, 0xbbf8, 0xbbf8, 0xbbf7, 0xbbf6,
            0xbbf6, 0xbbf5, 0xbbf5, 0xbbf4, 0xbbf3, 0xbbf2, 0xbbf1, 0xbbf0,
            0xbbef, 0xbbee, 0xbbed, 0xbbec, 0xbbeb, 0xbbe9, 0xbbe8, 0xbbe6,
            0xbbe5, 0xbbe3, 0xbbe1, 0xbbdf, 0xbbdd, 0xbbdb, 0xbbd8, 0xbbd6,
            0xbbd3, 0xbbd0, 0xbbcd, 0xbbca, 0xbbc6, 0xbbc3, 0xbbbf, 0xbbbb,
            0xbbb6, 0xbbb2, 0xbbad, 0xbba7, 0xbba2, 0xbb9c, 0xbb96, 0xbb8f,
            0xbb88, 0xbb80, 0xbb78, 0xbb70, 0xbb67, 0xbb5e, 0xbb54, 0xbb49,
            0xbb3e, 0xbb32, 0xbb25, 0xbb18, 0xbb0a, 0xbafb, 0xbaeb, 0xbadb,
            0xbac9, 0xbab7, 0xbaa3, 0xba8f, 0xba79, 0xba63, 0xba4b, 0xba32,
            0xba18, 0xb9fc, 0xb9df, 0xb9c1, 0xb9a2, 0xb981, 0xb95e, 0xb93a,
            0xb915, 0xb8ee, 0xb8c5, 0xb89b, 0xb870, 0xb843, 0xb814, 0xb7c8,
            0xb765, 0xb6ff, 0xb696, 0xb62a, 0xb5bc, 0xb54b, 0xb4d8, 0xb463,
            0xb3d6, 0xb2e4, 0xb1ee, 0xb0f6, 0xaff5, 0xadfc, 0xabfd, 0xa7ff,
            0x0000, 0x27ff, 0x2bfd, 0x2dfc, 0x2ff5, 0x30f6, 0x31ee, 0x32e4,
            0x33d6, 0x3463, 0x34d8, 0x354b, 0x35bc, 0x362a, 0x3696, 0x36ff,
            0x3765, 0x37c8, 0x3814, 0x3843, 0x3870, 0x389b, 0x38c5, 0x38ee,
            0x3915, 0x393a, 0x395e, 0x3981, 0x39a2, 0x39c1, 0x39df, 0x39fc,
            0x3a18, 0x3a32, 0x3a4b, 0x3a63, 0x3a79, 0x3a8f, 0x3aa3, 0x3ab7,
            0x3ac9, 0x3adb, 0x3aeb, 0x3afb, 0x3b0a, 0x3b18, 0x3b25, 0x3b32,
            0x3b3e, 0x3b49, 0x3b54, 0x3b5e, 0x3b67, 0x3b70, 0x3b78, 0x3b80,
            0x3b88, 0x3b8f, 0x3b96, 0x3b9c, 0x3ba2, 0x3ba7, 0x3bad, 0x3bb2,
            0x3bb6, 0x3bbb, 0x3bbf, 0x3bc3, 0x3bc6, 0x3bca, 0x3bcd, 0x3bd0,
            0x3bd3, 0x3bd6, 0x3bd8, 0x3bdb, 0x3bdd, 0x3bdf, 0x3be1, 0x3be3,
            0x3be5, 0x3be6, 0x3be8, 0x3be9, 0x3beb, 0x3bec, 0x3bed, 0x3bee,
            0x3bef, 0x3bf0, 0x3bf1, 0x3bf2, 0x3bf3, 0x3bf4, 0x3bf5, 0x3bf5,
            0x3bf6, 0x3bf6, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf9, 0x3bf9, 0x3bf9,
            0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc,
            0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfe,
            0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff,
            0x3bff,
        };

        // Program LE Lut
        for (size_t ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvS16* expTable = lutAcc.linearExpTable(ii);
            *expTable = 0x0000;
        }

        // Program LO Lut
        for (size_t ii=0; ii<lutAcc.numLinearOnlyTable(); ii++)
        {
            NvU16* linearTable = reinterpret_cast<NvU16*>(lutAcc.linearOnlyTable(ii));
            *linearTable = tanhFP16Lut[ii];
        }

        // Program the LE method
        *lutAcc.method() = lutAcc.method_Linear();

        // Program LE/LO offsets
        DLALUTOffsetAccessor lutLEOffsetAcc = lutAcc.linearExpOffsetAccessor();
        *lutLEOffsetAcc.fracBits() = -128; // disables lut
        DLALUTOffsetAccessor lutLOOffsetAcc = lutAcc.linearOnlyOffsetAccessor();
        *lutLOOffsetAcc.fracBits() = indexSel; // index_sel

        // Program LE/LO offsets, valid only for LINEAR method
        *lutAcc.linearExpStart() = 0; // disabled
        *lutAcc.linearExpEnd() = 0; // unused
        uint32_t* pIndexOffset = reinterpret_cast<uint32_t*>(&indexOffset);
        *lutAcc.linearOnlyStart() = static_cast<uint64_t>(*pIndexOffset); // index_offset
        *lutAcc.linearOnlyEnd() = 0; // unused

        // Program LE/LO slopes, 0.0f
        DLASlopeAccessor leuSlopeAcc = lutAcc.linearExpUnderflowSlopeAccessor();
        *leuSlopeAcc.dataF() = 0x0000;
        DLASlopeAccessor leoSlopeAcc = lutAcc.linearExpOverflowSlopeAccessor();
        *leoSlopeAcc.dataF() = 0x0000;
        DLASlopeAccessor louSlopeAcc = lutAcc.linearOnlyUnderflowSlopeAccessor();
        *louSlopeAcc.dataF() = 0x0000;
        DLASlopeAccessor looSlopeAcc = lutAcc.linearOnlyOverflowSlopeAccessor();
        *looSlopeAcc.dataF() = 0x0000;

        // Program priority bits
        *lutAcc.hybridPriority() = lutAcc.priority_LinearOnly();
        *lutAcc.underflowPriority() = lutAcc.priority_LinearOnly();
        *lutAcc.overflowPriority() = lutAcc.priority_LinearOnly();
    }
    else if (lutParams->precision == surface::NVDLA_PRECISION_INT8)
    {
        REPORT_ERROR(NvDlaError_BadValue, "Empty INT8 Tanh LUT");

        // TANH, INT8, N=any, alpha=any, beta=any, k=any
        // mode=EXP, start_index=0
        NvU32 startIndex = 0;

        // Program LE Lut
        for (size_t ii=0; ii<lutAcc.numLinearExpTable(); ii++)
        {
            NvS16* expTable = lutAcc.linearExpTable(ii);
            *expTable = 0x0000;
        }

        // Program LO Lut
        for (size_t ii=0; ii<lutAcc.numLinearOnlyTable(); ii++)
        {
            NvS16* linearTable = lutAcc.linearOnlyTable(ii);
            *linearTable = 0x0000;
        }

        // Program the LE method
        *lutAcc.method() = lutAcc.method_Exponential();

        // Program LE/LO offsets
        DLALUTOffsetAccessor lutLEOffsetAcc = lutAcc.linearExpOffsetAccessor();
        *lutLEOffsetAcc.expOffset() = startIndex; // (2^start_index)
        DLALUTOffsetAccessor lutLOOffsetAcc = lutAcc.linearOnlyOffsetAccessor();
        *lutLOOffsetAcc.fracBits() = -128; // disabled

        // Program LE/LO offsets, valid only for LINEAR method
        *lutAcc.linearExpStart() = 0;
        *lutAcc.linearExpEnd() = 0;
        *lutAcc.linearOnlyStart() = 1;
        *lutAcc.linearOnlyEnd() = 1;

        // Program LE/LO slopes, {0 >> 0}
        DLAFloatDataAccessor leuSlopeAcc = lutAcc.linearExpUnderflowSlopeAccessor().dataIAccessor();
        *leuSlopeAcc.scale() = 0x0000;
        *leuSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor leoSlopeAcc = lutAcc.linearExpOverflowSlopeAccessor().dataIAccessor();
        *leoSlopeAcc.scale() = 0x0000;
        *leoSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor louSlopeAcc = lutAcc.linearOnlyUnderflowSlopeAccessor().dataIAccessor();
        *louSlopeAcc.scale() = 0x0000;
        *louSlopeAcc.shifter() = 0x00;
        DLAFloatDataAccessor looSlopeAcc = lutAcc.linearOnlyOverflowSlopeAccessor().dataIAccessor();
        *looSlopeAcc.scale() = 0x0000;
        *looSlopeAcc.shifter() = 0x00;

        // Program priority bits
        *lutAcc.hybridPriority() = lutAcc.priority_LinearExp();
        *lutAcc.underflowPriority() = lutAcc.priority_LinearExp();
        *lutAcc.overflowPriority() = lutAcc.priority_LinearExp();
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Unsupported Tanh Precision: %d", lutParams->precision);
    }

fail:
    return e;
}

NvDlaError LutManager::genKey(const LutParams* params, std::string* key) const
{
    NvDlaError e = NvDlaSuccess;

    std::string type;
    std::string precision;
    std::string extra;

    if (!params || !key)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    switch(params->type)
    {
    case LutManager::LUT_TYPE_LRN:
        type += "LRN";
        break;
    case LutManager::LUT_TYPE_SIGMOID:
        type += "SIGMOID";
        break;
    case LutManager::LUT_TYPE_TANH:
        type += "TANH";
        break;
    default:
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown lut param type %d", params->type);
    }

    switch(params->precision)
    {
    case surface::NVDLA_PRECISION_INT8:
        precision += "INT8";
        break;
    case surface::NVDLA_PRECISION_INT16:
        precision += "INT16";
        break;
    case surface::NVDLA_PRECISION_FP16:
        precision += "FP16";
        break;
    default:
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown precision type %d", params->precision);
    }

    if (params->type == LutManager::LUT_TYPE_LRN)
    {
        extra += toString(params->lrnParams.localSize) + "-";
        extra += toString(params->lrnParams.alpha) + "-";
        extra += toString(params->lrnParams.beta) + "-";
        extra += toString(params->lrnParams.k);
    }

    *key = type;
    *key += precision;
    *key += extra;

fail:
    return e;
}

NvDlaError LutManager::getHandle(const LutParams* lutParams, LutHandle* hLut)
{
    NvDlaError e = NvDlaSuccess;
    std::string key;
    std::map<std::string, LutHandle>::iterator it;

    PROPAGATE_ERROR_FAIL(genKey(lutParams, &key));

    // Search for key, create and store if no hits available
    it = m_lutHandles.find(key);
    if (it != m_lutHandles.end())
    {
        *hLut = it->second;
    } else {
        // Store the next free lut handle
        LutHandle newHandle = m_hNextFree;

        m_lutHandles[key] = newHandle;
        m_lutParams[newHandle] = *lutParams;

        *hLut = newHandle;

        // Increment next free lut handle
        m_hNextFree += 1;
    }

fail:
    return e;
}

};  //nvdla::priv
};  //nvdla::
