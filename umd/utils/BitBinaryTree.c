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

// Common includes
#include <stdbool.h>
//#include <nvcommon.h>
#include "dlaerror.h"
#include "dlatypes.h"


// File includes
#include <stdlib.h> // malloc
#include <memory.h> // memset

#include "nvdla_os_inf.h"
#include "BitBinaryTree.h"
#include "ErrorMacros.h"

static inline NvU32 pow2i(const NvU32 x)
{
    return (1 << x);
}

static inline NvU32 roundUp(NvU32 numToRound, NvU32 multiple)
{
    if (multiple == 0)
    {
        return 0;
    }

    NvU32 result = numToRound;
    NvU32 remainder = numToRound % multiple;
    if (remainder != 0)
    {
        result += multiple - remainder;
    }

    return result;
}

static NvDlaError construct(NvDlaBitBinaryTreeInst* self, NvU8 numLevels)
{
    //NvDlaDebugPrintf("NvDlaBitBinaryTree::construct called %u levels\n", numLevels);
    if (!self || numLevels == 0 || numLevels > 31)
    {
        ORIGINATE_ERROR(NvDlaError_BadParameter);
    }

    self->numLevels = numLevels;
    NvU32 numElements = 0;

    for (NvU8 ii = 0; ii < numLevels; ii++)
    {
        numElements += pow2i(ii);
    }

    //NvDlaDebugPrintf("BinaryBitTree(%u levels, %u elements) ", numLevels, numElements);

    // Calculate number of bytes required
    NvU32 numBytes = roundUp(numElements, 8) / 8;

    //NvDlaDebugPrintf("=> %uB\n", numBytes);

    self->treeStorage = (NvU8*)malloc(numBytes);
    if (!self->treeStorage)
    {
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);
    }

    memset(self->treeStorage, 0, numBytes);

    return NvDlaSuccess;
}

static NvDlaError destruct(NvDlaBitBinaryTreeInst* self)
{
    //NvDlaDebugPrintf("NvDlaBitBinaryTree::destruct called\n");
    if (!self)
    {
        ORIGINATE_ERROR(NvDlaError_BadParameter);
    }

    free(self->treeStorage);

    return NvDlaSuccess;
}

static bool get(const NvDlaBitBinaryTreeInst* self, NvU8 level, NvU32 index)
{
    //NvDlaDebugPrintf("get called %u level %u index\n", level, index);
    NvU32 bitAddr = (pow2i(level) - 1) + index;

    NvU32 byteAddr = bitAddr / 8;
    NvU8 byteMask = 1 << (bitAddr % 8);

    NvU8 value = self->treeStorage[byteAddr] & byteMask;

    return value > 0 ? true : false;
}

static void set(NvDlaBitBinaryTreeInst* self, NvU8 level, NvU32 index, bool value)
{
    //NvDlaDebugPrintf("set called %u level %u index %u value\n", level, index, value);
    NvU32 bitAddr = (pow2i(level) - 1) + index;

    NvU32 byteAddr = bitAddr / 8;
    NvU8 byteMask = 1 << (bitAddr % 8);

    if (value)
    {
        self->treeStorage[byteAddr] |= byteMask;
    }
    else
    {
        self->treeStorage[byteAddr] &= ~byteMask;
    }
}

static bool flip(NvDlaBitBinaryTreeInst* self, NvU8 level, NvU32 index)
{
    //NvDlaDebugPrintf("flip called %u level %u index\n", level, index);
    NvU32 bitAddr = (pow2i(level) - 1) + index;

    NvU32 byteAddr = bitAddr / 8;
    NvU8 byteMask = 1 << (bitAddr % 8);

    NvU8 value = self->treeStorage[byteAddr] & byteMask;

    if (value > 0)
    {
        self->treeStorage[byteAddr] &= ~byteMask;
    }
    else
    {
        self->treeStorage[byteAddr] |= byteMask;
    }

    return value > 0 ? false : true;
}

static NvDlaError print(const NvDlaBitBinaryTreeInst* self)
{
    if (!self)
    {
        ORIGINATE_ERROR(NvDlaError_BadParameter);
    }

    for (NvU8 level = 0; level < self->numLevels; level++)
    {
        NvU32 numElements = pow2i(level);

        NvDlaDebugPrintf("Level %d ", level);
        NvDlaDebugPrintf("[");
        for (NvU32 ii = 0; ii < numElements; ii++)
        {
            bool value = get(self, level, ii);
            if (value)
            {
                NvDlaDebugPrintf("1");
            }
            else
            {
                NvDlaDebugPrintf("0");
            }

            NvU8 numLevels = self->numLevels;

            NvU32 numSpaces = pow2i(numLevels - level) - 1;
            for (NvU32 jj = 0; jj < numSpaces; jj++)
            {
                NvDlaDebugPrintf(" ");
            }
        }
        NvDlaDebugPrintf("]\n");
    }

    return NvDlaSuccess;
}

const NvDlaBitBinaryTreeClass NvDlaBitBinaryTree =
{
    .construct = construct,
    .destruct = destruct,

    .get = get,
    .set = set,
    .flip = flip,
    .print = print,
};
