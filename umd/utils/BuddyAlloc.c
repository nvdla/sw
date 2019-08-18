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
#include <stdint.h>
#include <stdbool.h>

// File includes
#include <stdlib.h> // malloc

#include "dlaerror.h"
#include "dlatypes.h"
#include "nvdla_os_inf.h" // NvDlaDebugPrintf
#include "BuddyAlloc.h"
#include "ErrorMacros.h"

#define GENERIC_MAX(x, y) ((x) > (y) ? (x) : (y))

#if 0
#define LOGD(...) \
    NvDlaDebugPrintf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

#if 0
#define LOGV(...) \
    NvDlaDebugPrintf(__VA_ARGS__)
#else
#define LOGV(...)
#endif


static inline NvU32 log2i(const NvU32 x)
{
#if defined(__i386__) || defined(__x86_64__)
    NvU32 y;
    __asm__ ( "\tbsr %1, %0\n"
        : "=r"(y)
        : "r" (x)
    );
    return y;
#else
    NvU32 v = x;  // 32-bit value to find the log2 of
    const NvU32 b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    const NvU32 S[] = {1, 2, 4, 8, 16};
    NvS32 i;

    register NvU32 r = 0; // result of log2(v) will go here
    for (i = 4; i >= 0; i--) // unroll for speed...
    {
      if (v & b[i])
      {
        v >>= S[i];
        r |= S[i];
      }
    }

    return r;
#endif
}

static inline bool isPowerOfTwo(const NvU64 x)
{
    return (x && !(x & (x - 1)));
}

static inline NvU8 getNumLevels(const NvDlaBuddyAllocInst* self)
{
    return self->maxElementSizeLog2 - self->minElementSizeLog2 + 1;
}

static inline NvU32 getBlockBytesFromLevel(const NvDlaBuddyAllocInst* self, NvU8 level)
{
    return 1 << (self->maxElementSizeLog2 - level);
}

static inline NvU32 getIndexFromAddrLevel(const NvDlaBuddyAllocInst* self, NvU32 addr, NvU8 level)
{
    return addr >> (self->maxElementSizeLog2 - level);
}

static inline NvU32 getAddrFromLevelIndex(const NvDlaBuddyAllocInst* self, NvU8 level, NvU32 index)
{
    NvU32 blockBytes = getBlockBytesFromLevel(self, level);

    return blockBytes * index;
}

static inline NvU8 getBlockLevelFromNumBytesLog2(const NvDlaBuddyAllocInst* self, NvU8 numBytesLog2)
{
    return self->maxElementSizeLog2 - numBytesLog2;
}

static NvU8 getBlockLevelFromAddr(const NvDlaBuddyAllocInst* self, NvUPtr addr)
{
    // Transform address to binary tree representation
    NvU8 level = getNumLevels(self) - 1;

    while (level > 0)
    {
        NvU32 index = getIndexFromAddrLevel(self, addr, level - 1);

        if (NvDlaBitBinaryTree.get(self->splitData, level - 1, index))
        {
            return level;
        }

        level = level - 1;
    }

    return 0;
}

static void printFreeList(const NvDlaBuddyAllocInst* self)
{
    for (NvU32 ii=0; ii<getNumLevels(self); ii++)
    {
        NvDlaFreeBlock* tmp = self->freeHead[ii];
        LOGV("freeHead[%u] = ", ii);

        if (tmp)
        {
            if (tmp->prev)
            {
                LOGV("[0x%x] ", (NvUPtr)tmp->prev);

            } else {
                LOGV("[NULL] ");
            }
        }

        while (tmp)
        {
            LOGV("0x%x ", (NvUPtr)tmp);
            tmp = tmp->next;
        }

        LOGV("NULL\n");
    }
}

static NvDlaError construct(NvDlaBuddyAllocInst* self,
                    const void* poolData, NvU32 poolSize,
                    NvU8 minElementSizeLog2)
{
    LOGD("NvDlaBuddyAlloc::construct called\n");
    NVDLA_UNUSED(&printFreeList);

    if (!self || !poolData ||
        !isPowerOfTwo(poolSize) ||
        poolSize < ((NvU64)(1LLU<<minElementSizeLog2)) ||
        minElementSizeLog2 < log2i(sizeof(NvDlaFreeBlock)))
    {
        ORIGINATE_ERROR(NvDlaError_BadParameter);
    }

    NvU8 poolSizeLog2 = log2i(poolSize);

    LOGV("NvDlaBuddyAlloc::poolSize %uB\n", poolSize);
    LOGV("NvDlaBuddyAlloc::poolSizeLog2 %u\n", poolSizeLog2);

    self->poolData = poolData;
    self->poolSize = poolSize;
    self->maxElementSizeLog2 = poolSizeLog2;
    self->minElementSizeLog2 = minElementSizeLog2;

    // Initialize buddy list
    self->freeHead[0] = (NvDlaFreeBlock*)poolData;
    self->freeHead[0]->prev = NULL;
    self->freeHead[0]->next = NULL;

    for (NvU8 ii = 1; ii < NVDLA_UTILS_BUDDY_ALLOC_MAX_NUM_BLOCKTYPE; ii++)
    {
        self->freeHead[ii] = NULL;
    }

    self->splitData = NULL;
    self->fxfData = NULL;

    NvU8 numLevels = getNumLevels(self);

    if (numLevels > 1)
    {
        self->splitData = (NvDlaBitBinaryTreeInst*)malloc(sizeof(*self->splitData));
        NvDlaBitBinaryTree.construct(self->splitData, numLevels - 1);

        self->fxfData = (NvDlaBitBinaryTreeInst*)malloc(sizeof(*self->fxfData));
        NvDlaBitBinaryTree.construct(self->fxfData, numLevels - 1);

        //self->allocData = (NvDlaBitBinaryTree*)malloc(sizeof(NvDlaBitBinaryTree));
        //*self->allocData = NvDlaBitBinaryTreeStatic;
        //self->allocData->construct(self->allocData, numLevels);
    }

    return NvDlaSuccess;
}

static NvDlaError destruct(NvDlaBuddyAllocInst* self)
{
    LOGD("NvDlaBuddyAlloc::destruct called\n");

    if (!self)
    {
        ORIGINATE_ERROR(NvDlaError_BadParameter);
    }

    if (self->fxfData)
    {
        NvDlaBitBinaryTree.destruct(self->fxfData);
        free(self->fxfData);
    }

    if (self->splitData)
    {
        NvDlaBitBinaryTree.destruct(self->splitData);
        free(self->splitData);
    }

    return NvDlaSuccess;
}


static void borrowFromBuddy(NvDlaBuddyAllocInst* self, NvU8 level)
{
    NvU32 blockBytesLevel = getBlockBytesFromLevel(self, level);
    NVDLA_UNUSED(blockBytesLevel);
    LOGV("%uB block trying to borrow from buddy\n", blockBytesLevel);

    if (level == 0)
    {
        LOGV("Already at root buddy\n");
        return;
    }

    NvU32 buddyLevel = level - 1;
    if (self->freeHead[buddyLevel] == NULL)
    {
        // No available buddy, ask buddy to also borrow
        borrowFromBuddy(self, buddyLevel);
    }

    // Check for buddy availability again
    if (self->freeHead[buddyLevel] == NULL)
    {
        // Still no available buddy, give up
        LOGV("No available buddy, giving up\n");
        return;
    }

    NvU32 blockBytesBuddyLevel = getBlockBytesFromLevel(self, buddyLevel);
    NVDLA_UNUSED(blockBytesBuddyLevel);
    LOGV("%uB block borrowing from buddy %uB block\n", blockBytesLevel, blockBytesBuddyLevel);

    // Reference the free blocks
    NvDlaFreeBlock* freeBlock0 = self->freeHead[buddyLevel];
    NvDlaFreeBlock* freeBlock1 = (NvDlaFreeBlock*)(((NvUPtr)freeBlock0) + getBlockBytesFromLevel(self, level));

    // Transform buddy address to binary tree representation
    NvU32 buddyAddr = (NvU32)((NvUPtr)self->freeHead[buddyLevel] - (NvUPtr)self->poolData);
    LOGV("buddyAddr = 0x%x\n", buddyAddr);

    NvU32 buddyIndex = getIndexFromAddrLevel(self, buddyAddr, buddyLevel);

    // Mark buddy block as split
    NvDlaBitBinaryTree.set(self->splitData, buddyLevel, buddyIndex, true);
    //splitData->print();

    // Remove free block from buddy
    if (self->freeHead[buddyLevel]->next)
    {
        self->freeHead[buddyLevel]->next->prev = self->freeHead[buddyLevel]->prev;
    }
    self->freeHead[buddyLevel] = self->freeHead[buddyLevel]->next;

    // Split the block and save both blocks here (the free list should already be empty)
    freeBlock1->prev = freeBlock0;
    freeBlock1->next = NULL;
    freeBlock0->prev = NULL;
    freeBlock0->next = freeBlock1;
    self->freeHead[level] = freeBlock0;

    // Mark buddy as allocated
    if (buddyLevel > 0)
    {
        NvU32 buddyParentLevel = buddyLevel - 1;
        NvU32 buddyParentIndex = getIndexFromAddrLevel(self, buddyAddr, buddyParentLevel);

        NvDlaBitBinaryTree.flip(self->fxfData, buddyParentLevel, buddyParentIndex);
    }

    //printFreeList(self);

    // Update alloc debug info
    //self->allocData->set(self->allocData, buddyLevel, buddyIndex, true);
}

static void* allocate(NvDlaBuddyAllocInst* self, NvU32 size)
{
    LOGV("NvDlaBuddyAlloc::allocate(%uB)\n", size);

    // Round numBytes up to the nearest power-of-two
    NvU8 allocNumBytesLog2 = log2i(size);
    if (!isPowerOfTwo(size))
    {
        allocNumBytesLog2 = allocNumBytesLog2 + 1;
    }

    // Clamp to legal min element value
    allocNumBytesLog2 = GENERIC_MAX(allocNumBytesLog2, self->minElementSizeLog2);

    void* myPointer = NULL;

    // convert allocNumBytesLog2 into FreeHead level
    NvU8 level = getBlockLevelFromNumBytesLog2(self, allocNumBytesLog2);

    // Requested allocation greater than max element value
    if (allocNumBytesLog2 > self->maxElementSizeLog2)
    {
        LOGD("Request size too large\n");
        goto fail;
    }

    if (self->freeHead[level] == NULL)
    {
        borrowFromBuddy(self, level);
    }
    else
    {
        LOGV("Easy Allocation!\n");
        //easyAllocations = easyAllocations + 1;
    }

    // Vanilla allocate attempt
    if (self->freeHead[level] != NULL)
    {
        // We have an available free block
        myPointer = self->freeHead[level];

        if (self->freeHead[level]->next != NULL)
        {
            self->freeHead[level]->next->prev = self->freeHead[level]->prev;
        }

        self->freeHead[level] = self->freeHead[level]->next;

        LOGV("Free block is available\n");
    }
    else
    {
        NvU32 blockBytes = getBlockBytesFromLevel(self, level);
        NVDLA_UNUSED(blockBytes);
        LOGV("No free %uB block available\n", blockBytes);
        goto fail;
    }

    NvUPtr allocAddr = 0;
    if (myPointer != NULL)
    {
        allocAddr = (NvUPtr)myPointer - (NvUPtr)self->poolData;

        // Update fxfData binary tree
        if (level == 0)
        {
            // Root has no buddies to a^b
        }
        else
        {
            NvU32 buddyLevel = level - 1;

            // Transform address to binary tree representation
            NvU32 buddyIndex = getIndexFromAddrLevel(self, allocAddr, buddyLevel);

            // Mark block as allocated, flip freeA ^ freeB bit
            NvDlaBitBinaryTree.flip(self->fxfData, buddyLevel, buddyIndex);
            //printf( "fxfData\n" );
            //self->fxfData->print(self->fxfData);
        }

        // Update alloc debug info
        //self->allocData->set(self->allocData, level, getIndexFromAddrLevel(self, allocAddr, level), true);
        //printf( "allocData\n" );
        //self->allocData->print(self->allocData);
    }
    else
    {
        goto fail;
    }

    LOGD("NvDlaBuddyAlloc::allocate(%uB) => SUCCESS [0x%x:0x%x] (%uB)\n", size, (NvUPtr)myPointer, allocAddr, 1 << allocNumBytesLog2);
    //printFreeList(self);
    return myPointer;


fail:
    LOGD("NvDlaBuddyAlloc::allocate(%uB) => FAILED\n", size);
    return NULL;
}

static NvDlaError reclaimFromPartners(NvDlaBuddyAllocInst* self, NvU32 addr, NvU8 level)
{
    LOGV("reclaimFromPartners 0x%x %u\n", addr, level);

    NvU32 index = getIndexFromAddrLevel(self, addr, level);
    NvU8 partnerLevel = level + 1;
    NvU32 partner0Index = getIndexFromAddrLevel(self, addr, partnerLevel);
    NvU32 partner1Index = 0;

    // Sort partner 0 and 1
    if (partner0Index % 2)
    {
        partner0Index = partner0Index - 1;
    }
    partner1Index = partner0Index + 1;

    LOGV("partnerIndex0 %d\n", partner0Index);
    LOGV("partnerIndex1 %d\n", partner1Index);

    // Clear split status bit
    NvDlaBitBinaryTree.set(self->splitData, level, index, false);

    LOGV("Reclaiming 0x%x for level %d\n", addr, level);
    NvU32 partner0Addr = getAddrFromLevelIndex(self, partnerLevel, partner0Index);
    NvU32 partner1Addr = getAddrFromLevelIndex(self, partnerLevel, partner1Index);

    // Remove partners from their free lists
    NvUPtr partner0Pointer = (NvUPtr)self->poolData + partner0Addr;
    NvUPtr partner1Pointer = (NvUPtr)self->poolData + partner1Addr;
    LOGV("Request to remove partner0 0x%x:0x%x\n", partner0Pointer, partner0Addr);
    LOGV("Request to remove partner1 0x%x:0x%x\n", partner1Pointer, partner1Addr);

    LOGV("Removing partner0...");

    NvDlaFreeBlock* temp = (NvDlaFreeBlock*)partner0Pointer;

    if (self->freeHead[partnerLevel] == temp)
    {
        self->freeHead[partnerLevel] = self->freeHead[partnerLevel]->next;
    }

    // Remove partner0 from the free list
    if (temp->prev != NULL)
    {
        temp->prev->next = temp->next;
    }
    if (temp->next != NULL)
    {
        temp->next->prev = temp->prev;
    }

    LOGV("Done\n");

    LOGV("Removing partner1...");

    temp = (NvDlaFreeBlock*)partner1Pointer;

    if (self->freeHead[partnerLevel] == temp)
    {
        self->freeHead[partnerLevel] = self->freeHead[partnerLevel]->next;
    }

    // Remove partner1 from the free list
    if (temp->prev != NULL)
    {
        temp->prev->next = temp->next;
    }
    if (temp->next != NULL)
    {
        temp->next->prev = temp->prev;
    }

    LOGV("Done\n");

    // Reclaim the block
    NvUPtr myPointer = (NvUPtr)self->poolData + partner0Addr;

    NvDlaFreeBlock* freeBlockPtr = (NvDlaFreeBlock*)myPointer;
    freeBlockPtr->prev = NULL;
    freeBlockPtr->next = self->freeHead[level];
    if (self->freeHead[level] != NULL)
    {
        self->freeHead[level]->prev = freeBlockPtr;
    }
    self->freeHead[level] = freeBlockPtr;

    // Mark self as free
    if (level > 0)
    {
        NvU32 buddyLevel = level - 1;
        NvU32 buddyIndex = getIndexFromAddrLevel(self, addr, buddyLevel);

        NvDlaBitBinaryTree.flip(self->fxfData, buddyLevel, buddyIndex);
    }

    // Update alloc debug info
    //self->allocData->set(self->allocData, level, getIndexFromAddrLevel(self, addr, level), false);

    return NvDlaSuccess;
}

static NvDlaError deallocate(NvDlaBuddyAllocInst* self, void* ptr)
{
    NvDlaError e;

    LOGV("NvDlaBuddyAlloc::deallocate([0x%x", (NvUPtr)ptr);
    if (!self || !ptr)
    {
        LOGV("])\n");
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter);
    }

    NvUPtr deallocAddr = (NvUPtr)ptr - (NvUPtr)self->poolData;

    // Determine the size of the block
    NvU8 level = getBlockLevelFromAddr(self, deallocAddr);

    NvU32 blockBytes = getBlockBytesFromLevel(self, level);
    NVDLA_UNUSED(blockBytes);
    LOGV(":0x%x]) (%uB)\n", deallocAddr, blockBytes);

    // Vanilla deallocate attempt
    NvDlaFreeBlock* freeBlockPtr = (NvDlaFreeBlock*)ptr;
    if (self->freeHead[level] != NULL)
    {
        freeBlockPtr->prev = NULL;
        freeBlockPtr->next = self->freeHead[level];
        self->freeHead[level]->prev = freeBlockPtr;
    } else {
        freeBlockPtr->prev = NULL;
        freeBlockPtr->next = NULL;
    }

    self->freeHead[level] = freeBlockPtr;

    // Update fxfData binary tree
    if (level == 0)
    {
        // Root has no buddies to a^b
    }
    else
    {
        NvU8 buddyLevel = level - 1;

        // Transform address to binary tree representation
        NvU32 buddyIndex = getIndexFromAddrLevel(self, deallocAddr, buddyLevel);

        // Mark block as allocated, flip freeA ^ freeB bit
        NvDlaBitBinaryTree.flip(self->fxfData, buddyLevel, buddyIndex);
        //printf( "fxfData\n" );
        //fxfData->print();
    }

    // Update alloc debug info
    //self->allocData->set(self->allocData, level, getIndexFromAddrLevel(self, deallocAddr, level), false);
    //printf("allocData\n");
    //self->allocData->print(self->allocData);

    NvU8 myLevel = level;

    while (myLevel > 0)
    {
        LOGV("myLevel is %u\n", myLevel);
        // Is our partner block free?
        if (myLevel == 0)
        {
            // We have no partner blocks
            break;
        }
        else
        {
            NvU8 buddyLevel = myLevel - 1;
            NvU32 buddyIndex = getIndexFromAddrLevel(self, deallocAddr, buddyLevel);

            if (NvDlaBitBinaryTree.get(self->fxfData, buddyLevel, buddyIndex))
            {
                // Partner block is not free
                break;
            }
            else
            {
                // Partner block is also free, attempt to merge blocks
                PROPAGATE_ERROR_FAIL(reclaimFromPartners(self, deallocAddr, buddyLevel));

                // Continue to check if our buddy wants to reclaim as well
                myLevel = buddyLevel;

                // Print alloc debug info
                //printf( "allocData\n" );
                //self->allocData->print(self->allocData);
            }
        }
    }

    //printf("allocData\n");
    //self->allocData->print(self->allocData);
    LOGD("NvDlaBuddyAlloc::deallocate([0x%x:0x%x]) => SUCCESS\n", (NvUPtr)ptr, deallocAddr);
    //printFreeList(self);
    return NvDlaSuccess;

fail:
    LOGD("NvDlaBuddyAlloc::deallocate([0x%x:0x%x]) (%uB) => FAILED\n", (NvUPtr)ptr, deallocAddr, blockBytes);
    return e;
}

const NvDlaBuddyAllocClass NvDlaBuddyAlloc =
{
    .construct = construct,
    .destruct = destruct,

    .allocate = allocate,
    .deallocate = deallocate,
};
