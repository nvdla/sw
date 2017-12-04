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

#define _GNU_SOURCE

#include <dlaerror.h>
#include <dlatypes.h>

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <drm.h>
#include <drm_mode.h>

#include "nvdla.h"
#include "nvdla_inf.h"
#include "nvdla_ioctl.h"
#include "nvdla_os_inf.h"

#define NVDLA_DEVICE_NODE "/dev/dri/renderD128"

#define NVDLA_MEM_READ (PROT_READ)
#define NVDLA_MEM_WRITE (PROT_WRITE)

static int nvdla_mem_map(void **pVirtAddr, int size, NvS64 offset, int fd, NvU32 flags)
{
    void *ptr;

    ptr = mmap(0, size, flags, MAP_SHARED, fd, offset);
    if (ptr == MAP_FAILED) {
        printf("Failed to map memory errno=%d\n", errno);
        return -1;
    }

    *pVirtAddr = ptr;

    return 0;
}

NvDlaError
NvDlaAllocMem(void *session_handle, void *device_handle, void **mem_handle,
                void **pData, NvU32 size, NvDlaHeap heap)
{
    int err = 0;
    NvDlaMemHandle hMem;
    struct drm_prime_handle req;
    struct nvdla_gem_create_args create_args;
    struct nvdla_gem_map_offset_args map_args;
    NvDlaDeviceHandle hDlaDev = (NvDlaDeviceHandle)device_handle;

    hMem = (NvDlaMemHandle)malloc(sizeof(struct NvDlaMemHandleRec));
    *mem_handle = hMem;

    memset(*mem_handle, 0, sizeof(struct NvDlaMemHandleRec));

    memset(&create_args, 0, sizeof(create_args));

    create_args.size = size;

    err = ioctl(hDlaDev->fd, DRM_IOCTL_NVDLA_GEM_CREATE, &create_args);
    if (err) {
        printf("Failed to allocate handle err=%d errno=%d\n", err, errno);
        err = -errno;
        goto free_mem_handle;
    }

    hMem->prime_handle = create_args.handle;

    req.handle = create_args.handle;
    req.flags = DRM_CLOEXEC;

    err = ioctl(hDlaDev->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &req);
    if (err) {
        printf("failed to get fd for handle errno=%d\n", errno);
        err = -errno;
        goto free_gem_handle;
    }

    hMem->fd = req.fd;

    memset(&map_args, 0, sizeof(map_args));

    map_args.handle = create_args.handle;

    err = ioctl(hDlaDev->fd, DRM_IOCTL_NVDLA_GEM_MMAP, &map_args);
    if (err) {
        err = -errno;
        goto free_mem_handle;
    }

    err = nvdla_mem_map(pData, size, map_args.offset, hDlaDev->fd, NVDLA_MEM_WRITE | NVDLA_MEM_READ);
    if (err) {
        goto free_mem_handle;
        return err;
    }

    return 0;

free_gem_handle:
    NvDlaFreeMem(session_handle, device_handle, *mem_handle, *pData, size);
free_mem_handle:
    free(hMem);
    *mem_handle = NULL;
    return err;
}

NvDlaError
NvDlaFreeMem(void *session_handle, void *device_handle, void *mem_handle, void *pData, NvU32 size)
{
    int err;
    struct nvdla_gem_destroy_args args;
    NvDlaMemHandle hMem = (NvDlaMemHandle)mem_handle;
    NvDlaDeviceHandle hDlaDev = (NvDlaDeviceHandle)device_handle;

    args.handle = hMem->prime_handle;

    err = ioctl(hDlaDev->fd, DRM_IOCTL_NVDLA_GEM_DESTROY, &args);
    if (err) {
        printf("Failed to destroy handle err=%d errno=%d\n", err, errno);
        return NvDlaError_IoctlFailed;
    }

    free(hMem);

    return NvDlaSuccess;
}

NvDlaError
NvDlaSubmit(void *session_handle, void *device_handle, NvDlaTask *pTasks, NvU32 num_tasks)
{
    NvDlaDeviceHandle dla_device = (NvDlaDeviceHandle)device_handle;
    struct nvdla_mem_handle address_list[num_tasks][NVDLA_MAX_BUFFERS_PER_TASK];
    struct nvdla_ioctl_submit_task tasks[num_tasks];
    struct nvdla_submit_args args;
    uint32_t i;

    memset(&args, 0, sizeof(args));
    args.tasks = (uintptr_t)tasks;
    args.num_tasks = num_tasks;

    for (i = 0; i < num_tasks; i++) {
        uint32_t num_addresses = tasks[i].num_addresses =
                            pTasks[i].num_addresses;
        uint32_t j;

        tasks[i].address_list = (uintptr_t)address_list[i];
        for (j = 0; j < num_addresses; j++) {
            NvDlaMemHandle mem_handle = (NvDlaMemHandle)pTasks[i].address_list[j].handle;

            address_list[i][j].handle = (uint32_t)mem_handle->fd;
            address_list[i][j].offset = pTasks[i].address_list[j].offset;
        }
    }

    if (ioctl(dla_device->fd, DRM_IOCTL_NVDLA_SUBMIT, &args) < 0) {
        printf("%s: Error IOCTL failed (%s)\n",
                        __func__, strerror(errno));
        return NvDlaError_IoctlFailed;
    }

    return NvDlaSuccess;
}

NvDlaError
NvDlaInitialize(void **session_handle)
{
    if(!session_handle)
        return NvDlaError_BadParameter;
    *session_handle = NULL;

    return NvDlaSuccess;
}

void
NvDlaDestroy(void *session_handle)
{
    (void)session_handle;

    return;
}

NvDlaError
NvDlaOpen(void *session_handle, NvU32 instance, void **device_handle)
{
    NvDlaContext *pContext = NULL;
    NvDlaError e = NvDlaSuccess;

    if (instance > 0)
        return NvDlaError_BadParameter;

    if (!device_handle)
        return NvDlaError_BadParameter;

    pContext = (NvDlaContext*)NvDlaAlloc(sizeof(NvDlaContext));
    if (!pContext) {
        return NvDlaError_InsufficientMemory;
    }

    NvDlaMemset(pContext, 0, sizeof(NvDlaContext));

    pContext->fd = open(NVDLA_DEVICE_NODE, O_RDWR);
    if (pContext->fd < 0) {
        e = NvDlaError_ResourceError;
        goto fail;
    }

    *device_handle = (void *)pContext;

    return NvDlaSuccess;

fail:
    NvDlaFree(pContext);
    return e;
}

void
NvDlaClose(void *hDlaDevice)
{
    NvDlaDeviceHandle device_handle = (NvDlaDeviceHandle)hDlaDevice;

    if (device_handle->fd != -1)
        (void)close(device_handle->fd);
    return;
}
