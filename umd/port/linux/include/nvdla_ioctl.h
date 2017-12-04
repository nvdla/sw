/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation; or, when distributed
 * separately from the Linux kernel or incorporated into other
 * software packages, subject to the following license:
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

#ifndef __LINUX_NVDLA_IOCTL_H
#define __LINUX_NVDLA_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#if !defined(__KERNEL__)
#define __user
#endif

/**
 * struct nvdla_mem_handle structure for memory handles
 *
 * @handle      handle to DMA buffer allocated in userspace
 * @reserved        Reserved for padding
 * @offset      offset in bytes from start address of buffer
 *
 */
struct nvdla_mem_handle {
    __u32 handle;
    __u32 reserved;
    __u64 offset;
};

/**
 * struct nvdla_ioctl_submit_task structure for single task information
 *
 * @num_addresses       total number of entries in address_list
 * @reserved            Reserved for padding
 * @address_list        pointer to array of struct nvdla_mem_handle
 *
 */
struct nvdla_ioctl_submit_task {
#define NVDLA_MAX_BUFFERS_PER_TASK (512)
    __u32 num_addresses;
#define NVDLA_NO_TIMEOUT    (0xffffffff)
    __u32 timeout;
    __u64 address_list;
};

/**
 * struct nvdla_submit_args structure for task submit
 *
 * @tasks       pointer to array of struct nvdla_ioctl_submit_task
 * @num_tasks       number of entries in tasks
 * @flags       flags for task submit, no flags defined yet
 * @version     version of task structure
 *
 */
struct nvdla_submit_args {
    __u64 tasks;
    __u16 num_tasks;
#define NVDLA_MAX_TASKS_PER_SUBMIT  24
#define NVDLA_SUBMIT_FLAGS_ATOMIC   (1 << 0)
    __u16 flags;
    __u32 version;
};

/**
 * struct nvdla_gem_create_args for allocating DMA buffer through GEM
 *
 * @handle      handle updated by kernel after allocation
 * @flags       implementation specific flags
 * @size        size of buffer to allocate
 */
struct nvdla_gem_create_args {
    __u32 handle;
    __u32 flags;
    __u64 size;
};

/**
 * struct nvdla_gem_map_offset_args for mapping DMA buffer
 *
 * @handle      handle of the buffer
 * @reserved        reserved for padding
 * @offset      offset updated by kernel after mapping
 */
struct nvdla_gem_map_offset_args {
    __u32 handle;
    __u32 reserved;
    __u64 offset;
};

/**
 * struct nvdla_gem_destroy_args for destroying DMA buffer
 *
 * @handle      handle of the buffer
 */
struct nvdla_gem_destroy_args {
    __u32 handle;
};

#define DRM_NVDLA_SUBMIT        0x00
#define DRM_NVDLA_GEM_CREATE        0x01
#define DRM_NVDLA_GEM_MMAP      0x02
#define DRM_NVDLA_GEM_DESTROY       0x03

#define DRM_IOCTL_NVDLA_SUBMIT DRM_IOWR(DRM_COMMAND_BASE + DRM_NVDLA_SUBMIT, struct nvdla_submit_args)
#define DRM_IOCTL_NVDLA_GEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + DRM_NVDLA_GEM_CREATE, struct nvdla_gem_create_args)
#define DRM_IOCTL_NVDLA_GEM_MMAP DRM_IOWR(DRM_COMMAND_BASE + DRM_NVDLA_GEM_MMAP, struct nvdla_gem_map_offset_args)
#define DRM_IOCTL_NVDLA_GEM_DESTROY DRM_IOWR(DRM_COMMAND_BASE + DRM_NVDLA_GEM_DESTROY, struct nvdla_gem_destroy_args)

#endif
