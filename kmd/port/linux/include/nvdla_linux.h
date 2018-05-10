/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __LINUX_NVDLA_LINUX_H_
#define __LINUX_NVDLA_LINUX_H_

#include <linux/completion.h>
#include <linux/device.h>
#include <linux/kref.h>
#include <linux/platform_device.h>
#include <linux/spinlock.h>

/**
 * @brief			Task information submitted from user space
 *
 * ref				Reference count for task
 * num_addresses		Number of addresses in address list
 * nvdla_dev			Pointer to NVDLA device
 * address_list			Address list
 * file				DRM file instance
 */
struct nvdla_task {
	struct kref ref;
	uint32_t num_addresses;
	struct nvdla_device *nvdla_dev;
	struct nvdla_mem_handle *address_list;
	struct drm_file *file;
};

/**
 * @brief			Configuration parameters supported by the engine
 *
 * atom_size			Memory smallest access size
 * bdma_enable			Defines whether bdma is supported
 * rubik_enable			Defines whether rubik is supported
 * weight_compress_support	Defines whether weight data compression is supported
 */
struct nvdla_config
{
	uint32_t atom_size;
	bool bdma_enable;
	bool rubik_enable;
	bool weight_compress_support;
};

/**
 * @brief			NVDLA device
 *
 * irq				Interrupt number associated with this device
 * ref				Reference count for device
 * base				IO mapped base address for device
 * nvdla_lock			Spinlock used for synchronization
 * drm				DRM device instance
 * task				Pointer to task in execution
 * config_data			Pointer to the configuration data
 * pdev				Pointer to NVDLA platform device
 * event_notifier		Completion object used to wait for events from HW
 * engine_context		Private data passed from engine in dla_engine_init
 */
struct nvdla_device {
	int32_t irq;
	struct kref ref;
	void __iomem *base;
	spinlock_t nvdla_lock;
	struct drm_device *drm;
	struct nvdla_task *task;
	struct nvdla_config *config_data;
	struct platform_device *pdev;
	struct completion event_notifier;

	void *engine_context;
};

/**
 * @brief			Submit task
 *
 * This function submits task to NVDLA engine.
 *
 * @param nvdla_dev		Pointer to NVDLA device
 * @param task			Pointer to task
 * @return			0 on success and negative on error
 *
 */
int32_t nvdla_task_submit(struct nvdla_device *nvdla_dev, struct nvdla_task *task);

/**
 * @brief			Get DMA address
 *
 * This function gets DMA address for given fd
 *
 * @param dev			DRM device instance
 * @param file			DRM file instance
 * @param fd			File desriptor for DMA buffer
 * @param addr			Pointer to update DMA address
 * @return			0 on success and negative on error
 *
 */
int32_t nvdla_gem_dma_addr(struct drm_device *dev, struct drm_file *file,
					uint32_t fd, dma_addr_t *addr);

/**
 * @brief			DRM probe
 *
 * Probe function for DRM device
 *
 * @param nvdla_dev		NVDLA device pointer
 * @return			0 on success and negative on error
 *
 */
int32_t nvdla_drm_probe(struct nvdla_device *nvdla_dev);

/**
 * @brief			DRM remove
 *
 * Remove function for DRM device
 *
 * @param nvdla_dev		NVDLA device pointer
 *
 */
void nvdla_drm_remove(struct nvdla_device *nvdla_dev);

#endif
