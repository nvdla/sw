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

#include <stdarg.h>

#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/irq.h>
#include <linux/irqdomain.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/of_irq.h>
#include <linux/of_platform.h>
#include <linux/platform_device.h>
#include <linux/printk.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/time.h>
#include <linux/uaccess.h>

#include <nvdla_interface.h>
#include <nvdla_linux.h>
#include <nvdla_ioctl.h>

void dla_debug(const char *str, ...)
{
	va_list args;
	va_start(args, str);
	vprintk(pr_fmt(str), args);
	va_end(args);
}

void dla_info(const char *str, ...)
{
	va_list args;
	va_start(args, str);
	vprintk(str, args);
	va_end(args);
}

void dla_warn(const char *str, ...)
{
	va_list args;
	va_start(args, str);
	vprintk(str, args);
	va_end(args);
}

void dla_error(const char *str, ...)
{
	va_list args;
	va_start(args, str);
	vprintk(str, args);
	va_end(args);
}

void *dla_memset(void *src, int ch, uint64_t len)
{
	return memset(src, ch, len);
}

void *dla_memcpy(void *dest, const void *src, uint64_t len)
{
	return memcpy(dest, src, len);
}

int64_t dla_get_time_us(void)
{
	return ktime_get_ns() / NSEC_PER_USEC;
}

void dla_reg_write(void *driver_context, uint32_t addr, uint32_t reg)
{
	struct nvdla_device *nvdla_dev =
			(struct nvdla_device *)driver_context;

	if (!nvdla_dev)
		return;

	writel(reg, nvdla_dev->base + addr);
}

uint32_t dla_reg_read(void *driver_context, uint32_t addr)
{
	struct nvdla_device *nvdla_dev =
			(struct nvdla_device *)driver_context;

	if (!nvdla_dev)
		return 0;

	return readl(nvdla_dev->base + addr);
}

static irqreturn_t nvdla_engine_isr(int32_t irq, void *data)
{
	unsigned long flags;
	struct nvdla_device *nvdla_dev = (struct nvdla_device *)data;

	if (!nvdla_dev)
		return IRQ_NONE;

	spin_lock_irqsave(&nvdla_dev->nvdla_lock, flags);
	dla_isr_handler(nvdla_dev->engine_context);
	complete(&nvdla_dev->event_notifier);
	spin_unlock_irqrestore(&nvdla_dev->nvdla_lock, flags);

	return IRQ_HANDLED;
}

static int32_t dla_read_dma_address(void *driver_context, void *task_data,
						int16_t index, void *dst)
{
	int32_t ret = 0;
	struct nvdla_mem_handle *handles;
	dma_addr_t *phys_addr = (dma_addr_t *)(dst);
	struct nvdla_device *nvdla_dev =
			(struct nvdla_device *)driver_context;
	struct nvdla_task *task = (struct nvdla_task *)task_data;

	if (index == -1 || index > task->num_addresses)
		return -EINVAL;

	handles = (struct nvdla_mem_handle *)task->address_list;
	ret = nvdla_gem_dma_addr(nvdla_dev->drm, task->file,
					handles[index].handle,
					phys_addr);

	return ret;
}

static int32_t dla_read_cpu_address(void *driver_context, void *task_data,
						int16_t index, void *dst)
{
	uint64_t *temp = (uint64_t *)dst;
	struct nvdla_task *task = (struct nvdla_task *)task_data;

	if (index == -1 || index > task->num_addresses)
		return -EINVAL;

	*temp = (uint64_t)index;
	return 0;
}

int32_t dla_get_dma_address(void *driver_context, void *task_data,
					int16_t index, void *dst_ptr,
					uint32_t destination)
{
	int32_t ret = 0;

	if (destination == DESTINATION_PROCESSOR) {
		ret = dla_read_cpu_address(driver_context, task_data,
						index, dst_ptr);
	} else if (destination == DESTINATION_DMA) {
		ret = dla_read_dma_address(driver_context, task_data,
						index, dst_ptr);
	} else {
		ret = -EINVAL;
	}

	return ret;
}

int32_t dla_data_write(void *driver_context, void *task_data,
				void *src, uint64_t dst,
				uint32_t size, uint64_t offset)
{
	int32_t ret;
	void *ptr = NULL;
	struct dma_buf *buf;
	struct nvdla_mem_handle *handles;
	struct nvdla_task *task = (struct nvdla_task *)task_data;

	handles = task->address_list;
	buf = dma_buf_get(handles[dst].handle);
	if (IS_ERR(buf)) {
		pr_err("%s: Failed get dma_buf for handle=%d\n", __func__,
						handles[dst].handle);
		return -EFAULT;
	}

	ret = dma_buf_begin_cpu_access(buf, DMA_BIDIRECTIONAL);
	if (ret)
		goto put_dma_buf;

	ptr = dma_buf_vmap(buf);
	if (!ptr) {
		pr_err("%s: Failed to vmap dma_buf for handle=%d\n", __func__,
						handles[dst].handle);
		ret = -ENOMEM;
		goto end_cpu_access;
	}


	memcpy((void *)((uint8_t *)ptr + offset), src, size);

	dma_buf_vunmap(buf, ptr);

end_cpu_access:
	dma_buf_end_cpu_access(buf, DMA_BIDIRECTIONAL);

put_dma_buf:
	dma_buf_put(buf);

	return ret;
}

int32_t dla_data_read(void *driver_context, void *task_data,
				uint64_t src, void *dst,
				uint32_t size, uint64_t offset)
{
	int32_t ret;
	void *ptr = NULL;
	struct dma_buf *buf;
	struct nvdla_mem_handle *handles;
	struct nvdla_task *task = (struct nvdla_task *)task_data;

	handles = task->address_list;

	buf = dma_buf_get(handles[src].handle);
	if (IS_ERR(buf)) {
		pr_err("%s: Failed get dma_buf for handle=%d\n", __func__,
						handles[src].handle);
		return -EFAULT;
	}

	ret = dma_buf_begin_cpu_access(buf, DMA_BIDIRECTIONAL);
	if (ret)
		goto put_dma_buf;

	ptr = dma_buf_vmap(buf);
	if (!ptr) {
		pr_err("%s: Failed to vmap dma_buf for handle=%d\n", __func__,
						handles[src].handle);
		ret = -ENOMEM;
		goto end_cpu_access;
	}

	memcpy(dst, (void *)(((uint8_t *)ptr) + offset), size);

	dma_buf_vunmap(buf, ptr);

end_cpu_access:
	dma_buf_end_cpu_access(buf, DMA_BIDIRECTIONAL);

put_dma_buf:
	dma_buf_put(buf);

	return ret;
}

int32_t nvdla_task_submit(struct nvdla_device *nvdla_dev, struct nvdla_task *task)
{
	int32_t err = 0;
	uint32_t task_complete = 0;

	nvdla_dev->task = task;

	err = dla_execute_task(nvdla_dev->engine_context, (void *)task);
	if (err) {
		pr_err("Task execution failed\n");
		return err;
	}

	pr_debug("Wait for task complete\n");

	while (1) {
		unsigned long flags;

		wait_for_completion(&nvdla_dev->event_notifier);

		spin_lock_irqsave(&nvdla_dev->nvdla_lock, flags);

		err = dla_process_events(nvdla_dev->engine_context, &task_complete);

		spin_unlock_irqrestore(&nvdla_dev->nvdla_lock, flags);

		if (err || task_complete)
			break;
	}

	pr_debug("Task complete\n");
	dla_clear_task(nvdla_dev->engine_context);

	return err;
}

/* driver probe and init */
static const struct of_device_id nvdla_of_match[] = {
	{ .name = "nvdla", .compatible = "nvidia,nvdla_os_initial", },
	{ },
};

static int32_t nvdla_probe(struct platform_device *pdev)
{
	int32_t err = 0;
	struct resource *res;
	struct nvdla_device *nvdla_dev;
	struct device *dev = &pdev->dev;

	nvdla_dev = devm_kzalloc(dev, sizeof(*nvdla_dev), GFP_KERNEL);
	if (!nvdla_dev)
		return -ENOMEM;

	platform_set_drvdata(pdev, nvdla_dev);
	nvdla_dev->pdev = pdev;

	init_completion(&nvdla_dev->event_notifier);

	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	nvdla_dev->base = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(nvdla_dev->base))
		return PTR_ERR(nvdla_dev->base);

	res = platform_get_resource(pdev, IORESOURCE_IRQ, 0);
	if (!res) {
		dev_err(&pdev->dev, "no irq resource\n");
		return -EINVAL;
	}
	nvdla_dev->irq = res->start;

	err = devm_request_irq(&pdev->dev, nvdla_dev->irq,
				nvdla_engine_isr, 0,
				dev_name(&pdev->dev), nvdla_dev);
	if (err)
		return err;

	dla_register_driver(&nvdla_dev->engine_context, (void *)nvdla_dev);
	dla_clear_task(nvdla_dev->engine_context);

	err = nvdla_drm_probe(nvdla_dev);
	if (err)
		dev_err(&pdev->dev, "failed to register drm device\n");

	return err;
}

static int32_t __exit nvdla_remove(struct platform_device *pdev)
{
	struct nvdla_device *nvdla_dev = dev_get_drvdata(&pdev->dev);

	nvdla_drm_remove(nvdla_dev);

	return 0;
}

static struct platform_driver nvdla_driver = {
	.probe = nvdla_probe,
	.remove = __exit_p(nvdla_remove),
	.driver = {
		.owner = THIS_MODULE,
		.name = "NVDLA",
		.of_match_table = nvdla_of_match,
	},
};
module_platform_driver(nvdla_driver);

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR("NVIDIA");
MODULE_DESCRIPTION("Nvidia Deep Learning Accelerator driver");
