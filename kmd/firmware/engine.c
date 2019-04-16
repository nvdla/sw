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

#include <opendla.h>
#include <dla_debug.h>
#include <dla_err.h>
#include <dla_interface.h>

#include "dla_engine_internal.h"
#include "common.h"

static const uint32_t map_rdma_ptr_addr[] = {
	0xFFFFFFFF,
	0xFFFFFFFF,
	SDP_REG(RDMA_S_POINTER),
	PDP_REG(RDMA_S_POINTER),
	CDP_REG(RDMA_S_POINTER),
	0xFFFFFFFF,
};

static const uint32_t map_sts_addr[] = {
	BDMA_REG(STATUS),
	CACC_REG(S_STATUS),
	SDP_REG(S_STATUS),
	PDP_REG(S_STATUS),
	CDP_REG(S_STATUS),
	RBK_REG(S_STATUS),
};

static const uint32_t map_ptr_addr[] = {
	BDMA_REG(STATUS),
	CACC_REG(S_POINTER),
	SDP_REG(S_POINTER),
	PDP_REG(S_POINTER),
	CDP_REG(S_POINTER),
	RBK_REG(S_POINTER),
};

int32_t dla_enable_intr(uint32_t mask)
{
	uint32_t reg = glb_reg_read(S_INTR_MASK);

	reg = reg & (~mask);
	glb_reg_write(S_INTR_MASK, reg);

	RETURN(0);
}

int32_t dla_disable_intr(uint32_t mask)
{
	uint32_t reg = glb_reg_read(S_INTR_MASK);

	reg = reg | mask;
	glb_reg_write(S_INTR_MASK, reg);

	RETURN(0);
}

uint8_t bdma_grp_sts[2] = {
	FIELD_ENUM(BDMA_STATUS_0, IDLE, YES),
	FIELD_ENUM(BDMA_STATUS_0, IDLE, YES)
};

struct dla_roi_desc roi_desc;

/**
 * Get DMA data cube address
 */
int32_t
dla_get_dma_cube_address(void *driver_context, void *task_data,
					int16_t index, uint32_t offset, void *dst_ptr,
					uint32_t destination)
{
	int32_t ret = 0;
	uint64_t *pdst = (uint64_t *)dst_ptr;
       ret = dla_get_dma_address(driver_context, task_data, index,
								dst_ptr, destination);
	if (ret)
		goto exit;

	pdst[0] += offset;

exit:
	return ret;
}

/**
 * Read input buffer address
 *
 * For input layer, in case of static ROI this address is read
 * from address list and index is specified in data cube. In case
 * dynamic ROI, it has to be read depending on ROI information
 * and using surface address
 *
 * For all other layers, this address is read from address list
 * using index specified in data cube
 */
int
dla_read_input_address(struct dla_data_cube *data,
		       uint64_t *address,
		       int16_t op_index,
		       uint8_t roi_index,
		       uint8_t bpp)
{
	uint64_t roi_desc_addr;
	int32_t ret = ERR(INVALID_INPUT);
	struct dla_engine *en = dla_get_engine();

	/**
	 * If memory type is HW then no address required
	 */
	if (data->type == DLA_MEM_HW) {
		ret = 0;
		goto exit;
	}

	/**
	 * If address list index is not -1 means this address has to
	 * be read from address list
	 */
	if (data->address != -1) {

		/**
		 * But if other parameters indicate that this is input layer
		 * for dynamic ROI then it is an error
		 */
		if (en->network->dynamic_roi &&
			en->network->input_layer == op_index)
			goto exit;
		ret = dla_get_dma_cube_address(en->driver_context,
						en->task->task_data,
						data->address,
						data->offset,
						(void *)address,
						DESTINATION_DMA);
		goto exit;
	}

	/**
	 * Check if it is dynamic ROI and this is input layer
	 */
	if (en->network->dynamic_roi && en->network->input_layer == op_index) {
		if (!en->task->surface_addr)
			goto exit;

		/* Calculate address of ROI descriptor in array */
		roi_desc_addr = en->task->roi_array_addr;

		/* Read ROI descriptor */
		ret = dla_data_read(en->driver_context,
				en->task->task_data,
				roi_desc_addr,
				(void *)&roi_desc,
				sizeof(roi_desc),
				sizeof(struct dla_roi_array_desc) +
				roi_index * sizeof(struct dla_roi_desc));
		if (ret)
			goto exit;

		/* Calculate ROI address */
		*address = en->task->surface_addr;
		*address += (roi_desc.top * data->line_stride) +
						(bpp * roi_desc.left);
	}

exit:
	RETURN(ret);
}

int
utils_get_free_group(struct dla_processor *processor,
		     uint8_t *group_id,
		     uint8_t *rdma_id)
{
	int32_t ret = 0;
	uint32_t pointer;
	uint32_t hw_consumer_ptr;
	uint32_t hw_rdma_ptr;

	hw_rdma_ptr = 0;

	if (processor->op_type == DLA_OP_BDMA) {
		pointer = reg_read(map_ptr_addr[processor->op_type]);
		hw_consumer_ptr = ((pointer & MASK(BDMA_STATUS_0, GRP0_BUSY)) >>
				SHIFT(BDMA_STATUS_0, GRP0_BUSY)) ==
				FIELD_ENUM(BDMA_STATUS_0, GRP0_BUSY, YES) ?
				1 : 0;
	} else {
		pointer = reg_read(map_ptr_addr[processor->op_type]);
		hw_consumer_ptr = (pointer & MASK(CDP_S_POINTER_0, CONSUMER)) >>
				SHIFT(CDP_S_POINTER_0, CONSUMER);

		/**
		 * Read current consumer pointer for RDMA only if processor
		 * has RDMA module
		 */
		if (map_rdma_ptr_addr[processor->op_type] != 0xFFFFFFFF) {
			pointer =
			reg_read(map_rdma_ptr_addr[processor->op_type]);
			hw_rdma_ptr = (pointer &
					MASK(CDP_S_POINTER_0, CONSUMER)) >>
					SHIFT(CDP_S_POINTER_0, CONSUMER);
		}
	}

	/**
	 * If both processors are programmed then exit
	 */
	if (processor->group_status == 0x3) {
		ret = ERR(PROCESSOR_BUSY);
		goto exit;
	}

	if (!processor->group_status)
		/**
		 * If both groups are idle then use consumer pointer
		 */
		*group_id = hw_consumer_ptr;
	else
		/**
		 * Here it is assumed that only one group is idle or busy
		 * and hence right shift will work to get correct
		 * group id
		 */
		*group_id = !(processor->group_status >> 1);

	/**
	 * If both groups are idle then read group id from pointer
	 */
	if (!processor->rdma_status)
		*rdma_id = hw_rdma_ptr;
	else
		*rdma_id = !(processor->rdma_status >> 1);

exit:
	RETURN(ret);
}
