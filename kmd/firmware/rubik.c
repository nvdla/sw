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

#include "common.h"
#include "dla_engine_internal.h"
#include "engine_debug.h"

static uint8_t map_rubik_mode[] = {
	FIELD_ENUM(RBK_D_MISC_CFG_0, RUBIK_MODE, CONTRACT),
	FIELD_ENUM(RBK_D_MISC_CFG_0, RUBIK_MODE, SPLIT),
	FIELD_ENUM(RBK_D_MISC_CFG_0, RUBIK_MODE, MERGE),
};

static uint8_t  map_ram_type[] = {
	FIELD_ENUM(RBK_D_DAIN_RAM_TYPE_0, DATAIN_RAM_TYPE, MCIF),
	FIELD_ENUM(RBK_D_DAIN_RAM_TYPE_0, DATAIN_RAM_TYPE, CVIF),
};

static uint8_t  map_precision[] = {
	FIELD_ENUM(RBK_D_MISC_CFG_0, IN_PRECISION, INT8),
	FIELD_ENUM(RBK_D_MISC_CFG_0, IN_PRECISION, INT16),
	FIELD_ENUM(RBK_D_MISC_CFG_0, IN_PRECISION, FP16),
};

static uint8_t map_bpe[] = {
	BPE_PRECISION_INT8,
	BPE_PRECISION_INT16,
	BPE_PRECISION_FP16,
};

#if STAT_ENABLE
void
dla_rubik_stat_data(struct dla_processor *processor,
					struct dla_processor_group *group)
{
	uint64_t end_time = 0;
	struct dla_rubik_stat_desc *rubik_stat;

	rubik_stat = &processor->stat_data_desc->rubik_stat;

	end_time = dla_get_time_us();

	rubik_stat->read_stall = rubik_reg_read(D_PERF_READ_STALL);
	rubik_stat->write_stall = rubik_reg_read(D_PERF_WRITE_STALL);
	rubik_stat->runtime = (uint32_t)(end_time - group->start_time);
}

void
dla_rubik_dump_stat(struct dla_processor *processor)
{
	struct dla_rubik_stat_desc *rubik_stat;

	rubik_stat = &processor->stat_data_desc->rubik_stat;

	dla_debug_rubik_stats(rubik_stat);
}
#endif /* STAT_ENABLE */

void
dla_rubik_set_producer(int32_t group_id, int32_t __unused)
{
	uint32_t reg;

	/**
	 * set producer pointer for all sub-modules
	 */
	reg = group_id << SHIFT(RBK_S_POINTER_0, PRODUCER);
	rubik_reg_write(S_POINTER, reg);
}

int
dla_rubik_enable(struct dla_processor_group *group)
{
	uint32_t reg;
	struct dla_engine *engine = dla_get_engine();

	dla_trace("Enter: %s", __func__);

	if (engine->stat_enable == (uint32_t)1) {
		rubik_reg_write(D_PERF_ENABLE, 1);
		group->start_time = dla_get_time_us();
	}

	/**
	 * enable all sub-modules
	 */
	reg = FIELD_ENUM(RBK_D_OP_ENABLE_0, OP_EN, ENABLE);
	rubik_reg_write(D_OP_ENABLE, reg);

	dla_trace("Exit: %s", __func__);

	RETURN(0);
}

void
dla_rubik_rdma_check(struct dla_processor_group *group)
{
	group->is_rdma_needed = 0;
}

static int32_t
processor_rubik_program(struct dla_processor_group *group)
{
	int32_t ret = 0;
	uint32_t reg, high, low;
	uint64_t input_address = 0;
	uint64_t output_address = 0;
	struct dla_engine *engine = dla_get_engine();
	struct dla_rubik_op_desc *rubik_op;
	struct dla_rubik_surface_desc *rubik_surface;

	dla_trace("Enter: %s", __func__);

	rubik_op = &group->operation_desc->rubik_op;
	rubik_surface = &group->surface_desc->rubik_surface;

	/* Argument check */
	ASSERT_GOTO((rubik_surface->src_data.type != DLA_MEM_HW),
		ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO((rubik_surface->dst_data.type != DLA_MEM_HW),
		ret, ERR(INVALID_INPUT), exit);

	/* get the addresses from task descriptor */
	ret = dla_read_input_address(&rubik_surface->src_data,
						&input_address,
						group->op_desc->index,
						group->roi_index,
						1);
	if (ret)
		goto exit;

	dla_get_dma_cube_address(engine->driver_context,
				engine->task->task_data,
				rubik_surface->dst_data.address,
				rubik_surface->dst_data.offset,
				(void *)&output_address,
				DESTINATION_DMA);

	/* config rubik */
	reg = (((uint32_t)map_rubik_mode[rubik_op->mode]) <<
			SHIFT(RBK_D_MISC_CFG_0, RUBIK_MODE)) |
			(((uint32_t)map_precision[rubik_op->precision]) <<
			SHIFT(RBK_D_MISC_CFG_0, IN_PRECISION));
	rubik_reg_write(D_MISC_CFG, reg);
	reg = (((uint32_t)map_ram_type[rubik_surface->src_data.type]) <<
			SHIFT(RBK_D_DAIN_RAM_TYPE_0, DATAIN_RAM_TYPE));
	rubik_reg_write(D_DAIN_RAM_TYPE, reg);
	reg =  ((rubik_surface->src_data.width-1) <<
			SHIFT(RBK_D_DATAIN_SIZE_0_0, DATAIN_WIDTH)) |
			((rubik_surface->src_data.height-1) <<
			SHIFT(RBK_D_DATAIN_SIZE_0_0, DATAIN_HEIGHT));
	rubik_reg_write(D_DATAIN_SIZE_0, reg);
	reg =  ((rubik_surface->src_data.channel-1) <<
			SHIFT(RBK_D_DATAIN_SIZE_1_0, DATAIN_CHANNEL));
	rubik_reg_write(D_DATAIN_SIZE_1, reg);

	high = HIGH32BITS(input_address);
	low = LOW32BITS(input_address);
	rubik_reg_write(D_DAIN_ADDR_LOW, low);
	rubik_reg_write(D_DAIN_ADDR_HIGH, high);
	if (rubik_op->mode == RUBIK_MODE_MERGE) {
		ASSERT_GOTO((rubik_surface->src_data.plane_stride != 0),
			ret, ERR(INVALID_INPUT), exit);
		ASSERT_GOTO(((rubik_surface->src_data.plane_stride&0x1F) == 0),
			ret, ERR(INVALID_INPUT), exit);
		rubik_reg_write(D_DAIN_PLANAR_STRIDE,
			rubik_surface->src_data.plane_stride);
	} else {
		rubik_reg_write(D_DAIN_SURF_STRIDE,
			rubik_surface->src_data.surf_stride);
	}
	rubik_reg_write(D_DAIN_LINE_STRIDE,
				rubik_surface->src_data.line_stride);

	reg = (((uint32_t)map_ram_type[rubik_surface->dst_data.type]) <<
			SHIFT(RBK_D_DAOUT_RAM_TYPE_0, DATAOUT_RAM_TYPE));
	rubik_reg_write(D_DAOUT_RAM_TYPE, reg);
	reg =  ((rubik_surface->dst_data.channel-1) <<
			SHIFT(RBK_D_DATAOUT_SIZE_1_0, DATAOUT_CHANNEL));
	rubik_reg_write(D_DATAOUT_SIZE_1, reg);

	high = HIGH32BITS(output_address);
	low = LOW32BITS(output_address);
	rubik_reg_write(D_DAOUT_ADDR_LOW, low);
	rubik_reg_write(D_DAOUT_ADDR_HIGH, high);

	rubik_reg_write(D_DAOUT_LINE_STRIDE,
			rubik_surface->dst_data.line_stride);
	if (rubik_op->mode != RUBIK_MODE_SPLIT) {
		rubik_reg_write(D_DAOUT_SURF_STRIDE,
				rubik_surface->dst_data.surf_stride);
		if (rubik_op->mode == RUBIK_MODE_CONTRACT) {
			reg = ((rubik_surface->dst_data.channel *
				map_bpe[rubik_op->precision] + 31) >> 5) *
				rubik_surface->src_data.surf_stride;
			rubik_reg_write(D_CONTRACT_STRIDE_0, reg);

			reg = rubik_op->stride_y *
				rubik_surface->dst_data.line_stride;
			rubik_reg_write(D_CONTRACT_STRIDE_1, reg);

			reg = (((uint32_t)(rubik_op->stride_x-1)) <<
			SHIFT(RBK_D_DECONV_STRIDE_0, DECONV_X_STRIDE)) |
				(((uint32_t)(rubik_op->stride_y-1)) <<
			SHIFT(RBK_D_DECONV_STRIDE_0, DECONV_Y_STRIDE));
			rubik_reg_write(D_DECONV_STRIDE, reg);
		}
	} else {
		rubik_reg_write(D_DAOUT_PLANAR_STRIDE,
				rubik_surface->dst_data.plane_stride);
	}

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

int
dla_rubik_is_ready(struct dla_processor *processor,
			     struct dla_processor_group *group)
{
	return 1;
}

void
dla_rubik_dump_config(struct dla_processor_group *group)
{
	struct dla_rubik_op_desc *rubik_op;
	struct dla_rubik_surface_desc *rubik_surface;

	rubik_surface = &group->surface_desc->rubik_surface;
	rubik_op = &group->operation_desc->rubik_op;

	dla_debug_rubik_surface_desc(rubik_surface, group->roi_index);
	dla_debug_rubik_op_desc(rubik_op, group->roi_index);
}

int
dla_rubik_program(struct dla_processor_group *group)
{
	int32_t ret = 0;
	struct dla_engine *engine = dla_get_engine();

	dla_trace("Enter: %s", __func__);

	if (!engine->config_data->rubik_enable) {
		dla_error("RUBIK is not supported for this configuration\n");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	dla_enable_intr(MASK(GLB_S_INTR_MASK_0, RUBIK_DONE_MASK1) |
			MASK(GLB_S_INTR_MASK_0, RUBIK_DONE_MASK0));

	ret = processor_rubik_program(group);
	if (ret)
		goto exit;

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}
