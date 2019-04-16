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

static const uint8_t map_ram[] = {
	FIELD_ENUM(CDP_RDMA_D_SRC_DMA_CFG_0, SRC_RAM_TYPE, MC),
	FIELD_ENUM(CDP_RDMA_D_SRC_DMA_CFG_0, SRC_RAM_TYPE, CV),
};

static const uint8_t map_precision[] = {
	FIELD_ENUM(CDP_RDMA_D_DATA_FORMAT_0, INPUT_DATA, INT8),
	FIELD_ENUM(CDP_RDMA_D_DATA_FORMAT_0, INPUT_DATA, INT16),
	FIELD_ENUM(CDP_RDMA_D_DATA_FORMAT_0, INPUT_DATA, FP16),
};

static const uint8_t map_perf_dma[] = {
	FIELD_ENUM(CDP_D_PERF_ENABLE_0, DMA_EN, DISABLE),
	FIELD_ENUM(CDP_D_PERF_ENABLE_0, DMA_EN, ENABLE),
};

static const uint8_t map_perf_lut[] = {
	FIELD_ENUM(CDP_D_PERF_ENABLE_0, LUT_EN, DISABLE),
	FIELD_ENUM(CDP_D_PERF_ENABLE_0, LUT_EN, ENABLE),
};

#if STAT_ENABLE
void
dla_cdp_stat_data(struct dla_processor *processor,
					struct dla_processor_group *group)
{
	uint64_t end_time = 0;
	struct dla_cdp_stat_desc *cdp_stat;

	cdp_stat = &processor->stat_data_desc->cdp_stat;

	end_time = dla_get_time_us();

	cdp_stat->write_stall = cdp_reg_read(D_PERF_WRITE_STALL);
	cdp_stat->lut_uflow = cdp_reg_read(D_PERF_LUT_UFLOW);
	cdp_stat->lut_oflow = cdp_reg_read(D_PERF_LUT_OFLOW);
	cdp_stat->lut_hybrid = cdp_reg_read(D_PERF_LUT_HYBRID);
	cdp_stat->lut_le_hit = cdp_reg_read(D_PERF_LUT_LE_HIT);
	cdp_stat->lut_lo_hit = cdp_reg_read(D_PERF_LUT_LO_HIT);
	cdp_stat->runtime = (uint32_t)(end_time - group->start_time);
}

void
dla_cdp_dump_stat(struct dla_processor *processor)
{
	struct dla_cdp_stat_desc *cdp_stat;

	cdp_stat = &processor->stat_data_desc->cdp_stat;

	dla_debug_cdp_stats(cdp_stat);
}
#endif /* STAT_ENABLE */

static uint32_t
map_local_size(uint8_t local_size)
{
	return ((local_size-1)/2)-1;
}

void
dla_cdp_set_producer(int32_t group_id, int32_t rdma_group_id)
{
	uint32_t reg;

	/**
	 * set producer pointer for all sub-modules
	 */
	reg = group_id << SHIFT(CDP_S_POINTER_0, PRODUCER);
	cdp_reg_write(S_POINTER, reg);
	reg = group_id << SHIFT(CDP_RDMA_S_POINTER_0, PRODUCER);
	cdp_rdma_reg_write(S_POINTER, reg);
}

int
dla_cdp_enable(struct dla_processor_group *group)
{
	uint32_t reg;
	uint8_t perf_reg;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s\n", __func__);

	if (engine->stat_enable == (uint32_t)1) {
		perf_reg = (map_perf_dma[1] <<
				SHIFT(CDP_D_PERF_ENABLE_0, DMA_EN)) |
			(map_perf_lut[1] <<
				SHIFT(CDP_D_PERF_ENABLE_0, LUT_EN));

		cdp_reg_write(D_PERF_ENABLE, perf_reg);
		group->start_time = dla_get_time_us();
	}

	/**
	 * enable all sub-modules
	 */
	reg = FIELD_ENUM(CDP_RDMA_D_OP_ENABLE_0, OP_EN, ENABLE);
	cdp_rdma_reg_write(D_OP_ENABLE, reg);
	reg = FIELD_ENUM(CDP_D_OP_ENABLE_0, OP_EN, ENABLE);
	cdp_reg_write(D_OP_ENABLE, reg);

	dla_debug("Exit: %s\n", __func__);

	RETURN(0);
}

void
dla_cdp_rdma_check(struct dla_processor_group *group)
{
	group->is_rdma_needed = 1;
}

static int32_t
processor_cdp_program(struct dla_processor_group *group)
{
	int32_t ret = 0;
	uint32_t reg, high, low;
	uint64_t input_address = 0;
	uint64_t output_address = 0;
	struct dla_lut_param lut;
	struct dla_engine *engine = dla_get_engine();
	struct dla_cdp_op_desc *cdp_op;
	struct dla_cdp_surface_desc *cdp_surface;

	dla_debug("Enter: %s\n", __func__);

	cdp_op = &group->operation_desc->cdp_op;
	cdp_surface = &group->surface_desc->cdp_surface;

	/* Argument check */
	if (cdp_surface->src_data.type == DLA_MEM_HW) {
		dla_error("Invalid source memory type\n");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}
	if (cdp_surface->dst_data.type == DLA_MEM_HW) {
		dla_error("Invalid destination memory type\n");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (cdp_op->in_precision != cdp_op->out_precision) {
		dla_error("CDP does not support precision conversion\n");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	/* get the addresses from task descriptor */
	ret = dla_read_input_address(&cdp_surface->src_data,
						&input_address,
						group->op_desc->index,
						group->roi_index,
						1);
	if (ret)
		goto exit;

	dla_get_dma_cube_address(engine->driver_context,
				engine->task->task_data,
				cdp_surface->dst_data.address,
				cdp_surface->dst_data.offset,
				(void *)&output_address,
				DESTINATION_DMA);
	if (cdp_op->lut_index >= 0) {
		group->lut_index = cdp_op->lut_index;
		dla_read_lut(engine, cdp_op->lut_index, (void *)&lut);
		dla_debug_lut_params(&lut);
	}

	/* config CDP RDMA registers */
	reg = ((cdp_surface->src_data.width - 1)
		<< SHIFT(CDP_RDMA_D_DATA_CUBE_WIDTH_0, WIDTH));
	cdp_rdma_reg_write(D_DATA_CUBE_WIDTH, reg);

	reg = ((cdp_surface->src_data.height - 1)
		<< SHIFT(CDP_RDMA_D_DATA_CUBE_HEIGHT_0, HEIGHT));
	cdp_rdma_reg_write(D_DATA_CUBE_HEIGHT, reg);

	reg = ((cdp_surface->src_data.channel - 1)
		<< SHIFT(CDP_RDMA_D_DATA_CUBE_CHANNEL_0, CHANNEL));
	cdp_rdma_reg_write(D_DATA_CUBE_CHANNEL, reg);

	high = HIGH32BITS(input_address);
	low = LOW32BITS(input_address);
	cdp_rdma_reg_write(D_SRC_BASE_ADDR_LOW, low);
	cdp_rdma_reg_write(D_SRC_BASE_ADDR_HIGH, high);

	cdp_rdma_reg_write(D_SRC_LINE_STRIDE,
			cdp_surface->src_data.line_stride);
	cdp_rdma_reg_write(D_SRC_SURFACE_STRIDE,
			cdp_surface->src_data.surf_stride);

	reg = (map_ram[cdp_surface->src_data.type]
		<< SHIFT(CDP_RDMA_D_SRC_DMA_CFG_0, SRC_RAM_TYPE));
	cdp_rdma_reg_write(D_SRC_DMA_CFG, reg);

	reg = (map_precision[cdp_op->in_precision]
		<< SHIFT(CDP_RDMA_D_DATA_FORMAT_0, INPUT_DATA));
	cdp_rdma_reg_write(D_DATA_FORMAT, reg);

	/* config CDP */
	if (cdp_op->lut_index >= 0)
		update_lut(CDP_S_LUT_ACCESS_CFG_0, &lut, cdp_op->in_precision);

	high = HIGH32BITS(output_address);
	low = LOW32BITS(output_address);
	cdp_reg_write(D_DST_BASE_ADDR_LOW, low);
	cdp_reg_write(D_DST_BASE_ADDR_HIGH, high);

	cdp_reg_write(D_DST_LINE_STRIDE, cdp_surface->dst_data.line_stride);
	cdp_reg_write(D_DST_SURFACE_STRIDE, cdp_surface->dst_data.surf_stride);

	reg = (map_ram[cdp_surface->dst_data.type]
		<< SHIFT(CDP_D_DST_DMA_CFG_0, DST_RAM_TYPE));
	cdp_reg_write(D_DST_DMA_CFG, reg);

	reg = (map_precision[cdp_op->in_precision]
		<< SHIFT(CDP_D_DATA_FORMAT_0, INPUT_DATA_TYPE));
	cdp_reg_write(D_DATA_FORMAT, reg);

	reg = (map_local_size(cdp_op->local_size)
		<< SHIFT(CDP_D_LRN_CFG_0, NORMALZ_LEN));
	cdp_reg_write(D_LRN_CFG, reg);

	reg = (cdp_op->in_cvt.offset
		<< SHIFT(CDP_D_DATIN_OFFSET_0, DATIN_OFFSET));
	cdp_reg_write(D_DATIN_OFFSET, reg);

	reg = (cdp_op->in_cvt.scale
		<< SHIFT(CDP_D_DATIN_SCALE_0, DATIN_SCALE));
	cdp_reg_write(D_DATIN_SCALE, reg);

	reg = (cdp_op->in_cvt.truncate
		<< SHIFT(CDP_D_DATIN_SHIFTER_0, DATIN_SHIFTER));
	cdp_reg_write(D_DATIN_SHIFTER, reg);

	reg = (cdp_op->out_cvt.offset
		<< SHIFT(CDP_D_DATOUT_OFFSET_0, DATOUT_OFFSET));
	cdp_reg_write(D_DATOUT_OFFSET, reg);

	reg = (cdp_op->out_cvt.scale
		<< SHIFT(CDP_D_DATOUT_SCALE_0, DATOUT_SCALE));
	cdp_reg_write(D_DATOUT_SCALE, reg);

	reg = (cdp_op->out_cvt.truncate
		<< SHIFT(CDP_D_DATOUT_SHIFTER_0, DATOUT_SHIFTER));
	cdp_reg_write(D_DATOUT_SHIFTER, reg);

	reg = ((cdp_op->bypass_sqsum ?
		FIELD_ENUM(CDP_D_FUNC_BYPASS_0, SQSUM_BYPASS, ENABLE) :
		FIELD_ENUM(CDP_D_FUNC_BYPASS_0, SQSUM_BYPASS, DISABLE)) <<
		SHIFT(CDP_D_FUNC_BYPASS_0, SQSUM_BYPASS)) |
		((cdp_op->bypass_out_mul ?
		FIELD_ENUM(CDP_D_FUNC_BYPASS_0, MUL_BYPASS, ENABLE) :
		FIELD_ENUM(CDP_D_FUNC_BYPASS_0, MUL_BYPASS, DISABLE)) <<
		SHIFT(CDP_D_FUNC_BYPASS_0, MUL_BYPASS));
	cdp_reg_write(D_FUNC_BYPASS, reg);

exit:
	dla_debug("Exit: %s", __func__);
	RETURN(ret);
}

int
dla_cdp_is_ready(struct dla_processor *processor,
		 struct dla_processor_group *group)
{
	struct dla_processor_group *next_group;
	struct dla_cdp_op_desc *cdp_op;

	cdp_op = &group->operation_desc->cdp_op;
	next_group = &processor->groups[!group->id];

	/**
	 * Single LUT is shared between two CDP groups, need to make
	 * sure that usage does not conflict. Also, LUT write
	 * access is locked when CDP sub-engine is active, so delay
	 * writing LUT when another group is active.
	 */

	/**
	 * if no LUT required for current group then it can be programmed
	 * without further checks
	 */
	if (cdp_op->lut_index == -1)
		return 1;

	/**
	 * if same LUT is used for both groups then it can be programmed
	 * without more checks. Even if another group is active and LUT
	 * is locked, it would have been programmed by another group.
	 */
	if (next_group->lut_index == cdp_op->lut_index)
		return 1;

	/**
	 * if LUT index of another group is not -1 means some LUT is programmed,
	 * then do not program current LUT as we already know current LUT is not
	 * -1 and neither same as another group.
	 */
	if (next_group->lut_index != -1)
		return 0;

	/**
	 * if current group needs LUT different than another group and that
	 * group is not active then program it.
	 */
	if (!next_group->active)
		return 1;

	/**
	 * if control is here it means current group is using LUT different than
	 * another group and that group is active. Wait for another group to
	 * become idle.
	 */

	return 0;
}

void
dla_cdp_dump_config(struct dla_processor_group *group)
{
	struct dla_cdp_op_desc *cdp_op;
	struct dla_cdp_surface_desc *cdp_surface;

	cdp_surface = &group->surface_desc->cdp_surface;
	cdp_op = &group->operation_desc->cdp_op;

	dla_debug_cdp_surface_desc(cdp_surface, group->roi_index);
	dla_debug_cdp_op_desc(cdp_op, group->roi_index);
}

int
dla_cdp_program(struct dla_processor_group *group)
{
	int32_t ret;

	dla_debug("Enter: %s", __func__);
	dla_enable_intr(MASK(GLB_S_INTR_MASK_0, CDP_DONE_MASK1) |
			MASK(GLB_S_INTR_MASK_0, CDP_DONE_MASK0));

	ret = processor_cdp_program(group);
	if (ret)
		goto exit;

exit:
	dla_debug("Exit: %s", __func__);
	RETURN(ret);
}
