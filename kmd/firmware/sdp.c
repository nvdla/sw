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
#include <dla_interface.h>

#include "common.h"
#include "dla_engine_internal.h"
#include "engine_debug.h"

static const uint8_t map_ena[] = {
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DISABLE, YES),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DISABLE, NO),
};

static const uint8_t map_prelu[] = {
	FIELD_ENUM(SDP_D_DP_BS_CFG_0, BS_MUL_PRELU, NO),
	FIELD_ENUM(SDP_D_DP_BS_CFG_0, BS_MUL_PRELU, YES),
};

static const uint8_t map_bypass[] = {
	FIELD_ENUM(SDP_D_DP_BS_CFG_0, BS_BYPASS, YES),
	FIELD_ENUM(SDP_D_DP_BS_CFG_0, BS_BYPASS, NO),
};

static const uint8_t map_alu_op[] = {
	FIELD_ENUM(SDP_D_DP_EW_CFG_0, EW_ALU_ALGO, MAX),
	FIELD_ENUM(SDP_D_DP_EW_CFG_0, EW_ALU_ALGO, MIN),
	FIELD_ENUM(SDP_D_DP_EW_CFG_0, EW_ALU_ALGO, SUM),
	FIELD_ENUM(SDP_D_DP_EW_CFG_0, EW_ALU_ALGO, EQL),
};

static const uint8_t map_alu_src[] = {
	FIELD_ENUM(SDP_D_DP_BS_ALU_CFG_0, BS_ALU_SRC, MEM),
	FIELD_ENUM(SDP_D_DP_BS_ALU_CFG_0, BS_ALU_SRC, REG),
};

static const uint8_t map_fly[] = {
	FIELD_ENUM(SDP_D_FEATURE_MODE_CFG_0, FLYING_MODE, OFF),
	FIELD_ENUM(SDP_D_FEATURE_MODE_CFG_0, FLYING_MODE, ON),
};

static const uint8_t map_dst[] = {
	FIELD_ENUM(SDP_D_FEATURE_MODE_CFG_0, OUTPUT_DST, MEM),
	FIELD_ENUM(SDP_D_FEATURE_MODE_CFG_0, OUTPUT_DST, PDP),
};


static const uint8_t map_wg[] = {
	FIELD_ENUM(SDP_D_FEATURE_MODE_CFG_0, WINOGRAD, OFF),
	FIELD_ENUM(SDP_D_FEATURE_MODE_CFG_0, WINOGRAD, ON),
};

static const uint8_t map_precision[] = {
	FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT8),
	FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT16),
	FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, FP16),
};

static const uint32_t map_proc_precision[3][3] = {
	{
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT8),
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT8),
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, FP16),
	},
	{
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT8),
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT16),
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, FP16),
	},
	{
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT8),
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, INT16),
		FIELD_ENUM(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION, FP16),
	},
};

static const uint8_t map_op_type[] = {
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_USE, MUL),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_USE, MUL),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_USE, ALU),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_USE, BOTH),
};

static const uint8_t map_element_size[] = {
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_SIZE, ONE_BYTE),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_SIZE, TWO_BYTE),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_SIZE, TWO_BYTE),
};

static const uint8_t map_op_mode[] = {
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_MODE, PER_ELEMENT),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_MODE, PER_KERNEL),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DATA_MODE, PER_ELEMENT),
};

static const uint8_t map_ram_type[] = {
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_RAM_TYPE, MC),
	FIELD_ENUM(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_RAM_TYPE, CV),
};

static const uint8_t map_perf_dma[] = {
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_DMA_EN, NO),
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_DMA_EN, YES),
};

static const uint8_t map_perf_lut[] = {
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_LUT_EN, NO),
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_LUT_EN, YES),
};

static const uint8_t map_perf_sat[] = {
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_SAT_EN, NO),
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_SAT_EN, YES),
};

static const uint8_t map_perf_nan_inf[] = {
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_NAN_INF_COUNT_EN, NO),
	FIELD_ENUM(SDP_D_PERF_ENABLE_0, PERF_NAN_INF_COUNT_EN, YES),
};

#if STAT_ENABLE
void
dla_sdp_stat_data(struct dla_processor *processor,
					struct dla_processor_group *group)
{
	uint64_t end_time = 0;
	struct dla_sdp_stat_desc *sdp_stat;

	sdp_stat = &processor->stat_data_desc->sdp_stat;

	end_time = dla_get_time_us();

	sdp_stat->nan_input_num = sdp_reg_read(D_STATUS_NAN_INPUT_NUM);
	sdp_stat->inf_input_num = sdp_reg_read(D_STATUS_INF_INPUT_NUM);
	sdp_stat->nan_output_num = sdp_reg_read(D_STATUS_NAN_OUTPUT_NUM);
	sdp_stat->wdma_write_stall = sdp_reg_read(D_PERF_WDMA_WRITE_STALL);
	sdp_stat->runtime = (uint32_t)(end_time - group->start_time);
}

void
dla_sdp_dump_stat(struct dla_processor *processor)
{
	struct dla_sdp_stat_desc *sdp_stat;

	sdp_stat = &processor->stat_data_desc->sdp_stat;

	dla_debug_sdp_stats(sdp_stat);
}
#endif /* STAT_ENABLE */

void
dla_sdp_set_producer(int32_t group_id, int32_t rdma_group_id)
{
	uint32_t reg;

	/**
	 * set producer pointer for all sub-modules
	 */
	reg = group_id << SHIFT(SDP_S_POINTER_0, PRODUCER);
	sdp_reg_write(S_POINTER, reg);
	reg = rdma_group_id << SHIFT(SDP_RDMA_S_POINTER_0, PRODUCER);
	sdp_rdma_reg_write(S_POINTER, reg);
}

int
dla_sdp_enable(struct dla_processor_group *group)
{
	uint32_t reg;
	uint8_t perf_reg;
	struct dla_engine *engine = dla_get_engine();

	dla_trace("Enter: %s", __func__);

	if (engine->stat_enable == (uint32_t)1) {
		perf_reg = (map_perf_dma[1] <<
			SHIFT(SDP_D_PERF_ENABLE_0, PERF_DMA_EN)) |
			(map_perf_lut[1] <<
			SHIFT(SDP_D_PERF_ENABLE_0, PERF_LUT_EN)) |
			(map_perf_sat[1] <<
			SHIFT(SDP_D_PERF_ENABLE_0, PERF_SAT_EN)) |
			(map_perf_nan_inf[1] <<
			SHIFT(SDP_D_PERF_ENABLE_0, PERF_NAN_INF_COUNT_EN));

		sdp_reg_write(D_PERF_ENABLE, perf_reg);
		group->start_time = dla_get_time_us();
	}

	/**
	 * enable all sub-modules
	 */
	if (group->is_rdma_needed) {
		reg = FIELD_ENUM(SDP_RDMA_D_OP_ENABLE_0, OP_EN, ENABLE);
		sdp_rdma_reg_write(D_OP_ENABLE, reg);
	}
	reg = FIELD_ENUM(SDP_D_OP_ENABLE_0, OP_EN, ENABLE);
	sdp_reg_write(D_OP_ENABLE, reg);

	dla_trace("Exit: %s", __func__);

	RETURN(0);
}

void
dla_sdp_rdma_check(struct dla_processor_group *group)
{
	uint8_t x1_rdma_ena;
	uint8_t x2_rdma_ena;
	uint8_t y_rdma_ena;
	uint8_t fly;
	struct dla_sdp_op_desc *sdp_op;
	struct dla_sdp_surface_desc *sdp_surface;

	sdp_op = &group->operation_desc->sdp_op;
	sdp_surface = &group->surface_desc->sdp_surface;

	x1_rdma_ena = sdp_op->x1_op.enable;
	x2_rdma_ena = sdp_op->x2_op.enable;
	y_rdma_ena  = sdp_op->y_op.enable;

	x1_rdma_ena &= (sdp_op->x1_op.mode != SDP_OP_PER_LAYER);
	x2_rdma_ena &= (sdp_op->x2_op.mode != SDP_OP_PER_LAYER);
	y_rdma_ena &= (sdp_op->y_op.mode != SDP_OP_PER_LAYER);

	fly = sdp_surface->src_data.type == DLA_MEM_HW;

	group->is_rdma_needed = (!fly) || (x1_rdma_ena ||
					x2_rdma_ena || y_rdma_ena);
}

static int32_t
processor_sdp_program(struct dla_processor_group *group)
{
	int32_t ret = 0;
	uint64_t src_addr = -1, x1_addr = -1, x2_addr = -1;
	uint64_t  y_addr = -1, dst_addr = -1;
	uint32_t reg, high, low;
	uint8_t fly;
	uint32_t atom_size;
	struct dla_sdp_op *x1_op;
	struct dla_sdp_op *x2_op;
	struct dla_sdp_op *y_op;
	uint8_t x1_rdma_ena;
	uint8_t x2_rdma_ena;
	uint8_t y_rdma_ena;
	uint8_t out_dma_ena;
	struct dla_lut_param lut;
	struct dla_engine *engine = dla_get_engine();
	struct dla_sdp_op_desc *sdp_op;
	struct dla_sdp_surface_desc *sdp_surface;

	dla_trace("Enter: %s", __func__);
	atom_size = engine->config_data->atom_size;

	sdp_op = &group->operation_desc->sdp_op;
	sdp_surface = &group->surface_desc->sdp_surface;

	fly = sdp_surface->src_data.type == DLA_MEM_HW;
	out_dma_ena = sdp_surface->dst_data.type != DLA_MEM_HW;
	x1_op = &sdp_op->x1_op;
	x2_op = &sdp_op->x2_op;
	y_op = &sdp_op->y_op;
	x1_rdma_ena = x1_op->enable && x1_op->type != SDP_OP_NONE;
	x2_rdma_ena = x2_op->enable && x2_op->type != SDP_OP_NONE;
	y_rdma_ena  = y_op->enable && y_op->type != SDP_OP_NONE;

	/* load address */
	if (!fly) {
		ret = dla_read_input_address(&sdp_surface->src_data,
						&src_addr,
						group->op_desc->index,
						group->roi_index,
					    1);
		if (ret)
			goto exit;
		CHECK_ALIGN(src_addr, atom_size);
	}

	if (out_dma_ena) {
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					sdp_surface->dst_data.address,
					sdp_surface->dst_data.offset,
					(void *)&dst_addr,
					DESTINATION_DMA);
		CHECK_ALIGN(dst_addr, atom_size);
	}

	if (sdp_op->lut_index >= 0) {
		group->lut_index = sdp_op->lut_index;
		dla_read_lut(engine, sdp_op->lut_index, (void *)&lut);
		dla_debug_lut_params(&lut);
	}


	x1_rdma_ena &= (x1_op->mode != SDP_OP_PER_LAYER);
	x2_rdma_ena &= (x2_op->mode != SDP_OP_PER_LAYER);
	y_rdma_ena &= (y_op->mode != SDP_OP_PER_LAYER);

	if (x1_rdma_ena) {
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					sdp_surface->x1_data.address,
					sdp_surface->x1_data.offset,
					(void *)&x1_addr,
					DESTINATION_DMA);
		CHECK_ALIGN(x1_addr, atom_size);
	}
	if (x2_rdma_ena) {
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					sdp_surface->x2_data.address,
					sdp_surface->x2_data.offset,
					(void *)&x2_addr,
					DESTINATION_DMA);
		CHECK_ALIGN(x2_addr, atom_size);
	}
	if (y_rdma_ena) {
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					sdp_surface->y_data.address,
					sdp_surface->y_data.offset,
					(void *)&y_addr,
					DESTINATION_DMA);
		CHECK_ALIGN(y_addr, atom_size);
	}

	reg = (map_fly[0] << SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, FLYING_MODE));
	sdp_rdma_reg_write(D_FEATURE_MODE_CFG, reg);

	reg = (map_ena[1] << SHIFT(SDP_RDMA_D_BRDMA_CFG_0, BRDMA_DISABLE));
	sdp_rdma_reg_write(D_BRDMA_CFG, reg);
	reg = (map_ena[1] << SHIFT(SDP_RDMA_D_NRDMA_CFG_0, NRDMA_DISABLE));
	sdp_rdma_reg_write(D_NRDMA_CFG, reg);
	reg = (map_ena[1] << SHIFT(SDP_RDMA_D_ERDMA_CFG_0, ERDMA_DISABLE));
	sdp_rdma_reg_write(D_ERDMA_CFG, reg);

	reg = (map_fly[fly] <<
			SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, FLYING_MODE)) |
	(map_wg[sdp_op->conv_mode == CONV_MODE_WINOGRAD] <<
			SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, WINOGRAD)) |
	(map_precision[sdp_op->src_precision] <<
			SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, IN_PRECISION)) |
	(map_precision[sdp_op->dst_precision] <<
			SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, OUT_PRECISION)) |
	(map_proc_precision[sdp_op->dst_precision][sdp_op->src_precision] <<
			SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, PROC_PRECISION)) |
	((sdp_op->batch_num-1) <<
			SHIFT(SDP_RDMA_D_FEATURE_MODE_CFG_0, BATCH_NUMBER));
	sdp_rdma_reg_write(D_FEATURE_MODE_CFG, reg);

	if (group->is_rdma_needed) {

		sdp_rdma_reg_write(D_DATA_CUBE_WIDTH,
					sdp_surface->src_data.width - 1);
		sdp_rdma_reg_write(D_DATA_CUBE_HEIGHT,
					sdp_surface->src_data.height - 1);
		sdp_rdma_reg_write(D_DATA_CUBE_CHANNEL,
					sdp_surface->src_data.channel - 1);

		/* config SDP source info */
		if (!fly) {
			/**
			 * if not on-the-fly, we have to config
			 * the source cube info
			 */
			high = HIGH32BITS(src_addr);
			low = LOW32BITS(src_addr);
			sdp_rdma_reg_write(D_SRC_BASE_ADDR_LOW, low);
			sdp_rdma_reg_write(D_SRC_BASE_ADDR_HIGH, high);
			sdp_rdma_reg_write(D_SRC_LINE_STRIDE,
					sdp_surface->src_data.line_stride);
			sdp_rdma_reg_write(D_SRC_SURFACE_STRIDE,
					sdp_surface->src_data.surf_stride);
			sdp_rdma_reg_write(D_SRC_DMA_CFG,
				map_ram_type[sdp_surface->src_data.type]);
		}

		/* config x1 source info */
		reg = (map_ena[x1_rdma_ena] <<
				SHIFT(SDP_RDMA_D_BRDMA_CFG_0,
				BRDMA_DISABLE)) |
			(map_op_type[x1_op->type] <<
				SHIFT(SDP_RDMA_D_BRDMA_CFG_0,
				BRDMA_DATA_USE)) |
			(map_element_size[x1_op->precision] <<
				SHIFT(SDP_RDMA_D_BRDMA_CFG_0,
				BRDMA_DATA_SIZE)) |
			(map_op_mode[x1_op->mode] <<
				SHIFT(SDP_RDMA_D_BRDMA_CFG_0,
				BRDMA_DATA_MODE)) |
			(map_ram_type[sdp_surface->x1_data.type] <<
				SHIFT(SDP_RDMA_D_BRDMA_CFG_0,
				BRDMA_RAM_TYPE));
		sdp_rdma_reg_write(D_BRDMA_CFG, reg);

		if (x1_rdma_ena) {
			high = HIGH32BITS(x1_addr);
			low = LOW32BITS(x1_addr);
			sdp_rdma_reg_write(D_BS_BASE_ADDR_LOW,
					low);
			sdp_rdma_reg_write(D_BS_BASE_ADDR_HIGH,
					high);
			sdp_rdma_reg_write(D_BS_LINE_STRIDE,
					sdp_surface->x1_data.line_stride);
			sdp_rdma_reg_write(D_BS_SURFACE_STRIDE,
					sdp_surface->x1_data.surf_stride);
		}

		/* config x2 source info */
		reg = (map_ena[x2_rdma_ena] <<
					SHIFT(SDP_RDMA_D_NRDMA_CFG_0,
					NRDMA_DISABLE)) |
			(map_op_type[x2_op->type] <<
					SHIFT(SDP_RDMA_D_NRDMA_CFG_0,
					NRDMA_DATA_USE)) |
			(map_element_size[x2_op->precision] <<
					SHIFT(SDP_RDMA_D_NRDMA_CFG_0,
					NRDMA_DATA_SIZE)) |
			(map_op_mode[x2_op->mode] <<
					SHIFT(SDP_RDMA_D_NRDMA_CFG_0,
					NRDMA_DATA_MODE)) |
			(map_ram_type[sdp_surface->x2_data.type] <<
					SHIFT(SDP_RDMA_D_NRDMA_CFG_0,
					NRDMA_RAM_TYPE));

		sdp_rdma_reg_write(D_NRDMA_CFG, reg);

		if (x2_rdma_ena) {
			high = HIGH32BITS(x2_addr);
			low = LOW32BITS(x2_addr);
			sdp_rdma_reg_write(D_BN_BASE_ADDR_LOW,
					low);
			sdp_rdma_reg_write(D_BN_BASE_ADDR_HIGH,
					high);
			sdp_rdma_reg_write(D_BN_LINE_STRIDE,
					sdp_surface->x2_data.line_stride);
			sdp_rdma_reg_write(D_BN_SURFACE_STRIDE,
					sdp_surface->x2_data.surf_stride);
		}

		/* config y source info */
		reg = (map_ena[y_rdma_ena] <<
				SHIFT(SDP_RDMA_D_ERDMA_CFG_0,
				ERDMA_DISABLE)) |
			(map_op_type[y_op->type] <<
				SHIFT(SDP_RDMA_D_ERDMA_CFG_0,
				ERDMA_DATA_USE)) |
			(map_element_size[y_op->precision] <<
				SHIFT(SDP_RDMA_D_ERDMA_CFG_0,
				ERDMA_DATA_SIZE)) |
			(map_op_mode[y_op->mode] <<
				SHIFT(SDP_RDMA_D_ERDMA_CFG_0,
				ERDMA_DATA_MODE)) |
			(map_ram_type[sdp_surface->y_data.type] <<
				SHIFT(SDP_RDMA_D_ERDMA_CFG_0,
				ERDMA_RAM_TYPE));

		sdp_rdma_reg_write(D_ERDMA_CFG, reg);
		if (y_rdma_ena) {
			high = HIGH32BITS(y_addr);
			low = LOW32BITS(y_addr);
			sdp_rdma_reg_write(D_EW_BASE_ADDR_LOW,
					low);
			sdp_rdma_reg_write(D_EW_BASE_ADDR_HIGH,
					high);
			sdp_rdma_reg_write(D_EW_LINE_STRIDE,
					sdp_surface->y_data.line_stride);
			sdp_rdma_reg_write(D_EW_SURFACE_STRIDE,
					sdp_surface->y_data.surf_stride);
		}
	}

	if (sdp_op->lut_index >= 0)
		update_lut(SDP_S_LUT_ACCESS_CFG_0, &lut,
					sdp_op->src_precision);

	sdp_reg_write(D_DATA_CUBE_WIDTH, sdp_surface->src_data.width - 1);
	sdp_reg_write(D_DATA_CUBE_HEIGHT, sdp_surface->src_data.height - 1);
	sdp_reg_write(D_DATA_CUBE_CHANNEL, sdp_surface->src_data.channel - 1);

	if (out_dma_ena) {
		high = HIGH32BITS(dst_addr);
		low = LOW32BITS(dst_addr);
		sdp_reg_write(D_DST_BASE_ADDR_HIGH,
				high);
		sdp_reg_write(D_DST_BASE_ADDR_LOW,
				low);
		sdp_reg_write(D_DST_LINE_STRIDE,
				sdp_surface->dst_data.line_stride);
		sdp_reg_write(D_DST_SURFACE_STRIDE,
				sdp_surface->dst_data.surf_stride);
	}

	/* Config BS module */
	reg = (map_bypass[x1_op->enable] <<
			SHIFT(SDP_D_DP_BS_CFG_0,
			BS_BYPASS)) |
		(map_bypass[x1_op->type != SDP_OP_MUL &&
				x1_op->type != SDP_OP_NONE] <<
			SHIFT(SDP_D_DP_BS_CFG_0,
			BS_ALU_BYPASS)) |
		(map_alu_op[x1_op->alu_type] <<
			SHIFT(SDP_D_DP_BS_CFG_0,
			BS_ALU_ALGO)) |
		(map_bypass[x1_op->type != SDP_OP_ADD &&
			x1_op->type != SDP_OP_NONE] <<
			SHIFT(SDP_D_DP_BS_CFG_0,
			BS_MUL_BYPASS)) |
		(map_prelu[x1_op->act == ACTIVATION_PRELU]
			<< SHIFT(SDP_D_DP_BS_CFG_0,
			BS_MUL_PRELU)) |
		(map_bypass[x1_op->act == ACTIVATION_RELU] <<
			SHIFT(SDP_D_DP_BS_CFG_0,
			BS_RELU_BYPASS));
	sdp_reg_write(D_DP_BS_CFG, reg);

	if (x1_op->enable) {
		if (x1_op->type == SDP_OP_ADD ||
				x1_op->type == SDP_OP_BOTH) {
			reg = (map_alu_src[x1_op->mode == SDP_OP_PER_LAYER] <<
					SHIFT(SDP_D_DP_BS_ALU_CFG_0,
					BS_ALU_SRC)) |
				(x1_op->shift_value <<
					SHIFT(SDP_D_DP_BS_ALU_CFG_0,
					BS_ALU_SHIFT_VALUE));
			sdp_reg_write(D_DP_BS_ALU_CFG, reg);
		}

		if (x1_op->mode == SDP_OP_PER_LAYER) {
			sdp_reg_write(D_DP_BS_ALU_SRC_VALUE,
					x1_op->alu_operand);
			sdp_reg_write(D_DP_BS_MUL_SRC_VALUE,
					x1_op->mul_operand);
		}

		/**
		 * MUL truncate will take effect no matter
		 * MUL is bypassed or not
		 */
		reg = (map_alu_src[x1_op->mode == SDP_OP_PER_LAYER] <<
			SHIFT(SDP_D_DP_BS_MUL_CFG_0,
			BS_MUL_SRC)) |
		(x1_op->truncate <<
			SHIFT(SDP_D_DP_BS_MUL_CFG_0,
			BS_MUL_SHIFT_VALUE));
		sdp_reg_write(D_DP_BS_MUL_CFG, reg);
	}

	/* Config BN module */
	reg = (map_bypass[x2_op->enable] <<
			SHIFT(SDP_D_DP_BN_CFG_0,
			BN_BYPASS)) |
		(map_bypass[x2_op->type != SDP_OP_MUL &&
			x2_op->type != SDP_OP_NONE] <<
			SHIFT(SDP_D_DP_BN_CFG_0,
			BN_ALU_BYPASS)) |
		(map_alu_op[x2_op->alu_type] <<
			SHIFT(SDP_D_DP_BN_CFG_0,
			BN_ALU_ALGO)) |
		(map_bypass[x2_op->type != SDP_OP_ADD &&
			x2_op->type != SDP_OP_NONE] <<
			SHIFT(SDP_D_DP_BN_CFG_0,
			BN_MUL_BYPASS)) |
		(map_prelu[x2_op->act == ACTIVATION_PRELU]
			<< SHIFT(SDP_D_DP_BN_CFG_0,
			BN_MUL_PRELU)) |
		(map_bypass[x2_op->act == ACTIVATION_RELU]
			<< SHIFT(SDP_D_DP_BN_CFG_0,
			BN_RELU_BYPASS));
	sdp_reg_write(D_DP_BN_CFG, reg);

	if (x2_op->enable) {
		if (x2_op->type == SDP_OP_ADD ||
			x2_op->type == SDP_OP_BOTH) {
			reg = (map_alu_src[x2_op->mode == SDP_OP_PER_LAYER] <<
					SHIFT(SDP_D_DP_BN_ALU_CFG_0,
					BN_ALU_SRC)) |
				(x2_op->shift_value <<
					SHIFT(SDP_D_DP_BN_ALU_CFG_0,
					BN_ALU_SHIFT_VALUE));
			sdp_reg_write(D_DP_BN_ALU_CFG, reg);
		}

		if (x2_op->mode == SDP_OP_PER_LAYER) {
			sdp_reg_write(D_DP_BN_ALU_SRC_VALUE,
					x2_op->alu_operand);
			sdp_reg_write(D_DP_BN_MUL_SRC_VALUE,
					x2_op->mul_operand);
		}

		reg = (map_alu_src[x2_op->mode == SDP_OP_PER_LAYER] <<
				SHIFT(SDP_D_DP_BN_MUL_CFG_0,
				BN_MUL_SRC)) |
			(x2_op->truncate <<
				SHIFT(SDP_D_DP_BN_MUL_CFG_0,
				BN_MUL_SHIFT_VALUE));
		sdp_reg_write(D_DP_BN_MUL_CFG, reg);
	}

	/* Config EW module */
	reg = (map_bypass[y_op->enable] <<
			SHIFT(SDP_D_DP_EW_CFG_0,
			EW_BYPASS)) |
		(map_bypass[y_op->type != SDP_OP_MUL &&
			y_op->type != SDP_OP_NONE] <<
			SHIFT(SDP_D_DP_EW_CFG_0,
			EW_ALU_BYPASS)) |
		(map_alu_op[y_op->alu_type] <<
			SHIFT(SDP_D_DP_EW_CFG_0,
			EW_ALU_ALGO)) |
		(map_bypass[y_op->type != SDP_OP_ADD &&
			y_op->type != SDP_OP_NONE] <<
			SHIFT(SDP_D_DP_EW_CFG_0,
			EW_MUL_BYPASS)) |
		((map_prelu[y_op->act == ACTIVATION_PRELU]) <<
			SHIFT(SDP_D_DP_EW_CFG_0,
			EW_MUL_PRELU)) |
		(map_bypass[y_op->act == ACTIVATION_LUT] <<
			SHIFT(SDP_D_DP_EW_CFG_0,
			EW_LUT_BYPASS));
	sdp_reg_write(D_DP_EW_CFG, reg);

	if (y_op->enable) {
		if (y_op->type == SDP_OP_ADD || y_op->type == SDP_OP_BOTH) {
			reg = (map_alu_src[y_op->mode == SDP_OP_PER_LAYER] <<
					SHIFT(SDP_D_DP_EW_ALU_CFG_0,
					EW_ALU_SRC)) |
				(map_bypass[y_op->cvt.alu_cvt.enable] <<
					SHIFT(SDP_D_DP_EW_ALU_CFG_0,
					EW_ALU_CVT_BYPASS));
			sdp_reg_write(D_DP_EW_ALU_CFG, reg);

			if (y_op->mode == SDP_OP_PER_LAYER) {
				sdp_reg_write(D_DP_EW_ALU_SRC_VALUE,
						y_op->alu_operand);
			} else {
				sdp_reg_write(D_DP_EW_ALU_CVT_OFFSET_VALUE,
						y_op->cvt.alu_cvt.offset);
				sdp_reg_write(D_DP_EW_ALU_CVT_SCALE_VALUE,
						y_op->cvt.alu_cvt.scale);
				sdp_reg_write(D_DP_EW_ALU_CVT_TRUNCATE_VALUE,
						y_op->cvt.alu_cvt.truncate);
			}
		}

		if (y_op->type == SDP_OP_MUL || y_op->type == SDP_OP_BOTH) {
			reg = (map_alu_src[y_op->mode == SDP_OP_PER_LAYER] <<
					SHIFT(SDP_D_DP_EW_MUL_CFG_0,
					EW_MUL_SRC)) |
				(map_bypass[y_op->cvt.mul_cvt.enable] <<
					SHIFT(SDP_D_DP_EW_MUL_CFG_0,
					EW_MUL_CVT_BYPASS));
			sdp_reg_write(D_DP_EW_MUL_CFG, reg);

			if (y_op->mode == SDP_OP_PER_LAYER) {
				sdp_reg_write(D_DP_EW_MUL_SRC_VALUE,
						y_op->mul_operand);
			} else {
				sdp_reg_write(D_DP_EW_MUL_CVT_OFFSET_VALUE,
						y_op->cvt.mul_cvt.offset);
				sdp_reg_write(D_DP_EW_MUL_CVT_SCALE_VALUE,
						y_op->cvt.mul_cvt.scale);
				sdp_reg_write(D_DP_EW_MUL_CVT_TRUNCATE_VALUE,
						y_op->cvt.mul_cvt.truncate);
			}
		}

		sdp_reg_write(D_DP_EW_TRUNCATE_VALUE, y_op->truncate);
	}

	reg = (map_fly[sdp_surface->src_data.type == DLA_MEM_HW] <<
			SHIFT(SDP_D_FEATURE_MODE_CFG_0,
			FLYING_MODE)) |
		(map_dst[sdp_surface->dst_data.type == DLA_MEM_HW] <<
			SHIFT(SDP_D_FEATURE_MODE_CFG_0,
			OUTPUT_DST)) |
		(map_wg[sdp_op->conv_mode == CONV_MODE_WINOGRAD] <<
			SHIFT(SDP_D_FEATURE_MODE_CFG_0,
			WINOGRAD)) |
		((sdp_op->batch_num - 1) <<
			SHIFT(SDP_D_FEATURE_MODE_CFG_0,
			BATCH_NUMBER));
	sdp_reg_write(D_FEATURE_MODE_CFG, reg);
	sdp_reg_write(D_DST_DMA_CFG,
			map_ram_type[sdp_surface->dst_data.type]);
	if (sdp_op->batch_num > 1)
		sdp_reg_write(D_DST_BATCH_STRIDE, sdp_op->batch_stride);

	reg =
	(map_proc_precision[sdp_op->dst_precision][sdp_op->src_precision] <<
			SHIFT(SDP_D_DATA_FORMAT_0,
			PROC_PRECISION)) |
		(map_precision[sdp_op->dst_precision] <<
			SHIFT(SDP_D_DATA_FORMAT_0,
			OUT_PRECISION));
	sdp_reg_write(D_DATA_FORMAT, reg);
	sdp_reg_write(D_CVT_OFFSET, sdp_op->out_cvt.offset);
	sdp_reg_write(D_CVT_SCALE, sdp_op->out_cvt.scale);
	sdp_reg_write(D_CVT_SHIFT, sdp_op->out_cvt.truncate);

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

int
dla_sdp_is_ready(struct dla_processor *processor,
			   struct dla_processor_group *group)
{
	struct dla_processor_group *next_group;
	struct dla_sdp_op_desc *sdp_op;

	sdp_op = &group->operation_desc->sdp_op;
	next_group = &processor->groups[!group->id];

	/**
	 * Single LUT is shared between two SDP groups, need to make
	 * sure that usage does not conflict. Also, LUT write
	 * access is locked when SDP sub-engine is active, so delay
	 * writing LUT when another group is active.
	 */

	/**
	 * if no LUT required for current group then it can be programmed
	 * without further checks
	 */
	if (sdp_op->lut_index == -1)
		return 1;

	/**
	 * if same LUT is used for both groups then it can be programmed
	 * without more checks. Even if another group is active and LUT
	 * is locked, it would have been programmed by another group.
	 */
	if (next_group->lut_index == sdp_op->lut_index)
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
dla_sdp_dump_config(struct dla_processor_group *group)
{
	struct dla_sdp_op_desc *sdp_op;
	struct dla_sdp_surface_desc *sdp_surface;

	sdp_surface = &group->surface_desc->sdp_surface;
	sdp_op = &group->operation_desc->sdp_op;

	dla_debug_sdp_surface_desc(sdp_surface, group->roi_index);
	dla_debug_sdp_op_desc(sdp_op, group->roi_index);
}

int
dla_sdp_program(struct dla_processor_group *group)
{
	int32_t ret;

	dla_trace("Enter: %s", __func__);
	dla_enable_intr(MASK(GLB_S_INTR_MASK_0, SDP_DONE_MASK1) |
			MASK(GLB_S_INTR_MASK_0, SDP_DONE_MASK0));

	ret = processor_sdp_program(group);
	if (ret)
		goto exit;

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}
