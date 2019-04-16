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

#define MAX_SPLIT_NUM	64
#define ARRAY_SIZE(a)	(sizeof(a) / sizeof((a[0])))

static const uint8_t map_ram[] = {
	FIELD_ENUM(PDP_RDMA_D_SRC_RAM_CFG_0, SRC_RAM_TYPE, MC),
	FIELD_ENUM(PDP_RDMA_D_SRC_RAM_CFG_0, SRC_RAM_TYPE, CV),
};

static const uint8_t map_pool[] = {
	FIELD_ENUM(PDP_D_OPERATION_MODE_CFG_0,
			POOLING_METHOD, POOLING_METHOD_AVERAGE),
	FIELD_ENUM(PDP_D_OPERATION_MODE_CFG_0,
			POOLING_METHOD, POOLING_METHOD_MAX),
	FIELD_ENUM(PDP_D_OPERATION_MODE_CFG_0,
			POOLING_METHOD, POOLING_METHOD_MIN),
};

static const uint8_t map_precision[] = {
	FIELD_ENUM(PDP_D_DATA_FORMAT_0, INPUT_DATA, INT8),
	FIELD_ENUM(PDP_D_DATA_FORMAT_0, INPUT_DATA, INT16),
	FIELD_ENUM(PDP_D_DATA_FORMAT_0, INPUT_DATA, FP16),
};

static const uint8_t map_pool_kernel[] = {
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_1),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_2),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_3),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_4),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_5),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_6),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_7),
	FIELD_ENUM(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH, KERNEL_WIDTH_8),
};

/* The reciprocal of kernel width: 1/1, 1/2, 1/3, ... */
static const uint32_t recip_kernel_size[2][8] = {
	/*
	 * INT8/16
	 * 1      1/2     1/3     1/4     1/5     1/6     1/7     1/8
	 */
	{0x10000, 0x8000, 0x5555, 0x4000, 0x3333, 0x2aaa, 0x2492, 0x2000},
	{0x7c00, 0x7800, 0x7555,  0x7400, 0x7266, 0x7155, 0x7092, 0x7000},
};

#if STAT_ENABLE
void
dla_pdp_stat_data(struct dla_processor *processor,
					struct dla_processor_group *group)
{
	uint64_t end_time = 0;
	struct dla_pdp_stat_desc *pdp_stat;

	pdp_stat = &processor->stat_data_desc->pdp_stat;

	end_time = dla_get_time_us();

	pdp_stat->write_stall = pdp_reg_read(D_PERF_WRITE_STALL);
	pdp_stat->runtime = (uint32_t)(end_time - group->start_time);
}

void
dla_pdp_dump_stat(struct dla_processor *processor)
{
	struct dla_pdp_stat_desc *pdp_stat;

	pdp_stat = &processor->stat_data_desc->pdp_stat;

	dla_debug_pdp_stats(pdp_stat);
}
#endif /* STAT_ENABLE */

static uint32_t
get_fly_mode(uint8_t type)
{
	uint32_t val;

	val = type == DLA_MEM_HW ?
			FIELD_ENUM(PDP_D_OPERATION_MODE_CFG_0,
						FLYING_MODE, ON_FLYING) :
			FIELD_ENUM(PDP_D_OPERATION_MODE_CFG_0,
						FLYING_MODE, OFF_FLYING);

	return val;
}

void
dla_pdp_set_producer(int32_t group_id, int32_t rdma_group_id)
{
	uint32_t reg;

	dla_trace("Enter: %s", __func__);

	dla_debug("group id %d rdma id %d\n", group_id, rdma_group_id);

	reg = group_id << SHIFT(PDP_S_POINTER_0, PRODUCER);
	pdp_reg_write(S_POINTER, reg);

	reg = rdma_group_id << SHIFT(PDP_RDMA_S_POINTER_0, PRODUCER);
	pdp_rdma_reg_write(S_POINTER, reg);

	dla_trace("Exit: %s", __func__);
}

int
dla_pdp_enable(struct dla_processor_group *group)
{
	int32_t ret = 0;
	uint32_t reg;
	struct dla_engine *engine = dla_get_engine();

	dla_trace("Enter: %s", __func__);

	if (!group) {
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (engine->stat_enable == (uint32_t)1) {
		reg = FIELD_ENUM(PDP_D_PERF_ENABLE_0, DMA_EN, ENABLE);
		pdp_reg_write(D_PERF_ENABLE, reg);
		group->start_time = dla_get_time_us();
	}

	dla_debug("rdma needed %u\n", group->is_rdma_needed);

	/**
	 * enable all sub-modules
	 */
	if (group->is_rdma_needed) {
		reg = FIELD_ENUM(PDP_RDMA_D_OP_ENABLE_0, OP_EN, ENABLE);
		pdp_rdma_reg_write(D_OP_ENABLE, reg);
	}
	reg = FIELD_ENUM(PDP_D_OP_ENABLE_0, OP_EN, ENABLE);
	pdp_reg_write(D_OP_ENABLE, reg);

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

void
dla_pdp_rdma_check(struct dla_processor_group *group)
{
	struct dla_pdp_surface_desc *pdp_surface;

	pdp_surface = &group->surface_desc->pdp_surface;

	group->is_rdma_needed = 0;

	if (pdp_surface->src_data.type != DLA_MEM_HW)
		group->is_rdma_needed = 1;
}

static int
validate_strides(uint8_t stride_x, uint8_t stride_y)
{
	int32_t ret = 0;

	if (stride_x < 1 || stride_y < 1 || stride_x > 8 || stride_y > 8) {
		dla_error("Invalid Stride (x[%d], y[%d])\n", stride_x, stride_y);
		ret = ERR(INVALID_INPUT);
	}

	RETURN(ret);
}

static int
vaildate_pdp_configs(struct dla_processor_group *group)
{
	int32_t ret = 0;
	struct dla_pdp_op_desc *pdp_op;
	struct dla_pdp_surface_desc *pdp_surface;

	dla_trace("Enter: %s", __func__);

	pdp_op = &group->operation_desc->pdp_op;
	pdp_surface = &group->surface_desc->pdp_surface;

	if (pdp_surface->dst_data.type == DLA_MEM_HW) {
		dla_error("Destination buffer for PDP has to be either MC or CV");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	ret = validate_data_cube(pdp_surface->src_data, pdp_surface->dst_data,
								DLA_MEM_HW);
	if (ret)
		goto exit;

	ret = validate_precision(pdp_op->precision, ARRAY_SIZE(map_precision));
	if (ret)
		goto exit;

	ret = validate_strides(pdp_op->stride_x, pdp_op->stride_y);
	if (ret)
		goto exit;

	if (pdp_op->split_num > MAX_SPLIT_NUM) {
		dla_error("Invalid split_num: %u\n", pdp_op->split_num);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (pdp_op->pool_width >= ARRAY_SIZE(map_pool_kernel)) {
		dla_error("Invalid pool_width: %u\n", pdp_op->pool_width);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (pdp_op->pool_height >= ARRAY_SIZE(map_pool_kernel)) {
		dla_error("Invalid pool_height: %u\n", pdp_op->pool_height);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (pdp_op->pool_mode >= ARRAY_SIZE(map_pool)) {
		dla_error("Invalid pool_mode: %u\n", pdp_op->pool_mode);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

static int
processor_pdp_program(struct dla_processor_group *group)
{
	int32_t ret = 0;
	uint32_t reg, high, low;
	uint64_t input_address = 0;
	uint64_t output_address = 0;
	struct dla_engine *engine = dla_get_engine();
	struct dla_pdp_op_desc *pdp_op;
	struct dla_pdp_surface_desc *pdp_surface;

	dla_trace("Enter: %s", __func__);

	pdp_op = &group->operation_desc->pdp_op;
	pdp_surface = &group->surface_desc->pdp_surface;

	ret = vaildate_pdp_configs(group);
	if (ret)
		goto exit;

	ret = dla_read_input_address(&pdp_surface->src_data,
					&input_address,
					group->op_desc->index,
					group->roi_index,
					1);
	if (ret)
		goto exit;

	if (pdp_surface->dst_data.address != -1)
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					pdp_surface->dst_data.address,
					pdp_surface->dst_data.offset,
					(void *)&output_address,
					DESTINATION_DMA);

	if (pdp_surface->src_data.type != DLA_MEM_HW) {
		/* PDP RDMA */
		pdp_rdma_reg_write(D_DATA_CUBE_IN_WIDTH,
				pdp_surface->src_data.width - 1);
		pdp_rdma_reg_write(D_DATA_CUBE_IN_HEIGHT,
				pdp_surface->src_data.height - 1);
		pdp_rdma_reg_write(D_DATA_CUBE_IN_CHANNEL,
				pdp_surface->src_data.channel - 1);

		high = HIGH32BITS(input_address);
		low  = LOW32BITS(input_address);
		pdp_rdma_reg_write(D_SRC_BASE_ADDR_HIGH, high);
		pdp_rdma_reg_write(D_SRC_BASE_ADDR_LOW, low);
		pdp_rdma_reg_write(D_SRC_LINE_STRIDE,
				pdp_surface->src_data.line_stride);
		pdp_rdma_reg_write(D_SRC_SURFACE_STRIDE,
				pdp_surface->src_data.surf_stride);

		reg = (map_precision[pdp_op->precision]
			<< SHIFT(PDP_RDMA_D_DATA_FORMAT_0, INPUT_DATA));
		pdp_rdma_reg_write(D_DATA_FORMAT, reg);

		reg = map_ram[pdp_surface->src_data.type]
			<< SHIFT(PDP_RDMA_D_SRC_RAM_CFG_0, SRC_RAM_TYPE);
		pdp_rdma_reg_write(D_SRC_RAM_CFG, reg);

		reg = ((pdp_op->split_num - 1)
			 << SHIFT(PDP_RDMA_D_OPERATION_MODE_CFG_0, SPLIT_NUM));
		pdp_rdma_reg_write(D_OPERATION_MODE_CFG, reg);

		reg = (map_pool_kernel[pdp_op->pool_width]
			<< SHIFT(PDP_RDMA_D_POOLING_KERNEL_CFG_0,
							KERNEL_WIDTH)) |
			((pdp_op->stride_x - 1)
			<< SHIFT(PDP_RDMA_D_POOLING_KERNEL_CFG_0,
							KERNEL_STRIDE_WIDTH));
		pdp_rdma_reg_write(D_POOLING_KERNEL_CFG, reg);

		reg = (pdp_op->pad_left
			<< SHIFT(PDP_RDMA_D_POOLING_PADDING_CFG_0, PAD_WIDTH));
		pdp_rdma_reg_write(D_POOLING_PADDING_CFG, reg);

		reg = ((pdp_op->partial_in_width_first == 0 ? 0 :
				pdp_op->partial_in_width_first - 1)
			<< SHIFT(PDP_RDMA_D_PARTIAL_WIDTH_IN_0,
				PARTIAL_WIDTH_IN_FIRST)) |
			((pdp_op->partial_in_width_mid == 0 ? 0 :
				pdp_op->partial_in_width_mid - 1)
			<< SHIFT(PDP_RDMA_D_PARTIAL_WIDTH_IN_0,
				PARTIAL_WIDTH_IN_MID)) |
			((pdp_op->partial_in_width_last == 0 ? 0 :
				pdp_op->partial_in_width_last - 1)
			<< SHIFT(PDP_RDMA_D_PARTIAL_WIDTH_IN_0,
				PARTIAL_WIDTH_IN_LAST));
		pdp_rdma_reg_write(D_PARTIAL_WIDTH_IN, reg);
	} else {
		ASSERT_GOTO(pdp_op->split_num == 1, ret,
					ERR(INVALID_INPUT), exit);
	}

	reg = ((pdp_surface->src_data.width - 1)
		<< SHIFT(PDP_D_DATA_CUBE_IN_WIDTH_0, CUBE_IN_WIDTH));
	pdp_reg_write(D_DATA_CUBE_IN_WIDTH, reg);

	reg = ((pdp_surface->src_data.height - 1)
		<< SHIFT(PDP_D_DATA_CUBE_IN_HEIGHT_0, CUBE_IN_HEIGHT));
	pdp_reg_write(D_DATA_CUBE_IN_HEIGHT, reg);

	reg = ((pdp_surface->src_data.channel - 1)
		<< SHIFT(PDP_D_DATA_CUBE_IN_CHANNEL_0, CUBE_IN_CHANNEL));
	pdp_reg_write(D_DATA_CUBE_IN_CHANNEL, reg);

	reg = ((pdp_surface->dst_data.width - 1)
		<< SHIFT(PDP_D_DATA_CUBE_OUT_WIDTH_0, CUBE_OUT_WIDTH));
	pdp_reg_write(D_DATA_CUBE_OUT_WIDTH, reg);

	reg = ((pdp_surface->dst_data.height - 1)
		<< SHIFT(PDP_D_DATA_CUBE_OUT_HEIGHT_0, CUBE_OUT_HEIGHT));
	pdp_reg_write(D_DATA_CUBE_OUT_HEIGHT, reg);

	reg = ((pdp_surface->dst_data.channel - 1)
		<< SHIFT(PDP_D_DATA_CUBE_OUT_CHANNEL_0, CUBE_OUT_CHANNEL));
	pdp_reg_write(D_DATA_CUBE_OUT_CHANNEL, reg);

	reg = (map_pool[pdp_op->pool_mode]
		<< SHIFT(PDP_D_OPERATION_MODE_CFG_0, POOLING_METHOD)) |
		(get_fly_mode(pdp_surface->src_data.type)
		<< SHIFT(PDP_D_OPERATION_MODE_CFG_0, FLYING_MODE)) |
		((pdp_op->split_num - 1)
		<< SHIFT(PDP_D_OPERATION_MODE_CFG_0, SPLIT_NUM));
	pdp_reg_write(D_OPERATION_MODE_CFG, reg);

	reg = ((pdp_op->partial_in_width_first == 0 ? 0 :
			pdp_op->partial_in_width_first-1)
		<< SHIFT(PDP_D_PARTIAL_WIDTH_IN_0, PARTIAL_WIDTH_IN_FIRST)) |
		((pdp_op->partial_in_width_mid == 0 ? 0 :
			pdp_op->partial_in_width_mid-1)
		<< SHIFT(PDP_D_PARTIAL_WIDTH_IN_0, PARTIAL_WIDTH_IN_MID)) |
		((pdp_op->partial_in_width_last == 0 ? 0 :
			pdp_op->partial_in_width_last-1)
		<< SHIFT(PDP_D_PARTIAL_WIDTH_IN_0, PARTIAL_WIDTH_IN_LAST));
	pdp_reg_write(D_PARTIAL_WIDTH_IN, reg);

	reg = ((pdp_op->partial_width_first == 0 ? 0 :
			pdp_op->partial_width_first-1)
		<< SHIFT(PDP_D_PARTIAL_WIDTH_OUT_0, PARTIAL_WIDTH_OUT_FIRST)) |
		((pdp_op->partial_width_mid == 0 ? 0 :
			pdp_op->partial_width_mid-1)
		<< SHIFT(PDP_D_PARTIAL_WIDTH_OUT_0, PARTIAL_WIDTH_OUT_MID))   |
		((pdp_op->partial_width_last == 0 ? 0 :
			pdp_op->partial_width_last-1)
		<< SHIFT(PDP_D_PARTIAL_WIDTH_OUT_0, PARTIAL_WIDTH_OUT_LAST));
	pdp_reg_write(D_PARTIAL_WIDTH_OUT, reg);

	reg = (map_pool_kernel[pdp_op->pool_width]
		<< SHIFT(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_WIDTH)) |
		(map_pool_kernel[pdp_op->pool_height]
		<< SHIFT(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_HEIGHT))|
		((pdp_op->stride_x - 1)
		<< SHIFT(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_STRIDE_WIDTH)) |
		((pdp_op->stride_y - 1)
		<< SHIFT(PDP_D_POOLING_KERNEL_CFG_0, KERNEL_STRIDE_HEIGHT));
	pdp_reg_write(D_POOLING_KERNEL_CFG, reg);

	pdp_reg_write(D_RECIP_KERNEL_WIDTH,
			recip_kernel_size[pdp_op->precision ==
					PRECISION_FP16][pdp_op->pool_width]);
	pdp_reg_write(D_RECIP_KERNEL_HEIGHT,
			recip_kernel_size[pdp_op->precision ==
					PRECISION_FP16][pdp_op->pool_height]);

	reg = (pdp_op->pad_left
		<< SHIFT(PDP_D_POOLING_PADDING_CFG_0, PAD_LEFT)) |
		(pdp_op->pad_right
		<< SHIFT(PDP_D_POOLING_PADDING_CFG_0, PAD_RIGHT)) |
		(pdp_op->pad_top
		<< SHIFT(PDP_D_POOLING_PADDING_CFG_0, PAD_TOP)) |
		(pdp_op->pad_bottom
		<< SHIFT(PDP_D_POOLING_PADDING_CFG_0, PAD_BOTTOM));
	if (pdp_op->precision == PRECISION_FP16) {
		int32_t i;

		for (i = 0; i < 7; i++)
			ASSERT_GOTO(pdp_op->padding_value[i] == 0, ret,
						ERR(INVALID_INPUT), exit);
	}

	pdp_reg_write(D_POOLING_PADDING_CFG, reg);
	pdp_reg_write(D_POOLING_PADDING_VALUE_1_CFG, pdp_op->padding_value[0]);
	pdp_reg_write(D_POOLING_PADDING_VALUE_2_CFG, pdp_op->padding_value[1]);
	pdp_reg_write(D_POOLING_PADDING_VALUE_3_CFG, pdp_op->padding_value[2]);
	pdp_reg_write(D_POOLING_PADDING_VALUE_4_CFG, pdp_op->padding_value[3]);
	pdp_reg_write(D_POOLING_PADDING_VALUE_5_CFG, pdp_op->padding_value[4]);
	pdp_reg_write(D_POOLING_PADDING_VALUE_6_CFG, pdp_op->padding_value[5]);
	pdp_reg_write(D_POOLING_PADDING_VALUE_7_CFG, pdp_op->padding_value[6]);

	if (pdp_surface->src_data.type != DLA_MEM_HW) {
		pdp_reg_write(D_SRC_LINE_STRIDE,
				pdp_surface->src_data.line_stride);
		pdp_reg_write(D_SRC_SURFACE_STRIDE,
				pdp_surface->src_data.surf_stride);
	}

	high = HIGH32BITS(output_address);
	low = LOW32BITS(output_address);
	pdp_reg_write(D_DST_BASE_ADDR_LOW, low);
	pdp_reg_write(D_DST_BASE_ADDR_HIGH, high);

	pdp_reg_write(D_DST_LINE_STRIDE, pdp_surface->dst_data.line_stride);
	pdp_reg_write(D_DST_SURFACE_STRIDE, pdp_surface->dst_data.surf_stride);

	reg = (map_ram[pdp_surface->dst_data.type]
		<< SHIFT(PDP_D_DST_RAM_CFG_0, DST_RAM_TYPE));
	pdp_reg_write(D_DST_RAM_CFG, reg);

	reg = (map_precision[pdp_op->precision]
		<< SHIFT(PDP_D_DATA_FORMAT_0, INPUT_DATA));
	pdp_reg_write(D_DATA_FORMAT, reg);

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

int
dla_pdp_is_ready(struct dla_processor *processor,
			   struct dla_processor_group *group)
{
	return 1;
}

void
dla_pdp_dump_config(struct dla_processor_group *group)
{
	struct dla_pdp_op_desc *pdp_op;
	struct dla_pdp_surface_desc *pdp_surface;

	pdp_surface = &group->surface_desc->pdp_surface;
	pdp_op = &group->operation_desc->pdp_op;

	dla_debug_pdp_surface_desc(pdp_surface, group->roi_index);
	dla_debug_pdp_op_desc(pdp_op, group->roi_index);
}

int
dla_pdp_program(struct dla_processor_group *group)
{
	int32_t ret;

	dla_trace("Enter: %s", __func__);

	if (!group) {
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	dla_enable_intr(MASK(GLB_S_INTR_MASK_0, PDP_DONE_MASK1) |
			MASK(GLB_S_INTR_MASK_0, PDP_DONE_MASK0));

	ret = processor_pdp_program(group);
	if (ret)
		goto exit;

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}
