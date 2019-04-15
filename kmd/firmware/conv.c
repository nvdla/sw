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

static const uint8_t map_precision[] = {
	FIELD_ENUM(CDMA_D_MISC_CFG_0, IN_PRECISION, INT8),
	FIELD_ENUM(CDMA_D_MISC_CFG_0, IN_PRECISION, INT16),
	FIELD_ENUM(CDMA_D_MISC_CFG_0, IN_PRECISION, FP16),
};

static const uint8_t map_conv[] = {
	FIELD_ENUM(CACC_D_MISC_CFG_0, CONV_MODE, DIRECT),
	FIELD_ENUM(CACC_D_MISC_CFG_0, CONV_MODE, WINOGRAD),
};

static const uint8_t map_weight_fmt[] = {
	FIELD_ENUM(CSC_D_WEIGHT_FORMAT_0, WEIGHT_FORMAT, UNCOMPRESSED),
	FIELD_ENUM(CSC_D_WEIGHT_FORMAT_0, WEIGHT_FORMAT, COMPRESSED),
};

static const uint8_t map_img_fmt[][2] = {
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R8), 1},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R10), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R12), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R16), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R16_I), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R16_F), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A16B16G16R16), 8},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_X16B16G16R16), 8},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A16B16G16R16_F), 8},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A16Y16U16V16), 8},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_V16U16Y16A16), 8},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A16Y16U16V16_F), 8},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A8B8G8R8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A8R8G8B8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_B8G8R8A8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R8G8B8A8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_X8B8G8R8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_X8R8G8B8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_B8G8R8X8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R8G8B8X8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A2B10G10R10), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A2R10G10B10), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_B10G10R10A2), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_R10G10B10A2), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A2Y10U10V10), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_V10U10Y10A2), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_A8Y8U8V8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_V8U8Y8A8), 4},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y8___U8V8_N444), 1},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y8___V8U8_N444), 1},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y10___U10V10_N444), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y10___V10U10_N444), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y12___U12V12_N444), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y12___V12U12_N444), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y16___U16V16_N444), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			PIXEL_FORMAT, T_Y16___V16U16_N444), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			DATAIN_FORMAT, FEATURE), 2},
	{FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
			DATAIN_FORMAT, PIXEL), 1},
};

static const uint8_t map_pixel[] = {
	FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0, PIXEL_MAPPING, PITCH_LINEAR),
};

static const uint8_t map_ram[] = {
	FIELD_ENUM(CDMA_D_DAIN_RAM_TYPE_0, DATAIN_RAM_TYPE, MCIF),
	FIELD_ENUM(CDMA_D_DAIN_RAM_TYPE_0, DATAIN_RAM_TYPE, CVIF),
};

static const uint8_t map_mean[] = {
	FIELD_ENUM(CDMA_D_MEAN_FORMAT_0, MEAN_FORMAT, DISABLE),
	FIELD_ENUM(CDMA_D_MEAN_FORMAT_0, MEAN_FORMAT, ENABLE),
};

#if STAT_ENABLE
void
dla_conv_stat_data(struct dla_processor *processor,
					struct dla_processor_group *group)
{
	uint64_t end_time = 0;
	struct dla_conv_stat_desc *conv_stat;

	conv_stat = &processor->stat_data_desc->conv_stat;

	end_time = dla_get_time_us();

	conv_stat->data_read_stall = cdma_reg_read(D_PERF_DAT_READ_STALL);
	conv_stat->weight_read_stall = cdma_reg_read(D_PERF_WT_READ_STALL);
	conv_stat->data_read_latency = cdma_reg_read(D_PERF_DAT_READ_LATENCY);
	conv_stat->weight_read_latency = cdma_reg_read(D_PERF_WT_READ_LATENCY);
	conv_stat->nan_data_num = cdma_reg_read(D_NAN_INPUT_DATA_NUM);
	conv_stat->nan_weight_num = cdma_reg_read(D_NAN_INPUT_WEIGHT_NUM);
	conv_stat->inf_data_num = cdma_reg_read(D_INF_INPUT_DATA_NUM);
	conv_stat->inf_weight_num = cdma_reg_read(D_INF_INPUT_WEIGHT_NUM);
	conv_stat->saturation_count = cacc_reg_read(D_OUT_SATURATION);
	conv_stat->runtime = (uint32_t)(end_time - group->start_time);
}

void
dla_conv_dump_stat(struct dla_processor *processor)
{
	struct dla_conv_stat_desc *conv_stat;

	conv_stat = &processor->stat_data_desc->conv_stat;

	dla_debug_conv_stats(conv_stat);
}
#endif /* STAT_ENABLE */

static uint32_t
get_in_format(uint8_t format)
{
	uint32_t in_format = 0;

	if (format >= FORMAT_T_R8 && format < FORMAT_FEATURE) {
		in_format = FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
						DATAIN_FORMAT, PIXEL);
	} else if (format == FORMAT_FEATURE) {
		in_format = FIELD_ENUM(CDMA_D_DATAIN_FORMAT_0,
						DATAIN_FORMAT, FEATURE);
	} else {
		assert(0);
	}

	return in_format;
}

void
dla_conv_set_producer(int32_t group_id, int32_t rdma_group_id)
{
	uint32_t reg;

	/* set producer pointer for all sub-modules */
	reg = group_id << SHIFT(CACC_S_POINTER_0, PRODUCER);
	cacc_reg_write(S_POINTER, reg);
	cmac_a_reg_write(S_POINTER, reg);
	cmac_b_reg_write(S_POINTER, reg);
	csc_reg_write(S_POINTER, reg);
	cdma_reg_write(S_POINTER, reg);
}

int
dla_conv_enable(struct dla_processor_group *group)
{
	uint32_t reg;
	struct dla_engine *engine = dla_get_engine();

	dla_trace("Enter: %s", __func__);

	do {
		reg = cdma_reg_read(S_CBUF_FLUSH_STATUS);
	} while (!(reg & MASK(CDMA_S_CBUF_FLUSH_STATUS_0, FLUSH_DONE)));

	if (engine->stat_enable == (uint32_t)1) {
		cdma_reg_write(D_PERF_ENABLE, 1);
		group->start_time = dla_get_time_us();
	}

	/* enable all sub-modules */
	reg = FIELD_ENUM(CACC_D_OP_ENABLE_0, OP_EN, ENABLE);
	cacc_reg_write(D_OP_ENABLE, reg);
	cmac_a_reg_write(D_OP_ENABLE, reg);
	cmac_b_reg_write(D_OP_ENABLE, reg);
	csc_reg_write(D_OP_ENABLE, reg);
	cdma_reg_write(D_OP_ENABLE, reg);

	dla_trace("Exit: %s", __func__);

	RETURN(0);
}

void
dla_conv_rdma_check(struct dla_processor_group *group)
{
	group->is_rdma_needed = 0;
}

static int32_t
processor_conv_program(struct dla_processor_group *group)
{
	int32_t ret = 0;
	uint32_t reg, high, low, shift, mask;
	uint32_t stride_x, stride_y, pad_x, pad_y;
	uint64_t weight_address = 0;
	uint64_t wmb_address = 0;
	uint64_t wgs_address = 0;
	uint64_t input_address = 0;
	uint64_t output_address = 0;
	uint32_t atom_size = 0;
	bool weight_compress_support = false;
	struct dla_engine *engine = dla_get_engine();
	struct dla_conv_op_desc *conv_op;
	struct dla_conv_surface_desc *conv_surface;

	dla_trace("Enter: %s", __func__);

	weight_compress_support = engine->config_data->weight_compress_support;
	atom_size = engine->config_data->atom_size;
	conv_op = &group->operation_desc->conv_op;
	conv_surface = &group->surface_desc->conv_surface;

	if (conv_op->weight_format == WEIGHT_FORMAT_COMPRESSED) {
		ASSERT_GOTO((weight_compress_support), ret, ERR(INVALID_INPUT), exit);
		ASSERT_GOTO((conv_surface->wmb_data.address != -1),
			ret, ERR(INVALID_INPUT), exit);
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					conv_surface->wmb_data.address,
					conv_surface->wmb_data.offset,
					(void *)&wmb_address,
					DESTINATION_DMA);
		CHECK_ALIGN(wmb_address, atom_size);
		CHECK_ALIGN(conv_surface->wmb_data.size, 128);

		ASSERT_GOTO((conv_surface->wgs_data.address != -1),
			ret, ERR(INVALID_INPUT), exit);
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					conv_surface->wgs_data.address,
					conv_surface->wgs_data.offset,
					(void *)&wgs_address,
					DESTINATION_DMA);
		CHECK_ALIGN(wgs_address, atom_size);
		CHECK_ALIGN(conv_surface->wgs_data.size, 4);
	}

	if (conv_surface->weight_data.address != -1) {
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					conv_surface->weight_data.address,
					conv_surface->weight_data.offset,
					(void *)&weight_address,
					DESTINATION_DMA);
		CHECK_ALIGN(weight_address, atom_size);
		CHECK_ALIGN(conv_surface->weight_data.size, 128);
	}

	if (conv_surface->dst_data.address != -1) {
		dla_get_dma_cube_address(engine->driver_context,
					engine->task->task_data,
					conv_surface->dst_data.address,
					conv_surface->dst_data.offset,
					(void *)&output_address,
					DESTINATION_DMA);
		CHECK_ALIGN(output_address, atom_size);
		CHECK_ALIGN(conv_surface->dst_data.size, atom_size);
		CHECK_ALIGN(conv_surface->dst_data.line_stride, atom_size);
		CHECK_ALIGN(conv_surface->dst_data.surf_stride, atom_size);
	}

	ret = dla_read_input_address(&conv_surface->src_data, &input_address,
					group->op_desc->index,
					group->roi_index,
					map_img_fmt[conv_op->data_format][1]);
	if (ret)
		goto exit;

	CHECK_ALIGN(input_address, atom_size);

	ASSERT_GOTO((conv_op->out_cvt.scale  == 1),
		ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO((conv_op->out_cvt.offset == 0),
		ret, ERR(INVALID_INPUT), exit);

	/* check if the register group is idle */
	reg = cacc_reg_read(S_STATUS);
	mask = group->id ? MASK(CACC_S_STATUS_0, STATUS_1) :
		MASK(CACC_S_STATUS_0, STATUS_0);
	shift = group->id ? SHIFT(CACC_S_STATUS_0, STATUS_1) :
		SHIFT(CACC_S_STATUS_0, STATUS_0);
	reg = (reg & mask) >> shift;
	ASSERT_GOTO((reg == FIELD_ENUM(CACC_S_STATUS_0, STATUS_0, IDLE)),
		ret, ERR(INVALID_INPUT), exit);

	reg = cmac_a_reg_read(S_STATUS);
	mask = group->id ? MASK(CMAC_A_S_STATUS_0, STATUS_1) :
        MASK(CMAC_A_S_STATUS_0, STATUS_0);
	shift = group->id ? SHIFT(CMAC_A_S_STATUS_0, STATUS_1) :
		SHIFT(CMAC_A_S_STATUS_0, STATUS_0);
	reg = (reg & mask) >> shift;
	ASSERT_GOTO((reg == FIELD_ENUM(CMAC_A_S_STATUS_0, STATUS_0, IDLE)),
		ret, ERR(INVALID_INPUT), exit);

	reg = cmac_b_reg_read(S_STATUS);
	mask = group->id ? MASK(CMAC_B_S_STATUS_0, STATUS_1) :
		MASK(CMAC_B_S_STATUS_0, STATUS_0);
	shift = group->id ? SHIFT(CMAC_B_S_STATUS_0, STATUS_1) :
		SHIFT(CMAC_B_S_STATUS_0, STATUS_0);
	reg = (reg & mask) >> shift;
	ASSERT_GOTO((reg == FIELD_ENUM(CMAC_B_S_STATUS_0, STATUS_0, IDLE)),
		ret, ERR(INVALID_INPUT), exit);

	reg = csc_reg_read(S_STATUS);
	mask = group->id ? MASK(CSC_S_STATUS_0, STATUS_1) :
		MASK(CSC_S_STATUS_0, STATUS_0);
	shift = group->id ? SHIFT(CSC_S_STATUS_0, STATUS_1) :
		SHIFT(CSC_S_STATUS_0, STATUS_0);
	reg = (reg & mask) >> shift;
	ASSERT_GOTO((reg == FIELD_ENUM(CSC_S_STATUS_0, STATUS_0, IDLE)),
		ret, ERR(INVALID_INPUT), exit);

	reg = cdma_reg_read(S_STATUS);
	mask = group->id ? MASK(CDMA_S_STATUS_0, STATUS_1) :
		MASK(CDMA_S_STATUS_0, STATUS_0);
	shift = group->id ? SHIFT(CDMA_S_STATUS_0, STATUS_1) :
		SHIFT(CDMA_S_STATUS_0, STATUS_0);
	reg = (reg & mask) >> shift;
	ASSERT_GOTO((reg == FIELD_ENUM(CDMA_S_STATUS_0, STATUS_0, IDLE)),
		ret, ERR(INVALID_INPUT), exit);

	/* reverse config each sub-module in CC */

	/* CACC */
	reg = (map_conv[conv_op->conv_mode]
		<< SHIFT(CACC_D_MISC_CFG_0, CONV_MODE)) |
		(map_precision[conv_op->out_precision]
		<< SHIFT(CACC_D_MISC_CFG_0, PROC_PRECISION));
	cacc_reg_write(D_MISC_CFG, reg);

	reg = ((conv_surface->dst_data.width - 1)
		<< SHIFT(CACC_D_DATAOUT_SIZE_0_0, DATAOUT_WIDTH)) |
		((conv_surface->dst_data.height - 1)
		<< SHIFT(CACC_D_DATAOUT_SIZE_0_0, DATAOUT_HEIGHT));
	cacc_reg_write(D_DATAOUT_SIZE_0, reg);

	reg = ((conv_surface->dst_data.channel - 1)
		<< SHIFT(CACC_D_DATAOUT_SIZE_1_0, DATAOUT_CHANNEL));
	cacc_reg_write(D_DATAOUT_SIZE_1, reg);

	low = LOW32BITS(output_address);
	cacc_reg_write(D_DATAOUT_ADDR, low);
	cacc_reg_write(D_BATCH_NUMBER, conv_op->batch - 1);
	cacc_reg_write(D_LINE_STRIDE, conv_surface->dst_data.line_stride);
	cacc_reg_write(D_SURF_STRIDE, conv_surface->dst_data.surf_stride);

	if (conv_surface->dst_data.width == 1 &&
				conv_surface->dst_data.height == 1) {
		ASSERT_GOTO((((uint32_t)conv_surface->dst_data.line_stride ==
			(uint32_t)(conv_surface->dst_data.width * atom_size))),
			ret, ERR(INVALID_INPUT), exit);
		reg = (CACC_D_DATAOUT_MAP_0_LINE_PACKED_TRUE <<
				SHIFT(CACC_D_DATAOUT_MAP_0, LINE_PACKED));
		reg |= (CACC_D_DATAOUT_MAP_0_SURF_PACKED_TRUE <<
				SHIFT(CACC_D_DATAOUT_MAP_0, SURF_PACKED));
	} else {
		reg = (FIELD_ENUM(CACC_D_DATAOUT_MAP_0, LINE_PACKED, FALSE) <<
				SHIFT(CACC_D_DATAOUT_MAP_0, LINE_PACKED));
		reg |= (FIELD_ENUM(CACC_D_DATAOUT_MAP_0, SURF_PACKED, FALSE) <<
				SHIFT(CACC_D_DATAOUT_MAP_0, SURF_PACKED));
	}
	cacc_reg_write(D_DATAOUT_MAP, reg);

	cacc_reg_write(D_CLIP_CFG, conv_op->out_cvt.truncate);

	/* CMAC */
	reg = (map_conv[conv_op->conv_mode]
		<< SHIFT(CMAC_A_D_MISC_CFG_0, CONV_MODE)) |
		(map_precision[conv_op->out_precision]
		<< SHIFT(CMAC_A_D_MISC_CFG_0, PROC_PRECISION));
	cmac_a_reg_write(D_MISC_CFG, reg);
	cmac_b_reg_write(D_MISC_CFG, reg);

	/* CSC */
	reg = (map_conv[conv_op->conv_mode]
		<< SHIFT(CSC_D_MISC_CFG_0, CONV_MODE)) |
		(map_precision[conv_op->out_precision]
		<< SHIFT(CSC_D_MISC_CFG_0, IN_PRECISION)) |
		(map_precision[conv_op->out_precision]
		<< SHIFT(CSC_D_MISC_CFG_0, PROC_PRECISION)) |
		(conv_op->data_reuse
		<< SHIFT(CSC_D_MISC_CFG_0, DATA_REUSE)) |
		(conv_op->weight_reuse
		<< SHIFT(CSC_D_MISC_CFG_0, WEIGHT_REUSE)) |
		(conv_op->skip_data_rls
		<< SHIFT(CSC_D_MISC_CFG_0, SKIP_DATA_RLS)) |
		(conv_op->skip_weight_rls
		<< SHIFT(CSC_D_MISC_CFG_0, SKIP_WEIGHT_RLS));
	csc_reg_write(D_MISC_CFG, reg);

	reg = (get_in_format(conv_op->data_format) <<
		SHIFT(CSC_D_DATAIN_FORMAT_0, DATAIN_FORMAT));
	csc_reg_write(D_DATAIN_FORMAT, reg);

	reg = ((conv_op->input_width_csc - 1)
		<< SHIFT(CSC_D_DATAIN_SIZE_EXT_0_0, DATAIN_WIDTH_EXT)) |
		((conv_op->input_height_csc - 1)
		<< SHIFT(CSC_D_DATAIN_SIZE_EXT_0_0, DATAIN_HEIGHT_EXT));
	csc_reg_write(D_DATAIN_SIZE_EXT_0, reg);

	reg = ((conv_op->input_channel_csc - 1)
		<< SHIFT(CSC_D_DATAIN_SIZE_EXT_1_0, DATAIN_CHANNEL_EXT));
	csc_reg_write(D_DATAIN_SIZE_EXT_1, reg);

	reg = ((conv_op->batch - 1)
		<< SHIFT(CSC_D_BATCH_NUMBER_0, BATCHES));
	csc_reg_write(D_BATCH_NUMBER, reg);
	reg = ((conv_op->post_extension)
		<< SHIFT(CSC_D_POST_Y_EXTENSION_0, Y_EXTENSION));
	csc_reg_write(D_POST_Y_EXTENSION, reg);

	reg = ((conv_op->entry_per_slice - 1)
		<< SHIFT(CSC_D_ENTRY_PER_SLICE_0, ENTRIES));
	csc_reg_write(D_ENTRY_PER_SLICE, reg);

	reg = (map_weight_fmt[conv_op->weight_format]
		<< SHIFT(CSC_D_WEIGHT_FORMAT_0, WEIGHT_FORMAT));
	csc_reg_write(D_WEIGHT_FORMAT, reg);

	reg = ((conv_op->kernel_width_csc - 1)
		<< SHIFT(CSC_D_WEIGHT_SIZE_EXT_0_0, WEIGHT_WIDTH_EXT)) |
		((conv_op->kernel_height_csc - 1)
		<< SHIFT(CSC_D_WEIGHT_SIZE_EXT_0_0, WEIGHT_HEIGHT_EXT));
	csc_reg_write(D_WEIGHT_SIZE_EXT_0, reg);

	reg = ((conv_op->kernel_channel_csc - 1)
		<< SHIFT(CSC_D_WEIGHT_SIZE_EXT_1_0, WEIGHT_CHANNEL_EXT)) |
		((conv_surface->dst_data.channel - 1)
		<< SHIFT(CSC_D_WEIGHT_SIZE_EXT_1_0, WEIGHT_KERNEL));
	csc_reg_write(D_WEIGHT_SIZE_EXT_1, reg);

	csc_reg_write(D_WEIGHT_BYTES, conv_surface->weight_data.size);
	csc_reg_write(D_WMB_BYTES, conv_surface->wmb_data.size);

	reg = ((conv_op->input_width_cmac - 1)
		<< SHIFT(CSC_D_DATAOUT_SIZE_0_0, DATAOUT_WIDTH)) |
		((conv_op->input_height_cmac - 1)
		<< SHIFT(CSC_D_DATAOUT_SIZE_0_0, DATAOUT_HEIGHT));
	csc_reg_write(D_DATAOUT_SIZE_0, reg);

	reg = ((conv_surface->dst_data.channel - 1)
		<< SHIFT(CSC_D_DATAOUT_SIZE_1_0, DATAOUT_CHANNEL));
	csc_reg_write(D_DATAOUT_SIZE_1, reg);

	reg = ((conv_surface->dst_data.width *
				conv_surface->dst_data.height - 1)
		<< SHIFT(CSC_D_ATOMICS_0, ATOMICS));
	csc_reg_write(D_ATOMICS, reg);
	reg = ((conv_op->release - 1)
		<< SHIFT(CSC_D_RELEASE_0, RLS_SLICES));
	csc_reg_write(D_RELEASE, reg);

	if (conv_op->conv_mode == CONV_MODE_DIRECT) {
		stride_x = conv_op->conv_stride_x - 1;
		stride_y = conv_op->conv_stride_y - 1;
		pad_x = conv_op->pad_x_left;
		pad_y = conv_op->pad_y_top;
	} else {
		stride_x = 0;
		stride_y = 0;
		pad_x = 0;
		pad_y = 0;
	}

	reg = (stride_x
		<< SHIFT(CSC_D_CONV_STRIDE_EXT_0, CONV_X_STRIDE_EXT)) |
		(stride_y
		<< SHIFT(CSC_D_CONV_STRIDE_EXT_0, CONV_Y_STRIDE_EXT));
	csc_reg_write(D_CONV_STRIDE_EXT, reg);

	reg = ((conv_op->dilation_x - 1)
		<< SHIFT(CSC_D_DILATION_EXT_0, X_DILATION_EXT)) |
		((conv_op->dilation_y - 1)
		<< SHIFT(CSC_D_DILATION_EXT_0, Y_DILATION_EXT));
	csc_reg_write(D_DILATION_EXT, reg);

	reg = (pad_x
		<< SHIFT(CSC_D_ZERO_PADDING_0, PAD_LEFT)) |
		(pad_y
		<< SHIFT(CSC_D_ZERO_PADDING_0, PAD_TOP));
	csc_reg_write(D_ZERO_PADDING, reg);

	reg = (conv_op->pad_val
		<< SHIFT(CSC_D_ZERO_PADDING_VALUE_0, PAD_VALUE)) &
		MASK(CSC_D_ZERO_PADDING_VALUE_0, PAD_VALUE);
	csc_reg_write(D_ZERO_PADDING_VALUE, reg);

	reg = ((conv_op->data_bank - 1)
		<< SHIFT(CSC_D_BANK_0, DATA_BANK)) |
		((conv_op->weight_bank - 1)
		<< SHIFT(CSC_D_BANK_0, WEIGHT_BANK));
	csc_reg_write(D_BANK, reg);
	csc_reg_write(D_PRA_CFG, conv_op->pra_truncate);

	/* CBUF */
	/* there's no CBUF register */

	/* CDMA */
	reg = (map_conv[conv_op->conv_mode]
		<< SHIFT(CDMA_D_MISC_CFG_0, CONV_MODE)) |
		(map_precision[conv_op->in_precision]
		<< SHIFT(CDMA_D_MISC_CFG_0, IN_PRECISION)) |
		(map_precision[conv_op->out_precision]
		<< SHIFT(CDMA_D_MISC_CFG_0, PROC_PRECISION)) |
		(conv_op->data_reuse
		<< SHIFT(CDMA_D_MISC_CFG_0, DATA_REUSE)) |
		(conv_op->weight_reuse
		<< SHIFT(CDMA_D_MISC_CFG_0, WEIGHT_REUSE)) |
		(conv_op->skip_data_rls
		<< SHIFT(CDMA_D_MISC_CFG_0, SKIP_DATA_RLS)) |
		(conv_op->skip_weight_rls
		<< SHIFT(CDMA_D_MISC_CFG_0, SKIP_WEIGHT_RLS));
	cdma_reg_write(D_MISC_CFG, reg);

	reg = (get_in_format(conv_op->data_format) <<
		SHIFT(CDMA_D_DATAIN_FORMAT_0, DATAIN_FORMAT)) |
		(map_img_fmt[conv_op->data_format][0]
		<< SHIFT(CDMA_D_DATAIN_FORMAT_0, PIXEL_FORMAT)) |
		(map_pixel[conv_op->pixel_mapping]
		<< SHIFT(CDMA_D_DATAIN_FORMAT_0, PIXEL_MAPPING)) |
		(conv_op->pixel_override
		<< SHIFT(CDMA_D_DATAIN_FORMAT_0, PIXEL_SIGN_OVERRIDE));
	cdma_reg_write(D_DATAIN_FORMAT, reg);

	reg = ((conv_surface->src_data.width - 1)
		<< SHIFT(CDMA_D_DATAIN_SIZE_0_0, DATAIN_WIDTH)) |
		((conv_surface->src_data.height - 1)
		<< SHIFT(CDMA_D_DATAIN_SIZE_0_0, DATAIN_HEIGHT));
	cdma_reg_write(D_DATAIN_SIZE_0, reg);

	reg = ((conv_surface->src_data.channel - 1)
		<< SHIFT(CDMA_D_DATAIN_SIZE_1_0, DATAIN_CHANNEL));
	cdma_reg_write(D_DATAIN_SIZE_1, reg);

	reg = ((conv_op->input_width_csc - 1)
		<< SHIFT(CDMA_D_DATAIN_SIZE_EXT_0_0, DATAIN_WIDTH_EXT)) |
		((conv_op->input_height_csc - 1)
		<< SHIFT(CDMA_D_DATAIN_SIZE_EXT_0_0, DATAIN_HEIGHT_EXT));
	cdma_reg_write(D_DATAIN_SIZE_EXT_0, reg);

	reg = (map_ram[conv_surface->src_data.type]
		<< SHIFT(CDMA_D_DAIN_RAM_TYPE_0, DATAIN_RAM_TYPE));
	cdma_reg_write(D_DAIN_RAM_TYPE, reg);

	high = HIGH32BITS(input_address);
	low = LOW32BITS(input_address);
	cdma_reg_write(D_DAIN_ADDR_HIGH_0, high);
	cdma_reg_write(D_DAIN_ADDR_LOW_0, low);

	high = HIGH32BITS((input_address + conv_surface->offset_u));
	low = LOW32BITS(input_address + conv_surface->offset_u);
	cdma_reg_write(D_DAIN_ADDR_HIGH_1, high);
	cdma_reg_write(D_DAIN_ADDR_LOW_1, low);

	cdma_reg_write(D_LINE_STRIDE, conv_surface->src_data.line_stride);
	cdma_reg_write(D_SURF_STRIDE, conv_surface->src_data.surf_stride);
	cdma_reg_write(D_LINE_UV_STRIDE, conv_surface->in_line_uv_stride);

	reg = ((conv_surface->src_data.line_stride ==
			((uint32_t)conv_surface->src_data.width * atom_size))
		<< SHIFT(CDMA_D_DAIN_MAP_0, LINE_PACKED));
	reg |= ((conv_surface->src_data.surf_stride ==
			((uint32_t)(conv_surface->src_data.width *
			conv_surface->src_data.height) * atom_size))
		<< SHIFT(CDMA_D_DAIN_MAP_0, SURF_PACKED));
	cdma_reg_write(D_DAIN_MAP, reg);

	reg = ((conv_op->batch - 1)
		<< SHIFT(CDMA_D_BATCH_NUMBER_0, BATCHES));
	cdma_reg_write(D_BATCH_NUMBER, reg);

	cdma_reg_write(D_BATCH_STRIDE, conv_op->batch_stride);

	reg = ((conv_op->entry_per_slice - 1)
		<< SHIFT(CDMA_D_ENTRY_PER_SLICE_0, ENTRIES));
	cdma_reg_write(D_ENTRY_PER_SLICE, reg);

	reg = ((conv_op->fetch_grain - 1)
		<< SHIFT(CDMA_D_FETCH_GRAIN_0, GRAINS));
	cdma_reg_write(D_FETCH_GRAIN, reg);

	reg = (map_weight_fmt[conv_op->weight_format]
		<< SHIFT(CDMA_D_WEIGHT_FORMAT_0, WEIGHT_FORMAT));
	cdma_reg_write(D_WEIGHT_FORMAT, reg);

	reg = ((conv_op->bytes_per_kernel - 1)
		<< SHIFT(CDMA_D_WEIGHT_SIZE_0_0, BYTE_PER_KERNEL));
	cdma_reg_write(D_WEIGHT_SIZE_0, reg);

	reg = ((conv_surface->dst_data.channel - 1)
		<< SHIFT(CDMA_D_WEIGHT_SIZE_1_0, WEIGHT_KERNEL));
	cdma_reg_write(D_WEIGHT_SIZE_1, reg);

	reg = (map_ram[conv_surface->weight_data.type]
		<< SHIFT(CDMA_D_WEIGHT_RAM_TYPE_0, WEIGHT_RAM_TYPE));
	cdma_reg_write(D_WEIGHT_RAM_TYPE, reg);

	high = HIGH32BITS(weight_address);
	low = LOW32BITS(weight_address);
	cdma_reg_write(D_WEIGHT_ADDR_HIGH, high);
	cdma_reg_write(D_WEIGHT_ADDR_LOW, low);
	cdma_reg_write(D_WEIGHT_BYTES, conv_surface->weight_data.size);

	if (conv_op->weight_format == WEIGHT_FORMAT_COMPRESSED) {
		high = HIGH32BITS(wgs_address);
		low = LOW32BITS(wgs_address);
		cdma_reg_write(D_WGS_ADDR_HIGH, high);
		cdma_reg_write(D_WGS_ADDR_LOW, low);

		high = HIGH32BITS(wmb_address);
		low = LOW32BITS(wmb_address);
		cdma_reg_write(D_WMB_ADDR_HIGH, high);
		cdma_reg_write(D_WMB_ADDR_LOW, low);
		cdma_reg_write(D_WMB_BYTES, conv_surface->wmb_data.size);
	}

	reg = (map_mean[conv_op->mean_format]
		<< SHIFT(CDMA_D_MEAN_FORMAT_0, MEAN_FORMAT));
	cdma_reg_write(D_MEAN_FORMAT, reg);

	if (conv_op->mean_format == MEAN_FORMAT_ENABLE) {
		reg = ((conv_op->mean_ry
			<< SHIFT(CDMA_D_MEAN_GLOBAL_0_0, MEAN_RY)) &
			MASK(CDMA_D_MEAN_GLOBAL_0_0, MEAN_RY)) |
			((conv_op->mean_gu
			<< SHIFT(CDMA_D_MEAN_GLOBAL_0_0, MEAN_GU)) &
			MASK(CDMA_D_MEAN_GLOBAL_0_0, MEAN_GU));
		cdma_reg_write(D_MEAN_GLOBAL_0, reg);

		reg = ((conv_op->mean_bv
			<< SHIFT(CDMA_D_MEAN_GLOBAL_1_0, MEAN_BV))&
			MASK(CDMA_D_MEAN_GLOBAL_1_0, MEAN_BV)) |
			((conv_op->mean_ax
			<< SHIFT(CDMA_D_MEAN_GLOBAL_1_0, MEAN_AX))&
			MASK(CDMA_D_MEAN_GLOBAL_1_0, MEAN_AX));
		cdma_reg_write(D_MEAN_GLOBAL_1, reg);
	}

	if (conv_op->in_cvt.enable) {
		reg = ((FIELD_ENUM(CDMA_D_CVT_CFG_0, CVT_EN, ENABLE))
			<< SHIFT(CDMA_D_CVT_CFG_0, CVT_EN)) |
			(conv_op->in_cvt.truncate
			<< SHIFT(CDMA_D_CVT_CFG_0, CVT_TRUNCATE));
		cdma_reg_write(D_CVT_CFG, reg);
		cdma_reg_write(D_CVT_OFFSET, conv_op->in_cvt.offset);
		cdma_reg_write(D_CVT_SCALE, conv_op->in_cvt.scale);
	} else {
		reg = ((FIELD_ENUM(CDMA_D_CVT_CFG_0, CVT_EN, DISABLE))
			<< SHIFT(CDMA_D_CVT_CFG_0, CVT_EN));
		cdma_reg_write(D_CVT_CFG, reg);
	}

	reg = ((conv_op->conv_stride_x - 1)
		<< SHIFT(CDMA_D_CONV_STRIDE_0, CONV_X_STRIDE)) |
		((conv_op->conv_stride_y - 1)
		<< SHIFT(CDMA_D_CONV_STRIDE_0, CONV_Y_STRIDE));
	cdma_reg_write(D_CONV_STRIDE, reg);

	reg = (conv_op->pad_x_left <<
		SHIFT(CDMA_D_ZERO_PADDING_0, PAD_LEFT)) |
		(conv_op->pad_x_right
		<< SHIFT(CDMA_D_ZERO_PADDING_0, PAD_RIGHT)) |
		(conv_op->pad_y_top
		<< SHIFT(CDMA_D_ZERO_PADDING_0, PAD_TOP)) |
		(conv_op->pad_y_bottom
		<< SHIFT(CDMA_D_ZERO_PADDING_0, PAD_BOTTOM));
	cdma_reg_write(D_ZERO_PADDING,   reg);

	reg = conv_op->pad_val <<
		SHIFT(CDMA_D_ZERO_PADDING_VALUE_0, PAD_VALUE) &
		MASK(CDMA_D_ZERO_PADDING_VALUE_0, PAD_VALUE);
	cdma_reg_write(D_ZERO_PADDING_VALUE, reg);
	reg = ((conv_op->weight_bank - 1)
		<< SHIFT(CDMA_D_BANK_0, WEIGHT_BANK)) |
		((conv_op->data_bank - 1)
		<< SHIFT(CDMA_D_BANK_0, DATA_BANK));
	cdma_reg_write(D_BANK, reg);

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

int
dla_conv_is_ready(struct dla_processor *processor,
			    struct dla_processor_group *group)
{
	return 1;
}

void
dla_conv_dump_config(struct dla_processor_group *group)
{
	struct dla_conv_op_desc *conv_op;
	struct dla_conv_surface_desc *conv_surface;

	conv_surface = &group->surface_desc->conv_surface;
	conv_op = &group->operation_desc->conv_op;

	dla_debug_conv_surface_desc(conv_surface, group->roi_index);
	dla_debug_conv_op_desc(conv_op, group->roi_index);
}

int
dla_conv_program(struct dla_processor_group *group)
{
	int32_t ret;

	dla_trace("Enter: %s", __func__);

	ret = processor_conv_program(group);
	if (ret)
		goto exit;

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}
