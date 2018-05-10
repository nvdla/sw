/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

static const uint8_t map_lut_method[] = {
	FIELD_ENUM(CDP_S_LUT_CFG_0, LUT_LE_FUNCTION, EXPONENT),
	FIELD_ENUM(CDP_S_LUT_CFG_0, LUT_LE_FUNCTION, LINEAR)
};
static const uint8_t map_lut_out[] = {
	FIELD_ENUM(CDP_S_LUT_CFG_0, LUT_UFLOW_PRIORITY, LE),
	FIELD_ENUM(CDP_S_LUT_CFG_0, LUT_UFLOW_PRIORITY, LO)
};

static const uint16_t access_data_offset[] = {
	CDP_S_LUT_ACCESS_DATA_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_ACCESS_DATA_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t lut_cfg_offset[] = {
	CDP_S_LUT_CFG_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_CFG_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t lut_info_offset[] = {
	CDP_S_LUT_INFO_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_INFO_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t le_start_offset[] = {
	CDP_S_LUT_LE_START_LOW_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LE_START_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t le_end_offset[] = {
	CDP_S_LUT_LE_END_LOW_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LE_END_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t lo_start_offset[] = {
	CDP_S_LUT_LO_START_LOW_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LO_START_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t lo_end_offset[] = {
	CDP_S_LUT_LO_END_LOW_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LO_END_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t le_slope_scale_offset[] = {
	CDP_S_LUT_LE_SLOPE_SCALE_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LE_SLOPE_SCALE_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t le_slope_shift_offset[] = {
	CDP_S_LUT_LE_SLOPE_SHIFT_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LE_SLOPE_SHIFT_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t lo_slope_scale_offset[] = {
	CDP_S_LUT_LO_SLOPE_SCALE_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LO_SLOPE_SCALE_0 - SDP_S_LUT_ACCESS_CFG_0,
};
static const uint16_t lo_slope_shift_offset[] = {
	CDP_S_LUT_LO_SLOPE_SHIFT_0 - CDP_S_LUT_ACCESS_CFG_0,
	SDP_S_LUT_LO_SLOPE_SHIFT_0 - SDP_S_LUT_ACCESS_CFG_0,
};

void update_lut(uint32_t reg_base, struct dla_lut_param *lut,
							uint8_t precision)
{
	int32_t i;
	uint32_t reg;
	uint32_t high, low;
	int32_t is_sdp = reg_base == SDP_S_LUT_ACCESS_CFG_0;
	struct dla_engine *engine = dla_get_engine();

	/* program raw table */
	reg = (FIELD_ENUM(CDP_S_LUT_ACCESS_CFG_0, LUT_TABLE_ID, LE)
		<< SHIFT(CDP_S_LUT_ACCESS_CFG_0, LUT_TABLE_ID)) |
		(FIELD_ENUM(CDP_S_LUT_ACCESS_CFG_0, LUT_ACCESS_TYPE, WRITE)
		<< SHIFT(CDP_S_LUT_ACCESS_CFG_0, LUT_ACCESS_TYPE));
	reg_write(reg_base, reg);

	for (i = 0; i < (1<<LUT_LINEAR_EXP_TABLE_ENTRY_LOG2)+1; i++) {
		dla_reg_write(engine->driver_context,
				reg_base + access_data_offset[is_sdp],
				lut->linear_exp_table[i]);
	}

	/* program density table */
	reg = (FIELD_ENUM(CDP_S_LUT_ACCESS_CFG_0, LUT_TABLE_ID, LO)
		<< SHIFT(CDP_S_LUT_ACCESS_CFG_0, LUT_TABLE_ID)) |
		(FIELD_ENUM(CDP_S_LUT_ACCESS_CFG_0, LUT_ACCESS_TYPE, WRITE)
		<< SHIFT(CDP_S_LUT_ACCESS_CFG_0, LUT_ACCESS_TYPE));
	dla_reg_write(engine->driver_context, reg_base, reg);

	for (i = 0; i < (1<<LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2)+1; i++) {
		dla_reg_write(engine->driver_context,
				reg_base + access_data_offset[is_sdp],
				lut->linear_only_table[i]);
	}

	/* program other configurations */
	reg = (map_lut_method[lut->method] <<
		SHIFT(CDP_S_LUT_CFG_0, LUT_LE_FUNCTION)) |
		(map_lut_out[lut->hybrid_priority] <<
		SHIFT(CDP_S_LUT_CFG_0, LUT_HYBRID_PRIORITY)) |
		(map_lut_out[lut->underflow_priority] <<
		SHIFT(CDP_S_LUT_CFG_0, LUT_UFLOW_PRIORITY)) |
		(map_lut_out[lut->overflow_priority] <<
		SHIFT(CDP_S_LUT_CFG_0, LUT_OFLOW_PRIORITY));
	dla_reg_write(engine->driver_context,
			reg_base + lut_cfg_offset[is_sdp], reg);

	if (lut->method == FIELD_ENUM(CDP_S_LUT_CFG_0,
					LUT_LE_FUNCTION, EXPONENT)) {
		reg = ((((uint32_t)lut->linear_exp_offset.exp_offset) <<
			SHIFT(CDP_S_LUT_INFO_0, LUT_LE_INDEX_OFFSET))&
		MASK(CDP_S_LUT_INFO_0, LUT_LE_INDEX_OFFSET)) |
			((((uint32_t)lut->linear_only_offset.frac_bits) <<
			SHIFT(CDP_S_LUT_INFO_0, LUT_LO_INDEX_SELECT))&
		MASK(CDP_S_LUT_INFO_0, LUT_LO_INDEX_SELECT));
	} else {
		reg = ((((uint32_t)lut->linear_exp_offset.frac_bits) <<
			SHIFT(CDP_S_LUT_INFO_0, LUT_LE_INDEX_SELECT))&
		MASK(CDP_S_LUT_INFO_0, LUT_LE_INDEX_SELECT)) |
			((((uint32_t)lut->linear_only_offset.frac_bits) <<
			SHIFT(CDP_S_LUT_INFO_0, LUT_LO_INDEX_SELECT))&
		MASK(CDP_S_LUT_INFO_0, LUT_LO_INDEX_SELECT));
	}
	dla_reg_write(engine->driver_context,
			reg_base + lut_info_offset[is_sdp], reg);
	high = HIGH32BITS(lut->linear_exp_start);
	low = LOW32BITS(lut->linear_exp_start);
	dla_reg_write(engine->driver_context,
			reg_base + le_start_offset[is_sdp], low);
	if (!is_sdp)
		dla_reg_write(engine->driver_context,
				reg_base + le_start_offset[is_sdp] + 4, high);

	high = HIGH32BITS(lut->linear_exp_end);
	low = LOW32BITS(lut->linear_exp_end);
	dla_reg_write(engine->driver_context,
				reg_base + le_end_offset[is_sdp], low);
	if (!is_sdp)
		dla_reg_write(engine->driver_context,
				reg_base + le_end_offset[is_sdp] + 4, high);

	high = HIGH32BITS(lut->linear_only_start);
	low = LOW32BITS(lut->linear_only_start);
	dla_reg_write(engine->driver_context,
				reg_base + lo_start_offset[is_sdp], low);
	if (!is_sdp)
		dla_reg_write(engine->driver_context,
				reg_base + lo_start_offset[is_sdp] + 4, high);

	high = HIGH32BITS(lut->linear_only_end);
	low = LOW32BITS(lut->linear_only_end);
	dla_reg_write(engine->driver_context,
				reg_base + lo_end_offset[is_sdp], low);
	if (!is_sdp)
		dla_reg_write(engine->driver_context,
				reg_base + lo_end_offset[is_sdp] + 4, high);

	if (precision == PRECISION_FP16) {
		reg = (lut->linear_exp_underflow_slope.data_f <<
			SHIFT(CDP_S_LUT_LE_SLOPE_SCALE_0,
					LUT_LE_SLOPE_UFLOW_SCALE)) |
			(lut->linear_exp_overflow_slope.data_f <<
			SHIFT(CDP_S_LUT_LE_SLOPE_SCALE_0,
					LUT_LE_SLOPE_OFLOW_SCALE));
		dla_reg_write(engine->driver_context,
				reg_base + le_slope_scale_offset[is_sdp], reg);

		reg = (lut->linear_only_underflow_slope.data_f <<
			SHIFT(CDP_S_LUT_LO_SLOPE_SCALE_0,
					LUT_LO_SLOPE_UFLOW_SCALE)) |
			(lut->linear_only_overflow_slope.data_f <<
			SHIFT(CDP_S_LUT_LO_SLOPE_SCALE_0,
					LUT_LO_SLOPE_OFLOW_SCALE));
		dla_reg_write(engine->driver_context,
				reg_base + lo_slope_scale_offset[is_sdp], reg);
	} else {
		union dla_slope *oslope;
		union dla_slope *uslope;

		uslope = &lut->linear_exp_underflow_slope;
		oslope = &lut->linear_exp_overflow_slope;
		reg = ((((uint32_t)uslope->data_i.scale)
			<< SHIFT(CDP_S_LUT_LE_SLOPE_SCALE_0,
					LUT_LE_SLOPE_UFLOW_SCALE))&
			MASK(CDP_S_LUT_LE_SLOPE_SCALE_0,
					LUT_LE_SLOPE_UFLOW_SCALE)) |
			((((uint32_t)oslope->data_i.scale)
			<< SHIFT(CDP_S_LUT_LE_SLOPE_SCALE_0,
					LUT_LE_SLOPE_OFLOW_SCALE))&
			MASK(CDP_S_LUT_LE_SLOPE_SCALE_0,
					LUT_LE_SLOPE_OFLOW_SCALE));
		dla_reg_write(engine->driver_context,
				reg_base + le_slope_scale_offset[is_sdp], reg);

		reg = ((((uint32_t)uslope->data_i.shifter) <<
			SHIFT(CDP_S_LUT_LE_SLOPE_SHIFT_0,
					LUT_LE_SLOPE_UFLOW_SHIFT))&
			MASK(CDP_S_LUT_LE_SLOPE_SHIFT_0,
					LUT_LE_SLOPE_UFLOW_SHIFT)) |
			((((uint32_t)oslope->data_i.shifter) <<
			SHIFT(CDP_S_LUT_LE_SLOPE_SHIFT_0,
					LUT_LE_SLOPE_OFLOW_SHIFT))&
			MASK(CDP_S_LUT_LE_SLOPE_SHIFT_0,
					LUT_LE_SLOPE_OFLOW_SHIFT));
		dla_reg_write(engine->driver_context,
				reg_base + le_slope_shift_offset[is_sdp], reg);

		uslope = &lut->linear_only_underflow_slope;
		oslope = &lut->linear_only_overflow_slope;
		reg = ((((uint32_t)uslope->data_i.scale) <<
			SHIFT(CDP_S_LUT_LO_SLOPE_SCALE_0,
					LUT_LO_SLOPE_UFLOW_SCALE))&
			MASK(CDP_S_LUT_LO_SLOPE_SCALE_0,
					LUT_LO_SLOPE_UFLOW_SCALE)) |
			((((uint32_t)oslope->data_i.scale) <<
			SHIFT(CDP_S_LUT_LO_SLOPE_SCALE_0,
					LUT_LO_SLOPE_OFLOW_SCALE))&
			MASK(CDP_S_LUT_LO_SLOPE_SCALE_0,
					LUT_LO_SLOPE_OFLOW_SCALE));
		dla_reg_write(engine->driver_context,
				reg_base + lo_slope_scale_offset[is_sdp], reg);
		reg = ((((uint32_t)uslope->data_i.shifter) <<
			SHIFT(CDP_S_LUT_LO_SLOPE_SHIFT_0,
					LUT_LO_SLOPE_UFLOW_SHIFT))&
			MASK(CDP_S_LUT_LO_SLOPE_SHIFT_0,
					LUT_LO_SLOPE_UFLOW_SHIFT)) |
			((((uint32_t)oslope->data_i.shifter) <<
			SHIFT(CDP_S_LUT_LO_SLOPE_SHIFT_0,
					LUT_LO_SLOPE_OFLOW_SHIFT))&
			MASK(CDP_S_LUT_LO_SLOPE_SHIFT_0,
					LUT_LO_SLOPE_OFLOW_SHIFT));
		dla_reg_write(engine->driver_context,
				reg_base + lo_slope_shift_offset[is_sdp], reg);
	}
}

int
validate_data_cube(struct dla_data_cube src_data_cube,
			struct dla_data_cube dst_data_cube,
			uint8_t mem_type)
{
	int32_t ret = 0;

	dla_trace("Enter: %s", __func__);

	if ((src_data_cube.width > DCUBE_MAX_WIDTH) ||
	    (src_data_cube.height > DCUBE_MAX_HEIGHT) ||
	    (src_data_cube.channel > DCUBE_MAX_CHANNEL)) {
		dla_error("Invalid SrcInput Cude[W: %u, H: %u, C: %u]",
				src_data_cube.width, src_data_cube.height,
				src_data_cube.channel);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if ((dst_data_cube.width > DCUBE_MAX_WIDTH) ||
	    (dst_data_cube.height > DCUBE_MAX_HEIGHT) ||
	    (dst_data_cube.channel > DCUBE_MAX_CHANNEL)) {
		dla_error("Invalid DstInput Cude[W: %u, H: %u, C: %u]",
				dst_data_cube.width, dst_data_cube.height,
				dst_data_cube.channel);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (src_data_cube.type > mem_type) {
		dla_error("Invalid src_data.mem_type: %u\n", src_data_cube.type);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	if (dst_data_cube.type > mem_type) {
		dla_error("Invalid dst_data.mem_type: %u\n", dst_data_cube.type);
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

exit:
	dla_trace("Exit: %s", __func__);
	RETURN(ret);
}

int
validate_precision(uint8_t precision, uint8_t map_precision)
{
	int32_t ret = 0;

	if (precision >= map_precision) {
		dla_error("Invalid precision: %u\n", precision);
		ret = ERR(INVALID_INPUT);
	}

	RETURN(ret);
}
