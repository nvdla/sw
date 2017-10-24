/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include <dla_debug.h>
#include <dla_interface.h>
#include <dla_sched.h>

#include "engine_debug.h"

#if DEBUG_NETWORK_DATA

void
dla_debug_network_desc(struct dla_network_desc *nd)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW dla_network_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("op desc index      = %d\n", nd->operation_desc_index);
	dla_debug("surface desc index = %d\n", nd->surface_desc_index);
	dla_debug("dep graph index    = %d\n", nd->dependency_graph_index);
	dla_debug("lut data index     = %d\n", nd->lut_data_index);
	dla_debug("stat_list_index    = %d\n", nd->stat_list_index);
	dla_debug("roi array index    = %d\n", nd->roi_array_index);
	dla_debug("surface index      = %d\n", nd->surface_index);
	dla_debug("num rois           = %u\n", nd->num_rois);
	dla_debug("num ops            = %u\n", nd->num_operations);
	dla_debug("num luts           = %u\n", nd->num_luts);
	dla_debug("num addr           = %u\n", nd->num_addresses);
	dla_debug("input layer        = %u\n", nd->input_layer);
	dla_debug("dynamic roi        = %u\n", nd->dynamic_roi);
}

static void
dla_debug_bdma_transfer(struct dla_bdma_transfer_desc *tr, int32_t id)
{
	dla_debug("transfer[%d]            = [ dla_bdma_transfer_desc =>\n", id);
	dla_debug("    source_address      = %x\n", tr->source_address);
	dla_debug("    destination_address = %x\n", tr->destination_address);
	dla_debug("    line_size           = %x\n", tr->line_size);
	dla_debug("    line_repeat         = %x\n", tr->line_repeat);
	dla_debug("    source_line         = %x\n", tr->source_line);
	dla_debug("    destination_line    = %x\n", tr->destination_line);
	dla_debug("    surface_repeat      = %x\n", tr->surface_repeat);
	dla_debug("    source_surface      = %x\n", tr->source_surface);
	dla_debug("    destination_surface = %x\n", tr->destination_surface);
}

void
dla_debug_bdma_surface_desc(struct dla_bdma_surface_desc *desc, int32_t roi)
{
	int32_t i;

	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_bdma_surface_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("source_type      = %u\n", desc->source_type);
	dla_debug("destination_type = %u\n", desc->destination_type);
	dla_debug("num_transfers    = %u\n", desc->num_transfers);
	for (i = 0; i < desc->num_transfers; i++)
		dla_debug_bdma_transfer(&desc->transfers[i], i);
}

void
dla_debug_bdma_op_desc(struct dla_bdma_op_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_bdma_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("num_transfers    = %u\n", desc->num_transfers);
}

void
dla_debug_address_info(struct dla_task *tk)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW address list\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("task base address        = %llu\n", tk->base);
	dla_debug("op desc address          = %llu\n", tk->operation_desc_addr);
	dla_debug("surface desc address     = %llu\n", tk->surface_desc_addr);
	dla_debug("dependency graph address = %llu\n", tk->dependency_graph_addr);
	dla_debug("LUT data address         = %llu\n", tk->lut_data_addr);
	dla_debug("stat address             = %llu\n", tk->stat_data_addr);
	dla_debug("ROI array address        = %llu\n", tk->roi_array_addr);
	dla_debug("surface address          = %llu\n", tk->surface_addr);
}

void
dla_debug_op_desc(struct dla_common_op_desc *desc, int32_t roi)
{
	int32_t i;

	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_common_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("[%p] Operation index %d ROI %d dep_count %d type %d\n",
			(unsigned int *)desc, desc->index, desc->roi_index,
			desc->dependency_count, desc->op_type);
	dla_debug("consumers = [ dla_consumer =>\n");
	for (i = 0; i < DLA_OP_NUM; i++)
		dla_debug(" [ %d %d ]", desc->consumers[i].index,
					desc->consumers[i].event);
	dla_debug("]");
	dla_debug("fused_parent = [ dla_consumer =>\n");
	dla_debug(" [ %d %d ]", desc->fused_parent.index,
					desc->fused_parent.event);
	dla_debug("]");
}

static void
dla_debug_data_cube(struct dla_data_cube *cube)
{
	dla_debug("    type          = %u\n", cube->type);
	dla_debug("    address       = %d\n", cube->address);
	dla_debug("    width         = %x\n", cube->width);
	dla_debug("    height        = %x\n", cube->height);
	dla_debug("    channel       = %x\n", cube->channel);
	dla_debug("    size          = %u\n", cube->size);
	dla_debug("    line_stride   = %u\n", cube->line_stride);
	dla_debug("    surf_stride   = %u\n", cube->surf_stride);
	dla_debug("    plane_stride  = %u\n", cube->plane_stride);
	dla_debug("]");
}

static void
dla_debug_converter(struct dla_cvt_param *cvt)
{
	dla_debug("[ scale = %d, truncate = %u, enable = %u, offset = %d ]\n",
			cvt->scale, cvt->truncate, cvt->enable, cvt->offset);
}

static void
dla_debug_float_data(struct dla_float_data *float_data)
{
	dla_debug("[ scale = %d, shifter = %d ]\n",
			float_data->scale, float_data->shifter);
}

static void
dla_debug_dla_slope(union dla_slope *slope)
{
	dla_debug("    data_i =\n");
	dla_debug_float_data(&slope->data_i);
	dla_debug("    data_f = %u\n", slope->data_f);
}

static void
dla_debug_lut_offset(union dla_lut_offset *offset)
{
	dla_debug("    exp_offset = %d\n", offset->exp_offset);
	dla_debug("    frac_bits  = %d\n", offset->frac_bits);
}

void
dla_debug_lut_params(struct dla_lut_param *lut_param)
{
	int32_t i, j;

	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW dla_lut_param\n");
	dla_debug("---------------------------------------------------------\n");

	dla_debug("linear_exp_table            = [\n");
	for (i = 0; i < (1<<LUT_LINEAR_EXP_TABLE_ENTRY_LOG2)+1; i++)
		dla_debug(" %u", lut_param->linear_exp_table[i]);
	dla_debug("]");

	dla_debug("linear_only_table           = [\n");
	for (j = 0; j < (1<<LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2)+1; j++)
		dla_debug(" %u\n", lut_param->linear_only_table[j]);
	dla_debug("]\n");

	dla_debug("linear_exp_offset           =\n");
	dla_debug_lut_offset(&lut_param->linear_exp_offset);
	dla_debug("linear_only_offset          =\n");
	dla_debug_lut_offset(&lut_param->linear_only_offset);
	dla_debug("linear_exp_start            = %llu\n",
				lut_param->linear_exp_start);
	dla_debug("linear_exp_end            = %llu\n",
				lut_param->linear_exp_end);
	dla_debug("linear_only_start           = %llu\n",
				lut_param->linear_only_start);
	dla_debug("linear_only_end           = %llu\n",
				lut_param->linear_only_end);
	dla_debug("linear_exp_underflow_slope  =\n");
	dla_debug_dla_slope(&lut_param->linear_exp_underflow_slope);
	dla_debug("linear_exp_overflow_slope   =\n");
	dla_debug_dla_slope(&lut_param->linear_exp_overflow_slope);
	dla_debug("linear_only_underflow_slope =\n");
	dla_debug_dla_slope(&lut_param->linear_only_underflow_slope);
	dla_debug("linear_only_overflow_slope  =\n");
	dla_debug_dla_slope(&lut_param->linear_only_overflow_slope);
	dla_debug("hybrid_priority             = %u\n",
				lut_param->hybrid_priority);
	dla_debug("underflow_priority          = %u\n",
				lut_param->underflow_priority);
	dla_debug("overflow_priority           = %u\n",
				lut_param->overflow_priority);
	dla_debug("method                      = %u\n",
				lut_param->method);
}

void
dla_debug_bdma_stats(struct dla_bdma_stat_desc *stat)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW STATS: dla_bdma_stat_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("read_stall   = %u\n", stat->read_stall);
	dla_debug("write_stall  = %u\n", stat->write_stall);
	dla_debug("runtime      = %u\n", stat->runtime);
}

void
dla_debug_conv_surface_desc(struct dla_conv_surface_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_conv_surface_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("weight_data         = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->weight_data);
	dla_debug("wmb_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->wmb_data);
	dla_debug("wgs_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->wgs_data);
	dla_debug("src_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->src_data);
	dla_debug("dst_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->dst_data);
	dla_debug("offset_u            = %lld\n", desc->offset_u);
	dla_debug("in_line_uv_stride   = %u\n", desc->in_line_uv_stride);
}

void
dla_debug_conv_op_desc(struct dla_conv_op_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_conv_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("conv_mode          = %u\n", desc->conv_mode);
	dla_debug("data_reuse         = %u\n", desc->data_reuse);
	dla_debug("weight_reuse       = %u\n", desc->weight_reuse);
	dla_debug("skip_data_rls      = %u\n", desc->skip_data_rls);
	dla_debug("skip_weight_rls    = %u\n", desc->skip_weight_rls);
	dla_debug("entry_per_slice    = %u\n", desc->entry_per_slice);
	dla_debug("data_format        = %u\n", desc->data_format);
	dla_debug("pixel_mapping      = %u\n", desc->pixel_mapping);
	dla_debug("fetch_grain        = %u\n", desc->fetch_grain);
	dla_debug("batch              = %u\n", desc->batch);
	dla_debug("weight_format      = %u\n", desc->weight_format);
	dla_debug("data_bank          = %u\n", desc->data_bank);
	dla_debug("weight_bank        = %u\n", desc->weight_bank);
	dla_debug("batch_stride       = %u\n", desc->batch_stride);
	dla_debug("post_extension     = %u\n", desc->post_extension);
	dla_debug("pixel_override     = %u\n", desc->pixel_override);
	dla_debug("release            = %u\n", desc->release);
	dla_debug("input_width_csc    = %u\n", desc->input_width_csc);
	dla_debug("input_height_csc   = %u\n", desc->input_height_csc);
	dla_debug("input_channel_csc  = %u\n", desc->input_channel_csc);
	dla_debug("kernel_width_csc   = %u\n", desc->kernel_width_csc);
	dla_debug("kernel_height_csc  = %u\n", desc->kernel_height_csc);
	dla_debug("kernel_channel_csc = %u\n", desc->kernel_channel_csc);
	dla_debug("input_width_cmac   = %u\n", desc->input_width_cmac);
	dla_debug("input_height_cmac  = %u\n", desc->input_height_cmac);
	dla_debug("bytes_per_kernel   = %u\n", desc->bytes_per_kernel);
	dla_debug("mean_ry            = %d\n", desc->mean_ry);
	dla_debug("mean_gu            = %d\n", desc->mean_gu);
	dla_debug("mean_bv            = %d\n", desc->mean_bv);
	dla_debug("mean_ax            = %d\n", desc->mean_ax);
	dla_debug("mean_format        = %u\n", desc->mean_format);
	dla_debug("conv_stride_x      = %u\n", desc->conv_stride_x);
	dla_debug("conv_stride_y      = %u\n", desc->conv_stride_y);
	dla_debug("pad_x_left         = %u\n", desc->pad_x_left);
	dla_debug("pad_x_right        = %u\n", desc->pad_x_right);
	dla_debug("pad_y_top          = %u\n", desc->pad_y_top);
	dla_debug("pad_y_bottom       = %u\n", desc->pad_y_bottom);
	dla_debug("dilation_x         = %u\n", desc->dilation_x);
	dla_debug("dilation_y         = %u\n", desc->dilation_y);
	dla_debug("pra_truncate       = %u\n", desc->pra_truncate);
	dla_debug("in_precision       = %u\n", desc->in_precision);
	dla_debug("out_precision      = %u\n", desc->out_precision);
	dla_debug("pad_val            = %d\n", desc->pad_val);
	dla_debug("in_cvt             =\n");
	dla_debug_converter(&desc->in_cvt);
	dla_debug("out_cvt            =\n");
	dla_debug_converter(&desc->out_cvt);
}

void
dla_debug_conv_stats(struct dla_conv_stat_desc *stat)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW STATS: dla_conv_stat_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("data_read_stall      = %u\n", stat->data_read_stall);
	dla_debug("weight_read_stall    = %u\n", stat->weight_read_stall);
	dla_debug("data_read_latency    = %u\n", stat->data_read_latency);
	dla_debug("weight_read_latency  = %u\n", stat->weight_read_latency);
	dla_debug("saturation_count     = %u\n", stat->saturation_count);
	dla_debug("nan_data_num         = %u\n", stat->nan_data_num);
	dla_debug("nan_weight_num       = %u\n", stat->nan_weight_num);
	dla_debug("inf_data_num         = %u\n", stat->inf_data_num);
	dla_debug("inf_weight_num       = %u\n", stat->inf_weight_num);
	dla_debug("runtime              = %u\n", stat->runtime);
}

void
dla_debug_pdp_surface_desc(struct dla_pdp_surface_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_pdp_surface_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("src_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->src_data);
	dla_debug("dst_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->dst_data);
}

void
dla_debug_pdp_op_desc(struct dla_pdp_op_desc *desc, int32_t roi)
{
	int32_t i;

	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_pdp_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("precision               = %u\n", desc->precision);
	dla_debug("padding_value           = [\n");
	for (i = 0; i < PDP_PAD_VAL_NUM; i++)
		dla_debug(" %d\n", desc->padding_value[i]);
	dla_debug("]\n");
	dla_debug("split_num               = %u\n", desc->split_num);
	dla_debug("partial_in_width_first  = %u\n",
					desc->partial_in_width_first);
	dla_debug("partial_in_width_mid    = %u\n", desc->partial_in_width_mid);
	dla_debug("partial_in_width_last   = %u\n", desc->partial_in_width_last);
	dla_debug("partial_width_first     = %u\n", desc->partial_width_first);
	dla_debug("partial_width_mid       = %u\n", desc->partial_width_mid);
	dla_debug("partial_width_last      = %u\n", desc->partial_width_last);
	dla_debug("pool_mode               = %u\n", desc->pool_mode);
	dla_debug("pool_width              = %u\n", desc->pool_width);
	dla_debug("pool_height             = %u\n", desc->pool_height);
	dla_debug("stride_x                = %u\n", desc->stride_x);
	dla_debug("stride_y                = %u\n", desc->stride_y);
	dla_debug("pad_left                = %u\n", desc->pad_left);
	dla_debug("pad_right               = %u\n", desc->pad_right);
	dla_debug("pad_top                 = %u\n", desc->pad_top);
	dla_debug("pad_bottom              = %u\n", desc->pad_bottom);
}

void
dla_debug_pdp_stats(struct dla_pdp_stat_desc *stat)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW STATS: dla_pdp_stat_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("inf_input_num   = %u\n", stat->inf_input_num);
	dla_debug("nan_input_num   = %u\n", stat->nan_input_num);
	dla_debug("nan_output_num  = %u\n", stat->nan_output_num);
	dla_debug("write_stall     = %u\n", stat->write_stall);
	dla_debug("runtime         = %u\n", stat->runtime);
}

void
dla_debug_cdp_surface_desc(struct dla_cdp_surface_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_cdp_surface_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("src_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->src_data);
	dla_debug("dst_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->dst_data);
}

void
dla_debug_cdp_op_desc(struct dla_cdp_op_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_cdp_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("in_precision      = %u\n", desc->in_precision);
	dla_debug("out_precision     = %u\n", desc->out_precision);
	dla_debug("lut_index         = %d\n", desc->lut_index);
	dla_debug("in_cvt             =\n");
	dla_debug_converter(&desc->in_cvt);
	dla_debug("out_cvt             =\n");
	dla_debug_converter(&desc->out_cvt);
	dla_debug("local_size        = %u\n", desc->local_size);
	dla_debug("bypass_sqsum      = %u\n", desc->bypass_sqsum);
	dla_debug("bypass_out_mul    = %u\n", desc->bypass_out_mul);
}

void
dla_debug_cdp_stats(struct dla_cdp_stat_desc *stat)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW STATS: dla_cdp_stat_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("nan_input_num     = %u\n", stat->nan_input_num);
	dla_debug("inf_input_num     = %u\n", stat->inf_input_num);
	dla_debug("nan_output_num    = %u\n", stat->nan_output_num);
	dla_debug("write_stall       = %u\n", stat->write_stall);
	dla_debug("lut_uflow         = %u\n", stat->lut_uflow);
	dla_debug("lut_oflow         = %u\n", stat->lut_oflow);
	dla_debug("lut_hybrid        = %u\n", stat->lut_hybrid);
	dla_debug("lut_le_hit        = %u\n", stat->lut_le_hit);
	dla_debug("lut_lo_hit        = %u\n", stat->lut_lo_hit);
	dla_debug("saturation_count  = %u\n", stat->saturation_count);
	dla_debug("runtime           = %u\n", stat->runtime);
}

void
dla_debug_rubik_surface_desc(struct dla_rubik_surface_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_rubik_surface_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("src_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->src_data);
	dla_debug("dst_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->dst_data);
}

void
dla_debug_rubik_op_desc(struct dla_rubik_op_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_rubik_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("mode       = %u\n", desc->mode);
	dla_debug("precision  = %u\n", desc->precision);
	dla_debug("stride_x   = %u\n", desc->stride_x);
	dla_debug("stride_y   = %u\n", desc->stride_y);
}

void
dla_debug_rubik_stats(struct dla_rubik_stat_desc *stat)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW STATS: dla_rubik_stat_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("read_stall   = %u\n", stat->read_stall);
	dla_debug("write_stall  = %u\n", stat->write_stall);
	dla_debug("runtime      = %u\n", stat->runtime);
}

static void
dla_debug_sdp_op(struct dla_sdp_op *sdp_op)
{
	dla_debug("    enable         = %u\n", sdp_op->enable);
	dla_debug("    alu_type       = %u\n", sdp_op->alu_type);
	dla_debug("    type           = %u\n", sdp_op->type);
	dla_debug("    mode           = %u\n", sdp_op->mode);
	dla_debug("    act            = %u\n", sdp_op->act);
	dla_debug("    shift_value    = %u\n", sdp_op->shift_value);
	dla_debug("    truncate       = %u\n", sdp_op->truncate);
	dla_debug("    precision      = %u\n", sdp_op->precision);
	dla_debug("    alu_operand    = %d\n", sdp_op->alu_operand);
	dla_debug("    mul_operand    = %d\n", sdp_op->mul_operand);
	dla_debug("cvt.alu_cvt          =\n");
	dla_debug_converter(&sdp_op->cvt.alu_cvt);
	dla_debug("cvt.mul_cvt          =\n");
	dla_debug_converter(&sdp_op->cvt.mul_cvt);
	dla_debug("]\n");
}

void
dla_debug_sdp_surface_desc(struct dla_sdp_surface_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_sdp_surface_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("src_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->src_data);
	dla_debug("x1_data             = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->x1_data);
	dla_debug("x2_data             = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->x2_data);
	dla_debug("y_data              = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->y_data);
	dla_debug("dst_data            = [ dla_data_cube =>\n");
	dla_debug_data_cube(&desc->dst_data);
}

void
dla_debug_sdp_op_desc(struct dla_sdp_op_desc *desc, int32_t roi)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW ROI[%d]: dla_sdp_op_desc\n", roi);
	dla_debug("---------------------------------------------------------\n");
	dla_debug("src_precision    = %u\n", desc->src_precision);
	dla_debug("dst_precision    = %u\n", desc->dst_precision);
	dla_debug("lut_index        = %d\n", desc->lut_index);
	dla_debug("out_cvt          =\n");
	dla_debug_converter(&desc->out_cvt);
	dla_debug("conv_mode        = %u\n", desc->conv_mode);
	dla_debug("batch_num        = %u\n", desc->batch_num);
	dla_debug("batch_stride     = %u\n", desc->batch_stride);
	dla_debug("x1_op            = [ dla_sdp_op =>\n");
	dla_debug_sdp_op(&desc->x1_op);
	dla_debug("x2_op            = [ dla_sdp_op =>\n");
	dla_debug_sdp_op(&desc->x2_op);
	dla_debug("y_op             = [ dla_sdp_op =>\n");
	dla_debug_sdp_op(&desc->y_op);
}

void
dla_debug_sdp_stats(struct dla_sdp_stat_desc *stat)
{
	dla_debug("*********************************************************\n");
	dla_debug("NVDLA FW STATS: dla_sdp_stat_desc\n");
	dla_debug("---------------------------------------------------------\n");
	dla_debug("nan_input_num     = %u\n", stat->nan_input_num);
	dla_debug("inf_input_num     = %u\n", stat->inf_input_num);
	dla_debug("nan_output_num    = %u\n", stat->nan_output_num);
	dla_debug("wdma_write_stall  = %u\n", stat->wdma_write_stall);
	dla_debug("lut_underflow     = %u\n", stat->lut_underflow);
	dla_debug("lut_overflow      = %u\n", stat->lut_overflow);
	dla_debug("lut_hybrid        = %u\n", stat->lut_hybrid);
	dla_debug("lut_le_hit        = %u\n", stat->lut_le_hit);
	dla_debug("lut_lo_hit        = %u\n", stat->lut_lo_hit);
	dla_debug("saturation_count  = %u\n", stat->saturation_count);
	dla_debug("runtime           = %u\n", stat->runtime);
}
#endif /* DEBUG_NETWORK_DATA */
