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

#include <nvdla_interface.h>
#include <dla_interface.h>

#include "dla_engine_internal.h"

static union dla_operation_container operation_desc[DLA_OP_NUM][DLA_NUM_GROUPS];
static union dla_surface_container surface_desc[DLA_OP_NUM][DLA_NUM_GROUPS];

static struct dla_task global_task;

static struct dla_engine engine = {
	.processors[DLA_OP_BDMA] = {
		.name = "BDMA",
		.op_type = DLA_OP_BDMA,
		.program = dla_bdma_program,
		.enable = dla_bdma_enable,
		.set_producer = dla_bdma_set_producer,
		.is_ready = dla_bdma_is_ready,
		.dump_config = dla_bdma_dump_config,
		.rdma_check = dla_bdma_rdma_check,
		.get_stat_data = dla_bdma_stat_data,
		.dump_stat = dla_bdma_dump_stat,
		.consumer_ptr = 0,
		.roi_index = 0,
		.group_status = 0,
		.rdma_status = 0,
		.last_group = 1,
		.groups[0] = {
			.id = 0,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_BDMA][0],
			.surface_desc = &surface_desc[DLA_OP_BDMA][0],
		},
		.groups[1] = {
			.id = 1,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_BDMA][1],
			.surface_desc = &surface_desc[DLA_OP_BDMA][1],
		},
	},
	.processors[DLA_OP_CONV] = {
		.name = "Convolution",
		.op_type = DLA_OP_CONV,
		.program = dla_conv_program,
		.enable = dla_conv_enable,
		.set_producer = dla_conv_set_producer,
		.is_ready = dla_conv_is_ready,
		.dump_config = dla_conv_dump_config,
		.rdma_check = dla_conv_rdma_check,
		.get_stat_data = dla_conv_stat_data,
		.dump_stat = dla_conv_dump_stat,
		.consumer_ptr = 0,
		.roi_index = 0,
		.group_status = 0,
		.rdma_status = 0,
		.last_group = 1,
		.groups[0] = {
			.id = 0,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_CONV][0],
			.surface_desc = &surface_desc[DLA_OP_CONV][0],
		},
		.groups[1] = {
			.id = 1,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_CONV][1],
			.surface_desc = &surface_desc[DLA_OP_CONV][1],
		},
	},
	.processors[DLA_OP_SDP] = {
		.name = "SDP",
		.op_type = DLA_OP_SDP,
		.program = dla_sdp_program,
		.enable = dla_sdp_enable,
		.set_producer = dla_sdp_set_producer,
		.is_ready = dla_sdp_is_ready,
		.dump_config = dla_sdp_dump_config,
		.rdma_check = dla_sdp_rdma_check,
		.get_stat_data = dla_sdp_stat_data,
		.dump_stat = dla_sdp_dump_stat,
		.consumer_ptr = 0,
		.roi_index = 0,
		.group_status = 0,
		.rdma_status = 0,
		.last_group = 1,
		.groups[0] = {
			.id = 0,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_SDP][0],
			.surface_desc = &surface_desc[DLA_OP_SDP][0],
		},
		.groups[1] = {
			.id = 1,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_SDP][1],
			.surface_desc = &surface_desc[DLA_OP_SDP][1],
		},
	},
	.processors[DLA_OP_PDP] = {
		.name = "PDP",
		.op_type = DLA_OP_PDP,
		.program = dla_pdp_program,
		.enable = dla_pdp_enable,
		.set_producer = dla_pdp_set_producer,
		.is_ready = dla_pdp_is_ready,
		.dump_config = dla_pdp_dump_config,
		.rdma_check = dla_pdp_rdma_check,
		.get_stat_data = dla_pdp_stat_data,
		.dump_stat = dla_pdp_dump_stat,
		.consumer_ptr = 0,
		.roi_index = 0,
		.group_status = 0,
		.rdma_status = 0,
		.last_group = 1,
		.groups[0] = {
			.id = 0,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_PDP][0],
			.surface_desc = &surface_desc[DLA_OP_PDP][0],
		},
		.groups[1] = {
			.id = 1,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_PDP][1],
			.surface_desc = &surface_desc[DLA_OP_PDP][1],
		},
	},
	.processors[DLA_OP_CDP] = {
		.name = "CDP",
		.op_type = DLA_OP_CDP,
		.program = dla_cdp_program,
		.enable = dla_cdp_enable,
		.set_producer = dla_cdp_set_producer,
		.is_ready = dla_cdp_is_ready,
		.dump_config = dla_cdp_dump_config,
		.rdma_check = dla_cdp_rdma_check,
		.get_stat_data = dla_cdp_stat_data,
		.dump_stat = dla_cdp_dump_stat,
		.consumer_ptr = 0,
		.roi_index = 0,
		.group_status = 0,
		.rdma_status = 0,
		.last_group = 1,
		.groups[0] = {
			.id = 0,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_CDP][0],
			.surface_desc = &surface_desc[DLA_OP_CDP][0],
		},
		.groups[1] = {
			.id = 1,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_CDP][1],
			.surface_desc = &surface_desc[DLA_OP_CDP][1],
		},
	},

	.processors[DLA_OP_RUBIK] = {
		.name = "RUBIK",
		.op_type = DLA_OP_RUBIK,
		.program = dla_rubik_program,
		.enable = dla_rubik_enable,
		.set_producer = dla_rubik_set_producer,
		.is_ready = dla_rubik_is_ready,
		.dump_config = dla_rubik_dump_config,
		.rdma_check = dla_rubik_rdma_check,
		.get_stat_data = dla_rubik_stat_data,
		.dump_stat = dla_rubik_dump_stat,
		.consumer_ptr = 0,
		.roi_index = 0,
		.group_status = 0,
		.rdma_status = 0,
		.last_group = 1,
		.groups[0] = {
			.id = 0,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_RUBIK][0],
			.surface_desc = &surface_desc[DLA_OP_RUBIK][0],
		},
		.groups[1] = {
			.id = 1,
			.rdma_id = 0,
			.active = 0,
			.events = 0,
			.roi_index = 0,
			.is_rdma_needed = 0,
			.lut_index = -1,
			.operation_desc = &operation_desc[DLA_OP_RUBIK][1],
			.surface_desc = &surface_desc[DLA_OP_RUBIK][1],
		},
	},

};

struct dla_engine *dla_get_engine(void)
{
	return &engine;
}

int32_t dla_register_driver(void **engine_context, void *driver_context)
{
	*engine_context = &engine;
	engine.task = &global_task;
	engine.driver_context = driver_context;
	engine.task->task_data = NULL;

	dla_init_op_cache(&engine);

	RETURN(0);
}

uint32_t reg_read(uint32_t addr)
{
	return dla_reg_read(engine.driver_context, addr);
}

void reg_write(uint32_t addr, uint32_t reg)
{
	dla_reg_write(engine.driver_context, addr, reg);
}
