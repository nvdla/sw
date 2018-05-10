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

#ifndef __DLA_ENGINE_H_
#define __DLA_ENGINE_H_

#include <dla_interface.h>
#include <dla_sched.h>

struct dla_processor_group {
	uint8_t id;
	uint8_t rdma_id;
	uint8_t active;
	uint8_t events;
	uint8_t roi_index;
	uint8_t is_rdma_needed;
	uint8_t pending;
	int32_t lut_index;
	uint8_t programming;
	uint64_t start_time;

	struct dla_common_op_desc *op_desc;
	struct dla_common_op_desc *consumers[DLA_OP_NUM];
	struct dla_common_op_desc *fused_parent;
	union dla_operation_container *operation_desc;
	union dla_surface_container *surface_desc;
};

struct dla_processor {
	const char *name;
	uint8_t op_type;
	uint8_t consumer_ptr;
	uint8_t roi_index;
	uint8_t group_status;
	uint8_t rdma_status;
	uint8_t last_group;

	struct dla_common_op_desc *tail_op;
	struct dla_processor_group groups[DLA_NUM_GROUPS];
	union dla_stat_container *stat_data_desc;

	int32_t (*is_ready)(struct dla_processor *processor,
				  struct dla_processor_group *group);
	int32_t (*enable)(struct dla_processor_group *group);
	int32_t (*program)(struct dla_processor_group *group);
	void (*set_producer)(int32_t group_id, int32_t rdma_id);
	void (*dump_config)(struct dla_processor_group *group);
	void (*rdma_check)(struct dla_processor_group *group);
	void (*get_stat_data)(struct dla_processor *processor,
				struct dla_processor_group *group);
	void (*dump_stat)(struct dla_processor *processor);
};

struct dla_engine {
	struct dla_task *task;
	struct dla_config *config_data;
	struct dla_network_desc *network;
	struct dla_processor processors[DLA_OP_NUM];

	uint16_t num_proc_hwl;
	int32_t status;
	uint32_t stat_enable;

	void *driver_context;
};

struct dla_engine *dla_get_engine(void);

#endif
