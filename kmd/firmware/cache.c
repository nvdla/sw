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
#include <dla_engine.h>
#include <dla_interface.h>

#include "dla_engine_internal.h"

#define DLA_OP_CACHE_SIZE (DLA_NUM_GROUPS * ((DLA_OP_NUM + 2) * 2))

static struct dla_common_op_desc desc_cache[DLA_OP_NUM][DLA_OP_CACHE_SIZE];
static int32_t desc_refcount[DLA_OP_NUM][DLA_OP_CACHE_SIZE];

void
dla_get_refcount(struct dla_common_op_desc *op_desc)
{
	int32_t i;
	struct dla_common_op_desc *desc = NULL;

	if (op_desc == NULL)
		return;

	if (op_desc->index == -1)
		return;

	desc = &desc_cache[op_desc->op_type][0];

	for (i = 0; i < DLA_OP_CACHE_SIZE; i++, desc++) {
		if (desc->index == op_desc->index &&
				desc->roi_index == op_desc->roi_index) {
			desc_refcount[op_desc->op_type][i]++;
			return;
		}
	}
}

struct dla_common_op_desc *
dla_get_op_desc(struct dla_task *task, int16_t index,
			uint8_t op_type, uint8_t roi_index)
{
	int32_t i;
	int32_t ret;
	uint64_t op_base;
	uint64_t dep_graph_addr;
	struct dla_common_op_desc *desc = NULL;
	struct dla_engine *engine = dla_get_engine();

	if (index == -1) {
		dla_debug("no desc get due to index==-1\n");
		goto exit;
	}

	dep_graph_addr = (sizeof(struct dla_common_op_desc) *
				engine->network->num_operations * roi_index);

	desc = &desc_cache[op_type][0];

	for (i = 0; i < DLA_OP_CACHE_SIZE; i++, desc++) {
		if (desc->index == index && desc->roi_index == roi_index) {
			if (desc->op_type != op_type) {
				dla_error("op_cache[op=%u] contains incorrect "
						"entry of op[%u]\n", op_type,
						desc->op_type);
				continue;
			}
			desc_refcount[op_type][i]++;
			goto exit;
		}
	}

	desc = &desc_cache[op_type][0];

	for (i = 0; i < DLA_OP_CACHE_SIZE; i++, desc++) {
		if (desc->index == -1) {
			op_base = dep_graph_addr +
					(sizeof(struct dla_common_op_desc) *
					(uint64_t)index);
			ret = dla_data_read(engine->driver_context,
					task->task_data,
					task->dependency_graph_addr,
					(void *)(desc),
					sizeof(struct dla_common_op_desc),
					op_base);
			if (ret) {
				desc = NULL;
				goto exit;
			}

			if (op_type != desc->op_type) {
				/*
				 * op_type of entry read from DRAM should not
				 * mismatch with given op_type. If they
				 * mismatches, then wrong entry is fetched, so
				 * report this issue by throwing error.
				 */
				dla_error("Fetched [op_type=%u] from DRAM doesn't "
					"match with op_type[%u]\n",
					desc->op_type,
					op_type);
				desc->op_type = op_type;
				desc->index = -1;
				desc->roi_index = -1;
				desc = NULL;
				goto exit;
			}

			desc->index = index;
			desc->roi_index = roi_index;

			/**
			 * Refcount must be 0 if we are reading it first time
			 * from DRAM
			 */
			assert(desc_refcount[op_type][i] == 0);

			desc_refcount[op_type][i]++;
			goto exit;
		}
	}

exit:
	return desc;
}

static void
dla_free_op_desc(struct dla_common_op_desc *op_desc)
{
	uint64_t op_base;
	uint64_t dep_graph_addr;
	struct dla_task *task;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s op desc index %u ROI %d\n", __func__,
				op_desc->index, op_desc->roi_index);

	task = engine->task;
	dep_graph_addr = (sizeof(struct dla_common_op_desc) *
				engine->network->num_operations *
				op_desc->roi_index);

	if (op_desc->index == -1)
		goto exit;

	if (op_desc == NULL)
		goto exit;

	/**
	 * TODO: keeping the depth value hardcoded as 0 for now,
	 * need to replace it once corresponding implementation is done.
	 */
	op_base = (dep_graph_addr +
			(sizeof(struct dla_common_op_desc) *
			(uint64_t)op_desc->index));

	/**
	 * Flush descriptor to DRAM
	 */
	dla_data_write(engine->driver_context,
			task->task_data,
			(void *)op_desc,
			task->dependency_graph_addr,
			sizeof(struct dla_common_op_desc),
			op_base);

	/**
	 * Release it
	 */
	op_desc->index = -1;
	op_desc->roi_index = -1;
exit:
	dla_debug("Exit: %s\n", __func__);
}

void
dla_put_op_desc(struct dla_common_op_desc *op_desc)
{
	int32_t i;
	struct dla_common_op_desc *desc;

	if (op_desc == NULL)
		return;

	if (op_desc->index == -1)
		return;

	desc = &desc_cache[op_desc->op_type][0];

	for (i = 0; i < DLA_OP_CACHE_SIZE; i++, desc++) {
		if (desc->index == op_desc->index &&
				desc->roi_index == op_desc->roi_index) {
			/**
			 * Refcount can't be 0 when we are trying to free it
			 */
			assert(desc_refcount[op_desc->op_type][i] > 0);

			desc_refcount[op_desc->op_type][i]--;

			/**
			 * Free desc if refcount is 0
			 */
			if (desc_refcount[op_desc->op_type][i] == 0)
				dla_free_op_desc(op_desc);

			return;
		}
	}
}

void
dla_init_op_cache(struct dla_engine *engine)
{
	int32_t i, j;
	struct dla_common_op_desc *desc = &desc_cache[0][0];

	dla_memset((uint8_t *)&desc_cache[0][0], 0, sizeof(desc_cache));
	dla_memset((uint8_t *)&desc_refcount[0][0], 0, sizeof(desc_refcount));

	for (i = 0; i < DLA_OP_NUM; i++) {
		for (j = 0; j < DLA_OP_CACHE_SIZE; j++) {
			desc->index = -1;
			desc->roi_index = -1;
			desc->op_type = (uint8_t)i;
			desc++;
		}
	}
}
