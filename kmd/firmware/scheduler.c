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
#include <dla_engine.h>
#include <dla_err.h>
#include <dla_interface.h>

#include "dla_engine_internal.h"
#include "engine_debug.h"

#define MAX_NUM_ADDRESSES	256

static uint64_t roi_array_length __aligned(8);
static struct dla_network_desc network;

static int
dla_update_consumers(struct dla_processor_group *group,
			struct dla_common_op_desc *op, uint8_t event);

static int32_t
dla_read_address_list(struct dla_engine *engine)
{
	RETURN(0);
}

int32_t
dla_read_lut(struct dla_engine *engine, int16_t index, void *dst)
{
	int32_t ret = 0;
	uint64_t src_addr;

	if (index == -1) {
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	src_addr = engine->task->lut_data_addr;

	ret = dla_data_read(engine->driver_context,
			engine->task->task_data,
			src_addr, (void *)dst,
			sizeof(struct dla_lut_param),
			(sizeof(struct dla_lut_param) * (uint64_t)index));

exit:
	RETURN(ret);
}

static int
dla_op_enabled(struct dla_processor_group *group)
{
	int32_t ret;
	struct dla_common_op_desc *op_desc;

	dla_debug("Enter: %s\n", __func__);
	op_desc = group->op_desc;

	group->active = 1;

	/* update dependency graph for this task */
	ret = dla_update_consumers(group, op_desc, DLA_EVENT_OP_ENABLED);
	dla_debug("Exit: %s\n", __func__);

	RETURN(ret);
}

static int
dla_op_programmed(struct dla_processor *processor,
		  struct dla_processor_group *group,
		  uint8_t rdma_id)
{
	int32_t ret;
	struct dla_common_op_desc *op_desc;

	dla_debug("Enter: %s\n", __func__);
	op_desc = group->op_desc;

	group->pending = 0;

	/* update dependency graph for this task */
	ret = dla_update_consumers(group, op_desc, DLA_EVENT_OP_PROGRAMMED);
	dla_debug("Exit: %s\n", __func__);

	RETURN(ret);
}

static int32_t
dla_read_config(struct dla_task *task, struct dla_processor *processor,
					struct dla_processor_group *group)
{
	int32_t ret;
	uint64_t base;
	int16_t index;
	uint8_t roi_index;
	struct dla_engine *engine;

	dla_debug("Enter: %s\n", __func__);

	engine = dla_get_engine();

	roi_index = group->roi_index;
	index = group->op_desc->index;

	base = (sizeof(union dla_operation_container) *
			(uint64_t)engine->network->num_operations *
			(uint64_t)roi_index);
	base = base + (sizeof(union dla_operation_container) *
			(uint64_t)index);

	LOG_EVENT(roi_index, group->id, processor->op_type,
					LOG_READ_OP_CONFIG_START);

	ret = dla_data_read(engine->driver_context, task->task_data,
				task->operation_desc_addr,
				(void *)group->operation_desc,
				sizeof(union dla_operation_container),
				base);
	if (ret)
		goto exit;

	LOG_EVENT(roi_index, group->id, processor->op_type,
					LOG_READ_OP_CONFIG_END);

	base = (sizeof(union dla_surface_container) *
			(uint64_t)engine->network->num_operations *
			(uint64_t)roi_index);

	base = base + (sizeof(union dla_surface_container) *
			(uint64_t)index);

	LOG_EVENT(roi_index, group->id, processor->op_type,
					LOG_READ_SURF_CONFIG_START);

	ret = dla_data_read(engine->driver_context, task->task_data,
				task->surface_desc_addr,
				(void *)group->surface_desc,
				sizeof(union dla_surface_container), base);
	if (ret)
		goto exit;

	LOG_EVENT(roi_index, group->id, processor->op_type,
					LOG_READ_SURF_CONFIG_END);

	processor->dump_config(group);

exit:
	dla_debug("Exit: %s\n", __func__);
	RETURN(ret);
}

static void
dla_reset_group(struct dla_processor_group *group)
{
	int32_t i;

	for (i = 0; i < DLA_OP_NUM; i++) {
		dla_put_op_desc(group->consumers[i]);
		group->consumers[i] = NULL;
	}

	dla_put_op_desc(group->fused_parent);
	group->fused_parent = NULL;

	dla_put_op_desc(group->op_desc);
	group->op_desc = NULL;
}

static int
dla_prepare_operation(struct dla_processor *processor,
			struct dla_common_op_desc *op_desc,
			uint8_t roi_index, uint32_t *group_number)
{
	int32_t ret = 0;
	uint8_t group_id;
	uint8_t rdma_id;
	struct dla_processor_group *group;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s\n", __func__);

	/*
	 * If not already programmed then find out if
	 * processor is free and which group is free
	 */
	ret = utils_get_free_group(processor, &group_id, &rdma_id);
	if (ret) {
		dla_debug("processor:%s register groups are busy\n",
			processor->name);
		goto exit;
	} else {
		dla_info("processor:%s group:%d, rdma_group:%d available\n",
				processor->name, group_id, rdma_id);
	}
	*group_number = group_id;
	group = &processor->groups[group_id];

	/*
	 * update operation descriptor
	 */
	group->op_desc = op_desc;
	dla_get_refcount(op_desc);
	group->id = group_id;
	group->roi_index = roi_index;
	group->rdma_id = rdma_id;

	ret = dla_read_config(engine->task, processor, group);
	if (ret)
		goto exit;

	group->pending = 1;

	processor->group_status |= (1 << group->id);

	processor->rdma_check(group);
	if (group->is_rdma_needed) {
		group->rdma_id = rdma_id;
		processor->rdma_status |= (1 << rdma_id);
	}

	processor->tail_op = op_desc;
exit:
	dla_debug("Exit: %s status=%d\n", __func__, ret);
	RETURN(ret);
}

static int
dla_program_operation(struct dla_processor *processor,
			struct dla_processor_group *group)
{
	int32_t i;
	int32_t ret = 0;
	struct dla_common_op_desc *op_desc;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s\n", __func__);

	dla_info("Program %s operation index %d ROI %d Group[%d]\n",
					processor->name,
					group->op_desc->index,
					group->roi_index,
					group->id);

	group->programming = 1;

	op_desc = group->op_desc;

	processor->set_producer(group->id, group->rdma_id);

	LOG_EVENT(group->roi_index, group->id, processor->op_type,
						LOG_PROGRAM_START);

	ret = processor->program(group);
	if (ret)
		goto exit;

	LOG_EVENT(group->roi_index, group->id, processor->op_type,
						LOG_PROGRAM_END);

	/**
	 * Pre-fetch consumers
	 */
	for (i = 0; i < DLA_OP_NUM; i++) {
		group->consumers[i] = dla_get_op_desc(engine->task,
					op_desc->consumers[i].index, i,
					group->roi_index);
	}

	group->fused_parent = dla_get_op_desc(engine->task,
					op_desc->fused_parent.index,
					op_desc->op_type - 1,
					group->roi_index);

	if (group->fused_parent != NULL) {
		if (group->fused_parent->op_type != (op_desc->op_type - 1)) {
			dla_warn("Invalid fused op type");
			ret = ERR(INVALID_INPUT);
			goto exit;
		}
	}

	ret = dla_op_programmed(processor, group, group->rdma_id);
	if (!ret)
		goto exit;

exit:
	group->programming = 0;
	dla_debug("Exit: %s status=%d\n", __func__, ret);
	RETURN(ret);
}

static int
dla_enable_operation(struct dla_processor *processor,
			struct dla_common_op_desc *op_desc)
{
	int32_t ret = 0;
	int32_t group_id;
	struct dla_engine *engine;
	struct dla_processor_group *group;

	dla_debug("Enter: %s\n", __func__);
	assert(op_desc->dependency_count == 0);

	/**
	 * If some operation has reported error then skip
	 * enabling next operations
	 */
	engine = dla_get_engine();
	if (engine->status)
		goto exit;

	/**
	 * Find out if operation is already programmed
	 */
	group_id = 0;
	group = &processor->groups[group_id];
	if ((processor->group_status & (1 << group_id)) &&
			group->op_desc->index == op_desc->index &&
			group->roi_index == op_desc->roi_index &&
			!group->pending)
		goto enable_op;

	group_id = 1;
	group = &processor->groups[group_id];
	if ((processor->group_status & (1 << group_id)) &&
			group->op_desc->index == op_desc->index &&
			group->roi_index == op_desc->roi_index &&
			!group->pending)
		goto enable_op;

	/**
	 * Operation is not programmed yet, ignore
	 */
	dla_debug("exit %s without actual enable due to processor "
				"hasn't been programmed\n", __func__);
	goto exit;

enable_op:
	/**
	 * If this event is triggered as part of programming same
	 * group then skip enable, it will get enabled after programming
	 * is complete
	 */
	if (group->programming)
		goto exit;

	if (group->active) {
		dla_debug("Processor:%s already enabled on group:%d\n",
			processor->name, group_id);
		goto exit;
	}

	dla_info("Enable %s operation index %d ROI %d\n",
					processor->name,
					group->op_desc->index,
					group->roi_index);

	processor->set_producer(group->id, group->rdma_id);

	LOG_EVENT(group->roi_index, group->id, processor->op_type,
						LOG_OPERATION_START);

	ret = processor->enable(group);
	if (ret)
		goto exit;

	ret = dla_op_enabled(group);
exit:
	dla_debug("Exit: %s status=%d\n", __func__, ret);
	RETURN(ret);
}

static int
dla_submit_operation(struct dla_processor *processor,
			struct dla_common_op_desc *op_desc,
			uint8_t roi_index)
{
	int32_t err;
	uint32_t group_id = 0;

	dla_debug("Enter: %s\n", __func__);

	dla_info("Prepare %s operation index %d ROI %d dep_count %d\n",
			processor->name, op_desc->index, roi_index,
			op_desc->dependency_count);
	err = dla_prepare_operation(processor, op_desc, roi_index, &group_id);
	if (err)
		goto exit;

	if (!processor->is_ready(processor, &processor->groups[group_id]))
		goto exit;

	err = dla_program_operation(processor, &processor->groups[group_id]);
	if (err)
		goto exit;

	if (op_desc->dependency_count == 0)
		err = dla_enable_operation(processor, op_desc);

exit:
	dla_debug("Exit: %s\n", __func__);
	RETURN(err);
}

/**
 * Dequeue next operation of same type from list of operations
 */
static int32_t
dla_dequeue_operation(struct dla_engine *engine,
			struct dla_processor *processor)
{
	int32_t ret = 0;
	int16_t index;
	struct dla_common_op_desc *consumer;

	dla_debug("Enter: %s\n", __func__);

	if (engine->status) {
		dla_debug("Skip dequeue op as engine has reported error\n");
		goto exit;
	}

	/**
	 * If we are done processing all ROIs for current op then
	 * load next op of same type otherwise reload same op for
	 * next ROI.
	 */
	if (processor->roi_index == (engine->network->num_rois - 1)) {
		index = processor->tail_op->consumers[processor->op_type].index;
		if (-1 == index) {
			/**
			 * It means we are done processing
			 * all ops of this type
			 */
			dla_debug("exit %s as there's no further operation\n",
				processor->name);
			goto exit;
		}
		processor->roi_index = 0;
	} else {
		processor->roi_index++;
		index = processor->tail_op->index;
	}

	dla_debug("Dequeue op from %s processor, index=%d ROI=%d\n",
			processor->name, index, processor->roi_index);

	/**
	 * Get operation descriptor
	 */
	consumer = dla_get_op_desc(engine->task, index,
				processor->op_type, processor->roi_index);
	if (consumer == NULL) {
		ret = ERR(NO_MEM);
		dla_error("Failed to allocate op_desc");
		goto exit;
	}

	ret = dla_submit_operation(processor, consumer, processor->roi_index);
	dla_put_op_desc(consumer);

exit:
	dla_debug("Exit: %s\n", __func__);
	RETURN(ret);
}

static int
dla_update_dependency(struct dla_consumer *consumer,
			struct dla_common_op_desc *op_desc,
			uint8_t event, uint8_t roi_index)
{
	int32_t ret = 0;
	struct dla_processor *processor;
	struct dla_engine *engine = dla_get_engine();

	if (consumer->index == -1)
		goto exit;

	/* Update dependency only if event matches */
	if (event != consumer->event)
		goto exit;

	/**
	 * If consumer index is valid but op desc is NULL means
	 * op desc for consumer was not pre-fetched
	 */
	if (op_desc == NULL) {
		ret = ERR(INVALID_INPUT);
		dla_error("Operation descriptor is NULL, consumer index %d",
				consumer->index);
		goto exit;
	}

	assert(op_desc->dependency_count > 0);

	dla_debug("Update dependency operation index %d ROI %d DEP_COUNT=%d\n",
					op_desc->index, op_desc->roi_index,
					op_desc->dependency_count);
	op_desc->dependency_count--;

	if (op_desc->dependency_count == 0) {
		processor = &engine->processors[op_desc->op_type];
		dla_debug("enable %s in %s as depdency are resolved\n",
			processor->name, __func__);

		ret = dla_enable_operation(processor, op_desc);
		if (ret)
			goto exit;
	}
exit:
	RETURN(ret);
}

static int
dla_update_consumers(struct dla_processor_group *group,
		     struct dla_common_op_desc *op,
		     uint8_t event)
{
	int32_t i;
	int32_t ret = 0;
	struct dla_engine *engine = dla_get_engine();

	if (engine->status) {
		dla_debug("Skip update as engine has reported error\n");
		goto exit;
	}

	for (i = 0; i < DLA_OP_NUM; i++) {
		ret = dla_update_dependency(&op->consumers[i],
						group->consumers[i],
						event, group->roi_index);
		if (ret) {
			dla_error("Failed to update dependency for "
				"consumer %d, ROI %d", i, group->roi_index);
			goto exit;
		}
	}

	ret = dla_update_dependency(&op->fused_parent,
					group->fused_parent,
					event, group->roi_index);
	if (ret) {
		dla_error("Failed to update dependency for "
			"fused parent, ROI %d", group->roi_index);
		goto exit;
	}

exit:
	RETURN(ret);
}

/**
 * Handle operation completion notification
 */
int
dla_op_completion(struct dla_processor *processor,
		  struct dla_processor_group *group)
{
	int32_t ret;
#if STAT_ENABLE
	uint64_t stat_data_address;
	uint64_t stat_base;
#endif /* STAT_ENABLE */
	struct dla_task *task;
	struct dla_common_op_desc *op_desc;
	struct dla_processor_group *next_group;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter:%s processor %s group%u\n", __func__,
					processor->name, group->id);

	dla_info("Completed %s operation index %d ROI %d\n",
					processor->name,
					group->op_desc->index,
					group->roi_index);

	task = engine->task;

	/**
	 * Mark OP as done only when all ROIs are done for that
	 * operation
	 */
	if (group->roi_index == (engine->network->num_rois - 1))
		engine->num_proc_hwl++;

	op_desc = group->op_desc;

#if STAT_ENABLE
	if (engine->stat_enable == (uint32_t)1) {
		processor->get_stat_data(processor, group);

		processor->dump_stat(processor);

		stat_data_address = (uint64_t)(engine->task->stat_data_addr +
				(sizeof(union dla_stat_container) *
				(uint64_t)(engine->network->num_operations) *
				(uint64_t)(op_desc->roi_index)));

		stat_base = (stat_data_address +
				(sizeof(union dla_stat_container) *
				(uint64_t)op_desc->index));

		/**
		 * Flush stat descriptor to DRAM
		 */
		ret = dla_data_write(engine->driver_context, task->task_data,
					(void *)(processor->stat_data_desc),
					stat_base,
					sizeof(union dla_stat_container),
					0);
		if (ret < 0)
			dla_error("Failed to write stats to DMA memory\n");
	}
#endif /* STAT_ENABLE */

	/**
	 * Get an extra reference count to keep op descriptor
	 * in cache until this operation completes
	 */
	dla_get_refcount(op_desc);

	LOG_EVENT(group->roi_index, group->id, processor->op_type,
						LOG_OPERATION_END);

	processor->group_status &= ~(1 << group->id);
	if (group->is_rdma_needed) {
		group->is_rdma_needed = 0;
		processor->rdma_status &= ~(1 << group->rdma_id);
		group->rdma_id = 0;
	}
	group->active = 0;
	group->lut_index = -1;
	processor->last_group = group->id;

	/**
	 * Switch consumer pointer to next group
	 */
	processor->consumer_ptr = !group->id;

	/**
	 * update dependency graph for this task
	 * TODO: Add proper error handling
	 */
	ret = dla_update_consumers(group, op_desc, DLA_EVENT_OP_COMPLETED);
	if (ret)
		goto exit;

	dla_info("%d HWLs done, totally %d layers\n",
				engine->num_proc_hwl,
				engine->network->num_operations);

	/* free operation descriptor from cache */
	dla_reset_group(group);

	/* if not hwl pending, means network completed */
	if (engine->network->num_operations == engine->num_proc_hwl) {
		dla_put_op_desc(op_desc);
		goto exit;
	}

	next_group = &processor->groups[!group->id];
	if (next_group->pending && !engine->status) {
		/**
		 * Next group must be ready here for programming,
		 * if not means it is an error
		 */
		if (!processor->is_ready(processor, next_group))
			goto dequeue_op;

		ret = dla_program_operation(processor, next_group);
		if (ret)
			goto exit;

		if (next_group->op_desc->dependency_count != 0)
			goto dequeue_op;

		ret = dla_enable_operation(processor,
					   next_group->op_desc);
		if (ret)
			goto exit;
	}

dequeue_op:
	/* dequeue operation from this processor */
	ret = dla_dequeue_operation(engine, processor);

exit:
	dla_put_op_desc(op_desc);
	dla_debug("Exit:%s processor %s group%u status=%d\n",
				__func__, processor->name,
				group->id, ret);

	RETURN(ret);
}

/**
 * Read network configuration from DRAM, network descriptor address
 * is always first in the address list. Network configuration contains
 * offset in address list for addresses of other lists used to
 * execute network
 *
 * @engine: Engine instance
 * @return: 0 for success
 */
static int
dla_read_network_config(struct dla_engine *engine)
{
	int32_t ret;
	uint64_t network_addr;
	struct dla_task *task = engine->task;

	dla_debug("Enter:%s\n", __func__);

	/**
	 * Read address list from DRAM to DMEM
	 */
	ret = dla_read_address_list(engine);
	if (ret) {
		dla_error("Failed to read address list");
		goto exit;
	}

	/**
	 * Read network descriptor address from address list. It is always
	 * at index 0.
	 */
	ret = dla_get_dma_address(engine->driver_context, task->task_data,
						0, (void *)&network_addr,
						DESTINATION_PROCESSOR);
	if (ret) {
		dla_error("Failed to read network desc address");
		goto exit;
	}

	/**
	 * Read network descriptor, it has information for a network
	 * such as all address indexes.
	 */
	ret = dla_data_read(engine->driver_context, task->task_data,
				network_addr, (void *)&network,
				sizeof(struct dla_network_desc),
				0);
	if (ret) {
		dla_error("Failed to read network descriptor");
		goto exit;
	}

	dla_debug_network_desc(&network);

	if (network.num_operations == 0)
		goto exit;

	/**
	 * Read operation descriptor list address from address list
	 */
	ret = dla_get_dma_address(engine->driver_context, task->task_data,
				network.operation_desc_index,
				(void *)&task->operation_desc_addr,
				DESTINATION_PROCESSOR);
	if (ret) {
		dla_error("Failed to read operation desc list address");
		goto exit;
	}

	/**
	 * Read surface descriptor list address from address list
	 */
	ret = dla_get_dma_address(engine->driver_context, task->task_data,
				network.surface_desc_index,
				(void *)&task->surface_desc_addr,
				DESTINATION_PROCESSOR);
	if (ret) {
		dla_error("Failed to read surface desc list address");
		goto exit;
	}

	/**
	 * Read dependency graph address from address list
	 */
	ret = dla_get_dma_address(engine->driver_context, task->task_data,
				network.dependency_graph_index,
				(void *)&task->dependency_graph_addr,
				DESTINATION_PROCESSOR);
	if (ret) {
		dla_error("Failed to ready dependency graph address");
		goto exit;
	}

	/**
	 * Read LUT data list address from address list
	 */
	if (network.num_luts) {
		ret = dla_get_dma_address(engine->driver_context,
					task->task_data,
					network.lut_data_index,
					(void *)&task->lut_data_addr,
					DESTINATION_PROCESSOR);
		if (ret) {
			dla_error("Failed to read LUT list address");
			goto exit;
		}
	}

	/**
	 * Read address for ROI information
	 */
	if (network.dynamic_roi) {
		/**
		 * Read ROI array address from address list
		 */
		ret = dla_get_dma_address(engine->driver_context,
					task->task_data,
					network.roi_array_index,
					(void *)&task->roi_array_addr,
					DESTINATION_PROCESSOR);
		if (ret) {
			dla_error("Failed to read ROI array address");
			goto exit;
		}

		ret = dla_data_read(engine->driver_context, task->task_data,
					task->roi_array_addr,
					(void *)&roi_array_length,
					sizeof(uint64_t),
					0);
		if (ret) {
			dla_error("Failed to read ROI array length");
			goto exit;
		}

		/**
		 * Number of ROIs detected can't be greater than maximum number
		 * ROIs this network can process
		 */
		if (roi_array_length > network.num_rois) {
			dla_error("Invalid number of ROIs detected");
			ret = ERR(INVALID_INPUT);
			goto exit;
		}

		network.num_rois = roi_array_length;

		/**
		 * Read surface address from address list
		 */
		ret = dla_get_dma_address(engine->driver_context,
						task->task_data,
						network.surface_index,
						(void *)&task->surface_addr,
						DESTINATION_DMA);
		if (ret) {
			dla_error("Failed to read surface address");
			goto exit;
		}
	}

#if STAT_ENABLE
	if (network.stat_list_index != -1) {
		ret = dla_get_dma_address(engine->driver_context,
						task->task_data,
						network.stat_list_index,
						(void *)&task->stat_data_addr,
						DESTINATION_PROCESSOR);
		if (ret) {
			dla_error("Failed to read stat address");
			goto exit;
		}
	}
#endif /* STAT_ENABLE */

exit:
	dla_debug("Exit:%s status=%d\n", __func__, ret);
	RETURN(ret);
}

static int
dla_initiate_processors(struct dla_engine *engine)
{
	int32_t i;
	int32_t ret = 0;
	int16_t index;
	struct dla_processor *processor;
	struct dla_common_op_desc *consumer;
	struct dla_network_desc *nw;

	dla_debug("Enter: %s\n", __func__);

	if (!engine) {
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	nw = engine->network;

	/* Validate operation heads before initiating processors */
	for (i = 0; i < DLA_OP_NUM; i++) {
		if (nw->op_head[i] >= nw->num_operations) {
			ret = ERR(INVALID_INPUT);
			dla_error("Invalid op_head %d for op %d",
						nw->op_head[i], i);
			goto exit;
		}
	}

	for (i = 0; i < DLA_OP_NUM; i++) {
		index = nw->op_head[i];

		/* If there is no op for this type then continue */
		if (-1 == index)
			continue;

		consumer = dla_get_op_desc(engine->task, index, i, 0);
		/*
		 * if consumer is NULL, it means either data copy error
		 * or cache insufficient - we should fix it
		 **/
		if (consumer == NULL) {
			dla_error("Failed to allocate memory for op_head[%d]=%d",
							i, index);
			ret = ERR(NO_MEM);
			goto exit;
		}

		processor = &engine->processors[consumer->op_type];

		ret = dla_submit_operation(processor, consumer, 0);
		dla_put_op_desc(consumer);
		if (ret && ret != ERR(PROCESSOR_BUSY)) {
			dla_error("Failed to submit %s op from index %u\n",
						processor->name, index);
			goto exit;
		}

		ret = dla_dequeue_operation(engine, processor);
		if (ret) {
			dla_error("Failed to dequeue op for %s processor",
							processor->name);
			goto exit;
		}
	}
exit:
	dla_debug("Exit: %s status=%d\n", __func__, ret);
	RETURN(ret);
}

static int
dla_handle_events(struct dla_processor *processor)
{
	int32_t j;
	int32_t ret = 0;
	uint8_t group_id;
	struct dla_processor_group *group;

	dla_debug("Enter:%s, processor:%s\n", __func__, processor->name);

	group_id = !processor->last_group;

	for (j = 0; j < DLA_NUM_GROUPS; j++) {
		group = &processor->groups[group_id];

		if ((1 << DLA_EVENT_CDMA_WT_DONE) & group->events) {
			dla_info("Handle cdma weight done event, processor %s "
				"group %u\n", processor->name, group->id);

			ret = dla_update_consumers(group,
						   group->op_desc,
						   DLA_EVENT_CDMA_WT_DONE);
			if (ret)
				goto exit;
		}

		if ((1 << DLA_EVENT_CDMA_DT_DONE) & group->events) {
			dla_info("Handle cdma data done event, processor %s "
				"group %u\n", processor->name, group->id);

			ret = dla_update_consumers(group,
						   group->op_desc,
						   DLA_EVENT_CDMA_DT_DONE);
			if (ret)
				goto exit;
		}

		/**
		 * Handle complete after all other events
		 */
		if ((1 << DLA_EVENT_OP_COMPLETED) & group->events) {
			dla_info("Handle op complete event, processor %s "
				"group %u\n", processor->name, group->id);

			ret = dla_op_completion(processor, group);
			if (ret)
				goto exit;
		}

		/**
		 * Clear all events
		 */
		group->events = 0;
		group_id = !group_id;
	}
exit:
	dla_debug("Exit:%s, ret:%x\n", __func__, ret);
	RETURN(ret);
}

int
dla_process_events(void *engine_context, uint32_t *task_complete)
{
	int32_t i;
	int32_t ret = 0;
	struct dla_engine *engine = (struct dla_engine *)engine_context;

	for (i = 0; i < DLA_OP_NUM; i++) {
		struct dla_processor *processor;

		processor = &engine->processors[i];
		ret = dla_handle_events(processor);
		/**
		 * Incase engine status is non-zero, then don't
		 * update the engine status. We should keep its
		 * status for later cleaning of engine.
		 */
		if (!engine->status)
			engine->status = ret;
	}

	if (engine->network->num_operations == engine->num_proc_hwl)
		*task_complete = 1;

	RETURN(ret);
}

/**
 * Execute task selected by task scheduler
 *
 * 1. Read network configuration for the task
 * 2. Initiate processors with head of list for same op
 * 3. Start processing events received
 */
int
dla_execute_task(void *engine_context, void *task_data, void *config_data)
{
	int32_t ret;
	struct dla_engine *engine = (struct dla_engine *)engine_context;

	if (engine == NULL) {
		dla_error("engine is NULL\n");
		ret = ERR(INVALID_INPUT);
		goto complete;
	}

	if (engine->task == NULL) {
		dla_error("task is NULL\n");
		ret = ERR(INVALID_INPUT);
		goto complete;
	}

	if (engine->task->task_data != NULL) {
		/* We have on the fly tasks running */
		dla_warn("Already some task in progress");
		ret = ERR(PROCESSOR_BUSY);
		goto complete;
	}

	engine->task->task_data = task_data;
	engine->config_data = config_data;
	engine->network = &network;
	engine->num_proc_hwl = 0;
	engine->stat_enable = 0;

	LOG_EVENT(0, 0, 0, LOG_TASK_START);

	ret = dla_read_network_config(engine);
	if (ret)
		goto complete;

	dla_debug_address_info(engine->task);

	/**
	 * If no operations in a task means nothing to do, NULL task
	 */
	if (engine->network->num_operations == 0)
		goto complete;

#if STAT_ENABLE
	if (network.stat_list_index != -1)
		engine->stat_enable = 1;
#endif /* STAT_ENABLE */

	ret = dla_initiate_processors(engine);
	engine->status = ret;

complete:
	LOG_EVENT(0, 0, 0, LOG_TASK_END);

	RETURN(ret);
}

void
dla_clear_task(void *engine_context)
{
	int32_t i, j;
	struct dla_engine *engine = (struct dla_engine *)engine_context;

	for (i = 0; i < DLA_OP_NUM; i++) {
		struct dla_processor *processor = &engine->processors[i];

		processor->roi_index = 0;
		processor->group_status = 0;
		processor->rdma_status = 0;

		processor->tail_op = NULL;

		for (j = 0; j < DLA_NUM_GROUPS; j++) {
			struct dla_processor_group *group =
						&processor->groups[j];

			group->rdma_id = group->id;
			group->active = 0;
			group->events = 0;
			group->roi_index = 0;
			group->is_rdma_needed = 0;
			group->lut_index = -1;
		}
	}

	engine->task->task_data = NULL;
	engine->network = NULL;
	engine->num_proc_hwl = 0;
	engine->status = 0;
	engine->stat_enable = 0;

	dla_info("reset engine done\n");
}
