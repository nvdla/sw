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

#include "dla_engine_internal.h"
#include "engine_debug.h"

static const uint8_t map_mem[] = {
	FIELD_ENUM(BDMA_CFG_CMD_0, SRC_RAM_TYPE, MC),
	FIELD_ENUM(BDMA_CFG_CMD_0, SRC_RAM_TYPE, CVSRAM),
};

#if STAT_ENABLE
void
dla_bdma_stat_data(struct dla_processor *processor,
					struct dla_processor_group *group)
{
	uint64_t end_time = 0;
	struct dla_bdma_stat_desc *bdma_stat;

	bdma_stat = &processor->stat_data_desc->bdma_stat;

	end_time = dla_get_time_us();

	if (group->id == (uint32_t)0) {
		bdma_stat->read_stall = bdma_reg_read(STATUS_GRP0_READ_STALL);
		bdma_stat->write_stall = bdma_reg_read(STATUS_GRP0_WRITE_STALL);
	} else {
		bdma_stat->read_stall = bdma_reg_read(STATUS_GRP1_READ_STALL);
		bdma_stat->write_stall = bdma_reg_read(STATUS_GRP1_WRITE_STALL);
	}
	bdma_stat->runtime = (uint32_t)(end_time - group->start_time);
}

void
dla_bdma_dump_stat(struct dla_processor *processor)
{
	struct dla_bdma_stat_desc *bdma_stat;

	bdma_stat = &processor->stat_data_desc->bdma_stat;

	dla_debug_bdma_stats(bdma_stat);
}
#endif /* STAT_ENABLE */

void
dla_bdma_set_producer(int32_t group_id, int32_t rdma_group_id)
{
	/**
	 * There is no producer bit for BDMA operation,
	 * interrupt pointer decides which outstanding request
	 * to use for this BDMA operation
	 */
}

int
dla_bdma_enable(struct dla_processor_group *group)
{
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s\n", __func__);

	if (group->surface_desc->bdma_surface.num_transfers == (uint16_t)0) {
		group->events |= ((uint8_t)1 << DLA_EVENT_OP_COMPLETED);
		goto exit;
	}

	if (engine->stat_enable == (uint32_t)1) {
		bdma_reg_write(CFG_STATUS, FIELD_ENUM(BDMA_CFG_STATUS_0,
							STALL_COUNT_EN, YES));
		group->start_time = dla_get_time_us();
	}

	/**
	 * Launch BDMA transfer
	 */
	if (group->id == 0)
		bdma_reg_write(CFG_LAUNCH0, FIELD_ENUM(BDMA_CFG_LAUNCH0_0,
							GRP0_LAUNCH, YES));
	else
		bdma_reg_write(CFG_LAUNCH1, FIELD_ENUM(BDMA_CFG_LAUNCH1_0,
							GRP1_LAUNCH, YES));

exit:
	dla_debug("Exit: %s\n", __func__);
	return 0;
}

void
dla_bdma_rdma_check(struct dla_processor_group *group)
{
	group->is_rdma_needed = 0;
}

/**
 * Program BDMA slot for transfer
 */
static int32_t
processor_bdma_program_slot(struct dla_bdma_surface_desc *bdma_surface,
				struct dla_bdma_transfer_desc *transfer)
{
	int32_t ret = 0;
	uint64_t source_addr = 0;
	uint64_t destination_addr = 0;
	uint32_t high, low, reg;
	uint8_t  bdma_free_slots = 0;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s\n", __func__);

	/* make sure there're enough free slots */
	if (bdma_free_slots <= 0) {
		do {
			reg = bdma_reg_read(STATUS);
			reg = (reg & MASK(BDMA_STATUS_0, FREE_SLOT)) >>
					SHIFT(BDMA_STATUS_0, FREE_SLOT);
		} while (reg == 0);
		bdma_free_slots = (uint8_t)reg;
	}

	dla_get_dma_address(engine->driver_context, engine->task->task_data,
						transfer->source_address,
						(void *)&source_addr,
						DESTINATION_DMA);
	dla_get_dma_address(engine->driver_context, engine->task->task_data,
						transfer->destination_address,
						(void *)&destination_addr,
						DESTINATION_DMA);

	ASSERT_GOTO((transfer->line_repeat <= 8192),
				ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO((transfer->surface_repeat <= 8192),
				ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO((transfer->line_size % 32) == 0,
				ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO(transfer->source_line >= transfer->line_size,
				ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO(transfer->destination_line >= transfer->line_size,
				ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO(transfer->source_surface >=
			(transfer->source_line * transfer->line_repeat),
				ret, ERR(INVALID_INPUT), exit);
	ASSERT_GOTO(transfer->destination_surface >=
			(transfer->destination_line * transfer->line_repeat),
				ret, ERR(INVALID_INPUT), exit);

	/* config registers */
	high = HIGH32BITS(source_addr);
	low = LOW32BITS(source_addr);
	bdma_reg_write(CFG_SRC_ADDR_LOW, low);
	bdma_reg_write(CFG_SRC_ADDR_HIGH, high);
	high = HIGH32BITS(destination_addr);
	low = LOW32BITS(destination_addr);
	bdma_reg_write(CFG_DST_ADDR_LOW, low);
	bdma_reg_write(CFG_DST_ADDR_HIGH, high);
	bdma_reg_write(CFG_LINE, (transfer->line_size >> 5) - 1);
	reg = (map_mem[bdma_surface->source_type] <<
				SHIFT(BDMA_CFG_CMD_0, SRC_RAM_TYPE)) |
		(map_mem[bdma_surface->destination_type] <<
				SHIFT(BDMA_CFG_CMD_0, DST_RAM_TYPE));
	bdma_reg_write(CFG_CMD, reg);
	bdma_reg_write(CFG_LINE_REPEAT, transfer->line_repeat - 1);
	bdma_reg_write(CFG_SRC_LINE, transfer->source_line);
	bdma_reg_write(CFG_DST_LINE, transfer->destination_line);
	bdma_reg_write(CFG_SURF_REPEAT, transfer->surface_repeat - 1);
	bdma_reg_write(CFG_SRC_SURF, transfer->source_surface);
	bdma_reg_write(CFG_DST_SURF, transfer->destination_surface);
	bdma_reg_write(CFG_OP, FIELD_ENUM(BDMA_CFG_OP_0, EN, ENABLE));

	dla_debug("Exit: %s\n", __func__);

exit:
	RETURN(ret);
}

int
dla_bdma_is_ready(struct dla_processor *processor,
			    struct dla_processor_group *group)
{
	struct dla_processor_group *next_group;

	next_group = &processor->groups[!group->id];

	/**
	 * If another group is already programmed but not active then
	 * do not program this operation as BDMA does not really
	 * have shadow copies for groups. It will end programming
	 * same group. Wait for another group to get enabled.
	 */
	if ((processor->group_status & (1 << next_group->id)) &&
						!next_group->active)
		return 0;

	return 1;
}

void
dla_bdma_dump_config(struct dla_processor_group *group)
{
	struct dla_bdma_op_desc *bdma_op;
	struct dla_bdma_surface_desc *bdma_surface;

	bdma_surface = &group->surface_desc->bdma_surface;
	bdma_op = &group->operation_desc->bdma_op;

	dla_debug_bdma_surface_desc(bdma_surface, group->roi_index);
	dla_debug_bdma_op_desc(bdma_op, group->roi_index);
}

int
dla_bdma_program(struct dla_processor_group *group)
{
	int32_t i;
	int32_t ret = 0;
	struct dla_bdma_surface_desc *bdma_surface;
	struct dla_engine *engine = dla_get_engine();

	dla_debug("Enter: %s\n", __func__);

	if (!engine->config_data->bdma_enable) {
		dla_error("BDMA is not supported for this configuration\n");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	bdma_surface = &group->surface_desc->bdma_surface;

	dla_debug("Num of transfers %u\n", bdma_surface->num_transfers);
	if (bdma_surface->num_transfers == (uint16_t)0)
		goto exit;

	if (bdma_surface->num_transfers > NUM_MAX_BDMA_OPS) {
		dla_error("Invalid number of transfers\n");
		ret = ERR(INVALID_INPUT);
		goto exit;
	}

	for (i = 0; i < bdma_surface->num_transfers; i++) {
		ret = processor_bdma_program_slot(bdma_surface,
					&bdma_surface->transfers[i]);
		if (ret)
			goto exit;
	}

	dla_enable_intr(MASK(GLB_S_INTR_MASK_0, BDMA_DONE_MASK1) |
			MASK(GLB_S_INTR_MASK_0, BDMA_DONE_MASK0));

exit:
	dla_debug("Exit: %s\n", __func__);
	RETURN(ret);
}
