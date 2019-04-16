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

#ifndef __FIRMWARE_DLA_ENGINE_INTERNAL_H_
#define __FIRMWARE_DLA_ENGINE_INTERNAL_H_

#include <opendla.h>
#include <dla_engine.h>
#include <dla_interface.h>
#include <dla_debug.h>

#include "nvdla_interface.h"

#define BITS(num, range) ((((0xFFFFFFFF >> (31 - (1 ? range))) & \
			(0xFFFFFFFF << (0 ? range))) & num) >> \
			(0 ? range))
#define HIGH32BITS(val64bit) ((uint32_t)(val64bit >> 32))
#define LOW32BITS(val64bit) ((uint32_t)(val64bit))

#ifdef MIN
#undef MIN
#endif /* MIN */

#ifdef MAX
#undef MAX
#endif /* MAX */

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*********************************************************/
/******************** Utilities **************************/
/*********************************************************/
#ifdef DEBUG
#define CHECK_ALIGN(val, align)		 assert((val&(align-1)) == 0)
#else
#define CHECK_ALIGN(val, align)
#endif /* DEBUG */

#define MASK(reg, field)		(reg##_##field##_FIELD)
#define FIELD_ENUM(r, f, e)		(r##_##f##_##e)
#define SHIFT(reg, field)		(reg##_##field##_SHIFT)

#define GLB_REG(name)                GLB_##name##_0
#define MCIF_REG(name)               MCIF_##name##_0
#define CVIF_REG(name)               CVIF_##name##_0
#define BDMA_REG(name)               BDMA_##name##_0
#define CDMA_REG(name)               CDMA_##name##_0
#define CSC_REG(name)                CSC_##name##_0
#define CMAC_A_REG(name)             CMAC_A_##name##_0
#define CMAC_B_REG(name)             CMAC_B_##name##_0
#define CACC_REG(name)               CACC_##name##_0
#define SDP_RDMA_REG(name)           SDP_RDMA_##name##_0
#define SDP_REG(name)                SDP_##name##_0
#define PDP_RDMA_REG(name)           PDP_RDMA_##name##_0
#define PDP_REG(name)                PDP_##name##_0
#define CDP_RDMA_REG(name)           CDP_RDMA_##name##_0
#define CDP_REG(name)                CDP_##name##_0
#define RBK_REG(name)                RBK_##name##_0

/* alias for register read for each sub-module */
#define glb_reg_read(reg)           reg_read(GLB_REG(reg))
#define bdma_reg_read(reg)          reg_read(BDMA_REG(reg))
#define cdma_reg_read(reg)          reg_read(CDMA_REG(reg))
#define csc_reg_read(reg)           reg_read(CSC_REG(reg))
#define cmac_a_reg_read(reg)        reg_read(CMAC_A_REG(reg))
#define cmac_b_reg_read(reg)        reg_read(CMAC_B_REG(reg))
#define cacc_reg_read(reg)          reg_read(CACC_REG(reg))
#define sdp_rdma_reg_read(reg)      reg_read(SDP_RDMA_REG(reg))
#define sdp_reg_read(reg)           reg_read(SDP_REG(reg))
#define pdp_rdma_reg_read(reg)      reg_read(PDP_RDMA_REG(reg))
#define pdp_reg_read(reg)           reg_read(PDP_REG(reg))
#define cdp_rdma_reg_read(reg)      reg_read(CDP_RDMA_REG(reg))
#define cdp_reg_read(reg)           reg_read(CDP_REG(reg))
#define rubik_reg_read(reg)         reg_read(RBK_REG(reg))

/* alias for register write for each sub-module */
#define glb_reg_write(reg, val)      reg_write(GLB_REG(reg), val)
#define bdma_reg_write(reg, val)     reg_write(BDMA_REG(reg), val)
#define cdma_reg_write(reg, val)     reg_write(CDMA_REG(reg), val)
#define csc_reg_write(reg, val)      reg_write(CSC_REG(reg), val)
#define cmac_a_reg_write(reg, val)   reg_write(CMAC_A_REG(reg), val)
#define cmac_b_reg_write(reg, val)   reg_write(CMAC_B_REG(reg), val)
#define cacc_reg_write(reg, val)     reg_write(CACC_REG(reg), val)
#define sdp_rdma_reg_write(reg, val) reg_write(SDP_RDMA_REG(reg), val)
#define sdp_reg_write(reg, val)      reg_write(SDP_REG(reg), val)
#define pdp_rdma_reg_write(reg, val) reg_write(PDP_RDMA_REG(reg), val)
#define pdp_reg_write(reg, val)      reg_write(PDP_REG(reg), val)
#define cdp_rdma_reg_write(reg, val) reg_write(CDP_RDMA_REG(reg), val)
#define cdp_reg_write(reg, val)      reg_write(CDP_REG(reg), val)
#define rubik_reg_write(reg, val)    reg_write(RBK_REG(reg), val)

void reg_write(uint32_t addr, uint32_t reg);
uint32_t reg_read(uint32_t addr);

/**
 * Operation descriptor cache functions
 */
void
dla_put_op_desc(struct dla_common_op_desc *op_desc);
struct dla_common_op_desc
*dla_get_op_desc(struct dla_task *task,
			   int16_t index,
			   uint8_t op_type,
			   uint8_t roi_index);
void
dla_dump_op_desc(struct dla_common_op_desc *desc);
void
dla_get_refcount(struct dla_common_op_desc *op_desc);
void
dla_init_op_cache(struct dla_engine *engine);

/**
 * Operation completion handler
 */
int
dla_op_completion(struct dla_processor *processor,
		      struct dla_processor_group *group);

int32_t
dla_read_lut(struct dla_engine *engine, int16_t index, void *dst);
int
dla_enable_intr(uint32_t mask);
int
dla_disable_intr(uint32_t mask);
int
utils_get_free_group(struct dla_processor *processor,
			uint8_t *group_id,
			uint8_t *rdma_id);
int32_t
dla_get_dma_cube_address(void *driver_context,
						void *task_data,
						int16_t index,
						uint32_t offset,
						void *dst_ptr,
						uint32_t destination);
int
dla_read_input_address(struct dla_data_cube *data,
		       uint64_t *address,
		       int16_t op_index,
		       uint8_t roi_index,
		       uint8_t bpp);

/**
 * BDMA operations
 */
void
dla_bdma_set_producer(int32_t group_id, int32_t rdma_group_id);
int
dla_bdma_enable(struct dla_processor_group *group);
int
dla_bdma_program(struct dla_processor_group *group);
int
dla_bdma_is_ready(struct dla_processor *processor,
			    struct dla_processor_group *group);
void
dla_bdma_dump_config(struct dla_processor_group *group);
void
dla_bdma_rdma_check(struct dla_processor_group *group);

#if STAT_ENABLE
void
dla_bdma_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group);
void
dla_bdma_dump_stat(struct dla_processor *processor);

#else
static inline void
dla_bdma_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group) {}
static inline void
dla_bdma_dump_stat(struct dla_processor *processor) {}
#endif

/**
 * Convolution operations
 */
void
dla_conv_set_producer(int32_t group_id, int32_t rdma_group_id);
int
dla_conv_enable(struct dla_processor_group *group);
int
dla_conv_program(struct dla_processor_group *group);
int
dla_conv_is_ready(struct dla_processor *processor,
			    struct dla_processor_group *group);
void
dla_conv_dump_config(struct dla_processor_group *group);
void
dla_conv_rdma_check(struct dla_processor_group *group);

#if STAT_ENABLE
void
dla_conv_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group);
void
dla_conv_dump_stat(struct dla_processor *processor);

#else
static inline void
dla_conv_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group) {}
static inline void
dla_conv_dump_stat(struct dla_processor *processor) {}
#endif /* STAT_ENABLE */

/**
 * SDP operations
 */
void
dla_sdp_set_producer(int32_t group_id, int32_t rdma_group_id);
int
dla_sdp_enable(struct dla_processor_group *group);
int
dla_sdp_program(struct dla_processor_group *group);
int
dla_sdp_is_ready(struct dla_processor *processor,
			   struct dla_processor_group *group);
void
dla_sdp_dump_config(struct dla_processor_group *group);
void
dla_sdp_rdma_check(struct dla_processor_group *group);

#if STAT_ENABLE
void
dla_sdp_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group);
void
dla_sdp_dump_stat(struct dla_processor *processor);

#else
static inline void
dla_sdp_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group) {}
static inline void
dla_sdp_dump_stat(struct dla_processor *processor) {}
#endif

/**
 * PDP operations
 */
void
dla_pdp_set_producer(int32_t group_id, int32_t rdma_group_id);
int
dla_pdp_enable(struct dla_processor_group *group);
int
dla_pdp_program(struct dla_processor_group *group);
int
dla_pdp_is_ready(struct dla_processor *processor,
			   struct dla_processor_group *group);
void
dla_pdp_dump_config(struct dla_processor_group *group);
void
dla_pdp_rdma_check(struct dla_processor_group *group);

#if STAT_ENABLE
void
dla_pdp_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group);
void
dla_pdp_dump_stat(struct dla_processor *processor);

#else
static inline void
dla_pdp_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group) {}
static inline void
dla_pdp_dump_stat(struct dla_processor *processor) {}
#endif

/**
 * CDP operations
 */
void
dla_cdp_set_producer(int32_t group_id, int32_t rdma_group_id);
int
dla_cdp_enable(struct dla_processor_group *group);
int
dla_cdp_program(struct dla_processor_group *group);
int
dla_cdp_is_ready(struct dla_processor *processor,
			   struct dla_processor_group *group);
void
dla_cdp_dump_config(struct dla_processor_group *group);
void
dla_cdp_rdma_check(struct dla_processor_group *group);

#if STAT_ENABLE
void
dla_cdp_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group);
void
dla_cdp_dump_stat(struct dla_processor *processor);

#else
static inline void
dla_cdp_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group) {}
static inline void
dla_cdp_dump_stat(struct dla_processor *processor) {}
#endif

/**
 * RUBIK operations
 */
void
dla_rubik_set_producer(int32_t group_id, int32_t rdma_group_id);
int
dla_rubik_enable(struct dla_processor_group *group);
int
dla_rubik_program(struct dla_processor_group *group);
int
dla_rubik_is_ready(struct dla_processor *processor,
			     struct dla_processor_group *group);
void
dla_rubik_dump_config(struct dla_processor_group *group);
void
dla_rubik_rdma_check(struct dla_processor_group *group);

#if STAT_ENABLE
void
dla_rubik_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group);
void
dla_rubik_dump_stat(struct dla_processor *processor);

#else
static inline void
dla_rubik_stat_data(struct dla_processor *processor,
				struct dla_processor_group *group) {}
static inline void
dla_rubik_dump_stat(struct dla_processor *processor) {}
#endif

#endif
