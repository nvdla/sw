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

#ifndef __FIRMWARE_ENGINE_DEBUG_H_
#define __FIRMWARE_ENGINE_DEBUG_H_

#include <dla_debug.h>
#include <dla_interface.h>

#if DEBUG_NETWORK_DATA
void
dla_debug_op_desc(struct dla_common_op_desc *desc, int32_t roi);
void
dla_debug_network_desc(struct dla_network_desc *network_desc);
void
dla_debug_address_info(struct dla_task *task);
void
dla_debug_bdma_surface_desc(struct dla_bdma_surface_desc *desc, int32_t roi);
void
dla_debug_bdma_op_desc(struct dla_bdma_op_desc *desc, int32_t roi);
void
dla_debug_bdma_stats(struct dla_bdma_stat_desc *stat);
void
dla_debug_conv_surface_desc(struct dla_conv_surface_desc *desc, int32_t roi);
void
dla_debug_conv_op_desc(struct dla_conv_op_desc *desc, int32_t roi);
void
dla_debug_conv_stats(struct dla_conv_stat_desc *stat);
void
dla_debug_sdp_op_desc(struct dla_sdp_op_desc *desc, int32_t roi);
void
dla_debug_sdp_surface_desc(struct dla_sdp_surface_desc *desc, int32_t roi);
void
dla_debug_sdp_stats(struct dla_sdp_stat_desc *stat);
void
dla_debug_pdp_surface_desc(struct dla_pdp_surface_desc *desc, int32_t roi);
void
dla_debug_pdp_op_desc(struct dla_pdp_op_desc *desc, int32_t roi);
void
dla_debug_pdp_stats(struct dla_pdp_stat_desc *stat);
void
dla_debug_cdp_surface_desc(struct dla_cdp_surface_desc *desc, int32_t roi);
void
dla_debug_cdp_op_desc(struct dla_cdp_op_desc *desc, int32_t roi);
void
dla_debug_cdp_stats(struct dla_cdp_stat_desc *stat);
void
dla_debug_rubik_op_desc(struct dla_rubik_op_desc *desc, int32_t roi);
void
dla_debug_rubik_surface_desc(struct dla_rubik_surface_desc *desc, int32_t roi);
void
dla_debug_rubik_stats(struct dla_rubik_stat_desc *stat);
void
dla_debug_lut_params(struct dla_lut_param *lut_param);

#else

static inline void
dla_debug_op_desc(struct dla_common_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_network_desc(struct dla_network_desc *network_desc) {}
static inline void
dla_debug_address_info(struct dla_task *task) {}
static inline void
dla_debug_bdma_surface_desc(struct dla_bdma_surface_desc *desc, int32_t roi) {}
static inline void
dla_debug_bdma_op_desc(struct dla_bdma_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_bdma_stats(struct dla_bdma_stat_desc *stat) {}
static inline void
dla_debug_conv_surface_desc(struct dla_conv_surface_desc *desc, int32_t roi) {}
static inline void
dla_debug_conv_op_desc(struct dla_conv_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_conv_stats(struct dla_conv_stat_desc *stat) {}
static inline void
dla_debug_sdp_op_desc(struct dla_sdp_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_sdp_surface_desc(struct dla_sdp_surface_desc *desc, int32_t roi) {}
static inline void
dla_debug_sdp_stats(struct dla_sdp_stat_desc *stat) {}
static inline void
dla_debug_pdp_surface_desc(struct dla_pdp_surface_desc *desc, int32_t roi) {}
static inline void
dla_debug_pdp_op_desc(struct dla_pdp_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_pdp_stats(struct dla_pdp_stat_desc *stat) {}
static inline void
dla_debug_cdp_surface_desc(struct dla_cdp_surface_desc *desc, int32_t roi) {}
static inline void
dla_debug_cdp_op_desc(struct dla_cdp_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_cdp_stats(struct dla_cdp_stat_desc *stat) {}
static inline void
dla_debug_rubik_op_desc(struct dla_rubik_op_desc *desc, int32_t roi) {}
static inline void
dla_debug_rubik_surface_desc(struct dla_rubik_surface_desc *desc, int32_t roi) {}
static inline void
dla_debug_rubik_stats(struct dla_rubik_stat_desc *stat) {}
static inline void
dla_debug_lut_params(struct dla_lut_param *lut_param) {}

#endif /* DEBUG_NETWORK_DATA */
#endif /* __FIRMWARE_ENGINE_DEBUG_H_ */
