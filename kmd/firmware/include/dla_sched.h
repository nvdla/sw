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

#ifndef __DLA_SCHED_H_
#define __DLA_SCHED_H_

struct dla_task {
	/* platform specific data to communicate with portability layer */
	void *task_data;
	/* task state */
	uint32_t state;
	/* Task base address */
	uint64_t base;
	/* start address of a list of dla_operation_container */
	uint64_t operation_desc_addr;
	/* start address of a list of dla_surface_container */
	uint64_t surface_desc_addr;
	/* start address of a list of dla_common_op_desc */
	uint64_t dependency_graph_addr;
	/* start address of a list of dla_lut_param */
	uint64_t lut_data_addr;
	/*
	 * start address of a list of dla_roi_desc,
	 * the first one is dla_roi_array_desc
	 * valid when network.dynamic_roi is true
	 */
	uint64_t roi_array_addr;
	/* start address of a list of dla_surface_container */
	uint64_t surface_addr;
	/* start address of a list of dla_stat_container */
	uint64_t stat_data_addr;
} __packed __aligned(256);

#endif
