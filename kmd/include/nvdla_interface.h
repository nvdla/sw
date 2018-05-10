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

#ifndef __NVDLA_INTERFACE_H_
#define __NVDLA_INTERFACE_H_

#include <linux/types.h>

/**
 * @brief			Register driver to firmware
 *
 * Implementation in firmware, called by portability layer
 *
 * This function must be called once during boot to initialize DLA
 * engine scheduler and register driver with firmware before submitting
 * any task. Pass pointer to driver context in @param driver_context
 * which is passed as param when firmware calls any function
 * of portability layer. It also updates pointer to engine context
 * which must be passed in any function call to firmware after this point.
 *
 * @param engine_context	Pointer to engine specific data
 * @param driver_context	Pointer to driver specific data
 *
 * @return			0 on success and negative on error
 */
int32_t dla_register_driver(void **engine_context, void *driver_context);

/**
 * @brief			Interrupt handler
 *
 * Implementation in firmware, called by portability layer
 *
 * This function is called when DLA interrupt is received. Portability layer
 * should register it's own handler using the mechanism supported by that platform
 * and call this function from the handler. Call to this function must be
 * protected by lock to prevent handling interrupt when firmware is programming
 * layers in process context.
 *
 * @param engine_context	Engine specific data received in dla_register_driver
 *
 * @return			0 on success and negative on error
 */
int32_t dla_isr_handler(void *engine_context);

/**
 * @brief			Process events recorded in interrupt handler
 *
 * Implementation in firmware, called by portability layer
 *
 * Interrupt handler just records events and does not process those events.
 * Portability layer must call this function in thread/process context after
 * interrupt handler is done.
 *
 * @param engine_context	Engine specific data received in dla_register_driver
 * @param task_complete		Pointer to parameter to indicate task complete,
				firmare writes 1 to it if all layers are processed.
 *
 * @return			0 on success and negative on error
 *
 */
int32_t dla_process_events(void *engine_context, uint32_t *task_complete);

/**
 * @brief			Clear task from firmware
 *
 * Implementation in firmware, called by portability layer
 *
 * This function resets engine scheduler state including op descriptor cache,
 * error values, sub-engine status, events etc and clears previous task state
 * from firmware. This function can be called by portability layer after
 * task completion. It is not mandatory to call it but calling it will
 * ensure clean state before next task execution.
 *
 * @param engine_context	Engine specific data received in dla_register_driver
 *
 * @return			0 on success and negative on error
 *
 */
void dla_clear_task(void *engine_context);

/**
 * @brief			Execute task
 *
 * Implementation in firmware, called by portability layer
 *
 * This function initializes sub-engines and starts task execution. Further
 * programming and layer scheduling is triggered by events received from
 * hardware.
 *
 * @param engine_context	Engine specific data received in dla_register_driver
 * @param task_data		Task specific data to be passed when reading task info
 * @param config_data		Configuration data to be passed
 *
 * @return			0 on success and negative on error
 *
 */
int32_t dla_execute_task(void *engine_context, void *task_data, void *config_data);

/**
 * @brief			Register read
 *
 * Implementation in portability layer, called by firmware
 *
 * Read DLA HW register. Portability layer is responsible to use correct
 * base address and for any IO mapping if required.
 *
 * @param engine_context	Driver specific data received in dla_register_driver
 * @param addr			Register offset
 *
 * @return			Register value
 *
 */
uint32_t dla_reg_read(void *driver_context, uint32_t addr);

/**
 * @brief			Register write
 *
 * Implementation in portability layer, called by firmware
 *
 * Write DLA HW registr. Portability layer is responsible to use correct
 * base address and for any IO mapping if required.
 *
 * @param driver_context	Driver specific data received in dla_register_driver
 * @param addr			Register offset
 * @param reg			Value to write
 *
 */
void dla_reg_write(void *driver_context, uint32_t addr, uint32_t reg);

/**
 * @brief			Read data from DMA mapped memory in local buffer
 *
 * Implementation in portability layer, called by firmware
 *
 * This function reads data from buffers passed by UMD in local memory.
 * Addresses for buffers passed by are shared in address list and network
 * descriptor contains index in address list for those buffers. Firmware
 * reads this data from buffer shared by UMD into local buffer to consume
 * the information.
 *
 * @param driver_context	Driver specific data received in dla_register_driver
 * @param task_data		Task specific data received in dla_execute_task
 * @param src			Index in address list
 * @param dst			Pointer to local memory
 * @param size			Size of data to copy
 * @param offset		Offset from start of UMD buffer
 *
 * @return			0 on success and negative on error
 *
 */
int32_t dla_data_read(void *driver_context, void *task_data,
				uint64_t src, void *dst,
				uint32_t size, uint64_t offset);

/**
 * @brief			Write data to DMA mapped memory from local buffer
 *
 * Implementation in portability layer, called by firmware
 *
 * This function writes data from local buffer to buffer passed by UMD.
 * Addresses for buffers passed by are shared in address list and network
 * descriptor contains index in address list for those buffers. Firmware
 * writes this data to buffer shared by UMD from local buffer to update
 * the information.
 *
 * @param driver_context	Driver specific data received in dla_register_driver
 * @param task_data		Task specific data received in dla_execute_task
 * @param src			Pointer to local memory
 * @param dst			Index in address list
 * @param size			Size of data to copy
 * @param offset		Offset from start of UMD buffer
 *
 * @return			0 on success and negative on error
 *
 */
int32_t dla_data_write(void *driver_context, void *task_data,
				void *src, uint64_t dst,
				uint32_t size, uint64_t offset);

/* Destination for DMA buffer */
#define DESTINATION_PROCESSOR	0
#define DESTINATION_DMA		1

/**
 * @brief			Read DMA address
 *
 * Implementation in portability layer, called by firmware
 *
 * Some buffers shared by UMD are accessed by processor responsible for
 * programming DLA HW. It would be companion micro-controller in case of
 * headed config while main CPU in case of headless config. Also, some
 * buffers are accessed by DLA DMA engines inside sub-engines. This function
 * should return proper address accessible by destination user depending
 * on config.
 *
 * @param driver_context	Driver specific data received in dla_register_driver
 * @param task_data		Task specific data received in dla_execute_task
 * @param index			Index in address list
 * @param dst_ptr		Pointer to update address
 * @param destination		Destination user for DMA address
 *
 * @return			0 on success and negative on error
 *
 */
int32_t dla_get_dma_address(void *driver_context, void *task_data,
					int16_t index, void *dst_ptr,
					uint32_t destination);

/**
 * @brief			Read time value in micro-seconds
 *
 * Implementation in portability layer, called by firmware
 *
 * Read system time in micro-seconds
 *
 * @return			Time value in micro-seconds
 *
 */
int64_t dla_get_time_us(void);

/**
 * @brief			Print debug message
 *
 * Implementation in portability layer, called by firmware
 *
 * Print debug message to console
 *
 * @param str			Format string and variable arguments
 *
 */
void dla_debug(const char *str, ...);

/**
 * @brief			Print information message
 *
 * Implementation in portability layer, called by firmware
 *
 * Print information message to console
 *
 * @param str			Format string and variable arguments
 *
 */
void dla_info(const char *str, ...);

/**
 * @brief			Print warning message
 *
 * Implementation in portability layer, called by firmware
 *
 * Print warning message to console
 *
 * @param str			Format string and variable arguments
 *
 */
void dla_warn(const char *str, ...);

/**
 * @brief			Print error message
 *
 * Implementation in portability layer, called by firmware
 *
 * Print error message to console
 *
 * @param str			Format string and variable arguments
 *
 */
void dla_error(const char *str, ...);

/**
 * @brief			Fill memory region
 *
 * Implementation in portability layer, called by firmware
 *
 * Fills the first len bytes of the memory area pointed to by src
 * with the constant byte ch.
 *
 * @param src			Memory area address
 * @param ch			Byte to fill
 * @param len			Length of memory area to fill
 *
 * @return			Memory area address
 *
 */
void *dla_memset(void *src, int ch, uint64_t len);

/**
 * @brief			Copy memory
 *
 * Implementation in portability layer, called by firmware
 *
 * Copies len bytes from memory area src to memory area dest.
 *
 * @param dest			Destination memory area address
 * @param src			Source memory area address
 * @param len			Length of memory area to copy
 *
 * @return			Destination memory area address
 *
 */
void *dla_memcpy(void *dest, const void *src, uint64_t len);

#endif
