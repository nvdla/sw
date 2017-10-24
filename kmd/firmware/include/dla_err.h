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

#ifndef __FIRMWARE_DLA_ERR_H_
#define __FIRMWARE_DLA_ERR_H_

#define ERR(code) -DLA_ERR_##code

#define DLA_ERR_NONE			0
#define DLA_ERR_INVALID_METHOD		1
#define DLA_ERR_INVALID_TASK		2
#define DLA_ERR_INVALID_INPUT		3
#define DLA_ERR_INVALID_FALC_DMA	4
#define DLA_ERR_INVALID_QUEUE		5
#define DLA_ERR_INVALID_PREACTION	6
#define DLA_ERR_INVALID_POSTACTION	7
#define DLA_ERR_NO_MEM			8
#define DLA_ERR_INVALID_DESC_VER	9
#define DLA_ERR_INVALID_ENGINE_ID	10
#define DLA_ERR_INVALID_REGION		11
#define DLA_ERR_PROCESSOR_BUSY		12
#define DLA_ERR_RETRY			13
#define DLA_ERR_TASK_STATUS_MISMATCH	14

#endif
