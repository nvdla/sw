# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# nvdla_compiler
#

LOCAL_DIR := $(GET_LOCAL_DIR)

MODULE_CC := gcc
MODULE_CPP := g++
MODULE_LD := ld

NVDLA_SRC_FILES := \
    main.cpp \
    CompileTest.cpp \
    GenerateTest.cpp \
    ParseTest.cpp \

INCLUDES += \
    -I$(ROOT)/include \
    -I$(ROOT)/port/linux \
    -I$(ROOT)/external/include \
    -I$(ROOT)/core/include \
    -I$(LOCAL_DIR)

MODULE_CPPFLAGS += \
    -DNVDLA_UTILS_ERROR_TAG="\"DLA\""
MODULE_CFLAGS += \
    -DNVDLA_UTILS_ERROR_TAG="\"DLA\""

SHARED_LIBS := \
    $(ROOT)/out/core/src/compiler/libnvdla_compiler/libnvdla_compiler.so

MODULE_SRCS := $(NVDLA_SRC_FILES)

include $(ROOT)/make/module.mk
