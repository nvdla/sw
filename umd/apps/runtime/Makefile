# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

ROOT := $(TOP)

TOOLCHAIN_PREFIX ?=

ifeq ($(TOOLCHAIN_PREFIX),)
$(error Toolchain prefix missing)
endif

MODULE := nvdla_runtime

include $(ROOT)/make/macros.mk

BUILDOUT ?= $(ROOT)/out/apps/runtime
BUILDDIR := $(BUILDOUT)/$(MODULE)
TEST_BIN := $(BUILDDIR)/$(MODULE)

INCLUDES :=
MODULE_COMPILEFLAGS := -W -Wall -Wno-multichar -Wno-unused-parameter -Wno-unused-function -Werror-implicit-function-declaration
MODULE_CFLAGS := --std=c99
MODULE_CPPFLAGS := --std=c++11 -fexceptions -fno-rtti

all:: $(TEST_BIN)

include rules.mk

# the logic to compile and link stuff is in here
$(TEST_BIN): $(ALLMODULE_OBJS) $(SHARED_LIBS)
	@echo building $(MODULE)  $@
	$(TOOLCHAIN_PREFIX)g++ $(ALLMODULE_OBJS) -pthread -L$(ROOT)/external/ -ljpeg -L$(ROOT)/out/core/src/runtime/libnvdla_runtime -lnvdla_runtime -o $@ -Wl,-rpath=.
