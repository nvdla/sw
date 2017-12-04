# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

MODULE_SRCDIR := $(MODULE)

# build with default toolchain/arch, unless module has overridden it
ifeq ($(MODULE_CC),)
MODULE_CC := $(TOOLCHAIN_PREFIX)gcc
endif
ifeq ($(MODULE_LD),)
MODULE_LD := $(TOOLCHAIN_PREFIX)ld
endif

$(info MODULE_CC $(MODULE_CC))

MODULE_DEFINES += MODULE_COMPILEFLAGS=\"$(subst $(SPACE),_,$(MODULE_COMPILEFLAGS))\"
MODULE_DEFINES += MODULE_CFLAGS=\"$(subst $(SPACE),_,$(MODULE_CFLAGS))\"
MODULE_DEFINES += MODULE_CPPFLAGS=\"$(subst $(SPACE),_,$(MODULE_CPPFLAGS))\"
MODULE_DEFINES += MODULE_LDFLAGS=\"$(subst $(SPACE),_,$(MODULE_LDFLAGS))\"
MODULE_DEFINES += MODULE_OPTFLAGS=\"$(subst $(SPACE),_,$(MODULE_OPTFLAGS))\"
MODULE_DEFINES += MODULE_INCLUDES=\"$(subst $(SPACE),_,$(MODULE_INCLUDES))\"

# include the rules to compile the module's object files
include $(ROOT)/make/compile.mk

# MODULE_OBJS is passed back from compile.mk
#$(info MODULE_OBJS = $(MODULE_OBJS))

# build a ld -r style combined object
MODULE_OBJECT := $(call TOBUILDDIR,$(MODULE_SRCDIR).mod.o)
$(MODULE_OBJECT): MODULE_LD:=$(MODULE_LD)
$(MODULE_OBJECT): $(MODULE_OBJS) $(MODULE_EXTRA_OBJS)
	@$(MKDIR)
	@echo linking $@
	$(ECHO)$(MODULE_LD) -r $^ -o $@

# track all the objects built
ALLOBJS += $(MODULE_OBJS)

# track the module object for make clean
GENERATED += $(MODULE_OBJECT)

ALLMODULE_OBJS := $(MODULE_OBJECT)

# empty out any vars set here
MODULE :=
MODULE_SRCDIR :=
MODULE_BUILDDIR :=
MODULE_DEPS :=
MODULE_SRCS :=
MODULE_OBJS :=
MODULE_DEFINES :=
MODULE_OPTFLAGS :=
MODULE_COMPILEFLAGS :=
MODULE_CFLAGS :=
MODULE_CPPFLAGS :=
MODULE_EXTRA_OBJS :=
MODULE_OBJECT :=
MODULE_CC :=
MODULE_LD :=
