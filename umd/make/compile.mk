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


# create a separate list of objects per source type
MODULE_CSRCS := $(filter %.c,$(MODULE_SRCS))
MODULE_CPPSRCS := $(filter %.cpp,$(MODULE_SRCS))

MODULE_COBJS := $(call TOBUILDDIR,$(patsubst %.c,%.o,$(MODULE_CSRCS)))
MODULE_CPPOBJS := $(call TOBUILDDIR,$(patsubst %.cpp,%.o,$(MODULE_CPPSRCS)))

MODULE_OBJS := $(MODULE_COBJS) $(MODULE_CPPOBJS)

#$(info MODULE_SRCS = $(MODULE_SRCS))
#$(info MODULE_CSRCS = $(MODULE_CSRCS))
#$(info MODULE_CPPSRCS = $(MODULE_CPPSRCS))

#$(info MODULE_OBJS = $(MODULE_OBJS))
#$(info MODULE_COBJS = $(MODULE_COBJS))
#$(info MODULE_CPPOBJS = $(MODULE_CPPOBJS))

$(MODULE_OBJS): MODULE_CC:=$(MODULE_CC)
$(MODULE_OBJS): MODULE_OPTFLAGS:=$(MODULE_OPTFLAGS)
$(MODULE_OBJS): MODULE_COMPILEFLAGS:=$(MODULE_COMPILEFLAGS)
$(MODULE_OBJS): MODULE_CFLAGS:=$(MODULE_CFLAGS)
$(MODULE_OBJS): MODULE_CPPFLAGS:=$(MODULE_CPPFLAGS)
$(MODULE_OBJS): MODULE_SRCDEPS:=$(MODULE_SRCDEPS)
$(MODULE_OBJS): SRCDEPS:=$(SRCDEPS)

$(MODULE_OBJS): $(MODULE_SRCDEPS) $(SRCDEPS)

$(MODULE_COBJS): $(BUILDDIR)/%.o: %.c $(SRCDEPS)
	@$(MKDIR)
	@echo compiling $<
	$(MODULE_CC) $(MODULE_OPTFLAGS) $(MODULE_COMPILEFLAGS) $(MODULE_CFLAGS) $(MODULE_INCLUDES) $(INCLUDES) -c $< -MD -MT $@ -MF $(@:%o=%d) -o $@

$(MODULE_CPPOBJS): $(BUILDDIR)/%.o: %.cpp $(SRCDEPS)
	@$(MKDIR)
	@echo compiling $<
	$(MODULE_CC) $(MODULE_OPTFLAGS) $(MODULE_COMPILEFLAGS) $(MODULE_CPPFLAGS) $(INCLUDES) $(MODULE_INCLUDES) -c $< -MD -MT $@ -MF $(@:%o=%d) -o $@

# clear some variables we set here
MODULE_CSRCS :=
MODULE_CPPSRCS :=
MODULE_COBJS :=
MODULE_CPPOBJS :=
