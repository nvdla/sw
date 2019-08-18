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
# libnvdla_compiler
#

LOCAL_DIR := $(GET_LOCAL_DIR)

MODULE_CC := gcc
MODULE_CPP := g++
MODULE_LD := ld

NVDLA_SRC_FILES := \
    caffe/CaffeParser.cpp \
    caffe/ditcaffe/protobuf-2.6.1/ditcaffe.pb.cpp \
    engine-ast/ActivationOp.cpp \
    engine-ast/BatchNormOp.cpp \
    engine-ast/BDMANode.cpp \
    engine-ast/BDMASingleOp.cpp \
    engine-ast/BDMAGroupOp.cpp \
    engine-ast/BiasOp.cpp \
    engine-ast/CDPLRNOp.cpp \
    engine-ast/CDPNode.cpp \
    engine-ast/ConcatOp.cpp \
    engine-ast/ConvCoreNode.cpp \
    engine-ast/ConvolutionOp.cpp \
    engine-ast/CPUNode.cpp \
    engine-ast/DeconvolutionOp.cpp \
    engine-ast/EngineAST.cpp \
    engine-ast/EngineEdge.cpp \
    engine-ast/EngineGraph.cpp \
    engine-ast/EngineNode.cpp \
    engine-ast/EngineNodeFactory.cpp \
    engine-ast/FullyConnectedOp.cpp \
    engine-ast/MultiOpsNode.cpp \
    engine-ast/NestedGraph.cpp \
    engine-ast/PDPNode.cpp \
    engine-ast/RubikNode.cpp \
    engine-ast/ScaleOp.cpp \
    engine-ast/SDPEltWiseOp.cpp \
    engine-ast/SDPNode.cpp \
    engine-ast/SDPNOP.cpp \
    engine-ast/SDPSuperOp.cpp \
    engine-ast/SoftMaxOp.cpp \
    engine-ast/SplitOp.cpp \
    AST.cpp \
    CanonicalAST.cpp \
    Check.cpp \
    Compiler.cpp \
    DlaPrototestInterface.pb.cpp \
    DLAResourceManager.cpp \
    DLAInterface.cpp \
    DLAInterfaceA.cpp \
    $(ROOT)/core/src/common/EMUInterface.cpp \
    $(ROOT)/core/src/common/EMUInterfaceA.cpp \
    Layer.cpp \
    $(ROOT)/core/src/common/Loadable.cpp \
    LutManager.cpp \
    Memory.cpp \
    Network.cpp \
    Profile.cpp \
    Profiler.cpp \
    Setup.cpp \
    Surface.cpp \
    TargetConfig.cpp \
    Tensor.cpp \
    TestPointParameter.cpp \
    Wisdom.cpp \
    WisdomContainer.cpp \
    $(ROOT)/utils/BitBinaryTree.c \
    $(ROOT)/utils/BuddyAlloc.c \
    $(ROOT)/utils/ErrorLogging.c \
    $(ROOT)/port/linux/nvdla_os.c

INCLUDES += \
    -I$(LOCAL_DIR)/include \
    -I$(ROOT)/include \
    -I$(ROOT)/core/include \
    -I$(ROOT)/external/include \
    -I$(ROOT)/port/linux/include \
    -I$(ROOT)/core/src/common/include \
    -I${PROTOBUF_INSTALL_DIR}/include \

MODULE_CPPFLAGS += \
    -DNVDLA_UTILS_ERROR_TAG="\"DLA\"" \
    -DGOOGLE_PROTOBUF_NO_RTTI \
    -DNVDLA_COMPILER_OUTPUT_FOR_PROTOTEST \

MODULE_CFLAGS += \
    -DNVDLA_UTILS_ERROR_TAG="\"DLA\"" \

MODULE_SRCS := $(NVDLA_SRC_FILES)

include $(ROOT)/make/module.mk
