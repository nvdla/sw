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

import test_plan
import settings


class Module(test_plan.Testplan):
    runScript = settings.KMD_RUNSCRIPT
    deviceTargets = ['sim', 'ufpga']

    def __init__(self):
        super(Module, self).__init__(__name__)


# Convenience globals
kmd = Module.runScript
devices = Module.deviceTargets
ces = ["Core Engine Scheduler"]
nn = ["Neural Network"]
bdma = ["BDMA HW"]
convd = ["CONV HW - Direct"]
convi = ["CONV HW - Image"]
convw = ["CONV HW - Winograd"]
convp = ["CONV HW - Pipeline"]
sdpx1 = ["SDP X1 HW"]
sdpx2 = ["SDP X2 HW"]
sdpy = ["SDP Y HW"]
sdpf = ["SDP HW - Full"]
cdp = ["CDP HW"]
pdp = ["PDP HW"]
rubik = ["RUBIK HW"]


def registerL0Tests(self, testplan):
    testplan.append(
        [0, "Written", kmd, "BDMA_L0_0", None, bdma, devices, "BDMA test - Sanity test for DRAM to DRAM transfer",
         "Single task for one DRAM to DRAM transfer. No need to check for data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Not Written", kmd, "BDMA_L0_1", None, bdma, devices, "BDMA test - Sanity test for DRAM to SRAM transfer",
         "Single task for one DRAM to CVSRAM transfer. No need to check for data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Not Written", kmd, "BDMA_L0_2", None, bdma, devices, "BDMA test - Sanity test for SRAM to DRAM transfer",
         "Single task for one SRAM to DRAM transfer. No need to check for data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Not Written", kmd, "BDMA_L0_3", None, bdma, devices, "BDMA test - Sanity test for SRAM to SRAM transfer",
         "Single task for one SRAM to CVSRAM transfer. No need to check for data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Written", kmd, "CONV_D_L0_0", None, convd, devices, "Convolution test - Sanity test direct convolution",
         "Direct convolution, 8x8x128 input cube, 3x3x128 kernel cube and 32 kernels input and weight read from DRAM, no mean and bias data, output written to DRAM through SDP."])
    testplan.append(
        [0, "Written", kmd, "SDP_X1_L0_0", None, sdpx1, devices,
         "SDP test - Sanity test for SDP, only X1 enabled with ALU, X2 and Y disable. No DMA used",
         "Element wise sum operation in X1, 8x8x32 input cube and 8x8x32 bias cube. Activation function as ReLU"])
    testplan.append(
        [0, "Written", kmd, "CDP_L0_0", None, cdp, devices, "CDP test - Sanity test for CDP",
         "Use only linear table with LUT configured with all 1. 8x8x32 input cube and 8x8x32 output cube."])
    testplan.append(
        [0, "Written", kmd, "PDP_L0_0", None, pdp, devices, "PDP test - Sanity test for PDP with max pooling",
         "Max pooling, 8x8x32 input cube, 8x8x32 output cube, no padding, 1x1 kernel size. No need to compare data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Written", kmd, "RBK_L0_0", None, rubik, devices, "Rubik test - Sanity test for Rubik contract mode",
         "Contract mode, 8x8x640 input cube, 80x32x16 output cube, 10x4 stride, INT16 precision. No need to compare data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Not Written", kmd, "RBK_L0_1", None, rubik, devices, "Rubik test - Sanity test for Rubik contract mode",
         "Contract mode, 8x8x640 input cube, 16x16x160 output cube, 2x2 stride, INT8 precision. No need to compare data. It is enough if task succeeds to pass this test."])
    testplan.append(
        [0, "Written", kmd, "NN_L0_0", None, nn, devices, "MNIST", "MNIST"])
    testplan.append(
        [0, "Written", kmd, "NN_L0_1", None, nn, devices, "AlexNet", "AlexNet"])

def registerFirmwareTests(self):
    testplan = []
    registerL0Tests(self, testplan)

    for item in testplan:
        test = test_plan.Test()
        test.level = item[0]
        test.status = item[1]
        test.runscript = item[2]
        test.name = item[3]
        test.options = item[4]
        test.features = item[5]
        test.targets = item[6]
        test.description = item[7]
        test.dependencies = None
        self.add_test(test)


def registerTests(self):
    registerFirmwareTests(self)


Module.register_tests = registerTests
