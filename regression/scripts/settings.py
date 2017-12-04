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

"""Defining global configurations

"""
import os

DEFAULT_TOP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

# Set your `tests` path below, like '/home/<user>/dla/tests'
DLA_HOME = '' or DEFAULT_TOP

TESTPLAN_HOME = os.path.join(DLA_HOME, 'testplan')

TEST_HOME = os.path.join(DLA_HOME, 'flatbufs')

GOLD_HOME = os.path.join(DLA_HOME, 'golden')

REF_HOME = os.path.join(DLA_HOME, 'ref')

KMD_RUNSCRIPT = os.path.join(DLA_HOME, 'scripts', 'kmdrun.py')

OUTPUT_DIR = os.path.join(os.getcwd(), 'out')

CLIENT_SCRIPT = os.path.join(DLA_HOME, 'scripts', 'dla_client.py')

TEST_SERVER_EXE = '/mnt/nvdla_runtime -s'
TEST_SERVER_IP = '127.0.0.1'
TEST_SERVER_PORT = '6667'
TEST_SERVER_USER = 'root'
TEST_SERVER_PASSWD = 'nvdla'
