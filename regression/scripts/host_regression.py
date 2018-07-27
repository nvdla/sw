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

from __future__ import print_function

import os
import shlex
import subprocess

import settings


def execute_test(test, options):
    scratch = settings.OUTPUT_DIR
    guid = test.guid()[:6].lower()
    # add os.sep: in case low-level script needs it
    out_dir = os.path.join(scratch, test.name + "_" + guid) + os.sep

    cmd_exec = test.runscript
    cmd_args = [
                   '--ssh',
                   '--odir', out_dir,
                   '--guid', guid,
                   '-o', 'output.log',
                   '-t', test.name,
               ] + test.options

    dict_opts = vars(options)
    forward_keys = (
        'project',
        'target',
        'os',
        'debug',
        'testhome',
        'goldhome',
    )
    for arg in forward_keys:
        cmd_args.append('--%s %s' % (arg, dict_opts[arg]))

    switch_keys = (
        'noclean',
        'savelead',
        'savegold',
    )
    for arg in switch_keys:
        if dict_opts[arg]:
            cmd_args.append('--%s' % arg)

    cmd = cmd_exec + ' ' + ' '.join(cmd_args)
    cmd = os.path.expandvars(cmd)

    if options.show:
        print('\nRun cmd: %s' % cmd)

    test_process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if options.show:
        for line in test_process.stdout:
            print(line, end='')

    return test_process.wait()
