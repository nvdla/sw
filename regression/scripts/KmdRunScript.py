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

"""KMD unit test runscript.

Runscript specialized for KMD testing
"""
from __future__ import print_function

import argparse
import os
import pexpect
import shlex
import subprocess
import time
from threading import Thread

import sys

import settings
from RunScript import RunScript


class KmdRunScript(RunScript):

    def __init__(self):
        super(KmdRunScript, self).__init__()

        self.loadable = None
        self.serverProcess = None

    def ssh(self, host, port, cmd, user, password, timeout=5):
        ssh_newkey = 'Are you sure you want to continue connecting'

        ssh_cmd = "ssh " + user + "@" + host + " -p " + str(port) + " " + cmd
        self.log('ssh cmd: %s' % ssh_cmd)

        p = pexpect.spawn(ssh_cmd, timeout=timeout, ignore_sighup=False)

        i = p.expect([ssh_newkey, 'password:', pexpect.EOF, pexpect.TIMEOUT])
        if i == 0:
            p.sendline('yes')
            i = p.expect([ssh_newkey, 'password:', pexpect.EOF, pexpect.TIMEOUT])

        if i == 1:
            p.sendline(password)
        elif i == 4:
            print("Connection timed out")
            exit(1)

        return p

    def parseArguments(self, argv=None):
        super(KmdRunScript, self).parseArguments(argv)

        options = self.options
        options.module = 'kmd'

        test_top = os.path.join(options.testHome, options.module)
        test_unit = options.testName.split('_')[0]  # BDMA_L0_0 -> BDMA/, PDP_L1_2_3 -> PDP/
        test_file = options.testName + "_fbuf"      # BDMA_L0_0 -> BDMA_L0_0_fbuf
        self.loadable = os.path.join(test_top, test_unit, test_file)
        if not os.path.exists(self.loadable):
            self.testError = True
            raise RuntimeError('Test file %s NOT found!' % self.loadable)

    def startSshServer(self, sudo=False):
        args = settings.TEST_SERVER_EXE
        if sudo:
            args = "sudo " + args
        args = "export LD_LIBRARY_PATH=/mnt" + " && " + args

        self.serverProcess = self.ssh(
            settings.TEST_SERVER_IP,
            settings.TEST_SERVER_PORT,
            args,
            settings.TEST_SERVER_USER,
            settings.TEST_SERVER_PASSWD
        )

        self.log("Waiting for server ready...")
        i = self.serverProcess.expect(["Ready for Client Connection", pexpect.EOF, pexpect.TIMEOUT])
        if i == 0 and self.serverProcess.isalive():
            self.log("Done connecting\n")
        else:
            raise RuntimeError("Failed to launch test server")

        self.serverlinebuffer = []
        t = Thread(target=self.reader, args=(self.serverProcess, self.serverlinebuffer))
        t.daemon = True
        t.start()

    def setupEnvironment(self):
        super(KmdRunScript, self).setupEnvironment()
        options = self.options

        self.serverLogFile = open(
            os.path.join(self.options.outputDir, 'DLAServer.log'),
            'w'
        )

        if options.ssh:
            start_server = self.startSshServer
        else:
            raise NotImplementedError("Only SSH connection is supported now.")

        try_limit = 3
        while(try_limit):
            try:
                start_server()
                break
            except RuntimeError as e:
                self.log(e.message)

            if try_limit:
                try_limit -= 1
                self.log("Let's try again")
                self.killTestServer()
            else:
                raise e

    def reader(self, f, buffer):
        while True:
            line = f.readline()
            if line:
                buffer.append(line)
            else:
                break

    def runTest(self):
        options = self.options

        exe = settings.CLIENT_SCRIPT
        args = ["python", exe, "-i", self.loadable, "-o", options.outputDir]

        self.log("Launching test: " + " ".join(args))
        testProcess = subprocess.Popen(args, stderr=subprocess.PIPE)

        linebuffer = []
        t = Thread(target=self.reader, args=(testProcess.stderr, linebuffer))
        t.daemon = True
        t.start()

        while testProcess.poll() is None:
            while linebuffer:
                self.log(linebuffer.pop(0).rstrip())

            while self.serverlinebuffer:
                serverline = self.serverlinebuffer.pop(0)
                self.serverLogFile.write(serverline)

            time.sleep(0.1)

        while linebuffer:
            self.log(linebuffer.pop(0).rstrip())

        while self.serverlinebuffer:
            serverline = self.serverlinebuffer.pop(0)
            self.serverLogFile.write(serverline)

        if testProcess.returncode != 0:
            self.log("Test reported an error: " + str(testProcess.returncode))
            self.testError = True
        else:
            self.log("Test completed\n")

    def shutdownSshServer(self):
        self.serverProcess.close(True)
        self.serverProcess.terminate(True)

        if self.testError:
            if self.options.ssh:
                self.killTestServer()

        self.log('Test server shutdown.')

    def killTestServer(self):
        # be very careful with the double quote
        # TODO: we need better way to do this
        kill_server_cmd = '''"ps | grep '%s' | awk '{print $1}' | xargs kill"''' % settings.TEST_SERVER_EXE
        p = self.ssh(
            settings.TEST_SERVER_IP,
            settings.TEST_SERVER_PORT,
            kill_server_cmd,
            settings.TEST_SERVER_USER,
            settings.TEST_SERVER_PASSWD
        )
        p.wait()
        p.close()

    def cleanUp(self):
        self.log("Shutting down test server")
        options = self.options

        if options.ssh:
            self.shutdownSshServer()

        if options.showSimLogs:
            self.log("-------------------")
            self.log("Dumping server logs")
            self.log("-------------------")

            for line in self.serverLogFile:
                self.log(line)
        self.serverLogFile.close()

        super(KmdRunScript, self).cleanUp()

