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

"""Base runscript.

"""
from __future__ import print_function

import argparse
import difflib
import errno
import os
import socket
import stat
import sys
from datetime import datetime

import settings


class RunScript(object):
    """Common base class for all RunScripts

    Providing basic options, and their defaults; help init test directory, logging
    """

    def __init__(self):
        self.argParser = None
        self.options = None
        self.gild = "NONE"
        self.testMiscompare = False
        self.testError = False
        self.testResult = 0

    def printResult(self):
        if not self.testError and not self.testMiscompare:
            self.log("PPPP     A     SSSSS  SSSSS\n"
                     "P   P   A A    S      S    \n"
                     "PPPP   AAAAA   SSSSS  SSSSS\n"
                     "P     A     A      S      S\n"
                     "P     A     A  SSSSS  SSSSS\n"),
        else:
            self.log("FFFFF    A     IIIII  L    \n"
                     "F       A A      I    L    \n"
                     "FFFF   AAAAA     I    L    \n"
                     "F     A     A    I    L    \n"
                     "F     A     A  IIIII  LLLLL\n"),

    def initialize(self):
        self.argParser = argparse.ArgumentParser(
            description='Runscript to help setup, run, check unit test'
        )

    def parseArguments(self, argv=None):
        default_msg = ' [default %(default)s]'
        parser = self.argParser

        parser.add_argument(
            '-P', '--project', type=str, default='NVDLA',
            help='Set project name' + default_msg
        )
        parser.add_argument(
            '--os', choices=['linux'], default='linux',
            help='Choose device OS' + default_msg
        )
        parser.add_argument(
            '-m', '--module', type=str,
            help='Choose module under testing'
        )
        parser.add_argument(
            '-t', '--test', dest='testName', type=str,
            required=True,
            help='Test name'
        )
        parser.add_argument(
            '--guid', dest='testguid', type=str,
            help='Set test guid'
        )
        parser.add_argument(
            '--target', type=str,
            choices=['sim', 'ufpga'], default='sim',
            help='Set target' + default_msg
        )
        parser.add_argument(
            '--fuzzy', action='store_true', default=False,
            help='Allow image comparisons to be imprecise' + default_msg
        )
        parser.add_argument(
            '-o', '--olog', dest='logFile', default='testout.log',
            help='Log file name'
        )
        parser.add_argument(
            '--odir', dest='outputDir',
            type=str, default=settings.OUTPUT_DIR,
            help='Temporary output directory'
        )
        parser.add_argument(
            '--cmpgold', dest='compareGold',
            type=str, metavar='proj',
            help='Compare gold files, [proj] defaults to the current project'
        )
        parser.add_argument(
            '--cmpref', dest='compareRef',
            type=str, metavar='proj',
            help='Compare reference files'
        )
        parser.add_argument(
            '--testhome', dest='testHome',
            type=str, default=settings.TEST_HOME,
            help="Path to find test files" + default_msg
        )
        parser.add_argument(
            '--goldhome', dest='goldHome',
            type=str, default=settings.GOLD_HOME,
            help="Path to find gold files" + default_msg
        )
        parser.add_argument(
            '--refhome', dest='refHome',
            type=str, default=settings.REF_HOME,
            help="Path to find reference files" + default_msg
        )

        debug_group = parser.add_argument_group(
            title='Debug options',
            description='Options to help debug test, infra, flow'
        )
        debug_group.add_argument(
            '--noclean',
            action='store_true', default=False,
            help="Don't clean output directory when test completes [default to clean]"
        )
        debug_group.add_argument(
            '-x', '--timeout', type=int,
            help=' Number of seconds before test timeouts'
        )
        debug_group.add_argument(
            '--show', dest='showSimLogs',
            action='store_true', default=False,
            help="Fork test output to stdout" + default_msg
        )
        debug_group.add_argument(
            '-d', '--debug', dest='debugLevel',
            type=int, default=0,
            help="Set debug level" + default_msg
        )

        save_group = parser.add_mutually_exclusive_group()
        save_group.add_argument(
            '--savelead',
            action='store_true', default=False,
            help='Update lead files with test results if they mismatch'
        )
        save_group.add_argument(
            '--savegold',
            action='store_true', default=False,
            help='Update gold files with test results if they mismatch'
        )

        # we may add more connection type, so the group is required,
        # but only ssh here for now
        con_group = parser.add_mutually_exclusive_group(required=True)
        con_group.add_argument(
            '--ssh', action='store_true',
            help='Connect to device using ssh'
        )

        options = parser.parse_args(argv)
        self.options = options

        options.resultDir = os.path.join(
            options.outputDir, "results"
        )
        options.goldDir = os.path.join(
            options.goldHome,
            options.testName + "_" + options.testguid,
            options.project
        )

    def setReRunScript(self):
        rerunScript = os.path.join(
            self.options.outputDir,
            "rerun_script"
        )
        with open(rerunScript, 'w') as f:
            f.write(sys.executable + " ")
            f.write(" ".join(sys.argv))
            f.write('\n')

        # Mark rerun_script as executable
        st = os.stat(rerunScript)
        os.chmod(rerunScript, st.st_mode | stat.S_IEXEC)

    def setupTestoutDirs(self):
        options = self.options
        out_dir = options.outputDir
        res_dir = options.resultDir

        # Make the directory that the test will run in
        # We don't want to clear directory contents beforehand
        #   for fear of something silly like 'rm -rf /'
        RunUtils.mkdir_p(out_dir)

        # Clear and recreate the results directory
        if os.path.exists(res_dir):
            RunUtils.rmdir_rf(res_dir)
        RunUtils.mkdir_p(res_dir)

    def setupEnvironment(self):
        options = self.options

        self.setupTestoutDirs()

        self.logFile = open(
            os.path.join(options.outputDir, options.logFile),
            'w'
        )

        self.setReRunScript()

        self.printPrologue()

    def printPrologue(self):
        date = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        hostname = socket.gethostname()
        testname = 'Test   ' + self.options.testName

        banner = '#' * 80
        self.log(banner)
        self.log(testname.center(80))
        self.log(banner)
        self.log(date + " on " + hostname)
        self.log("Command: " + " ".join(sys.argv))

    def runTest(self):
        pass

    def compareResults(self):
        # Skip over result comparison if test already failed
        if self.testError:
            return

        res_dir = self.options.resultDir
        gol_dir = self.options.goldDir

        # Generate md5 checksums
        test_fname = "test.md5"
        lead_fname = "lead.md5"
        gold_fname = "gold.md5"

        for filename in sorted(os.listdir(res_dir)):
            os.system(
                "cd %s && md5sum %s >> %s" %
                (res_dir, filename, test_fname)
            )

        test_path = os.path.join(res_dir, test_fname)
        lead_path = os.path.join(gol_dir, lead_fname)
        gold_path = os.path.join(gol_dir, gold_fname)

        test_exists = os.path.exists(test_path)
        lead_exists = os.path.exists(lead_path)
        gold_exists = os.path.exists(gold_path)

        if lead_exists:
            self.gild = "LEAD"
        elif gold_exists:
            self.gild = "GOLD"
        else:
            self.gild = "NONE"

        self.testMiscompare = False

        if not test_exists:
            if lead_exists or gold_exists:
                self.log("MISCOMPARE: No %s found\n" % test_fname)
                self.testMiscompare = True
        else:
            if not lead_exists and not gold_exists:
                self.log("GILD MISCOMPARE: No leads or golds\n")
                self.testMiscompare = True
            elif gold_exists:
                if RunUtils.diff(test_path, gold_path):
                    self.log("GOLD MISCOMPARE: %s mismatches %s\n" % (test_fname, gold_fname))
                    self.testMiscompare = True
                else:
                    self.log("GOLD PASS: %s matches %s\n" % (test_fname, gold_fname))
            elif lead_exists:
                if RunUtils.diff(test_path, lead_path):
                    self.log("LEAD MISCOMPARE: %s mismatches %s\n" % (test_fname, lead_fname))
                    self.testMiscompare = True
                else:
                    self.log("LEAD PASS: %s matches %s\n" % (test_fname, lead_fname))

    def gildTest(self):
        options = self.options
        if options.savelead:
            self.log("Saving leads...")
        else:
            self.log("Saving golds...")

        test_fname = "test.md5"
        test_path = os.path.join(options.resultDir, test_fname)

        if not os.path.exists(test_path):
            self.log("BYPASSING (no %s found)\n" % test_fname)
            return 0

        if os.path.exists(options.goldDir):
            RunUtils.rmdir_rf(options.goldDir)
        RunUtils.mkdir_p(options.goldDir)

        for filename in os.listdir(options.resultDir):
            if filename == test_fname:
                gild_fname = "lead.md5" if options.savelead else "gold.md5"
            else:
                gild_fname = filename

            file_path = os.path.join(options.resultDir, filename)
            gild_path = os.path.join(options.goldDir, gild_fname)
            os.system("cp %s %s" % (file_path, gild_path))

        self.log("COMPLETE\n")

    def testFinish(self):
        # Save leads and golds if we miscompared
        if not self.testError and self.testMiscompare:
            if self.options.savelead or self.options.savegold:
                self.gildTest()

        self.printResult()

        if self.testError:
            self.testResult = 1  # FAIL
        else:
            code = {
                'NONE': 0,
                'LEAD': 16,
                'GOLD': 32
            }
            self.testResult = code[self.gild] + self.testMiscompare

    def cleanUp(self):
        self.logFile.close()

        if not self.options.noclean:
            RunUtils.rmdir_rf(self.options.outputDir)

    def log(self, message):
        self.logFile.write(message)
        print(message)

    def exit(self, status):
        sys.exit(status)


class RunUtils(object):

    @staticmethod
    def mkdir_p(path):
        # Commented code below does not handle making more than one directory
        # Fallback to os.system() calls
        os.system("mkdir -p %s" % path)
        # try:
        #    os.mkdir(path)
        # except OSError as err:
        #    if err.errno == errno.EEXIST and os.path.isdir(path):
        #        pass
        #    else:
        #        raise

    @staticmethod
    def symlink_f(source, link_name):
        try:
            os.symlink(source, link_name)
        except OSError as err:
            if err.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(source, link_name)
            else:
                raise

    @staticmethod
    def rmdir_rf(path):
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except OSError as err:
                        if err.errno == errno.ENOTDIR:
                            # This is a symlink
                            os.remove(os.path.join(root, dir))
                        else:
                            raise
            os.rmdir(path)
        except OSError as err:
            print(err)

    @staticmethod
    def diff(afilepath, bfilepath):
        afile = open(afilepath, 'r')
        bfile = open(bfilepath, 'r')
        diff = difflib.ndiff(afile.readlines(), bfile.readlines())

        result = False
        for line in diff:
            if line[0:2] != "  ":
                # Line is not common to both sequences
                result = True

        afile.close()
        bfile.close()

        return result
