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
import re


class Test(object):
    """Single test"""

    def __init__(self):
        self.uid = None
        self.level = None
        self.name = None
        self.options = []
        self.targets = []
        self.features = []
        self.description = None
        self.dependencies = None
        self.runscript = None
        self.status = None

    def __repr__(self):
        return os.path.basename(self.runscript) + ' ' + self.name + \
               ' '.join(self.options)

    def __hash__(self):
        return hash((self.name, tuple(self.options)))

    def __eq__(self, other):
        return (self.name, self.options) == (other.name, other.options)

    def guid(self):
        code = hex(hash(self) & (2 ** 64 - 1))
        code = code[2:-1]
        return code.upper()

    def pprint(self):
        print("Test:", str(self))
        print("Test Description:", self.description)
        print("Level:", self.level)
        print("Features:", self.features)
        print("Targets:", self.targets)
        print("Runscript:", self.runscript)
        print("Status:", self.status)
        print("Hash:", hash(self))


class Testplan(object):
    """Container of tests"""

    def __init__(self, name):
        self.name = name
        self.test_list = []

    def register_tests(self):
        raise NotImplementedError("This call should be overridden")

    def add_test(self, test):
        # TODO: improve Test design
        test.uid = len(self.test_list) - 1
        # workaround for the legacy code
        test.level = str(test.level)
        test.options = test.options or []

        self.test_list.append(test)

    def get_test(self, uid):
        return self.test_list[uid]

    def num_written(self, level, target):
        # TODO: improve testplan design
        tests = tuple(self.valid_tests(level, target))
        return len(tests)

    def num_total(self, level, target):
        # TODO: improve testplan design
        tests = tuple(self.valid_tests(level, target, False))
        return len(tests)

    def match(self, test, kwd):
        """Match by key word"""
        if not kwd:
            return True

        kwd = kwd.upper()
        return (
            kwd in test.guid() or
            kwd in test.runscript.upper() or
            kwd in test.name.upper() or
            kwd in ' '.join(test.options).upper()
        )

    def re_match(self, test, pattern):
        """Match by regex"""
        if not pattern:
            return True

        m = re.search(pattern, test.name, re.IGNORECASE)
        return bool(m)

    def get_testlist(self, level, target, kwd=None, rex_pattern=None):
        testlist = []
        for test in self.valid_tests(level, target):
            if self.match(test, kwd) and self.re_match(test, rex_pattern):
                testlist.append(test)
        return testlist

    def valid_tests(self, level, target, status_check=True):
        for test in self.test_list:
            valid = (
                level == test.level and
                target in test.targets
            )
            if status_check:
                valid = valid and test.status in ('Written', 'Staged')

            if valid:
                yield test
