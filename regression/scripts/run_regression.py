#!/usr/bin/python

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


from __future__ import print_function, division

import importlib
import os
import sys
from argparse import ArgumentParser

import settings
import regression


def parseArguments(argv=None):
    default_msg = ' [default %(default)s]'

    parser = ArgumentParser(
        description='Help run regression tests with given testplan name and test level(s)'
    )
    parser.add_argument(
        'testplan',
        help='testplan name'
    )
    parser.add_argument(
        'levels',
        type=str,
        help='test level(s), eg. 0 or 0,1,3'
    )
    parser.add_argument(
        '-P', '--project',
        type=str, default='dla',
        help='Set project name' + default_msg
    )
    parser.add_argument(
        '--os',
        choices=['linux'], default='linux',
        help='Choose device OS' + default_msg
    )
    parser.add_argument(
        '--target',
        choices=['sim', 'ufpga'], default='sim',
        help='Set the target' + default_msg
    )
    parser.add_argument(
        '--noclean',
        action='store_true', default=False,
        help="Don't clean output directory when test completes [default to clean]"
    )
    parser.add_argument(
        '--show',
        action='store_true', default=False,
        help="Fork test output to stdout" + default_msg
    )
    parser.add_argument(
        '-d', '--debug',
        type=int, default=0,
        help="Set debug level" + default_msg
    )
    parser.add_argument(
        '-l', '--list',
        action='store_true', default=False,
        help="Print tests in testlist"
    )
    parser.add_argument(
        '--force', dest='forceTests',
        action='store_true', default=False,
        help="Force running staged tests"
    )
    parser.add_argument(
        '--testhome',
        type=str, default=settings.TEST_HOME,
        help="Path to find test files" + default_msg
    )
    parser.add_argument(
        '--goldhome',
        type=str, default=settings.GOLD_HOME,
        help="Path to find gold files" + default_msg
    )

    filter_group = parser.add_argument_group(
        title='Filter options',
        description='Options to help run tests selectively'
    )
    filter_group.add_argument(
        '-f', '--filter',
        type=str, default=None,
        help="Apply a match string to filter tests"
    )
    filter_group.add_argument(
        '-r', '--rfilter',
        type=str, default=None,
        help="Apply a regular expression to filter tests"
    )

    save_group = parser.add_argument_group()
    save_group.add_argument(
        '--savelead',
        action='store_true', default=False,
        help='Update lead files with test results if they mismatch'
    )
    save_group.add_argument(
        '--savegold', dest='saveGold',
        action='store_true', default=False,
        help='Update gold files with test results if they mismatch'
    )

    options = parser.parse_args(argv)
    return options


def setup_env(options):
    if 'DLA_TOP' not in os.environ:
        os.environ['DLA_TOP'] = settings.DLA_HOME
    sys.path.insert(0, settings.TESTPLAN_HOME)

    if 'OUT' not in os.environ:
        os.environ['OUT'] = settings.OUTPUT_DIR


def list_tests(options, testplan):
    """ List tests in given testplan with given level num"""
    for level in options.levels.split(','):
        testlist = testplan.get_testlist(
            level,
            options.target,
            options.filter,
            options.rfilter
        )
        regression.list_tests(testlist, options)


def deploy_tests(options):
    """Resolve testing dependency on target running env

    KUM/firmware: always run in host mode
    UMD: from device or host
    """
    pass


def run_tests(options, testplan):
    """Run all given tests, and list result stats"""
    deploy_tests(options)

    target = options.target

    for level in options.levels.split(','):
        print("Running testplan[%s] level[%s] [%s] tests"
              % (testplan.name, level, target))
        testlist = testplan.get_testlist(
            level,
            target,
            options.filter,
            options.rfilter
        )

        # Execute testlist
        results = regression.run_tests(testlist, options)

        # List result stats
        written = testplan.num_written(level, target)
        total = testplan.num_total(level, target)
        regression.list_results(results, written, total)


def main():
    options = parseArguments()

    # set environment vars
    setup_env(options)

    # obtain testplan object
    mod = importlib.import_module(options.testplan)
    testplan = mod.Module()

    testplan.register_tests()

    if options.list:
        list_tests(options, testplan)
    else:
        run_tests(options, testplan)


if __name__ == "__main__":
    main()
