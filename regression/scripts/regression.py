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

import warnings
from collections import OrderedDict

import host_regression


def list_tests(testlist, options):
    msg = 'Test (%3d/%3d)   %s   %-25s ... %s'
    test_num = len(testlist)

    for idx, test in enumerate(testlist):
        print(msg % (
            idx + 1, test_num, test.guid()[:6], str(test), test.status.upper()
        ))


def run_tests(testlist, options):
    test_num = len(testlist)

    all_results = {}
    ret_map = {
        0: 'PASS',
        16: 'PASS_LEAD',
        17: 'FAIL_LEAD',
        32: 'PASS_GOLD',
        33: 'FAIL_GOLD',
    }

    msg = 'Test (%d/%d) %s %s ...'
    for idx, test in enumerate(testlist):
        print(msg % (idx + 1, test_num, test.guid()[:6], str(test)), end=' ')

        if not options.forceTests and test.status == 'Staged':
            result = "STAGED"
        else:
            # Always in host mode for open-sourcing
            ret = host_regression.execute_test(test, options)
            result = ret_map.get(ret, 'FAIL')

        print(result)
        all_results[test.uid] = result

    return all_results


def percent(numerator, denominator):
    """Return (numerator/denominator)*100% in string

    :param numerator:
    :param denominator:
    :return: string
    """
    # Notice the / operator is from future as real division, aka same as Py3,
    return '{}%'.format(numerator * 100 / denominator)


def list_results(results, written, total):
    passed, failed, unknown = 0, 0, 0
    for r in results.values():
        if "PASS" in r:
            passed += 1
        elif "FAIL" in r:
            failed += 1
        else:
            unknown += 1

    done = passed + failed + unknown
    if done == 0:
        warnings.warn("No tests ran!")
    else:
        stats = OrderedDict()
        stats['Pass'] = (percent(passed, done), passed)
        stats['Fail'] = (percent(failed, done), failed)
        stats['Ran'] = (percent(done, written), done)
        stats['Written'] = (percent(written, total), written)

        seg_line = '-' * 40
        print(seg_line, 'Testing Stats'.center(40), sep='\n')
        title = ''.join('%-10s' % k for k in stats.keys())
        content = ''.join('%-10s' % v[0] for v in stats.values())
        print(seg_line, title, content, seg_line, sep='\n')

        for k, v in stats.items():
            print(k, '=', v[1])
        print('Total =', total)
        print(seg_line)

    if passed and passed == done:
        print("DLA sanity passed")
    else:
        print("DLA sanity failed")
