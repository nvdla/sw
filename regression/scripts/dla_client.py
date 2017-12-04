#!/usr/bin/env python


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


"""
Dla Client.
"""

import os
import sys
import subprocess
import logging
import socket
import time
import optparse
import TestUtils as tu
import dlaSocket as ds

def parse_options(argv):
    """
    Parses and checks the command-line options.

    Returns:
      A tuple containing the options structure.
    """

    usage = 'Usage: %prog [options]'
    desc = 'Example: %prog -i flatbufFile -d'
    parser = optparse.OptionParser(usage=usage, description=desc)
    parser.add_option('-i', '--flatbuf', dest='input_file', action='store',
                    help='Test flatbuf', metavar='PROTOTEXT_FILE')
    parser.add_option('-o', '--output', dest='output_dir', action='store',
                    default='./', help='Output directory', metavar='OUTPUT_DIR')
    parser.add_option('-d', '--debug', action="store_true", dest="log_level",
                    default=False, help='Log Level')
    parser.add_option('-p', '--port', action="store", dest="port_addr",
                    default=6666, help='Client Port.')
    parser.add_option('-s', '--doNotShutServer', action="store_false", dest="shut_server",
                    default=False, help='Don\'t Shut Down Server.')
    parser.add_option('--image', '--img', dest='image_file', action='store',
                    default=None, help='Image file')
    parser.add_option('--softmax', dest='softmax', action='store_true',
                    default=False, help='Perform software softmax on test output')
    parser.add_option('--shift', dest='image_shift', action='store', type="float",
                    default=0.0, help='Perform shift operation on image input')
    parser.add_option('--scale', dest='image_scale', action='store', type="float",
                    default=1.0, help='Perform scale operation on image input')
    parser.add_option('--power', dest='image_power', action='store', type="float",
                    default=1.0, help='Perform power operation on image input')

    options, categories = parser.parse_args(argv[1:])
    _input = options.input_file
    outfile_list = list()

    if options.input_file is None:
        parser.error('Input is missing')
    else:
        options.input_file = os.path.abspath(options.input_file)
        if not os.path.exists(options.input_file):
            print ("Invalid input directory: ", options.input_file)
            sys.exit(1)
        else:
            options.input_file = tu.get_input_files(options.input_file)

    if options.image_file is None:
        print("No img to run")
    else:
        options.image_file = os.path.abspath(options.image_file)
        if not os.path.exists(options.image_file):
            print("Invalid image location: ", options.image_file)
            sys.exit(1)
        else:
            options.image_file = tu.get_input_files(options.image_file)

    return (options, categories)

def getFlatBufData(input_file):
    """
    Collect fbuf data from input file and return.
    """
    data = ""

    with open(input_file, 'r') as _file:
        data = _file.read()

    return data

def getImageData(image_file):
    """
    Collect image data from image file and return.
    """
    image = ""

    with open(image_file, 'r') as _file:
        image = _file.read()

    return image

def validateTestExecution(msg):
    """
    Helper API to validate the Test Execution output.
    """
    if "PASSED" not in msg:
        passed = "FAIL"
    else:
        passed = "PASS"

    if passed == "FAIL":
        sys.exit(1)

    return

def getWelcomeMessage(sock, timeout=1000):
    logging.info("Requesting welcome message");
    cmd = "GET_WELCOME"
    sock.send(cmd)

    # Wait to receive welcome message
    sock.setTimeout(timeout)
    welcome = sock.receive()
    if "ERR" in welcome:
        logging.error("Unable to receive welcome message")
        sock.closeConnection()
        sys.exit(1)

    logging.info("Received welcome message: {%s}" % welcome);

def runFlatbuf(sock, fbuf_file, timeout=1000):
    logging.info("Sending and executing flatbuffer");
    cmd = "RUN_FLATBUF"
    sock.send(cmd)

    # Wait to receive test results
    sock.setTimeout(timeout)
    testResults = sock.receive()
    if "ERR" in testResults:
        logging.error("Unable to receive test results")
        sock.closeConnection()
        sys.exit(1)

    file_name = fbuf_file.split("/")[-1]
    logging.info("Received [%s] test results: {%s}" % (file_name, testResults))

    validateTestExecution(testResults)

def queryFlatbuf(sock, timeout=1000):
    logging.info("Querying if flatbuf is cached.")
    cmd = "QUERY_FLATBUF"
    sock.send(cmd)

    sock.setTimeout(timeout)
    msg = sock.receive()

    logging.info("Received flatbuf query response from Server: {0}".format(msg))

    return msg

def readFlatbuf(sock, fbuf_file):
    data = getFlatBufData(fbuf_file)
    if data == "":
        logging.error("Unable to read the flatbuf: %s".format(fbuf_file))
        sock.closeConnection()
        sys.exit(1)

    logging.info("Sending and loading flatbuffer");
    cmd = "READ_FLATBUF"
    sock.send(cmd)
    sock.send(data)

def preProcessImage(sock, shift=0.0, scale=1.0, power=1.0):
    logging.info("Pre process image with shift [{0}], scaling factor [{1}] and power factor [{2}]".format(shift, scale, power))
    shift_data = "%.4f" % shift
    cmd = "PERFORM_SHIFT"
    sock.send(cmd)
    sock.send(shift_data)

    cmd = "PERFORM_SCALE"
    scale_data = "%.4f" % scale
    sock.send(cmd)
    sock.send(scale_data)

    cmd = "PERFORM_POWER"
    power_data = "%.4f" % power
    sock.send(cmd)
    sock.send(power_data)

def postProcessImage(sock, softmax=False):
    if softmax:
        logging.info("Perform softmax on test output")
        cmd = "PERFORM_SOFTMAX"
        data = 'YES'
        sock.send(cmd)
        sock.send(data)
    else:
        logging.info("Don't perform softmax on test output")
        cmd = "PERFORM_SOFTMAX"
        data = 'NO'
        sock.send(cmd)
        sock.send(data)

def runImage(sock, img_file, shift=0.0, scale=1.0, power=1.0, softmax=False, timeout=1000):
    image = getImageData(img_file)
    if image == "":
        logging.error("Unable to read the image: %s".format(img_file))
        sock.closeConnection()
        sys.exit(1)

    preProcessImage(sock, shift, scale, power)
    postProcessImage(sock, softmax)

    file_name = img_file.split("/")[-1]
    logging.info("Seding and running image");
    cmd = "RUN_IMAGE_" + file_name
    sock.send(cmd)
    sock.send(image)

    # Wait to receive test results
    sock.setTimeout(timeout)
    testResults = sock.receive()
    if "ERR" in testResults:
        logging.error("Unable to receive test results")
        sock.closeConnection()
        sys.exit(1)

    logging.info("Received [%s] test results: {%s}" % (file_name, testResults))

    validateTestExecution(testResults)

def getNumOutputs(sock, timeout=1000):
    logging.info("Requesting number of test outputs");
    cmd = "GET_NUMOUTPUTS"
    sock.send(cmd)

    # Wait to receive the number of test outputs
    sock.setTimeout(timeout)
    numOutputs = sock.receive()
    if "ERR" in numOutputs:
        logging.error("Unable to receive the number of test outputs")
        sock.closeConnection()
        sys.exit(1)

    logging.info("Received number of test outputs: {%s}" % numOutputs)

    return int(numOutputs)

def writeOutput(sock, index, resultsDir, timeout=1000):
    logging.info("Requesting test output[%d]" % index);
    cmd = "GET_OUTPUT"
    sock.send(cmd)
    sock.send(str(index))

    # Wait to receive the number of test outputs
    # We need a proper way to check for communication failure here
    sock.setTimeout(timeout)
    dimg = sock.receive()

    logging.info("Received test output[%d]" % index)

    filename = resultsDir + ('o_%06d.dimg' % index)
    f = open(filename, 'w')
    f.write(dimg)
    f.close()

def shutDownServer(sock, timeout=1000):
    logging.info("Requesting to Shutdown the server.");
    cmd = "SHUTDOWN"
    sock.send(cmd)

    # Wait to receive the ACK from Server
    sock.setTimeout(timeout)
    msg = sock.receive()

    logging.info("Received response from Server: {0}".format(msg))

    return

def main():
    options, categories = parse_options(sys.argv)
    clientlogfile = options.output_dir + "/DLAClientLogger.log"
    resultsdir = options.output_dir + "/results/"

    tu.logger_setup(options.log_level, clientlogfile)

    dlasocket = ds.dlaSocket()
    #Set Client Port
    dlasocket.setPort(options.port_addr)

    dlasocket.connect(dlasocket.HOST, dlasocket.PORT)
    dlasocket.setTimeout(dlasocket.getTimeout())

    logging.info("DLA Client open at PORT: {0}.".format(dlasocket.getPort()))

    getWelcomeMessage(dlasocket)

    test_i = 0
    while test_i < len(options.input_file):
        fbuf_file = options.input_file[test_i]
        fbuf_size = os.stat(fbuf_file).st_size

        if options.image_file is None:
            fbuf_file_name = options.input_file[test_i].split("/")[-1]
            logging.info("Attempting to read flatbuf: [{0}], " \
                     "size[{1}].".format(fbuf_file_name, fbuf_size))

            readFlatbuf(dlasocket, fbuf_file)

            logging.info("Attempting to run flatbuf: [{0}], " \
                     "size[{1}].".format(fbuf_file_name, fbuf_size))

            runFlatbuf(dlasocket, fbuf_file)
        else:
            img_file  = options.image_file[test_i]
            img_size  = os.stat(img_file).st_size

            fbuf_file_name = options.input_file[test_i].split("/")[-1]
            logging.info("Attempting to read flatbuf: [{0}], " \
                     "size[{1}].".format(fbuf_file_name, fbuf_size))

            readFlatbuf(dlasocket, fbuf_file)

            image_file_name = options.image_file[test_i].split("/")[-1]
            logging.info("Attempting to run image: [{0}], " \
                     "size[{1}].".format(image_file_name, img_size))

            runImage(dlasocket, img_file, \
                     options.image_shift, options.image_scale, options.image_power, \
                     options.softmax)

        numOutputs = getNumOutputs(dlasocket)
        for ii in range(numOutputs):
            writeOutput(dlasocket, ii, resultsdir)

        test_i += 1

    #Send ShutDown command to Server if shut_server is True.
    if options.shut_server:
        shutDownServer(dlasocket)

    #Close dlsSocket
    dlasocket.closeConnection()

    return 0

if __name__ == "__main__":
    sys.exit(main())
