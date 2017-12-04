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
Dla Socket Class and utilities.
"""

import os
import sys
import logging
import socket
import time
import struct

class dlaSocket:
    """
    Dla Socket Base Class.
    """

    PORT = 39485
    HOST = 'localhost'
    MSGLEN = 256
    TIMEOUT = 100000
    SEP = '\n'

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def getPort(self):
        return self.PORT

    def setPort(self, port):
        self.PORT = port

    def getHost(self):
        return self.HOST

    def setHost(self, host):
        self.HOST = host

    def getTimeout(self):
        return self.TIMEOUT

    def setTimeout(self, timeout):
        self.sock.settimeout(timeout)
        return

    def getRecvBufSize(self):
        return self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

    def getSendBufSize(self):
        return self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)

    def closeConnection(self):
        self.sock.close()

        return

    def connect(self, host, port):
        try:
            # Add linger to avoid entering in TIME_WAIT state.
            l_onoff = 1
            l_linger = 0
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                struct.pack('ii', l_onoff, l_linger))

            self.sock.connect((str(host), int(port)))

        except socket.error as err:
            msg = "Couldn't Connect with the socket-server: {0}\n" \
                  " terminating program\n".format(err)
            logging.error(msg)
            sys.exit(1)

        logging.info("Connection accepted")

    def send(self, msg):
        self.sendSize(len(msg))
        self.sendData(msg)

    def receive(self):
        size = self.receiveSize()
        msg = self.receiveData(size)

        return msg

    def sendSize(self, size):
        sent = 0

        try:
            logging.info("sending size %u bytes" % size)
            sent = self.sock.send(str(size) + self.SEP)
        except socket.error as err:
            msg = "sendSize failed: {0}\n".format(err.strerror)
            logging.error(msg)

        if sent == 0:
            msg = "Socket Connection Broken."
            logging.error(msg)
            raise RuntimeError(msg)

    def sendData(self, msg):
        maxSendSize = self.getSendBufSize()
        sentSize = 0
        remainingSize = 0
        sendSize = 0

        while sentSize < len(msg):
            curSize = 0
            remainingSize = len(msg) - sentSize
            sendSize = maxSendSize if (maxSendSize < remainingSize) else remainingSize

            try:
                curSize = self.sock.send(msg[sentSize:(sentSize+sendSize)])
            except socket.error as err:
                msg = "sendSize failed: {0}\n".format(err.strerror)
                logging.error(msg)

            if curSize > 0:
                sentSize += curSize

            if curSize <= 0:
                msg = "Socket Connection Broken."
                logging.error(msg)
                raise RuntimeError(msg)

    def receiveSize(self):
        sizestr = ""

        while True:
            # Read one byte at a time until we hit our separator
            sizechar = self.sock.recv(1)

            if len(sizechar) <= 0:
                msg = "Socket Connection Broken."
                logging.error(msg)
                raise RuntimeError(msg)

            if sizechar == self.SEP:
                break
            else:
                sizestr += sizechar

        size = -1
        # Convert string size to integer
        try:
            size = int(sizestr)
        except ValueError:
            logging.error("received non-integer size (%s)" % sizestr)

        return size

    def receiveData(self, size):
        maxRecvSize = self.getRecvBufSize()

        buf = ""
        receivedSize = 0
        remainingSize = 0

        logging.info("reading %u bytes from Server." % size)
        while receivedSize < size:
            remainingSize = size - receivedSize
            recvSize = maxRecvSize if (maxRecvSize < remainingSize) else remainingSize

            recvdata = self.sock.recv(recvSize)
            if len(recvdata) <= 0:
                msg = "Socket Connection Broken."
                logging.error(msg)
                raise RuntimeError(msg)

            buf += recvdata
            receivedSize = len(buf)

        return buf
