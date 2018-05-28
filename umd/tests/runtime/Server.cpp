/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ErrorMacros.h"
#include "RuntimeTest.h"
#include "Server.h"

#include "nvdla/IRuntime.h"

#include "nvdla_os_inf.h"

#include "dlaerror.h"
#include "dlatypes.h"

#include <cstdio> // snprintf, fopen
#include <fstream>
#include <string>
#include <string.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

#define MAX_TRY 10
#define CLOSE_SLEEP 5

const uint32_t TCPSERVER_MSGLEN = 256;
const char TCPSERVER_SEPARATOR = '\n';
const uint32_t TCPSERVER_MAXREADSIZE = 1 << 29; // 512 MB

#define prepareReplyMsg(args...)					\
do {									\
    memset(replyMsg, 0, sizeof(replyMsg));				\
    snprintf(replyMsg, sizeof(replyMsg), "" args);		\
} while (0)

char replyMsg[256] = {0};               ///< Reply message to client.

static NvDlaError getRecvBufSize(const TestAppArgs* appArgs, TestInfo *testInfo, int32_t *size)
{
    NvDlaError e = NvDlaSuccess;
    socklen_t m;
    int retcode;

    if (!size)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRecvBufSize(): -1\n");
    }

    m = sizeof(*size);
    retcode = getsockopt(testInfo->dlaRemoteSock, SOL_SOCKET, SO_RCVBUF, (void*)size, &m);
    if (retcode != 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getsockopt() failed\n");
    }

fail:
    return e;
}

static NvDlaError getSendBufSize(const TestAppArgs* appArgs, TestInfo *testInfo, int32_t *size)
{
    NvDlaError e = NvDlaSuccess;
    socklen_t m;
    int retcode;

    if (!size)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getSendBufSize(): -1\n");
    }

    m = sizeof(*size);
    retcode = getsockopt(testInfo->dlaRemoteSock, SOL_SOCKET, SO_SNDBUF, (void*)size, &m);
    if (retcode != 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getsockopt() failed\n");
    }

fail:
    return e;
}

static int64_t getSizeInInteger(char *strSize)
{
    char *end;
    int64_t val = 0;

    char str[TCPSERVER_MSGLEN] = {'\0'};
    errno = 0; /* To distinguish success/failure after call */
    strncpy(str, strSize, sizeof(str));
    val = strtol(str, &end, 10);

    if ((errno == ERANGE) && (val == LONG_MAX || val == LONG_MIN)) {
        NvDlaDebugPrintf("strtol failed\n");
        val = -1;
        goto fail;
    }

    if (end == str) {
        NvDlaDebugPrintf("No digits were found\n");
        val = -1;
        goto fail;
    }

    if (val == 0) {
        NvDlaDebugPrintf("Data Size can't be %ld!\n", val);
        val = -1;
        goto fail;
    }

fail:
    return val;
}

static NvDlaError sendSize(const TestAppArgs* appArgs, TestInfo *testInfo, uint32_t len)
{
    NvDlaError e = NvDlaSuccess;

    NvDlaDebugPrintf("Sending size %u bytes.\n", len);

    char buf[256];
    snprintf(buf, sizeof(buf), "%u\n", len);
    uint32_t buflen = strlen(buf);

    size_t sentSize = 0;
    while (sentSize < buflen)
    {
        size_t curSize = send(testInfo->dlaRemoteSock, (const char*)buf + sentSize, buflen - sentSize, 0);

        if (curSize > 0)
            sentSize += curSize;

        if (curSize <= 0)
            break;
    }

    return e;
}

static NvDlaError sendData(const TestAppArgs* appArgs, TestInfo *testInfo, const void* buf, size_t len)
{
    NvDlaError e = NvDlaSuccess;
    size_t sentSize = 0;
    size_t remainingSize = 0;
    size_t sendSize = 0;
    int32_t maxSendSize = -1;

    PROPAGATE_ERROR_FAIL(getSendBufSize(appArgs, testInfo, &maxSendSize));
    if (maxSendSize == -1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Error encountered: %u\n", e);
    }

    while (sentSize < len)
    {
        remainingSize = len - sentSize;
        sendSize = (maxSendSize < (int32_t)remainingSize) ? maxSendSize : remainingSize;

        size_t curSize = send(testInfo->dlaRemoteSock, (const char*)buf + sentSize, sendSize, 0);

        if (curSize > 0)
            sentSize += curSize;

        if (curSize <= 0)
            break;
    }

fail:
    return e;
}

static NvDlaError readSize(const TestAppArgs* appArgs, TestInfo *testInfo, uint32_t* len)
{
    NvDlaError e = NvDlaSuccess;
    char sizebuf[TCPSERVER_MSGLEN];
    uint32_t ii = 0;
    int64_t size = 0;

    if (!len)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readSize(): Invalid Arguments.\n");
    }

    memset(sizebuf, 0, sizeof(sizebuf));

    *len = 0;

    do
    {
        if (ii > sizeof(sizebuf))
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readSize(): No separator found after size.\n");
        }

        // Read one byte at a time until we hit our separator
        size_t received = recv(testInfo->dlaRemoteSock, &sizebuf[ii], 1, 0);

        if (received > 0)
            ii += 1;

        if (received <= 0)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readSize(): Returned zero, connection possibly terminated.\n");
        }

    } while(sizebuf[ii-1] != TCPSERVER_SEPARATOR);

    // Override separator with \0
    sizebuf[ii-1] = '\0';

    size = getSizeInInteger(sizebuf);
    if (size < 0)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readSize(): Failed to convert size in string to integer.\n");
    }

    *len = size;

fail:
    return e;
}

static NvDlaError readData(const TestAppArgs* appArgs, TestInfo *testInfo, uint32_t size, void** buf, uint32_t* len)
{
    NvDlaError e = NvDlaSuccess;
    int32_t maxRecvSize = -1;
    void* tmpbuf;
    uint32_t receivedSize = 0;
    size_t remainingSize = 0;
    size_t recvSize = 0;

    PROPAGATE_ERROR_FAIL(getRecvBufSize(appArgs, testInfo, &maxRecvSize));

    if (maxRecvSize == -1)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Error encountered: %u\n", e);
    }

    if (!buf || !len)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readData(): Invalid Arguments.\n");
    }

    if (size > TCPSERVER_MAXREADSIZE)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readData(): size (%u bytes) exceeds maximum buffer size requirements (%u bytes).\n", size, TCPSERVER_MAXREADSIZE);
    }

    // Allocate read buffer to copy into
    tmpbuf = (void*)malloc(size);
    if (!tmpbuf)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readData(): Allocation failed.\n");
    }

    // Read the buffer
    while (receivedSize < size)
    {
        remainingSize = size - receivedSize;
        recvSize = (maxRecvSize < (int32_t)remainingSize) ? maxRecvSize : remainingSize;

        size_t curSize = recv(testInfo->dlaRemoteSock, (void*)((size_t)tmpbuf + receivedSize), recvSize, 0);

        if (curSize > 0)
            receivedSize += curSize;

        if (curSize <= 0)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "readData(): Returned zero, connection possibly terminated.\n");
        }
    }

    *buf = tmpbuf;
    *len = receivedSize;

fail:
    return e;
}

static NvDlaError send(const TestAppArgs* appArgs, TestInfo *testInfo, const void* buf, uint32_t len)
{
    NvDlaError e = NvDlaSuccess;

    if (!buf)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "send(): Invalid Arguments.\n");
    }

    PROPAGATE_ERROR_FAIL(sendSize(appArgs, testInfo, len));

    PROPAGATE_ERROR_FAIL(sendData(appArgs, testInfo, buf, len));

fail:
    return e;
}

static NvDlaError read(const TestAppArgs* appArgs, TestInfo *testInfo, void** buf, uint32_t* len)
{
    uint32_t size;
    NvDlaError e = NvDlaSuccess;

    if (!buf || !len)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read(): Invalid Arguments.\n");
    }

    PROPAGATE_ERROR_FAIL(readSize(appArgs, testInfo, &size));

    PROPAGATE_ERROR_FAIL(readData(appArgs, testInfo, size, buf, len));

fail:
    return e;
}

static NvDlaError startServer(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;
    struct sockaddr_in dlaServerAddr;
    int attempt;

    /* Re-attempt for binding, incase bind() fails. */
    for (attempt = 0; attempt < MAX_TRY; attempt++) {
        /* Create socket file descriptor. */
        if ((testInfo->dlaServerSock = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
            NvDlaDebugPrintf("Unable to create socket in attempt[%d].\n", attempt);
            continue;

            if (attempt >= MAX_TRY) {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "maximum tries reached\n");
            }
        }

        /* bind to an address. */
        memset(&dlaServerAddr, 0, sizeof(dlaServerAddr));
        dlaServerAddr.sin_family = AF_INET;
        dlaServerAddr.sin_addr.s_addr = htonl(INADDR_ANY);
        dlaServerAddr.sin_port = htons(6666);

        NvDlaDebugPrintf("using %s, listening at port:%d\n",
            inet_ntoa(dlaServerAddr.sin_addr), appArgs->serverPort);

        /* Bind to the given port on any address. */
        if (bind(testInfo->dlaServerSock, (struct sockaddr *)&dlaServerAddr,
            sizeof(dlaServerAddr))) {
            NvDlaDebugPrintf("Binding Failed with errno: %d: Unable " \
                "to bind server at port: %d\n", errno, appArgs->serverPort);
            /* Close the socket incase bind failed. */
            close(testInfo->dlaServerSock);
            /*
             * Below sleep is to provide enough time for socket to
             * to enter in CLOSE state.
             */
            sleep(CLOSE_SLEEP);
            continue;

            /* If max tries are expired report failure to system. */
            if (attempt >= MAX_TRY) {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "maximum tries reached\n");
            }
        } else {
            NvDlaDebugPrintf("binding successful.\n");
        }
        break;
    }

    /* Start listening on the socket. */
    if (listen(testInfo->dlaServerSock, 10) < 0) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "unable to listen on the socket\n");
    } else {
        NvDlaDebugPrintf("Ready for Client Connection...\n");
        fflush(stdout);
    }

    testInfo->dlaServerRunning = true;

fail:
    return e;
}

static NvDlaError processWelcomeMessage(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    /* Send the welcome message */
    char welcomeMessage[] = "Hello World!";
    NvDlaDebugPrintf("Sending welcome message: {%s}\n", welcomeMessage);

    PROPAGATE_ERROR_FAIL(send(appArgs, testInfo, (void*)welcomeMessage, strlen(welcomeMessage)));

fail:
    return e;
}

static NvDlaError executeTest(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    NvDlaDebugPrintf("Executing the Test...\n");

    e = run(appArgs, testInfo);
    if (e == NvDlaSuccess)
        prepareReplyMsg("[OK] Test PASSED!");
    else
        prepareReplyMsg("[OK] Test FAILED!");

    NvDlaDebugPrintf("Execution Done: %s\n", replyMsg);

    /* Send the test completion message */
    NvDlaDebugPrintf("Sending test completion message: {%s}\n", replyMsg);

    PROPAGATE_ERROR_FAIL(send(appArgs, testInfo, (void*)replyMsg, strlen(replyMsg)));

fail:
    return e;
}

static NvDlaError processRunFlatbuf(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(executeTest(appArgs, testInfo));

fail:
    return e;
}

static NvDlaError processQueryCachedFlatbuf(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    const char* queryBuf = testInfo->pData != NULL ? "YES" : "NO";

    PROPAGATE_ERROR_FAIL(send(appArgs, testInfo, (void*)queryBuf, strlen(queryBuf)));

fail:
    return e;
}

static NvDlaError processReadFlatbuf(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;
    void* buf = NULL;
    uint32_t len = 0;

    PROPAGATE_ERROR_FAIL(read(appArgs, testInfo, &buf, &len));

    testInfo->pData = (NvU8 *)buf;

fail:
    return e;
}

static NvDlaError processRunImage(const TestAppArgs* appArgs, TestInfo* testInfo)
{
    NvDlaError e = NvDlaSuccess;
    std::ofstream imgFile(appArgs->inputName.c_str(), std::ofstream::binary);

    void* buf = NULL;
    uint32_t len = 0;

    PROPAGATE_ERROR_FAIL(read(appArgs, testInfo, &buf, &len));

    NvDlaDebugPrintf("starting to run image %s of size %d\n", appArgs->inputName.c_str(), len);

    imgFile.write((const char*)buf, len);
    imgFile.close();

    PROPAGATE_ERROR_FAIL(executeTest(appArgs, testInfo));

fail:
    if (buf != NULL)
        free(buf);
    return e;

}

static NvDlaError processGetNumOutputs(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    char buf[32];
    snprintf(buf, sizeof(buf), "%d", testInfo->numOutputs);

    PROPAGATE_ERROR_FAIL(send(appArgs, testInfo, reinterpret_cast<void*>(buf), strlen(buf)));

fail:
    return e;
}

static NvDlaError processGetOutput(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;
    std::stringstream sstream;
    void* buf = NULL;
    uint32_t len = 0;
    int32_t index = 0;

    /* Get the output index */
    NvDlaDebugPrintf("Receiving output index...\n");

    PROPAGATE_ERROR_FAIL(read(appArgs, testInfo, &buf, &len));

    NvDlaDebugPrintf("Receiving output index...Done\n");
    index = atoi((char*)buf);

    /* Prepare NvDlaImage serialization stream */
    e = testInfo->outputImage->serialize(sstream, true);
    if (e != 0)
        return e;

    NvDlaDebugPrintf("Sending test output[%d]\n", index);
    PROPAGATE_ERROR_FAIL(send(appArgs, testInfo, reinterpret_cast<const void*>(sstream.str().c_str()), sstream.str().length()));

fail:
    if (buf != NULL)
        free(buf);
    return e;
}

static NvDlaError getCommand(const TestAppArgs* appArgs, TestInfo *testInfo, char* cmd)
{
    NvDlaError e = NvDlaSuccess;
    void* buf = NULL;
    uint32_t len = 0;

    NvDlaDebugPrintf("Waiting for command from Client...\n");

    PROPAGATE_ERROR_FAIL(read(appArgs, testInfo, &buf, &len));

    memcpy(cmd, buf, len);
    cmd[len] = '\0';

    NvDlaDebugPrintf("Received command: {%s}\n", cmd);

    free(buf);

fail:
    return e;
}

static NvDlaError processShutDownServer(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    char shutbuf[] = "ACK_SHUTDOWN";

    /* Send the number of outputs */
    NvDlaDebugPrintf("Sending ACK_SHUTDOWN msg to client.\n");

    testInfo->dlaServerRunning = false;

    PROPAGATE_ERROR_FAIL(send(appArgs, testInfo, (void*)shutbuf, strlen(shutbuf)));

fail:
    return e;
}

static bool isClientConnected(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    int optval, ret;
    socklen_t optlen = sizeof(optval);

    ret = getsockopt(testInfo->dlaServerSock, SOL_SOCKET, SO_ERROR, &optval, &optlen);
    if(optval == 0 && ret == 0) {
        NvDlaDebugPrintf("Client is OFF optval: %d, ret: %d\n", optval, ret);
        return true;
    }

    return false;
}

static NvDlaError mainEventLoop(TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    /* Main event loop */
    while(true)
    {
        char cmd[256];
        std::string strCmd = "";

        PROPAGATE_ERROR_FAIL(getCommand(appArgs, testInfo, cmd));
        strCmd = std::string(cmd);

        if (strcmp(cmd, "GET_WELCOME") == 0)
        {
            /* Send Welcome message to client. */
            PROPAGATE_ERROR_FAIL(processWelcomeMessage(appArgs, testInfo));
        } else if (strcmp(cmd, "QUERY_FLATBUF") == 0) {
            /* Query if a flatbuf is already cached and reply accordingly */
            PROPAGATE_ERROR_FAIL(processQueryCachedFlatbuf(appArgs, testInfo));
        } else if (strcmp(cmd, "RUN_FLATBUF") == 0) {
            /* Get the flatbuffer and run the test. */
            PROPAGATE_ERROR_FAIL(processRunFlatbuf(appArgs, testInfo));
        } else if (strcmp(cmd, "READ_FLATBUF") == 0) {
            /* Get the flatbuffer and read it. */
            PROPAGATE_ERROR_FAIL(processReadFlatbuf(appArgs, testInfo));
        } else if (strCmd.find(std::string("RUN_IMAGE")) != std::string::npos) {
            /* Extract file name from command */
            appArgs->inputName = strCmd.substr(strCmd.find_last_of("_") + 1);
            /* Get the image and run inference on it using the flatbuffer cached in memory */
            PROPAGATE_ERROR_FAIL(processRunImage(appArgs, testInfo));
        } else if (strcmp(cmd, "GET_NUMOUTPUTS") == 0) {
            /* Get the flatbuffer and run the test. */
            PROPAGATE_ERROR_FAIL(processGetNumOutputs(appArgs, testInfo));
        } else if (strcmp(cmd, "GET_OUTPUT") == 0) {
            /* Get the flatbuffer and run the test. */
            PROPAGATE_ERROR_FAIL(processGetOutput(appArgs, testInfo));
        } else if (strcmp(cmd, "SHUTDOWN") == 0) {
            /* SHUTDOWN the server. */
            PROPAGATE_ERROR_FAIL(processShutDownServer(appArgs, testInfo));
            if (!testInfo->dlaServerRunning)
                break;
        } else {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "invalid command");
        }
    }

fail:
    /* Clear the flatbuffer cached data if available */
    if (testInfo->pData != NULL) {
        free(testInfo->pData);
        testInfo->pData = NULL;
    }
    return e;
}

NvDlaError runServer(const TestAppArgs* appArgs, TestInfo *testInfo)
{
    NvDlaError e = NvDlaSuccess;

    if (appArgs->serverPort <= 1024)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, " port:%d is a reserved port.\n", appArgs->serverPort);
    }

    PROPAGATE_ERROR_FAIL(startServer(appArgs, testInfo));

    while (testInfo->dlaServerRunning)
    {
        if( (testInfo->dlaRemoteSock = accept(testInfo->dlaServerSock, (struct sockaddr*)NULL, NULL)) == -1)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "accept socket error: %s(errno: %d)", strerror(errno),errno);
        }

        mainEventLoop(const_cast<TestAppArgs*>(appArgs), testInfo);
    }

    /* We have quit out of the event loop
     * Close the current connection and wait for a new connection
     */
    close(testInfo->dlaRemoteSock);
    close(testInfo->dlaServerSock);

fail:
    return e;
}
