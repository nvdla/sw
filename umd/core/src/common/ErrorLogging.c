/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "nvdla_os_inf.h"

const char* NvDlaUtilsGetNvErrorString(NvDlaError e)
{
    return NULL;
}

/**
 * The default error logging function to use when NVDLA_UTILS_ERROR_TAG is defined.
 */
#if defined (NVDLA_UTILS_ERROR_TAG)
void NvDlaUtilsLogError(const char* tag, const char* path, NvDlaError e, const char* file, const char* func,
                        uint32_t line, bool propagating, const char* format, ...)
{
    static const uint32_t MAX_LENGTH = 1024;

    char msg[MAX_LENGTH];
    char *cur = msg;

    // Remove the leading file path.
    if (strstr(file, path))
        file = strstr(file, path) + strlen(path);

    // Use the error string if found, otherwise just use the raw error code.
    const char* errorString = NvDlaUtilsGetNvErrorString(e);
    if (errorString)
        cur += snprintf(cur, msg + sizeof(msg) - cur, "(%s) Error %s:", tag, errorString);
    else
        cur += snprintf(cur, msg + sizeof(msg) - cur, "(%s) Error 0x%08x:", tag, e);

    // Append the error message.
    if (format)
    {
        cur += snprintf(cur, msg + sizeof(msg) - cur, " ");

        va_list args;
        va_start(args, format);
        cur += vsnprintf(cur, msg + sizeof(msg) - cur, format, args);
        va_end(args);
    }

    // Append the error location.
    if (propagating)
        cur += snprintf(cur, msg + sizeof(msg) - cur, " (propagating from ");
    else
        cur += snprintf(cur, msg + sizeof(msg) - cur, " (in ");
    snprintf(cur, msg + sizeof(msg) - cur, "%s, function %s(), line %d)", file, func, line);

    // Output the error.
    NvDlaDebugPrintf("%s\n", msg);
}
#endif // defined(NVDLA_UTILS_ERROR_TAG)
