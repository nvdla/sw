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

#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
#include <errno.h>

#include <time.h>

#include "nvdla_os_inf.h"

static NvDlaError getOpenMode(NvU32 flags, int *mode)
{
    switch( flags )
    {
        case NVDLA_OPEN_READ:
            *mode = O_RDONLY | O_LARGEFILE;
            break;
        case NVDLA_OPEN_WRITE:
        case NVDLA_OPEN_CREATE | NVDLA_OPEN_WRITE:
            *mode = O_CREAT | O_WRONLY | O_TRUNC | O_LARGEFILE;
            break;
        case NVDLA_OPEN_READ | NVDLA_OPEN_WRITE:
            *mode = O_RDWR | O_LARGEFILE;
            break;
        case NVDLA_OPEN_CREATE | NVDLA_OPEN_READ | NVDLA_OPEN_WRITE:
            *mode = O_CREAT | O_RDWR | O_TRUNC | O_LARGEFILE;
            break;
        case NVDLA_OPEN_APPEND:
        case NVDLA_OPEN_CREATE | NVDLA_OPEN_APPEND:
        case NVDLA_OPEN_WRITE | NVDLA_OPEN_APPEND:
        case NVDLA_OPEN_CREATE | NVDLA_OPEN_WRITE | NVDLA_OPEN_APPEND:
            *mode = O_CREAT | O_WRONLY | O_APPEND | O_LARGEFILE;
            break;
        case NVDLA_OPEN_READ | NVDLA_OPEN_APPEND:
        case NVDLA_OPEN_CREATE | NVDLA_OPEN_READ | NVDLA_OPEN_APPEND:
        case NVDLA_OPEN_READ | NVDLA_OPEN_WRITE | NVDLA_OPEN_APPEND:
        case NVDLA_OPEN_CREATE | NVDLA_OPEN_READ | NVDLA_OPEN_WRITE | NVDLA_OPEN_APPEND:
            *mode = O_CREAT | O_RDWR | O_APPEND | O_LARGEFILE;
            break;
        default:
            return NvDlaError_BadParameter;
    }

    return NvDlaSuccess;
}

void *NvDlaAlloc(size_t size)
{
    return malloc(size);
}

void NvDlaFree(void *ptr)
{
    if(!ptr)
        free(ptr);
}

void NvDlaSleepMS(NvU32 msec)
{
    struct timespec ts = {msec/1000, (msec % 1000) * 1000000};

    while (nanosleep(&ts, &ts)) {
        if (ts.tv_sec == 0 && ts.tv_nsec == 0)
            break;
        if (errno != EINTR) {
            NvDlaDebugPrintf("\n\nNvDlaSleepMS failure:%s\n", strerror(errno));
            return;
        }
    }

    return;
}

NvU32 NvDlaGetTimeMS(void)
{
    struct timespec ts;
    NvU64 time;

    if (clock_gettime(CLOCK_MONOTONIC, &ts)) {
        NvDlaDebugPrintf("\n\nCLOCK_MONOTONIC unsupported\n");
        return 0;
    }

    time = ((NvU64)ts.tv_sec * 1000000 + (NvU64)ts.tv_nsec / 1000)/1000;
    return (NvU32)time;
}

void NvDlaDebugPrintf(const char *format, ... )
{
    va_list ap;

    va_start( ap, format );
    vprintf(format, ap);
    va_end( ap );
}

NvDlaError NvDlaStat(const char *filename, NvDlaStatType *stat)
{
    NvDlaFileHandle file;
    NvDlaDirHandle dir;
    NvDlaError err;

    if (!filename || !stat)
        return NvDlaError_BadParameter;

    stat->size = 0;
    stat->type = NvDlaFileType_Unknown;

    if (NvDlaFopen(filename, NVDLA_OPEN_READ, &file) == NvDlaSuccess) {
        err = NvDlaFstat(file, stat);
        NvDlaFclose(file);
        return err;
    }

    if (NvDlaOpendir(filename, &dir) == NvDlaSuccess) {
        stat->type = NvDlaFileType_Directory;
        stat->size = 0;
        NvDlaClosedir(dir);
        return NvDlaSuccess;
    }
    return NvDlaError_BadParameter;
}

NvDlaError NvDlaMkdir(char *dirname)
{
    int err;

    err = mkdir(dirname, S_IRWXU);
    if (err != 0)
        return NvDlaError_FileOperationFailed;

    return NvDlaSuccess;
}

NvDlaError NvDlaFremove(const char *filename)
{
    int err;
    if (!filename)
        return NvDlaError_BadParameter;

    err = unlink(filename);
    if(err != 0)
        return NvDlaError_FileOperationFailed;
    return NvDlaSuccess;
}

NvDlaError NvDlaFopen(const char *path, NvU32 flags,
    NvDlaFileHandle *file)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaFile *f = NULL;
    int fd = -1;
    int permissionFlags = (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    int mode;

    if (!path || !file)
        return NvDlaError_BadParameter;

    e = getOpenMode(flags, &mode);
    if (e != NvDlaSuccess)
        return NvDlaError_BadParameter;

    f = NvDlaAlloc(sizeof(NvDlaFile));
    if (!f)
        return NvDlaError_InsufficientMemory;

    fd = open(path, mode, permissionFlags);
    if (fd < 0) {
        e = NvDlaError_FileOperationFailed;
        goto fail;
    }
    f->fd = fd;

    *file = f;
    return NvDlaSuccess;

fail:
    NvDlaFree(f);
    return e;
}

void NvDlaFclose(NvDlaFileHandle stream)
{
    if (!stream)
        return;

    // TODO: what if close fails. Insert assertions??
    (void)close(stream->fd);
    NvDlaFree(stream);
}

// TODO: Should the FIFO device be considered?
NvDlaError NvDlaFwrite(NvDlaFileHandle stream, const void *ptr, size_t size)
{
    ssize_t len;
    size_t s;
    char *p;

    if (!stream || !ptr)
        return NvDlaError_BadParameter;

    if (!size)
        return NvDlaSuccess;

    s = size;
    p = (char *)ptr;
    do
    {
        len = write(stream->fd, p, s);
        if (len > 0)
        {
            p += len;
            s -= len;
        }
    } while ((len < 0 && (errno == EINTR)) || (s > 0 && len > 0));

    if (len < 0)
        return NvDlaError_FileWriteFailed;

    return NvDlaSuccess;
}

// TODO: Should the FIFO device be considered?
NvDlaError NvDlaFread(NvDlaFileHandle stream, void *ptr,
                size_t size, size_t *bytes)
{
    ssize_t len;
    size_t s;
    char *p;

    if(!stream || !ptr)
        return NvDlaError_BadParameter;

    if (!size) {
        if (bytes) *bytes = 0;
        return NvDlaSuccess;
    }
    if (size > SSIZE_MAX)
        return NvDlaError_BadValue;

    s = size;
    p = (char *)ptr;

    do
    {
        len = read(stream->fd, p, s);
        if (len > 0)
        {
            p += len;
            s -= len;
        }
    } while ((len < 0 && (errno == EINTR)) || (s > 0 && len > 0));

    if (len < 0)
        return NvDlaError_FileReadFailed;

    if (bytes)
        *bytes = size - s;

    if (!len)
        return NvDlaError_EndOfFile;
    return NvDlaSuccess;
}

NvDlaError NvDlaFseek(NvDlaFileHandle file, NvS64 offset, NvDlaSeekEnum whence)
{
    loff_t off;
    int seekMode;

    if(!file)
        return NvDlaError_BadParameter;

    switch(whence) {
    case NvDlaSeek_Set: seekMode = SEEK_SET; break;
    case NvDlaSeek_Cur: seekMode = SEEK_CUR; break;
    case NvDlaSeek_End: seekMode = SEEK_END; break;
    default:
        return NvDlaError_BadParameter;
    }

    off = lseek64(file->fd, (loff_t)offset, seekMode);
    if (off < 0)
        return NvDlaError_FileOperationFailed;
    return NvDlaSuccess;
}

NvDlaError NvDlaFstat(NvDlaFileHandle file, NvDlaStatType *stat)
{
    struct stat64 fs;
    int err;

    if (!stat || !file)
        return NvDlaError_BadParameter;

    err = fstat64(file->fd, &fs);
    if (err != 0)
        return NvDlaError_FileOperationFailed;

    stat->size = (NvU64)fs.st_size;
    stat->mtime = (NvU64)fs.st_mtime;

    if( S_ISREG( fs.st_mode ) ) {
        stat->type = NvDlaFileType_File;
    }
    else if( S_ISDIR( fs.st_mode ) ) {
        stat->type = NvDlaFileType_Directory;
    }
    else if( S_ISFIFO( fs.st_mode ) ) {
        stat->type = NvDlaFileType_Fifo;
    }
    else if( S_ISCHR( fs.st_mode ) ) {
        stat->type = NvDlaFileType_CharacterDevice;
    }
    else if( S_ISBLK( fs.st_mode ) ) {
        stat->type = NvDlaFileType_BlockDevice;
    }
    else {
        stat->type = NvDlaFileType_Unknown;
    }

    return NvDlaSuccess;
}

NvU64 NvDlaStatGetSize(NvDlaStatType *stat)
{
    return stat->size;
}

NvDlaError NvDlaFgetc(NvDlaFileHandle stream, NvU8 *c)
{
    return NvDlaFread(stream, c, 1, NULL);
}

void NvDlaMemset( void *s, NvU8 c, size_t size )
{
    // s should not be NULL!! Assert if required
    if (!s)
        return;

    (void)memset(s, (int)c, size);
}

NvDlaError NvDlaOpendir(const char *path, NvDlaDirHandle *dirHandle)
{
    NvDlaDir *d;
    if (!path || !dirHandle)
        return NvDlaError_BadParameter;

    d = (NvDlaDir *)NvDlaAlloc(sizeof(NvDlaDir));
    if (!d)
        return NvDlaError_InsufficientMemory;

    d->dir = opendir(path);
    if (d->dir == NULL)
        return NvDlaError_DirOperationFailed;

    *dirHandle = d;
    return NvDlaSuccess;
}

NvDlaError NvDlaReaddir(NvDlaDirHandle dirHandle, char *name, size_t size)
{
    struct dirent *d;

    if(!dirHandle || !name)
        return NvDlaError_BadParameter;

    if(!dirHandle->dir)
        return NvDlaError_BadValue;

    d = readdir(dirHandle->dir);
    if (!d)
        return NvDlaError_EndOfDirList;

    (void) strncpy(name, d->d_name, size);
    name[size-1] = '\0';

    return NvDlaSuccess;
}

void NvDlaClosedir(NvDlaDirHandle dirHandle)
{
    if (!dirHandle)
        return;

    if (dirHandle->dir)
        (void) closedir(dirHandle->dir);

    NvDlaFree(dirHandle);
    return;
}
