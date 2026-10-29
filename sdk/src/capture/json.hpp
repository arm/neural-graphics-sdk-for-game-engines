/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <string>

namespace arm::capture
{
    class JsonFileOps
    {
    public:
        virtual ~JsonFileOps() = default;

        virtual FILE* Open(const std::string& path)                       = 0;
        virtual int   Write(FILE* file, const char* format, va_list args) = 0;
        virtual int   Flush(FILE* file)                                   = 0;
        virtual long  Tell(FILE* file)                                    = 0;
        virtual int   Seek(FILE* file, long offset, int origin)           = 0;
        virtual int   Close(FILE* file)                                   = 0;
    };

    // Base mechanics for a JSON metadata file with a fixed-width count placeholder and
    // a streamed array. Derived writers own their provider-specific JSON schema.
    class JsonWriter
    {
    public:
        JsonWriter();
        explicit JsonWriter(JsonFileOps& fileOps);
        virtual ~JsonWriter();

        JsonWriter(const JsonWriter&) = delete;
        JsonWriter& operator=(const JsonWriter&) = delete;

        bool     IsOpen() const;
        uint32_t GetWrittenEntryCount() const;
        void     Reset();

    protected:
        bool Open(const std::string& path);
        void Writef(const char* format, ...);
        bool HasError() const;
        void SetCountOffset();
        void BeginEntry();
        void CommitEntry();
        bool BackfillCountAndClose();

    private:
        FILE*        file_ = nullptr;
        JsonFileOps* fileOps_;
        long         countOffset_       = 0;
        uint32_t     writtenEntryCount_ = 0;
        bool         writeFailed_       = false;
    };

}  // namespace arm::capture
