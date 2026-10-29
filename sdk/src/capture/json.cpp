/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "json.hpp"

#include "file.h"
#include "log.h"

#include <cstdarg>
#include <locale.h>

namespace arm::capture
{
    namespace
    {
        class ClassicNumericLocale final
        {
        public:
            ClassicNumericLocale()
            {
#if defined(_WIN32)
                locale_ = _create_locale(LC_NUMERIC, "C");
#else
                locale_ = newlocale(LC_NUMERIC_MASK, "C", nullptr);
#endif
            }

            ~ClassicNumericLocale()
            {
#if defined(_WIN32)
                if (locale_)
                    _free_locale(locale_);
#else
                if (locale_)
                    freelocale(locale_);
#endif
            }

            ClassicNumericLocale(const ClassicNumericLocale&) = delete;
            ClassicNumericLocale& operator=(const ClassicNumericLocale&) = delete;

            int Write(FILE* file, const char* format, va_list args) const
            {
                if (!locale_)
                    return -1;
#if defined(_WIN32)
                return _vfprintf_l(file, format, locale_, args);
#else
                const locale_t previousLocale = uselocale(locale_);
                const int      result         = std::vfprintf(file, format, args);
                uselocale(previousLocale);
                return result;
#endif
            }

        private:
#if defined(_WIN32)
            _locale_t locale_ = nullptr;
#else
            locale_t locale_ = nullptr;
#endif
        };

        ClassicNumericLocale& GetClassicNumericLocale()
        {
            static ClassicNumericLocale locale;
            return locale;
        }

        class StdioJsonFileOps final : public JsonFileOps
        {
        public:
            FILE* Open(const std::string& path) override
            {
                return OpenFileForWrite(path, "w");
            }

            int Write(FILE* file, const char* format, va_list args) override
            {
                return GetClassicNumericLocale().Write(file, format, args);
            }

            int Flush(FILE* file) override
            {
                return std::fflush(file);
            }

            long Tell(FILE* file) override
            {
                return std::ftell(file);
            }

            int Seek(FILE* file, long offset, int origin) override
            {
                return std::fseek(file, offset, origin);
            }

            int Close(FILE* file) override
            {
                return std::fclose(file);
            }
        };

        JsonFileOps& DefaultJsonFileOps()
        {
            static StdioJsonFileOps fileOps;
            return fileOps;
        }
    }  // namespace

    JsonWriter::JsonWriter()
        : fileOps_(&DefaultJsonFileOps())
    {
    }

    JsonWriter::JsonWriter(JsonFileOps& fileOps)
        : fileOps_(&fileOps)
    {
    }

    JsonWriter::~JsonWriter()
    {
        Reset();
    }

    bool JsonWriter::IsOpen() const
    {
        return file_ != nullptr;
    }

    bool JsonWriter::HasError() const
    {
        return writeFailed_;
    }

    uint32_t JsonWriter::GetWrittenEntryCount() const
    {
        return writtenEntryCount_;
    }

    void JsonWriter::Reset()
    {
        if (file_ && fileOps_->Close(file_) != 0)
            ARM_CAPTURE_LOG_ERROR("CAPTURE", "failed to close metadata file");
        file_              = nullptr;
        countOffset_       = 0;
        writtenEntryCount_ = 0;
        writeFailed_       = false;
    }

    bool JsonWriter::Open(const std::string& path)
    {
        Reset();
        file_ = fileOps_->Open(path);
        if (!file_)
            writeFailed_ = true;
        return !writeFailed_;
    }

    void JsonWriter::Writef(const char* format, ...)
    {
        if (!file_ || writeFailed_)
            return;

        va_list args;
        va_start(args, format);
        if (fileOps_->Write(file_, format, args) < 0)
            writeFailed_ = true;
        va_end(args);
    }

    void JsonWriter::SetCountOffset()
    {
        if (!file_ || writeFailed_)
            return;
        countOffset_ = fileOps_->Tell(file_);
        if (countOffset_ < 0)
            writeFailed_ = true;
    }

    void JsonWriter::BeginEntry()
    {
        if (writtenEntryCount_ != 0)
            Writef(",\n");
    }

    void JsonWriter::CommitEntry()
    {
        if (!file_ || writeFailed_)
            return;
        if (fileOps_->Flush(file_) != 0)
            writeFailed_ = true;
        else
            ++writtenEntryCount_;
    }

    bool JsonWriter::BackfillCountAndClose()
    {
        if (!file_)
            return false;

        if (!writeFailed_)
        {
            if (fileOps_->Flush(file_) != 0)
                writeFailed_ = true;
            else if (fileOps_->Seek(file_, countOffset_, SEEK_SET) != 0)
                writeFailed_ = true;
            else
            {
                Writef("%10u", writtenEntryCount_);
                if (!writeFailed_ && fileOps_->Flush(file_) != 0)
                    writeFailed_ = true;
            }
        }

        const int closeResult = fileOps_->Close(file_);
        file_                 = nullptr;
        if (closeResult != 0)
            writeFailed_ = true;
        return !writeFailed_;
    }

}  // namespace arm::capture
