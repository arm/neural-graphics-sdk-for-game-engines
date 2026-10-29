/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <limits>
#include <string>
#include <vector>

namespace unit_test
{
    using TestFunction = bool (*)();

    struct TestCase
    {
        std::string  suite;
        std::string  name;
        std::string  fullName;
        const char*  file            = "";
        int          line            = 0;
        TestFunction function        = nullptr;
        bool         expectedFailure = false;
    };

    class Registry
    {
    public:
        static Registry& Instance();

        void Register(const char* suite, const char* name, TestFunction function, bool expectedFailure, const char* file, int line);

        const std::vector<TestCase>& GetAll() const;
        const TestCase*              FindByName(const std::string& fullName) const;

    private:
        std::vector<TestCase> tests_;
    };

    struct RunnerOptions
    {
        std::string              executablePath;
        std::vector<std::string> originalArgs;
        std::vector<std::string> suites;
        std::vector<std::string> tests;
        int                      suiteRangeStart     = 1;
        int                      suiteRangeEnd       = std::numeric_limits<int>::max();
        int                      testRangeStart      = 1;
        int                      testRangeEnd        = std::numeric_limits<int>::max();
        int                      benchmarkIterations = 1;
        bool                     hasSuiteRange       = false;
        bool                     hasTestRange        = false;
        bool                     listSuitesOnly      = false;
        bool                     listTestsOnly       = false;
        bool                     listDevicesOnly     = false;
        bool                     isolate             = true;
        bool                     verbose             = false;
        bool                     runExpectedFailures = false;
        bool                     showHelp            = false;
        bool                     hasError            = false;
        std::string              errorMessage;
        std::string              internalRunName;
        std::string              selectedDevice;
    };

    enum class TestRunStatus
    {
        Pass,
        Skip,
        Fail,
        Crash,
    };

    struct SingleRunResult
    {
        TestRunStatus status = TestRunStatus::Fail;
        std::string   message;
        double        durationMs = 0.0;
    };

    class AutoRegister
    {
    public:
        AutoRegister(const char* suite, const char* name, TestFunction function, bool expectedFailure, const char* file, int line);
    };

    RunnerOptions ParseOptions(int argc, char** argv);
    void          PrintHelp();
    int           RunAll(const RunnerOptions& options);
    int           RunInternal(const std::string& fullName);
    void          ReportCheckFailure(const char* expression, const char* file, int line);
    void          MarkTestFailed(const std::string& reason);
    void          MarkTestSkipped(const std::string& reason);

    class ScopedCaptureEnvironment
    {
    public:
        ScopedCaptureEnvironment();
        ~ScopedCaptureEnvironment();

        ScopedCaptureEnvironment(const ScopedCaptureEnvironment&) = delete;
        ScopedCaptureEnvironment& operator=(const ScopedCaptureEnvironment&) = delete;

        static bool SetVariable(const char* name, const char* value);
        static bool ClearVariable(const char* name);

    private:
        struct Entry
        {
            std::string name;
            std::string value;
            bool        hadValue = false;
        };

        std::vector<Entry> savedEntries_;
    };

}  // namespace unit_test

#define UNIT_TEST_IMPL_CONCAT_INNER(a, b)         a##b
#define UNIT_TEST_IMPL_CONCAT(a, b)               UNIT_TEST_IMPL_CONCAT_INNER(a, b)
#define UNIT_TEST_IMPL_MAKE_NAME(prefix, counter) UNIT_TEST_IMPL_CONCAT(prefix, counter)

#define UNIT_TEST_CASE_INTERNAL(suiteName, testName, expectFailure, counter)                                        \
    static bool                    UNIT_TEST_IMPL_MAKE_NAME(unit_test_fn_, counter)();                              \
    static unit_test::AutoRegister UNIT_TEST_IMPL_MAKE_NAME(unit_test_reg_, counter)(                               \
        suiteName, testName, &UNIT_TEST_IMPL_MAKE_NAME(unit_test_fn_, counter), expectFailure, __FILE__, __LINE__); \
    static bool UNIT_TEST_IMPL_MAKE_NAME(unit_test_fn_, counter)()

#define UNIT_TEST_CASE(suiteName, testName)             UNIT_TEST_CASE_INTERNAL(#suiteName, #testName, false, __COUNTER__)
#define UNIT_TEST_CASE_EXPECT_FAIL(suiteName, testName) UNIT_TEST_CASE_INTERNAL(#suiteName, #testName, true, __COUNTER__)

#define UNIT_TEST_CHECK(expression)                                         \
    do                                                                      \
    {                                                                       \
        if (!(expression))                                                  \
        {                                                                   \
            unit_test::ReportCheckFailure(#expression, __FILE__, __LINE__); \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define UNIT_TEST_FAIL(reason)             \
    do                                     \
    {                                      \
        unit_test::MarkTestFailed(reason); \
        return false;                      \
    } while (0)

#define UNIT_TEST_SKIP(reason)              \
    do                                      \
    {                                       \
        unit_test::MarkTestSkipped(reason); \
        return true;                        \
    } while (0)
