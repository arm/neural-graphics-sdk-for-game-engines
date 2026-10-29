/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "test_framework.h"
#include "vk_test_utils.h"
#include <capture/config.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace
{
    constexpr int    kExitPass     = 0;
    constexpr int    kExitFail     = 1;
    constexpr int    kExitSkip     = 77;
    constexpr size_t kSectionWidth = 87;

    thread_local std::string g_failureReason;
    thread_local std::string g_skipReason;

#if defined(__ANDROID__)
    constexpr const char* kCapturePrefixes[] = {
        arm::capture::kAndroidNssPropertyPrefix,
        arm::capture::kAndroidNfruPropertyPrefix,
        "debug.ngsdk.test.",
    };
#else
    constexpr const char* kCapturePrefixes[] = {
        arm::capture::kDesktopNssEnvironmentPrefix,
        arm::capture::kDesktopNfruEnvironmentPrefix,
        "TEST_CAPTURE_",
        "TEST_HAS_CONFIG_",
    };
#endif

    struct SuiteEntry
    {
        int                                     id = 0;
        std::string                             name;
        std::vector<const unit_test::TestCase*> tests;
    };

    struct TestEntry
    {
        const unit_test::TestCase* test    = nullptr;
        int                        suiteId = 0;
        int                        testId  = 0;
    };

    struct TestCatalog
    {
        std::vector<SuiteEntry> suites;
        std::vector<TestEntry>  tests;
    };

    struct TestSelection
    {
        std::vector<const SuiteEntry*> suites;
        std::vector<TestEntry>         tests;
    };

    struct AggregatedRunResult
    {
        unit_test::TestRunStatus status = unit_test::TestRunStatus::Fail;
        std::string              message;
        double                   averageMs = 0.0;
    };

    struct FailureRecord
    {
        std::string name;
        std::string reason;
    };

    constexpr const char* kSelectedVulkanDeviceEnv      = "SDK_UNIT_TEST_VK_DEVICE";
    constexpr const char* kSelectedVulkanDeviceLabelEnv = "SDK_UNIT_TEST_VK_DEVICE_LABEL";

    std::string MakeSeparator(const char fill)
    {
        return std::string(kSectionWidth, fill);
    }

    bool ApplySelectedVulkanDevice(const std::string& deviceSelector, std::string* errorMessage)
    {
#if defined(_WIN32)
        const int rc = _putenv_s(kSelectedVulkanDeviceEnv, deviceSelector.c_str());
        if (rc != 0)
        {
            if (errorMessage)
            {
                *errorMessage = "failed to set Vulkan device environment variable";
            }
            return false;
        }
        return true;
#else
        if (deviceSelector.empty())
        {
            if (unsetenv(kSelectedVulkanDeviceEnv) != 0)
            {
                if (errorMessage)
                {
                    *errorMessage = "failed to clear Vulkan device environment variable";
                }
                return false;
            }
            return true;
        }

        if (setenv(kSelectedVulkanDeviceEnv, deviceSelector.c_str(), 1) != 0)
        {
            if (errorMessage)
            {
                *errorMessage = "failed to set Vulkan device environment variable";
            }
            return false;
        }
        return true;
#endif
    }

    std::string QuoteArg(const std::string& value)
    {
#if defined(_WIN32)
        return "\"" + value + "\"";
#else
        std::string out = "\"";
        for (char ch : value)
        {
            if (ch == '\\' || ch == '"')
            {
                out.push_back('\\');
            }
            out.push_back(ch);
        }
        out.push_back('"');
        return out;
#endif
    }

    std::string Trim(const std::string& value)
    {
        size_t begin = 0;
        while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0)
        {
            ++begin;
        }

        size_t end = value.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
        {
            --end;
        }

        return value.substr(begin, end - begin);
    }

    std::string ReadTextFile(const std::filesystem::path& path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            return {};
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    std::filesystem::path ResolveGitDirectory()
    {
        std::error_code             gitPathError;
        const std::filesystem::path gitPath = ".git";
        if (std::filesystem::is_directory(gitPath, gitPathError) && !gitPathError)
        {
            return gitPath;
        }

        std::error_code gitFileError;
        if (!std::filesystem::is_regular_file(gitPath, gitFileError) || gitFileError)
        {
            return {};
        }

        const std::string     gitFileContents = Trim(ReadTextFile(gitPath));
        constexpr const char* kGitDirPrefix   = "gitdir:";
        if (gitFileContents.rfind(kGitDirPrefix, 0) != 0)
        {
            return {};
        }

        std::filesystem::path gitDir = Trim(gitFileContents.substr(std::strlen(kGitDirPrefix)));
        if (gitDir.is_relative())
        {
            std::error_code cwdError;
            const auto      cwd = std::filesystem::current_path(cwdError);
            if (cwdError)
            {
                return {};
            }
            gitDir = cwd / gitDir;
        }

        return gitDir;
    }

    std::string ResolvePackedRef(const std::filesystem::path& gitDirectory, const std::string& refName)
    {
        std::istringstream packedRefs(ReadTextFile(gitDirectory / "packed-refs"));
        std::string        line;
        while (std::getline(packedRefs, line))
        {
            const std::string trimmed = Trim(line);
            if (trimmed.empty() || trimmed[0] == '#' || trimmed[0] == '^')
            {
                continue;
            }

            const size_t separator = trimmed.find(' ');
            if (separator == std::string::npos)
            {
                continue;
            }

            const std::string currentRef = Trim(trimmed.substr(separator + 1));
            if (currentRef == refName)
            {
                return Trim(trimmed.substr(0, separator));
            }
        }

        return {};
    }

    std::filesystem::path ResolveCommonGitDirectory(const std::filesystem::path& gitDirectory)
    {
        const std::string commondirContents = Trim(ReadTextFile(gitDirectory / "commondir"));
        if (commondirContents.empty())
        {
            return gitDirectory;
        }

        std::filesystem::path commonDir = commondirContents;
        if (commonDir.is_relative())
        {
            commonDir = gitDirectory / commonDir;
        }
        return commonDir;
    }

    std::string GetRepositoryCommit()
    {
        const std::filesystem::path gitDirectory = ResolveGitDirectory();
        if (gitDirectory.empty())
        {
            return "unknown";
        }
        const std::filesystem::path commonGitDirectory = ResolveCommonGitDirectory(gitDirectory);

        const std::string headContents = Trim(ReadTextFile(gitDirectory / "HEAD"));
        if (headContents.empty())
        {
            return "unknown";
        }

        std::string commit;
        if (headContents.rfind("ref:", 0) == 0)
        {
            const std::string refName = Trim(headContents.substr(4));
            commit                    = Trim(ReadTextFile(gitDirectory / refName));
            if (commit.empty() && commonGitDirectory != gitDirectory)
            {
                commit = Trim(ReadTextFile(commonGitDirectory / refName));
            }
            if (commit.empty())
            {
                commit = ResolvePackedRef(gitDirectory, refName);
            }
            if (commit.empty() && commonGitDirectory != gitDirectory)
            {
                commit = ResolvePackedRef(commonGitDirectory, refName);
            }
        }
        else
        {
            commit = headContents;
        }

        if (commit.empty())
        {
            return "unknown";
        }

        if (commit.size() > 12)
        {
            commit.resize(12);
        }

        return commit;
    }

    std::string BuildCommandLineString(const unit_test::RunnerOptions& options)
    {
        std::ostringstream stream;
        stream << QuoteArg(options.executablePath);
        for (const std::string& arg : options.originalArgs)
        {
            stream << ' ' << QuoteArg(arg);
        }
        return stream.str();
    }

    void VerboseLog(const unit_test::RunnerOptions& options, const std::string& message)
    {
        if (!options.verbose)
        {
            return;
        }

        std::cout << "[verbose] " << message << '\n';
    }

    std::string GetSelectedDeviceLabel(const unit_test::RunnerOptions& options)
    {
        if (const char* label = std::getenv(kSelectedVulkanDeviceLabelEnv))
        {
            const std::string trimmedLabel = Trim(label);
            if (!trimmedLabel.empty())
            {
                return trimmedLabel;
            }
        }

        const std::string selector = unit_test::vk::ResolveVulkanDeviceSelector(options.selectedDevice, std::getenv(kSelectedVulkanDeviceEnv));

        if (selector.empty())
        {
            return "<default>";
        }

        std::vector<unit_test::vk::VulkanPhysicalDeviceInfo> devices;
        std::string                                          deviceError;
        if (!unit_test::vk::EnumerateVulkanDevices(devices, &deviceError) || devices.empty())
        {
            return selector;
        }

        uint32_t    selectedIndex = 0;
        std::string selectionError;
        if (unit_test::vk::SelectVulkanDeviceIndex(devices, selector, selectedIndex, &selectionError))
        {
            const unit_test::vk::VulkanPhysicalDeviceInfo& device = devices[selectedIndex];
            return "[" + std::to_string(device.index) + "] " + device.name;
        }

        return selector;
    }

    void PrintMetadataSection(const unit_test::RunnerOptions& options)
    {
        const std::string plusLine = MakeSeparator('+');
        std::cout << plusLine << '\n';
        std::cout << "Selected Vulkan device/backend: " << GetSelectedDeviceLabel(options) << '\n';
        std::cout << "Parameters: " << BuildCommandLineString(options) << '\n';
        std::cout << "SDK commit: " << GetRepositoryCommit() << '\n';
        std::cout << plusLine << '\n';
    }

    void PrintSuiteSectionHeader(const int suiteId, const std::string& suiteName)
    {
        const std::string equalsLine = MakeSeparator('=');
        std::cout << '\n' << equalsLine << '\n';
        std::cout << "                  Running suite " << suiteId << " by `--suite " << suiteName << "`\n";
        std::cout << equalsLine << '\n';
    }

    void PrintTestSectionHeader(const std::string& fullTestName)
    {
        const std::string dashLine = MakeSeparator('-');
        std::cout << '\n' << dashLine << '\n';
        std::cout << "--test " << fullTestName << '\n';
        std::cout << dashLine << '\n';
    }

    void PrintSummarySectionHeader()
    {
        const std::string plusLine = MakeSeparator('+');
        std::cout << '\n' << plusLine << '\n';
        std::cout << "                                      Summary\n";
        std::cout << plusLine << '\n';
    }

    std::string ExtractFailureReason(const std::string& output)
    {
        std::istringstream stream(output);
        std::string        line;
        std::string        lastNonEmptyLine;

        while (std::getline(stream, line))
        {
            const std::string trimmed = Trim(line);
            if (!trimmed.empty())
            {
                lastNonEmptyLine = trimmed;
            }
        }

        return lastNonEmptyLine;
    }

    bool TryParsePositiveInt(const std::string& value, int& outValue)
    {
        if (value.empty())
        {
            return false;
        }

        for (char ch : value)
        {
            if (!std::isdigit(static_cast<unsigned char>(ch)))
            {
                return false;
            }
        }

        char*      endPtr = nullptr;
        const long parsed = std::strtol(value.c_str(), &endPtr, 10);
        if (endPtr == nullptr || *endPtr != '\0' || parsed < 1 || parsed > std::numeric_limits<int>::max())
        {
            return false;
        }

        outValue = static_cast<int>(parsed);
        return true;
    }

    bool ParseRange(const std::string& value, int& outStart, int& outEnd)
    {
        const size_t separator = value.find(':');
        if (separator == std::string::npos || separator == 0 || separator == value.size() - 1)
        {
            return false;
        }

        const std::string startToken = value.substr(0, separator);
        const std::string endToken   = value.substr(separator + 1);
        if (!TryParsePositiveInt(startToken, outStart) || !TryParsePositiveInt(endToken, outEnd))
        {
            return false;
        }

        return outEnd >= outStart;
    }

    TestCatalog BuildCatalog()
    {
        TestCatalog                             catalog;
        std::unordered_map<std::string, size_t> suiteIndexByName;

        for (const unit_test::TestCase& test : unit_test::Registry::Instance().GetAll())
        {
            auto it = suiteIndexByName.find(test.suite);
            if (it == suiteIndexByName.end())
            {
                SuiteEntry suite;
                suite.id   = static_cast<int>(catalog.suites.size()) + 1;
                suite.name = test.suite;
                catalog.suites.push_back(std::move(suite));
                it = suiteIndexByName.emplace(test.suite, catalog.suites.size() - 1).first;
            }

            SuiteEntry& suite = catalog.suites[it->second];
            suite.tests.push_back(&test);

            TestEntry entry;
            entry.test    = &test;
            entry.suiteId = suite.id;
            entry.testId  = static_cast<int>(suite.tests.size());
            catalog.tests.push_back(entry);
        }

        return catalog;
    }

    const SuiteEntry* FindSuiteById(const TestCatalog& catalog, int suiteId)
    {
        if (suiteId < 1 || suiteId > static_cast<int>(catalog.suites.size()))
        {
            return nullptr;
        }
        return &catalog.suites[static_cast<size_t>(suiteId - 1)];
    }

    const SuiteEntry* FindSuiteByName(const TestCatalog& catalog, const std::string& suiteName)
    {
        for (const SuiteEntry& suite : catalog.suites)
        {
            if (suite.name == suiteName)
            {
                return &suite;
            }
        }
        return nullptr;
    }

    bool ResolveSuiteId(const TestCatalog& catalog, const std::string& token, int& outSuiteId)
    {
        if (TryParsePositiveInt(token, outSuiteId))
        {
            return FindSuiteById(catalog, outSuiteId) != nullptr;
        }

        const SuiteEntry* suite = FindSuiteByName(catalog, token);
        if (suite == nullptr)
        {
            return false;
        }

        outSuiteId = suite->id;
        return true;
    }

    TestSelection CollectSelection(const TestCatalog& catalog, const unit_test::RunnerOptions& options, std::string* errorMessage)
    {
        TestSelection     selection;
        std::vector<bool> selectedSuiteByIndex(catalog.suites.size(), false);
        const bool        hasSuiteFilters = !options.suites.empty() || options.hasSuiteRange;

        if (!hasSuiteFilters)
        {
            std::fill(selectedSuiteByIndex.begin(), selectedSuiteByIndex.end(), true);
        }

        for (const std::string& suiteToken : options.suites)
        {
            int suiteId = 0;
            if (!ResolveSuiteId(catalog, suiteToken, suiteId))
            {
                if (errorMessage)
                {
                    *errorMessage = "unknown suite: " + suiteToken;
                }
                return {};
            }
            selectedSuiteByIndex[static_cast<size_t>(suiteId - 1)] = true;
        }

        if (options.hasSuiteRange)
        {
            for (const SuiteEntry& suite : catalog.suites)
            {
                if (suite.id >= options.suiteRangeStart && suite.id <= options.suiteRangeEnd)
                {
                    selectedSuiteByIndex[static_cast<size_t>(suite.id - 1)] = true;
                }
            }
        }

        for (size_t i = 0; i < selectedSuiteByIndex.size(); ++i)
        {
            if (selectedSuiteByIndex[i])
            {
                selection.suites.push_back(&catalog.suites[i]);
            }
        }

        std::unordered_set<std::string> selectedTests;
        if (!options.tests.empty())
        {
            selectedTests.insert(options.tests.begin(), options.tests.end());
        }

        for (const TestEntry& entry : catalog.tests)
        {
            if (!selectedSuiteByIndex[static_cast<size_t>(entry.suiteId - 1)])
            {
                continue;
            }

            if (options.hasTestRange && (entry.testId < options.testRangeStart || entry.testId > options.testRangeEnd))
            {
                continue;
            }

            if (!selectedTests.empty() && selectedTests.find(entry.test->fullName) == selectedTests.end())
            {
                continue;
            }

            selection.tests.push_back(entry);
        }

        std::sort(selection.tests.begin(), selection.tests.end(), [](const TestEntry& lhs, const TestEntry& rhs) {
            if (lhs.suiteId != rhs.suiteId)
            {
                return lhs.suiteId < rhs.suiteId;
            }
            return lhs.testId < rhs.testId;
        });

        return selection;
    }

    void PrintSuites(const std::vector<const SuiteEntry*>& suites)
    {
        for (const SuiteEntry* suite : suites)
        {
            std::cout << "[" << suite->id << "] " << suite->name << " (" << suite->tests.size() << " tests)\n";
        }
    }

    void PrintTests(const std::vector<TestEntry>& tests)
    {
        for (const TestEntry& entry : tests)
        {
            std::cout << "[" << entry.suiteId << ":" << entry.testId << "] " << entry.test->fullName;
            if (entry.test->expectedFailure)
            {
                std::cout << " [expected-fail]";
            }
            std::cout << '\n';
        }
    }

    unit_test::SingleRunResult RunInProcess(const unit_test::TestCase& test)
    {
        unit_test::SingleRunResult result;
        g_failureReason.clear();
        g_skipReason.clear();
        const auto start = std::chrono::steady_clock::now();

        try
        {
            bool passed = false;
            {
                unit_test::ScopedCaptureEnvironment captureEnvironmentScope;
                passed = test.function();
            }
            if (!g_failureReason.empty())
            {
                result.status  = unit_test::TestRunStatus::Fail;
                result.message = g_failureReason;
            }
            else if (!passed)
            {
                result.status  = unit_test::TestRunStatus::Fail;
                result.message = "test returned false";
            }
            else if (!g_skipReason.empty())
            {
                result.status  = unit_test::TestRunStatus::Skip;
                result.message = g_skipReason;
            }
            else
            {
                result.status = unit_test::TestRunStatus::Pass;
            }
        }
        catch (const std::exception& ex)
        {
            result.status  = unit_test::TestRunStatus::Fail;
            result.message = std::string("exception: ") + ex.what();
        }
        catch (...)
        {
            result.status  = unit_test::TestRunStatus::Crash;
            result.message = "unknown fatal exception";
        }

        const auto end    = std::chrono::steady_clock::now();
        result.durationMs = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }

    unit_test::SingleRunResult RunIsolated(const unit_test::TestCase& test, const unit_test::RunnerOptions& options)
    {
        unit_test::SingleRunResult result;
        const auto                 start = std::chrono::steady_clock::now();

        std::string command = QuoteArg(options.executablePath) + " --internal-run " + QuoteArg(test.fullName);
        if (options.verbose)
        {
            command += " --verbose";
        }
        if (!options.selectedDevice.empty())
        {
            command += " --select-device " + QuoteArg(options.selectedDevice);
        }

        std::filesystem::path outputPath;
        std::error_code       pathError;
        const auto            tempDir = std::filesystem::temp_directory_path(pathError);
        if (!pathError)
        {
            outputPath = tempDir / ("sdk_unit_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".log");
            command += " > " + QuoteArg(outputPath.string()) + " 2>&1";
        }

#if defined(_WIN32)
        command = "\"" + command + "\"";
#endif

        VerboseLog(options, "Running isolated test: " + test.fullName);
        VerboseLog(options, "Subprocess command: " + command);

        const int   systemResult = std::system(command.c_str());
        std::string subprocessOutput;
        if (!outputPath.empty())
        {
            subprocessOutput = ReadTextFile(outputPath);
            std::error_code removeError;
            std::filesystem::remove(outputPath, removeError);
        }

        if (options.verbose && !subprocessOutput.empty())
        {
            std::cout << subprocessOutput;
            if (subprocessOutput.back() != '\n')
            {
                std::cout << '\n';
            }
        }

        const std::string capturedReason = ExtractFailureReason(subprocessOutput);
        if (systemResult == -1)
        {
            result.status  = unit_test::TestRunStatus::Fail;
            result.message = capturedReason.empty() ? "std::system failed to launch test subprocess" : capturedReason;
        }
        else
        {
#if defined(_WIN32)
            if (systemResult == kExitPass)
            {
                result.status = unit_test::TestRunStatus::Pass;
            }
            else if (systemResult == kExitSkip)
            {
                result.status  = unit_test::TestRunStatus::Skip;
                result.message = capturedReason.empty() ? "test skipped without a reason" : capturedReason;
            }
            else
            {
                result.status  = unit_test::TestRunStatus::Fail;
                result.message = capturedReason.empty() ? "subprocess exit code " + std::to_string(systemResult) : capturedReason;
            }
#else
            if (WIFEXITED(systemResult))
            {
                const int exitCode = WEXITSTATUS(systemResult);
                if (exitCode == kExitPass)
                {
                    result.status = unit_test::TestRunStatus::Pass;
                }
                else if (exitCode == kExitSkip)
                {
                    result.status  = unit_test::TestRunStatus::Skip;
                    result.message = capturedReason.empty() ? "test skipped without a reason" : capturedReason;
                }
                else
                {
                    result.status  = unit_test::TestRunStatus::Fail;
                    result.message = capturedReason.empty() ? "subprocess exit code " + std::to_string(exitCode) : capturedReason;
                }
            }
            else if (WIFSIGNALED(systemResult))
            {
                result.status  = unit_test::TestRunStatus::Crash;
                result.message = capturedReason.empty() ? "terminated by signal " + std::to_string(WTERMSIG(systemResult)) : capturedReason;
            }
            else
            {
                result.status  = unit_test::TestRunStatus::Crash;
                result.message = capturedReason.empty() ? "abnormal subprocess termination" : capturedReason;
            }
#endif
        }

        const auto end    = std::chrono::steady_clock::now();
        result.durationMs = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }

    AggregatedRunResult RunWithBenchmark(const unit_test::TestCase& test, const unit_test::RunnerOptions& options)
    {
        AggregatedRunResult aggregate;
        bool                sawPass  = false;
        bool                sawSkip  = false;
        bool                sawFail  = false;
        bool                sawCrash = false;
        std::string         firstFailureMessage;
        std::string         firstSkipMessage;
        double              totalDurationMs = 0.0;

        for (int runIndex = 0; runIndex < options.benchmarkIterations; ++runIndex)
        {
            const unit_test::SingleRunResult runResult = options.isolate ? RunIsolated(test, options) : RunInProcess(test);
            totalDurationMs += runResult.durationMs;

            switch (runResult.status)
            {
            case unit_test::TestRunStatus::Pass:
                sawPass = true;
                break;
            case unit_test::TestRunStatus::Skip:
                sawSkip = true;
                if (firstSkipMessage.empty() && !runResult.message.empty())
                {
                    firstSkipMessage = runResult.message;
                }
                break;
            case unit_test::TestRunStatus::Fail:
                sawFail = true;
                if (firstFailureMessage.empty() && !runResult.message.empty())
                {
                    firstFailureMessage = runResult.message;
                }
                break;
            case unit_test::TestRunStatus::Crash:
                sawCrash = true;
                if (firstFailureMessage.empty() && !runResult.message.empty())
                {
                    firstFailureMessage = runResult.message;
                }
                break;
            }
        }

        if (sawCrash)
        {
            aggregate.status  = unit_test::TestRunStatus::Crash;
            aggregate.message = firstFailureMessage;
        }
        else if (sawFail)
        {
            aggregate.status  = unit_test::TestRunStatus::Fail;
            aggregate.message = firstFailureMessage;
        }
        else if (sawPass && sawSkip)
        {
            aggregate.status  = unit_test::TestRunStatus::Fail;
            aggregate.message = "test had inconsistent pass/skip results across benchmark iterations";
            if (!firstSkipMessage.empty())
            {
                aggregate.message += ": " + firstSkipMessage;
            }
        }
        else if (sawSkip)
        {
            aggregate.status  = unit_test::TestRunStatus::Skip;
            aggregate.message = firstSkipMessage;
        }
        else if (sawPass)
        {
            aggregate.status = unit_test::TestRunStatus::Pass;
        }

        aggregate.averageMs = totalDurationMs / static_cast<double>(options.benchmarkIterations);
        return aggregate;
    }

}  // namespace

namespace unit_test
{
    Registry& Registry::Instance()
    {
        static Registry registry;
        return registry;
    }

    void Registry::Register(const char* suite, const char* name, TestFunction function, bool expectedFailure, const char* file, int line)
    {
        TestCase test{};
        test.suite           = suite;
        test.name            = name;
        test.fullName        = test.suite + "." + test.name;
        test.file            = file;
        test.line            = line;
        test.function        = function;
        test.expectedFailure = expectedFailure;
        tests_.push_back(test);
    }

    const std::vector<TestCase>& Registry::GetAll() const
    {
        return tests_;
    }

    const TestCase* Registry::FindByName(const std::string& fullName) const
    {
        for (const TestCase& test : tests_)
        {
            if (test.fullName == fullName)
            {
                return &test;
            }
        }
        return nullptr;
    }

    AutoRegister::AutoRegister(const char* suite, const char* name, TestFunction function, bool expectedFailure, const char* file, int line)
    {
        Registry::Instance().Register(suite, name, function, expectedFailure, file, line);
    }

    RunnerOptions ParseOptions(int argc, char** argv)
    {
        RunnerOptions options{};

        if (argc > 0 && argv[0] != nullptr)
        {
            std::error_code error;
            const auto      absolutePath = std::filesystem::absolute(argv[0], error);
            options.executablePath       = error ? std::string(argv[0]) : absolutePath.string();
        }

        for (int i = 1; i < argc; ++i)
        {
            options.originalArgs.emplace_back(argv[i]);
        }

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--list" || arg == "--list-tests")
            {
                options.listTestsOnly = true;
            }
            else if (arg == "--list-device")
            {
                options.listDevicesOnly = true;
            }
            else if (arg == "--list-suites")
            {
                options.listSuitesOnly = true;
            }
            else if (arg == "--expected_failure")
            {
                options.runExpectedFailures = true;
            }
            else if (arg == "--select-device")
            {
                if (i + 1 >= argc)
                {
                    options.hasError     = true;
                    options.errorMessage = "--select-device requires an argument";
                    return options;
                }
                options.selectedDevice = Trim(argv[++i]);
                if (options.selectedDevice.empty())
                {
                    options.hasError     = true;
                    options.errorMessage = "--select-device requires a non-empty value";
                    return options;
                }
            }
            else if (arg.rfind("--select-device=", 0) == 0)
            {
                options.selectedDevice = Trim(arg.substr(std::string("--select-device=").size()));
                if (options.selectedDevice.empty())
                {
                    options.hasError     = true;
                    options.errorMessage = "--select-device requires a non-empty value";
                    return options;
                }
            }
            else if (arg == "--suite")
            {
                if (i + 1 >= argc)
                {
                    options.hasError     = true;
                    options.errorMessage = "--suite requires an argument";
                    return options;
                }
                options.suites.emplace_back(argv[++i]);
            }
            else if (arg == "--test")
            {
                if (i + 1 >= argc)
                {
                    options.hasError     = true;
                    options.errorMessage = "--test requires an argument";
                    return options;
                }
                options.tests.emplace_back(argv[++i]);
            }
            else if (arg == "--suite-range")
            {
                if (i + 1 >= argc)
                {
                    options.hasError     = true;
                    options.errorMessage = "--suite-range requires <start:end>";
                    return options;
                }
                if (!ParseRange(argv[++i], options.suiteRangeStart, options.suiteRangeEnd))
                {
                    options.hasError     = true;
                    options.errorMessage = "invalid --suite-range format (expected start:end)";
                    return options;
                }
                options.hasSuiteRange = true;
            }
            else if (arg == "--test-range" || arg == "--range")
            {
                if (i + 1 >= argc)
                {
                    options.hasError     = true;
                    options.errorMessage = "--test-range requires <start:end>";
                    return options;
                }
                if (!ParseRange(argv[++i], options.testRangeStart, options.testRangeEnd))
                {
                    options.hasError     = true;
                    options.errorMessage = "invalid --test-range format (expected start:end)";
                    return options;
                }
                options.hasTestRange = true;
            }
            else if (arg == "--benchmark")
            {
                if (i + 1 >= argc || !TryParsePositiveInt(argv[i + 1], options.benchmarkIterations))
                {
                    options.hasError     = true;
                    options.errorMessage = "--benchmark requires a positive integer";
                    return options;
                }
                ++i;
            }
            else if (arg.rfind("--benchmark=", 0) == 0)
            {
                if (!TryParsePositiveInt(arg.substr(std::string("--benchmark=").size()), options.benchmarkIterations))
                {
                    options.hasError     = true;
                    options.errorMessage = "--benchmark requires a positive integer";
                    return options;
                }
            }
            else if (arg == "--from")
            {
                if (i + 1 >= argc || !TryParsePositiveInt(argv[i + 1], options.testRangeStart))
                {
                    options.hasError     = true;
                    options.errorMessage = "--from requires a positive integer";
                    return options;
                }
                ++i;
                options.hasTestRange = true;
            }
            else if (arg == "--to")
            {
                if (i + 1 >= argc || !TryParsePositiveInt(argv[i + 1], options.testRangeEnd))
                {
                    options.hasError     = true;
                    options.errorMessage = "--to requires a positive integer";
                    return options;
                }
                ++i;
                options.hasTestRange = true;
            }
            else if (arg == "--no-isolation")
            {
                options.isolate = false;
            }
            else if (arg == "--verbose" || arg == "-v")
            {
                options.verbose = true;
            }
            else if (arg == "--help" || arg == "-h")
            {
                options.showHelp = true;
                return options;
            }
            else if (arg == "--internal-run")
            {
                if (i + 1 >= argc)
                {
                    options.hasError     = true;
                    options.errorMessage = "--internal-run requires <suite.test>";
                    return options;
                }
                options.internalRunName = argv[++i];
            }
            else
            {
                options.hasError     = true;
                options.errorMessage = "unknown option: " + arg;
                return options;
            }
        }

        if (options.hasSuiteRange && options.suiteRangeEnd < options.suiteRangeStart)
        {
            options.hasError     = true;
            options.errorMessage = "invalid suite range values";
            return options;
        }

        if (options.hasTestRange && options.testRangeEnd < options.testRangeStart)
        {
            options.hasError     = true;
            options.errorMessage = "invalid test range values";
            return options;
        }

        if (options.benchmarkIterations < 1)
        {
            options.hasError     = true;
            options.errorMessage = "benchmark iterations must be >= 1";
        }

        return options;
    }

    void PrintHelp()
    {
        std::cout << "Usage: sdk_unit_test [options]\n"
                  << "  --list-device                 List Vulkan physical devices\n"
                  << "  --select-device <id|name>     Select Vulkan physical device by index or name match\n"
                  << "  --list-tests | --list         List tests in scope\n"
                  << "  --list-suites                 List suites in scope\n"
                  << "  --suite <id|name>             Select suite by ID or name (repeatable)\n"
                  << "  --suite-range <start:end>     Select one-based inclusive suite IDs\n"
                  << "  --test <suite.test>           Select specific tests (repeatable)\n"
                  << "  --test-range <start:end>      Select one-based inclusive test IDs per suite\n"
                  << "  --benchmark <N>               Repeat each selected test N times (or --benchmark=N)\n"
                  << "  --expected_failure            Run expected-failure tests as normal tests\n"
                  << "  --no-isolation                Run tests in-process\n"
                  << "  --verbose, -v                 Print extra details\n"
                  << "  --help                        Show this help\n";
    }

    int RunInternal(const std::string& fullName)
    {
        const TestCase* test = Registry::Instance().FindByName(fullName);
        if (test == nullptr)
        {
            std::cerr << "Internal run failed: unknown test " << fullName << '\n';
            return kExitFail;
        }

        const SingleRunResult result = RunInProcess(*test);
        if (result.status != TestRunStatus::Pass && !result.message.empty())
        {
            std::cerr << result.message << '\n';
        }
        if (result.status == TestRunStatus::Skip)
        {
            return kExitSkip;
        }
        return result.status == TestRunStatus::Pass ? kExitPass : kExitFail;
    }

    int RunAll(const RunnerOptions& options)
    {
        if (options.listDevicesOnly)
        {
            std::vector<vk::VulkanPhysicalDeviceInfo> devices;
            std::string                               errorMessage;
            if (!vk::EnumerateVulkanDevices(devices, &errorMessage))
            {
                std::cerr << "Error: " << errorMessage << '\n';
                return 1;
            }

            for (const vk::VulkanPhysicalDeviceInfo& device : devices)
            {
                std::cout << "[" << device.index << "] " << device.name << '\n';
            }
            return 0;
        }

        std::string envError;
        if (!options.selectedDevice.empty() && !ApplySelectedVulkanDevice(options.selectedDevice, &envError))
        {
            std::cerr << "Error: " << envError << '\n';
            return 1;
        }

        const TestCatalog catalog = BuildCatalog();
        if (catalog.tests.empty())
        {
            std::cout << "No tests registered.\n";
            return 1;
        }

        std::string         selectionError;
        const TestSelection selection = CollectSelection(catalog, options, &selectionError);
        if (!selectionError.empty())
        {
            std::cerr << "Error: " << selectionError << '\n';
            return 1;
        }

        if (options.listSuitesOnly)
        {
            if (selection.suites.empty())
            {
                std::cout << "No suites matched the current filter.\n";
            }
            else
            {
                PrintSuites(selection.suites);
            }

            if (!options.listTestsOnly)
            {
                return 0;
            }

            std::cout << '\n';
        }

        if (options.listTestsOnly)
        {
            if (selection.tests.empty())
            {
                std::cout << "No tests matched the current filter.\n";
            }
            else
            {
                PrintTests(selection.tests);
            }
            return 0;
        }

        if (selection.tests.empty())
        {
            std::cout << "No tests matched the current filter.\n";
            return 1;
        }

        PrintMetadataSection(options);
        VerboseLog(options, "Isolation mode: " + std::string(options.isolate ? "subprocess" : "in-process"));
        VerboseLog(options, "Benchmark iterations: " + std::to_string(options.benchmarkIterations));
        VerboseLog(options, "Selected tests: " + std::to_string(selection.tests.size()));

        size_t                     passed         = 0;
        size_t                     skipped        = 0;
        size_t                     failed         = 0;
        size_t                     expectedFailed = 0;
        std::vector<FailureRecord> failures;
        int                        currentSuiteId = -1;

        for (size_t i = 0; i < selection.tests.size(); ++i)
        {
            const TestEntry& entry = selection.tests[i];
            const TestCase&  test  = *entry.test;

            if (entry.suiteId != currentSuiteId)
            {
                currentSuiteId = entry.suiteId;
                PrintSuiteSectionHeader(entry.suiteId, test.suite);
            }
            PrintTestSectionHeader(test.fullName);
            VerboseLog(options, "Executing test " + test.fullName);

            if (test.expectedFailure && !options.runExpectedFailures)
            {
                ++expectedFailed;
                std::cout << "[" << std::setw(3) << (i + 1) << "/" << std::setw(3) << selection.tests.size() << "] "
                          << "EXPECTED_FAIL [" << entry.suiteId << ":" << entry.testId << "] " << test.fullName << " (not run)\n";
                continue;
            }

            const AggregatedRunResult runResult   = RunWithBenchmark(test, options);
            const bool                passedTest  = runResult.status == TestRunStatus::Pass;
            const bool                skippedTest = runResult.status == TestRunStatus::Skip;
            const char*               statusText  = passedTest ? "PASS" : (skippedTest ? "SKIP" : "FAIL");

            if (passedTest)
            {
                ++passed;
            }
            else if (skippedTest)
            {
                ++skipped;
            }
            else
            {
                ++failed;
                FailureRecord failure;
                failure.name   = test.fullName;
                failure.reason = runResult.message.empty() ? "No failure reason captured." : runResult.message;
                failures.push_back(std::move(failure));
            }

            std::ostringstream durationStream;
            durationStream << std::fixed << std::setprecision(2) << runResult.averageMs << " ms";
            if (options.benchmarkIterations > 1)
            {
                durationStream << " avg/" << options.benchmarkIterations << " runs";
            }

            std::string detail;
            if (!runResult.message.empty())
            {
                detail = " - " + runResult.message;
            }

            std::cout << "[" << std::setw(3) << (i + 1) << "/" << std::setw(3) << selection.tests.size() << "] " << statusText << " [" << entry.suiteId << ":"
                      << entry.testId << "] " << test.fullName << " (" << durationStream.str() << ")" << detail << '\n';
        }

        PrintSummarySectionHeader();
        std::cout << "Summary: "
                  << "pass=" << passed << ", "
                  << "skipped=" << skipped << ", "
                  << "fail=" << failed << ", "
                  << "expectedFail=" << expectedFailed << '\n';

        if (!failures.empty())
        {
            std::cout << "Failed tests:\n";
            for (const FailureRecord& failure : failures)
            {
                std::cout << "  - " << failure.name << ": " << failure.reason << '\n';
            }
        }

        return failed == 0 ? 0 : 1;
    }

    void ReportCheckFailure(const char* expression, const char* file, int line)
    {
        std::cerr << "Check failed: (" << expression << ") at " << file << ":" << line << '\n';
    }

    void MarkTestFailed(const std::string& reason)
    {
        g_failureReason = reason;
    }

    void MarkTestSkipped(const std::string& reason)
    {
        if (g_skipReason.empty())
        {
            g_skipReason = reason.empty() ? "test skipped without a reason" : reason;
        }
    }

    ScopedCaptureEnvironment::ScopedCaptureEnvironment()
    {
        std::vector<std::string> names;
        for (const char* prefix : kCapturePrefixes)
        {
            for (size_t suffixIndex = 0; suffixIndex < std::size(arm::capture::kConfigSuffixes); ++suffixIndex)
            {
#if defined(__ANDROID__)
                names.emplace_back(std::string(prefix) + arm::capture::kConfigSuffixes[suffixIndex]);
#else
                names.emplace_back(std::string(prefix) + arm::capture::kConfigEnvironmentSuffixes[suffixIndex]);
#endif
            }
        }
#if defined(__ANDROID__)
        names.emplace_back(arm::capture::kAndroidPerformanceProperty);
        names.emplace_back(arm::capture::kAndroidPoolHostVisibleRatioProperty);
#else
        names.emplace_back(arm::capture::kDesktopPerformanceEnvironment);
        names.emplace_back(arm::capture::kDesktopPoolHostVisibleRatioEnvironment);
#endif

        for (const std::string& name : names)
        {
#if defined(__ANDROID__)
            char      value[PROP_VALUE_MAX] = {};
            const int length                = __system_property_get(name.c_str(), value);
            savedEntries_.push_back({name, length > 0 ? std::string(value) : std::string(), length > 0});
#else
            const char* value = std::getenv(name.c_str());
            savedEntries_.push_back({name, value ? std::string(value) : std::string(), value != nullptr});
#endif
            if (!ClearVariable(name.c_str()))
                MarkTestFailed("failed to clear capture test variable: " + name);
        }
    }

    ScopedCaptureEnvironment::~ScopedCaptureEnvironment()
    {
        for (auto entry = savedEntries_.rbegin(); entry != savedEntries_.rend(); ++entry)
        {
            if (entry->hadValue)
            {
                if (!SetVariable(entry->name.c_str(), entry->value.c_str()))
                    MarkTestFailed("failed to restore capture test variable: " + entry->name);
            }
            else
            {
                if (!ClearVariable(entry->name.c_str()))
                    MarkTestFailed("failed to clear capture test variable: " + entry->name);
            }
        }
    }

    bool ScopedCaptureEnvironment::SetVariable(const char* name, const char* value)
    {
#if defined(__ANDROID__)
        return __system_property_set(name, value ? value : "") == 0;
#elif defined(_WIN32)
        return _putenv_s(name, value ? value : "") == 0;
#else
        if (value)
            return setenv(name, value, 1) == 0;
        return unsetenv(name) == 0;
#endif
    }

    bool ScopedCaptureEnvironment::ClearVariable(const char* name)
    {
#if defined(__ANDROID__)
        return __system_property_set(name, "") == 0;
#elif defined(_WIN32)
        return _putenv_s(name, "") == 0;
#else
        return unsetenv(name) == 0;
#endif
    }

}  // namespace unit_test
