# spdlog source provenance

- Upstream: https://github.com/gabime/spdlog
- Version: `v1.17.0`
- Commit: `79524ddd08a4ec981b7fea76afd08ee05f83755d`
- License: MIT; see `LICENSE`.
- Bundled fmt: 12.1.0 (`FMT_VERSION 120100`); see
  `include/spdlog/fmt/bundled/fmt.license.rst`.

This SDK-private snapshot retains the complete `include/spdlog` header tree and
only the compiled sources required by `ffx_logging`: `spdlog.cpp`,
`stdout_sinks.cpp`, `color_sinks.cpp`, `async.cpp`, and
`bundled_fmtlib_format.cpp`.
Upstream examples, tests, benchmarks, documentation, CI configuration, and
packaging files are intentionally excluded.

The private `ffx_spdlog` target compiles this snapshot with the `spdlog`
namespace renamed to `ffx_sdk_spdlog`. This build-time isolation permits the
unmodified samples' spdlog revision to coexist in the same executable without
altering any upstream source file.
