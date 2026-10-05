# York revision maintenance and automation

This is an unofficial independent **general-purpose** customization of Qiskit Aer.
The default distribution runs on CPU without GPU hardware or CUDA. GPU builds
are an optional acceleration path. The original
source history, Apache-2.0 license, and contributor notices are retained.
It does not replace or close upstream PRs #2458 and #2459 in
`Qiskit/qiskit-aer`; their existing branches and AI disclosure are unchanged.

## Source and distribution identity

- Bootstrap source: upstream commit `755381a015037c66af3e3d0cc54ff1a56dbe7ace`.
- Integrated patches: the CUDA/C++ compiler-selection fix and local-wheel
  `$ORIGIN` RPATH fix, including both verification tools and release notes.
- Development version: `0.17.2.post2.dev0` (accelerator API 1).
- Default general-purpose CPU name: `qiskit-aer-york-rev`.
- Optional plugins: `qiskit-aer-york-opencl` and `qiskit-aer-york-cupy`, using
  separate `qiskit_aer_york_*` namespaces. They coexist with the CPU base.
- Older standalone CUDA 12 name: `qiskit-aer-york-rev-gpu`.
- The base and older standalone GPU build both own `qiskit_aer`. Install only
  one of these base distributions. The new plugins do not overwrite their files.
- There is no published York wheel yet. CI builds are not release artifacts;
  the CUDA CI build targets compute capability 8.6, not all NVIDIA GPUs.

The upstream patch disclosure is retained verbatim for provenance:

AI tool used: OpenAI Codex (model: `gpt-daybreak-blue-latest` for the initial implementation; GPT-6 for follow-up review and verification tooling).

York automation was also prepared with OpenAI Codex. Its automated review is
deterministic testing/reporting, **not** an AI code review.

## Schedule and scope

`York Compatibility and Review` runs on:

- Every **3 hours**, at **00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00,
  and 21:00 Asia/Seoul**. The UTC schedule uses the same three-hour boundaries.
- Pushes to `main` and pull requests targeting `main`.
- Manual **Run workflow** requests.

GitHub schedules can be delayed/dropped under load and public-repository
schedules can be disabled after 60 days without repository activity. This is
not a hard real-time scheduler. The separate Codex heartbeat still monitors
the two upstream PRs and is not replaced by this workflow.

Each run:

1. Resolves the newest non-yanked stable Qiskit and IBM Runtime releases from
   public PyPI metadata. Exact versions are installed; older resolver fallbacks
   must not be reported as compatibility with the newest version.
2. Builds the **York source**, not the upstream PyPI Aer wheel, on Linux x86-64
   with Python 3.10–3.14 and setup-python's latest stable `3.x`, and on Windows
   x86-64 and macOS arm64 with Python 3.12. The CPU build uses the same Python
   helper on all platforms and explicitly clears inherited CUDA build settings.
3. Rejects accelerator dependencies in the default wheel. Runs `pip check`,
   Bell shot sampling with automatic/statevector/density-matrix/stabilizer/
   extended-stabilizer/MPS methods, exact statevector/density-matrix/unitary/
   superoperator comparisons, a deterministic noise model, Aer SamplerV2 and
   EstimatorV2, Runtime SamplerV2/EstimatorV2 local mode, and Runtime's
   client-side Sampler when available in 0.50 or later. No IBM API credentials
   or paid QPU calls are used.
4. Builds a CUDA 12/Python 3.12 wheel in an NVIDIA CUDA 12.4.1 development
   container. Checks distribution identity, CUDA dependency metadata,
   generated C++/nvcc compiler selection, and the six relative ELF RPATHs.
5. Checks Python syntax/undefined names, formats York tooling, and runs offline
   monitor regression tests. This is not an exhaustive source/security review.
   The upstream legacy deferred `BackendProperties` annotation has a
   line-specific lint baseline; it does not suppress undefined-name checks
   elsewhere in the module or in York tooling.
6. Saves a Markdown job summary and JSON/Markdown artifacts for 30 days. A
   failed trusted run creates/updates **one** automated issue; an unchanged
   failure is not reposted, and recovery closes only that automated issue.
   PR runs have no issue-write job. Nothing is automatically merged or released.
   Results from superseded source commits cannot reopen/update the current
   failure issue. Notification jobs are serialized to avoid creation races.

Upstream CPU/GPU PyPI release differences are recorded as inventory information,
not incorrectly attributed to a York patch regression. Unsupported Python
requirements, install/build errors, and failed runtime checks remain visible
as failed jobs rather than being hidden as successful compatibility.

## Default CPU build

Use a clean source checkout for each build so stale `_skbuild` caches cannot
mix CPU/GPU compiler settings. `tools/york_build.py` refuses to reuse existing
build caches, virtual environments, or wheels. It creates separate
`.venv-york-build` and `.venv-york` environments: Conan 1.x's build-only
`urllib3<1.27` dependency must not be mixed with current IBM Runtime's
`urllib3>=2.4` dependency.

The helper uses system C++ libraries by default, without CUDA or Thrust:

- Linux: a C++ compiler, OpenBLAS, spdlog, and nlohmann-json development packages.
  Ubuntu/Debian CI uses `libopenblas-dev libspdlog-dev nlohmann-json3-dev`.
- macOS: Xcode command-line tools and Homebrew `spdlog nlohmann-json libomp`.
  Apple Accelerate provides BLAS. See the workflow for OpenMP CMake hints.
- Windows: Visual Studio 2022 C++ build tools and vcpkg `spdlog` and
  `nlohmann-json` with the `x64-windows-static-md` triplet. Set
  `CMAKE_ARGS` to the vcpkg CMake toolchain and target triplet as in the workflow.
  The existing bundled Windows OpenBLAS library is included in the wheel.

```text
python tools/york_build.py --qiskit <exact-qiskit-version> --runtime <exact-runtime-version>
```

`--use-conan` is available for existing Conan 1.x build setups instead of system
libraries. The helper does not install system packages or compilers. This is
a development wheel for the build platform, not a portable release wheel.
Existing Aer APIs/imports and simulation methods are retained. IBM Runtime is
installed for verification only; the package itself does not require it.

## Active maintenance

A separate Codex scheduled task returns to the existing project chat every
three hours. It checks new stable releases, upstream changes, York issues and
reviews, and CI evidence. Reproducible compatibility failures, GPU packaging
or execution bugs, functional regressions, and missing regression coverage
can be repaired with focused tests on a York maintenance branch and submitted
as a pull request. No change is required when there is no concrete problem.

Maintenance preserves the Aer API, licenses, attribution, and AI disclosure.
New user-facing features, public API changes, large refactors, and upstream
synchronization require a concrete proposal and user decision. The task does
not directly update `main`, merge pull requests, publish releases, or perform
paid QPU calls. The two upstream PR branches and their separate monitor remain
untouched. Only meaningful findings, completed fixes, and decisions are reported.

This local maintenance task requires the computer and Codex desktop app to
remain running, with the project available. GitHub compatibility checks run
independently. See the [scheduled-task documentation](https://learn.chatgpt.com/docs/automations).

## Optional GPU execution boundary

The new [accelerator plugins](accelerators/README.md) add native Windows GPU
execution to the CPU base and a common vendor-neutral OpenCL path. CUDA and
ROCm/HIP have separate entry points. CUDA requires a CUDA CuPy build, ROCm
requires a HIP CuPy build, and Intel uses OpenCL. API 1 supports ideal
statevector circuits, terminal measurements, and supported save instructions.
Hosted CI builds both plugin wheels and runs contract tests against the built
CPU package on Linux, Windows and macOS. Those tests are not GPU evidence.

The remaining instructions below describe the **older standalone native CUDA
build**, which has broader Aer GPU features but shares the base import namespace.

**Hosted CI does not execute on a physical GPU.** Successful compilation,
dependency resolution, and ELF metadata do not prove GPU availability, driver
compatibility, sampling execution, or portability to other GPU architectures.

Use a separately provisioned environment with an NVIDIA GPU and driver to run
the strict GPU smoke check. It fails if GPU is unavailable or Aer's simulator
result metadata reports a different device. Do not install a permanent public
PR-accessible self-hosted runner on a personal machine as part of this setup.

After installing a York GPU wheel and the exact Qiskit/Runtime versions into a
fresh virtual environment, install the smoke tool's `packaging` dependency
(this is not an Aer runtime requirement) and run **outside the source tree**:

```bash
/path/to/venv/bin/python -m pip install 'packaging>=24,<27'
env -u LD_LIBRARY_PATH /path/to/venv/bin/python -I \
  /path/to/source/tools/york_smoke.py --device GPU --output /tmp/york-gpu-smoke.json
```

For a local CUDA build, provide the same system development libraries as the
workflow (`libopenblas-dev`, `libspdlog-dev`, and `nlohmann-json3-dev`), an
appropriate compiler, and a CUDA 12 toolkit. The optional GPU build remains
Linux-specific and uses the existing Bash helper:

```bash
CUDACXX=/path/to/cuda/bin/nvcc bash tools/york_build.sh gpu <exact-qiskit-version> <exact-runtime-version> local
```

Upstream deployment workflows are archived in
`contrib/upstream-workflows/` and are not automatically executed in this repo.
PyPI publishing, releases, broader CUDA/Python/OS support, and permanent GPU CI
require separate validation and configuration.
