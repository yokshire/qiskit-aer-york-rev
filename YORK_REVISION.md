# York revision maintenance and automation

This is an unofficial independent customization of Qiskit Aer. The original
source history, Apache-2.0 license, and contributor notices are retained.
It does not replace or close upstream PRs #2458 and #2459 in
`Qiskit/qiskit-aer`; their existing branches and AI disclosure are unchanged.

## Source and distribution identity

- Bootstrap source: upstream commit `755381a015037c66af3e3d0cc54ff1a56dbe7ace`.
- Integrated patches: the CUDA/C++ compiler-selection fix and local-wheel
  `$ORIGIN` RPATH fix, including both verification tools and release notes.
- Development version: `0.17.2.post1.dev0`.
- CPU name: `qiskit-aer-york-rev`; CUDA 12 name: `qiskit-aer-york-rev-gpu`.
- Import namespace: `qiskit_aer`. Install only one Aer distribution in an
  isolated environment. Renaming a distribution does not isolate import files.
- There is no published York wheel yet. CI builds are not release artifacts;
  the CUDA CI build targets compute capability 8.6, not all NVIDIA GPUs.

The upstream patch disclosure is retained verbatim for provenance:

AI tool used: OpenAI Codex (model: `gpt-daybreak-blue-latest` for the initial implementation; GPT-6 for follow-up review and verification tooling).

York automation was also prepared with OpenAI Codex. Its automated review is
deterministic testing/reporting, **not** an AI code review.

## Schedule and scope

`York Compatibility and Review` runs on:

- Every day at **00:00 and 12:00 Asia/Seoul** (`03:00/15:00 UTC`).
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
   with Python 3.10–3.14 and setup-python's latest stable `3.x`.
3. Runs `pip check`, Bell shot sampling, exact statevector/unitary comparisons,
   Aer SamplerV2, Runtime SamplerV2/EstimatorV2 local mode, and Runtime's
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

Upstream CPU/GPU PyPI release differences are recorded as inventory information,
not incorrectly attributed to a York patch regression. Unsupported Python
requirements, install/build errors, and failed runtime checks remain visible
as failed jobs rather than being hidden as successful compatibility.

## GPU execution boundary

**Hosted CI does not execute on a physical GPU.** Successful compilation,
dependency resolution, and ELF metadata do not prove GPU availability, driver
compatibility, sampling execution, or portability to other GPU architectures.

Use a separately provisioned environment with an NVIDIA GPU and driver to run
the strict GPU smoke check. It fails if GPU is unavailable or Aer's simulator
result metadata reports a different device. Do not install a permanent public
PR-accessible self-hosted runner on a personal machine as part of this setup.

After installing a York GPU wheel and the exact Qiskit/Runtime versions into a
fresh virtual environment, run **outside the source tree**:

```bash
env -u LD_LIBRARY_PATH /path/to/venv/bin/python -I \
  /path/to/source/tools/york_smoke.py --device GPU --output /tmp/york-gpu-smoke.json
```

For a local build, provide the same system development libraries as the
workflow (`libopenblas-dev`, `libspdlog-dev`, `nlohmann-json3-dev`, and
`libthrust-dev` for CPU), an appropriate compiler, and a CUDA 12 toolkit for
GPU. The script creates `.venv-york` and builds a development wheel:

```bash
bash tools/york_build.sh cpu <exact-qiskit-version> <exact-runtime-version> local
CUDACXX=/path/to/cuda/bin/nvcc bash tools/york_build.sh gpu <exact-qiskit-version> <exact-runtime-version> local
```

Use a clean source checkout for each build so stale `_skbuild` caches cannot
mix CPU/GPU compiler settings. Upstream deployment workflows are archived in
`contrib/upstream-workflows/` and are not automatically executed in this repo.
PyPI publishing, releases, broader CUDA/Python/OS support, and permanent GPU CI
require separate validation and configuration.
