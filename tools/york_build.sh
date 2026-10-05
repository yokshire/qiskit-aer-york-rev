#!/usr/bin/env bash
# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# Build this source tree in a fresh virtual environment. CI artifacts are not releases.
set -euo pipefail

build_kind="${1:?cpu or gpu}"
qiskit_version="${2:?exact Qiskit version}"
runtime_version="${3:?exact Runtime version}"
evidence_label="${4:-$build_kind}"
workspace="$(pwd)"
python -m venv .venv-york-build
python -m venv .venv-york
runtime_python="$workspace/.venv-york/bin/python"
export PATH="$workspace/.venv-york-build/bin:$PATH"
python -m pip install --upgrade pip
python -m pip install -r tools/york-build-requirements.txt
# Conan 1.x's urllib3 pin must never contaminate the Runtime execution environment.

export CMAKE_ARGS="-DDISABLE_CONAN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
if [[ "$build_kind" == gpu ]]; then
    export QISKIT_AER_PACKAGE_NAME=qiskit-aer-york-rev-gpu
    export QISKIT_AER_CUDA_MAJOR=12
    export QISKIT_ADD_CUDA_REQUIREMENTS=true
    export AER_THRUST_BACKEND=CUDA
    export AER_CUDA_ARCH=8.6
    export AER_PYTHON_CUDA_ROOT="$workspace/.venv-york-build"
    export CMAKE_BUILD_PARALLEL_LEVEL=1
    export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
    python -m pip install \
        'nvidia-cuda-runtime-cu12>=12.1.105' nvidia-nvjitlink-cu12 \
        'nvidia-cublas-cu12>=12.1.3.1' 'nvidia-cusolver-cu12>=11.4.5.107' \
        'nvidia-cusparse-cu12>=12.1.0.106' 'cuquantum-cu12>=23.3.0,<24.11.0'
elif [[ "$build_kind" == cpu ]]; then
    export QISKIT_AER_PACKAGE_NAME=qiskit-aer-york-rev
    export AER_THRUST_BACKEND=OMP
    export QISKIT_ADD_CUDA_REQUIREMENTS=false
else
    printf 'Unknown build kind: %s\n' "$build_kind" >&2
    exit 2
fi

python -m build --wheel --no-isolation --outdir artifacts/wheels
shopt -s failglob
wheel_files=(artifacts/wheels/*.whl)
if [[ ${#wheel_files[@]} != 1 ]]; then
    printf 'Expected one newly built wheel\n' >&2
    exit 1
fi
wheel="${wheel_files[0]}"
"$runtime_python" -m pip install --upgrade pip
# One resolver invocation with exact versions prevents an older Qiskit fallback.
"$runtime_python" -m pip install "$wheel" "qiskit==$qiskit_version" "qiskit-ibm-runtime==$runtime_version"
"$runtime_python" -m pip check
"$runtime_python" -m pip freeze > artifacts/installed-versions.txt
if [[ "$build_kind" == gpu ]]; then
    python tools/york_ci.py check-wheel "$wheel" --gpu > artifacts/gpu-wheel.json
    compilation_database="$(find _skbuild -name compile_commands.json -print -quit)"
    test -n "$compilation_database"
    python tools/verify_cuda_compilation.py "$compilation_database"
    python tools/verify_cuda_wheel.py "$wheel"
    printf 'CUDA packaging checks passed; physical GPU execution was NOT performed.\n'
else
    python tools/york_ci.py check-wheel "$wheel" > artifacts/cpu-wheel.json
    # -I and a different cwd ensure the installed extension is tested, not the source tree.
    cd "${RUNNER_TEMP:-/tmp}"
    env -u LD_LIBRARY_PATH "$workspace/.venv-york/bin/python" -I \
        "$workspace/tools/york_smoke.py" --device CPU \
        --output "$workspace/artifacts/smoke-$evidence_label.json"
fi
