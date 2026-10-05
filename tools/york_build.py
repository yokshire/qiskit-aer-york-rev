# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Build and verify the default CPU distribution on Linux, macOS, or Windows."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv


def environment_python(directory):
    """Locate an isolated interpreter without requiring shell activation."""
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def build_environment(source, use_conan=False):
    """Select the general-purpose CPU build even in a CUDA-configured shell."""
    environment = os.environ.copy()
    environment.update(
        QISKIT_AER_PACKAGE_NAME="qiskit-aer-york-rev",
        QISKIT_ADD_CUDA_REQUIREMENTS="false",
        AER_THRUST_BACKEND="",
    )
    for name in ("CUDACXX", "AER_CUDA_ARCH", "AER_PYTHON_CUDA_ROOT", "QISKIT_AER_CUDA_MAJOR"):
        environment.pop(name, None)
    # Append explicit CPU selection after caller-supplied platform/toolchain options.
    environment["CMAKE_ARGS"] = (
        environment.get("CMAKE_ARGS", "")
        + f" -DDISABLE_CONAN={'OFF' if use_conan else 'ON'}"
        + " -DAER_THRUST_BACKEND:STRING= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ).strip()
    environment.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    binary_directory = environment_python(source / ".venv-york-build").parent
    environment["PATH"] = str(binary_directory) + os.pathsep + environment.get("PATH", "")
    return environment


def run(python, *arguments, source, environment, stdout=None):
    """Propagate failures from builds, dependency resolution, and runtime checks."""
    subprocess.run(
        [str(python), *map(str, arguments)],
        cwd=source,
        env=environment,
        stdout=stdout,
        check=True,
    )


def build(qiskit_version, runtime_version, label, use_conan=False):
    """Keep legacy build-only dependencies out of the Runtime environment."""
    source = Path(__file__).resolve().parents[1]
    wheels = source / "artifacts" / "wheels"
    # Do not silently reuse stale GPU CMake settings or overwrite an existing environment.
    existing = [
        path
        for path in (source / "_skbuild", source / ".venv-york-build", source / ".venv-york")
        if path.exists()
    ]
    if existing or (wheels.exists() and any(wheels.glob("*.whl"))):
        raise ValueError(
            "Use a clean source checkout: build environments/caches/wheels already exist"
        )
    for name in (".venv-york-build", ".venv-york"):
        venv.EnvBuilder(with_pip=True).create(source / name)
    build_python = environment_python(source / ".venv-york-build")
    runtime_python = environment_python(source / ".venv-york")
    environment = build_environment(source, use_conan)
    run(
        build_python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        source=source,
        environment=environment,
    )
    run(
        build_python,
        "-m",
        "pip",
        "install",
        "-r",
        "tools/york-build-requirements.txt",
        source=source,
        environment=environment,
    )
    run(
        build_python,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        wheels,
        source=source,
        environment=environment,
    )
    wheel_files = list(wheels.glob("*.whl"))
    if len(wheel_files) != 1:
        raise ValueError("Expected exactly one newly built CPU wheel")
    wheel = wheel_files[0]
    with (source / "artifacts/cpu-wheel.json").open("w", encoding="utf-8") as stream:
        run(
            build_python,
            "tools/york_ci.py",
            "check-wheel",
            wheel,
            source=source,
            environment=environment,
            stdout=stream,
        )

    runtime_environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
        runtime_environment.pop(name, None)
    run(
        runtime_python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        source=source,
        environment=runtime_environment,
    )
    run(
        runtime_python,
        "-m",
        "pip",
        "install",
        wheel,
        f"qiskit=={qiskit_version}",
        f"qiskit-ibm-runtime=={runtime_version}",
        "packaging>=24,<27",
        source=source,
        environment=runtime_environment,
    )
    run(runtime_python, "-m", "pip", "check", source=source, environment=runtime_environment)
    with (source / "artifacts/installed-versions.txt").open("w", encoding="utf-8") as stream:
        run(
            runtime_python,
            "-m",
            "pip",
            "freeze",
            source=source,
            environment=runtime_environment,
            stdout=stream,
        )
    with tempfile.TemporaryDirectory(prefix="york-cpu-smoke-") as outside_source:
        run(
            runtime_python,
            "-I",
            source / "tools/york_smoke.py",
            "--device",
            "CPU",
            "--output",
            source / f"artifacts/smoke-{label}.json",
            source=outside_source,
            environment=runtime_environment,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qiskit", required=True, help="Exact Qiskit version to verify")
    parser.add_argument("--runtime", required=True, help="Exact IBM Runtime version to verify")
    parser.add_argument("--label", default="cpu", help="Evidence label (a filename component)")
    parser.add_argument(
        "--use-conan", action="store_true", help="Resolve C++ libraries with Conan 1.x"
    )
    args = parser.parse_args()
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.label
    ):
        parser.error("--label must contain only letters, digits, dots, underscores, or hyphens")
    build(args.qiskit, args.runtime, args.label, args.use_conan)


if __name__ == "__main__":
    main()
