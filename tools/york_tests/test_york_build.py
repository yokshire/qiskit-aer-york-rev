# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Protect CPU builds from inherited CUDA settings and platform-specific shells."""

import importlib.util
import os
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "york_build", Path(__file__).parents[1] / "york_build.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class YorkBuildTests(unittest.TestCase):
    def test_cuda_shell_cannot_change_default_distribution_or_backend(self):
        original = {
            "QISKIT_AER_PACKAGE_NAME": "qiskit-aer-york-rev-gpu",
            "QISKIT_ADD_CUDA_REQUIREMENTS": "true",
            "AER_THRUST_BACKEND": "CUDA",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
            "AER_CUDA_ARCH": "8.6",
            "AER_PYTHON_CUDA_ROOT": "/old/gpu/environment",
            "CMAKE_ARGS": "-DCMAKE_TOOLCHAIN_FILE=custom.cmake -DAER_THRUST_BACKEND=CUDA",
            "PATH": "existing-path",
        }
        with patch.dict(os.environ, original, clear=True):
            environment = BUILD.build_environment(Path.cwd())
            self.assertEqual(os.environ["AER_THRUST_BACKEND"], "CUDA")
        self.assertEqual(environment["QISKIT_AER_PACKAGE_NAME"], "qiskit-aer-york-rev")
        self.assertEqual(environment["QISKIT_ADD_CUDA_REQUIREMENTS"], "false")
        self.assertEqual(environment["AER_THRUST_BACKEND"], "")
        for name in ("CUDACXX", "AER_CUDA_ARCH", "AER_PYTHON_CUDA_ROOT"):
            self.assertNotIn(name, environment)
        self.assertNotIn("CMAKE_ARGS", environment)
        options = environment["SKBUILD_CONFIGURE_OPTIONS"]
        self.assertIn("-DCMAKE_TOOLCHAIN_FILE=custom.cmake", options)
        self.assertGreater(
            options.index("-DAER_THRUST_BACKEND:STRING="),
            options.index("-DAER_THRUST_BACKEND=CUDA"),
        )
        self.assertIn("-DDISABLE_CONAN=ON", options)

    def test_quoted_compiler_flags_and_paths_reach_cmake_as_single_arguments(self):
        original = {
            "CMAKE_ARGS": '-DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp"',
            "SKBUILD_CONFIGURE_OPTIONS": '-DCMAKE_TOOLCHAIN_FILE="C:/Build Tools/vcpkg.cmake"',
        }
        with patch.dict(os.environ, original, clear=True):
            environment = BUILD.build_environment(Path.cwd())
        options = shlex.split(environment["SKBUILD_CONFIGURE_OPTIONS"])
        self.assertNotIn("CMAKE_ARGS", environment)
        self.assertIn("-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp", options)
        self.assertIn("-DCMAKE_TOOLCHAIN_FILE=C:/Build Tools/vcpkg.cmake", options)

    def test_native_venv_interpreter_path(self):
        directory = Path("venv")
        windows_python = directory / "Scripts/python.exe"
        posix_python = directory / "bin/python"
        with patch.object(BUILD.os, "name", "nt"):
            self.assertEqual(BUILD.environment_python(directory), windows_python)
        with patch.object(BUILD.os, "name", "posix"):
            self.assertEqual(BUILD.environment_python(directory), posix_python)


if __name__ == "__main__":
    unittest.main()
