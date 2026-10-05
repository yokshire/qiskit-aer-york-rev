# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Offline regression tests for the compatibility monitor's failure semantics."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("york_ci", Path(__file__).parents[1] / "york_ci.py")
CI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CI)


class YorkCITests(unittest.TestCase):
    """No network, GPU, or IBM account is used by these tests."""

    def successful_evidence(self, directory):
        """Create evidence for six Linux versions plus Windows and macOS."""
        path = Path(directory)
        CI.write_json(
            path / "inventory.json",
            {"packages": {name: {"version": "1.0.0"} for name in CI.PACKAGES}},
        )
        for index, system in enumerate(["Linux"] * 6 + ["Windows", "Darwin"]):
            CI.write_json(
                path / f"smoke-{index}.json",
                {
                    "status": "passed",
                    "device": "CPU",
                    "platform": {"system": system},
                    "versions": {name: "1.0.0" for name in ("qiskit", "qiskit-ibm-runtime")},
                },
            )
        CI.write_json(path / "gpu-wheel.json", {"name": "qiskit-aer-york-rev-gpu"})
        for label in ("ubuntu-24.04", "windows-2022", "macos-14"):
            CI.write_json(
                path / f"modular-{label}.json",
                {
                    "passed": True,
                    "label": label,
                    "core_dependencies": [],
                    "native_aer_installed": False,
                    "nature_without_pyscf": True,
                    "array_transport": True,
                },
            )

        CI.write_json(
            path / "nature-native.json",
            {
                "passed": True,
                "runtime": "native",
                "native_aer_installed": False,
                "array_transport": True,
                "binary_matches_fcidump": True,
                "rohf_spin_verified": True,
            },
        )
        for system in ("Linux", "Windows", "Darwin"):
            CI.write_json(
                path / f"data-{system}.json",
                {"passed": True, "platform_system": system, "array_views_share_payload": True},
            )

    def test_missing_array_evidence_is_not_a_pass(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as temporary:
            self.successful_evidence(temporary)
            path = Path(temporary)
            (path / "data-Windows.json").unlink()
            self.assertFalse(CI.make_report(path, needs)["passed"])

    def test_missing_modular_install_evidence_is_not_a_pass(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            self.successful_evidence(directory)
            path = Path(directory)
            (path / "modular-windows-2022.json").unlink()
            self.assertFalse(CI.make_report(path, needs)["passed"])

    def test_stable_selection_excludes_prerelease_yanked_and_empty(self):
        releases = {
            "2.5.2": [{"yanked": False}],
            "3.0.0rc1": [{"yanked": False}],
            "2.9.0": [{"yanked": True}],
            "2.6.0": [],
            "garbage": [{}],
        }
        self.assertEqual(CI.latest_stable(releases), "2.5.2")

    def test_no_stable_release_fails(self):
        with self.assertRaises(ValueError):
            CI.latest_stable({"3.0.0.dev1": [{}]})

    def test_metadata_rejects_missing_cuda_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("qiskit_aer/VERSION.txt", "0.17.2.post1.dev0")
                archive.writestr(
                    "york.dist-info/METADATA",
                    "Name: qiskit-aer-york-rev-gpu\nVersion: 0.17.2.post1.dev0\nRequires-Dist: qiskit\n",
                )
            with self.assertRaisesRegex(ValueError, "Missing CUDA"):
                CI.check_wheel(wheel, True)

    def test_correct_gpu_identity_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("qiskit_aer/VERSION.txt", "0.17.2.post1.dev0")
                archive.writestr(
                    "york.dist-info/METADATA",
                    "Name: qiskit-aer-york-rev-gpu\nVersion: 0.17.2.post1.dev0\nRequires-Dist: qiskit\n"
                    + "".join(f"Requires-Dist: {name}\n" for name in CI.CUDA_REQUIREMENTS),
                )
            self.assertTrue(CI.check_wheel(wheel, True)["cuda_dependencies"])

    def test_report_never_claims_gpu_execution(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            self.successful_evidence(directory)
            report = CI.make_report(Path(directory), needs)
        self.assertTrue(report["passed"])
        self.assertIn("GPU execution: NOT TESTED", report["body"])

    def test_skipped_or_missing_job_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(CI.make_report(Path(directory), {})["passed"])
            self.assertFalse(
                CI.make_report(Path(directory), {"inventory": {"result": "skipped"}})["passed"]
            )

    def test_smoke_failure_overrides_successful_job_status(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.successful_evidence(directory)
            CI.write_json(path / "smoke-cpu.json", {"status": "failed", "error": "ImportError"})
            self.assertFalse(CI.make_report(path, needs)["passed"])

    def test_successful_jobs_without_artifacts_are_not_a_pass(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(CI.make_report(Path(directory), needs)["passed"])

    def test_report_rejects_old_dependency_fallback(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            self.successful_evidence(directory)
            path = Path(directory)
            CI.write_json(
                path / "smoke-0.json",
                {"status": "passed", "device": "CPU", "versions": {"qiskit": "0.1.0"}},
            )
            self.assertFalse(CI.make_report(path, needs)["passed"])

    def test_report_fingerprint_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            CI.write_json(path / "inventory.json", {"checked_at": "first", "packages": {}})
            first = CI.make_report(path, {})["fingerprint"]
            CI.write_json(path / "inventory.json", {"checked_at": "second", "packages": {}})
            self.assertEqual(first, CI.make_report(path, {})["fingerprint"])

    def test_linux_only_evidence_does_not_prove_os_portability(self):
        needs = {
            name: {"result": "success"}
            for name in (
                "inventory",
                "static-review",
                "modular-install",
                "nature-driver",
                "cpu-compatibility",
                "cpu-portability",
                "cuda-wheel",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            self.successful_evidence(directory)
            path = Path(directory)
            for index in (6, 7):
                CI.write_json(
                    path / f"smoke-{index}.json",
                    {"status": "passed", "device": "CPU", "platform": {"system": "Linux"}},
                )
            self.assertFalse(CI.make_report(path, needs)["passed"])

    def test_cpu_wheel_rejects_accelerator_dependencies_from_other_cuda_versions(self):
        for dependency in ("nvidia-cuda-runtime-cu11", "cuquantum-cu13", "custatevec-cu12"):
            with self.subTest(dependency=dependency), tempfile.TemporaryDirectory() as directory:
                wheel = Path(directory) / "test.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("qiskit_aer/VERSION.txt", "0.17.2.post1.dev0")
                    archive.writestr(
                        "york.dist-info/METADATA",
                        "Name: qiskit-aer-york-rev\nVersion: 0.17.2.post1.dev0\nRequires-Dist: qiskit\n"
                        + f"Requires-Dist: {dependency}\n",
                    )
                with self.assertRaisesRegex(ValueError, "accelerator packages"):
                    CI.check_wheel(wheel, False)

    def test_source_change_updates_failure_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.dict(CI.os.environ, {"GITHUB_SHA": "first"}):
                first = CI.make_report(path, {})["fingerprint"]
            with patch.dict(CI.os.environ, {"GITHUB_SHA": "second"}):
                self.assertNotEqual(first, CI.make_report(path, {})["fingerprint"])


if __name__ == "__main__":
    unittest.main()
