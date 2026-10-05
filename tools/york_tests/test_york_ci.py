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
        """Create all evidence required by the six-row CPU matrix."""
        path = Path(directory)
        CI.write_json(
            path / "inventory.json",
            {"packages": {name: {"version": "1.0.0"} for name in CI.PACKAGES}},
        )
        for index in range(6):
            CI.write_json(
                path / f"smoke-{index}.json",
                {
                    "status": "passed",
                    "device": "CPU",
                    "versions": {name: "1.0.0" for name in ("qiskit", "qiskit-ibm-runtime")},
                },
            )
        CI.write_json(path / "gpu-wheel.json", {"name": "qiskit-aer-york-rev-gpu"})

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
            for name in ("inventory", "static-review", "cpu-compatibility", "cuda-wheel")
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
            for name in ("inventory", "static-review", "cpu-compatibility", "cuda-wheel")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.successful_evidence(directory)
            CI.write_json(path / "smoke-cpu.json", {"status": "failed", "error": "ImportError"})
            self.assertFalse(CI.make_report(path, needs)["passed"])

    def test_successful_jobs_without_artifacts_are_not_a_pass(self):
        needs = {
            name: {"result": "success"}
            for name in ("inventory", "static-review", "cpu-compatibility", "cuda-wheel")
        }
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(CI.make_report(Path(directory), needs)["passed"])

    def test_report_rejects_old_dependency_fallback(self):
        needs = {
            name: {"result": "success"}
            for name in ("inventory", "static-review", "cpu-compatibility", "cuda-wheel")
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

    def test_source_change_updates_failure_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.dict(CI.os.environ, {"GITHUB_SHA": "first"}):
                first = CI.make_report(path, {})["fingerprint"]
            with patch.dict(CI.os.environ, {"GITHUB_SHA": "second"}):
                self.assertNotEqual(first, CI.make_report(path, {})["fingerprint"])


if __name__ == "__main__":
    unittest.main()
