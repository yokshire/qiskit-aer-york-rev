# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Regression coverage for the hosted installed-wheel execution environment."""

from pathlib import Path
import unittest


class YorkWorkflowTests(unittest.TestCase):
    def test_pull_request_retarget_runs_dependency_validation(self):
        """Changing a stacked PR's base must invalidate the previous CI evidence."""
        workflow = (
            Path(__file__).parents[2] / ".github/workflows/york-compatibility.yml"
        ).read_text(encoding="utf-8")
        trigger = workflow.split("\n  pull_request:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("    branches: [main, codex/york-mps-truncation]", trigger)
        self.assertIn("\n    types: [opened, synchronize, reopened, edited]", trigger)

    def test_cpu_job_reserves_static_tls_before_python_starts(self):
        """Qiskit 2.6/Python 3.15 otherwise exhaust TLS before libgomp loads."""
        workflow = (
            Path(__file__).parents[2] / ".github/workflows/york-compatibility.yml"
        ).read_text(encoding="utf-8")
        cpu_job = workflow.split("\n  cpu-compatibility:\n", 1)[1].split("\n  cuda-wheel:\n", 1)[0]
        environment = cpu_job.split("\n    env:\n", 1)[1].split("\n    steps:\n", 1)[0]
        self.assertIn("\n      GLIBC_TUNABLES: glibc.rtld.optional_static_tls=4096", environment)


if __name__ == "__main__":
    unittest.main()
