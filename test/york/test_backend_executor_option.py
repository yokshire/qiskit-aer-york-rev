# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""York regression: forward existing backend/run executor options to AerJob."""

import unittest
from concurrent.futures import ThreadPoolExecutor

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.jobs import utils


class TestBackendExecutorOption(unittest.TestCase):
    @staticmethod
    def circuit():
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.measure(0, 0)
        return circuit

    def test_backend_executor(self):
        with ThreadPoolExecutor(1) as executor:
            job = AerSimulator(executor=executor).run(self.circuit(), shots=8)
            self.assertIs(job._executor, executor)
            self.assertEqual(job.result(timeout=10).get_counts(), {"1": 8})

    def test_run_executor_override(self):
        with ThreadPoolExecutor(1) as configured, ThreadPoolExecutor(1) as override:
            job = AerSimulator(executor=configured).run(self.circuit(), shots=8, executor=override)
            self.assertIs(job._executor, override)
            self.assertEqual(job.result(timeout=10).get_counts(), {"1": 8})

    def test_default_executor(self):
        job = AerSimulator().run(self.circuit(), shots=8)
        self.assertIs(job._executor, utils.DEFAULT_EXECUTOR)
        self.assertEqual(job.result(timeout=10).get_counts(), {"1": 8})


if __name__ == "__main__":
    unittest.main()
