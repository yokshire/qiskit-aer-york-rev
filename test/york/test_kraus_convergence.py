# This code is part of Qiskit.
# (C) Copyright IBM 2026.
# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York adaptations of the physical-channel regressions from upstream PR #2500.
"""Valid channels with degenerate Choi eigenvalues must be convertible."""

import subprocess
import sys
import unittest

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error


class TestKrausConvergence(unittest.TestCase):
    def run_case(self, case):
        result = subprocess.run(
            [sys.executable, "-I", __file__, case], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_noise_conversion(self):
        self.run_case("check_noise_conversion")

    def test_kraus_fusion(self):
        self.run_case("check_kraus_fusion")

    def assert_probabilities(self, result, target, shots):
        self.assertTrue(result.success, result.status)
        counts = result.get_counts()
        for key in set(target) | set(counts):
            self.assertAlmostEqual(counts.get(key, 0) / shots, target.get(key, 0), delta=0.05)

    def check_noise_conversion(self):
        depolarizing = depolarizing_error(0.1, 3)
        damping = amplitude_damping_error(0.1)
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing, ["ccx"])
        noise.add_all_qubit_quantum_error(damping, ["x"])
        circuit = QuantumCircuit(3)
        circuit.x(0)
        target_circuit = circuit.copy()
        target_circuit.append(Kraus(damping), [0])
        target_circuit.ccx(0, 1, 2)
        target_circuit.append(Kraus(depolarizing), [0, 1, 2])
        target = DensityMatrix(target_circuit).probabilities_dict()
        circuit.ccx(0, 1, 2)
        circuit.measure_all()
        for threads in (1, 2):
            with self.subTest(threads=threads):
                result = (
                    AerSimulator(method="statevector", max_parallel_threads=threads)
                    .run(circuit, noise_model=noise, shots=2000, seed_simulator=2455)
                    .result()
                )
                self.assert_probabilities(result, target, 2000)

    def check_kraus_fusion(self):
        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.append(Kraus(depolarizing_error(0.1, 3)), [0, 1, 2])
        target = DensityMatrix(circuit).probabilities_dict()
        circuit.measure_all()
        result = (
            AerSimulator(
                method="statevector",
                fusion_enable=True,
                fusion_threshold=1,
                fusion_max_qubit=3,
                fusion_verbose=True,
            )
            .run(circuit, shots=2000, seed_simulator=2455)
            .result()
        )
        self.assert_probabilities(result, target, 2000)
        self.assertTrue(result.results[0].metadata["fusion"]["applied"])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("check_noise_conversion", "check_kraus_fusion"):
        getattr(TestKrausConvergence(), sys.argv[1])()
    else:
        unittest.main()
