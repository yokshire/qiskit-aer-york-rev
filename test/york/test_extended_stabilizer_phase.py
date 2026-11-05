# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regressions for the existing-gate phase bug in upstream Aer PR #2425.
"""Saved amplitudes must retain phase independently of decomposition rank."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, SXGate, SXdgGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


class TestExtendedStabilizerPhase(unittest.TestCase):
    def run_circuit(self, circuit, threads=1):
        result = (
            AerSimulator(
                method="extended_stabilizer",
                max_parallel_threads=threads,
                extended_stabilizer_parallel_threshold=1,
                seed_simulator=117,
            )
            .run(circuit, shots=1)
            .result()
        )
        self.assertTrue(result.success, result.status)
        self.assertEqual(result.results[0].metadata["method"], "extended_stabilizer")
        self.assertEqual(result.results[0].metadata["device"], "CPU")
        return result.data()

    def test_clifford_phases_and_circuit_phase(self):
        gates = [SXGate(), SXdgGate()]
        gates += [RZGate(k * np.pi / 2) for k in range(-5, 6)]
        for gate in gates:
            with self.subTest(gate=gate):
                circuit = QuantumCircuit(2, global_phase=0.37)
                circuit.h(0)
                circuit.cx(0, 1)
                circuit.append(gate, [0])
                expected = Statevector(circuit).data
                circuit.save_statevector()
                actual = self.run_circuit(circuit)["statevector"]
                # Statevector.equiv ignores precisely the phase under test.
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_decomposed_state_phase_serial_and_parallel(self):
        for threads in (1, 2):
            for gate in (SXGate(), SXdgGate(), RZGate(np.pi / 2)):
                with self.subTest(threads=threads, gate=gate):
                    circuit = QuantumCircuit(1, global_phase=0.37)
                    circuit.h(0)
                    circuit.t(0)  # Force multiple decomposition terms.
                    circuit.save_statevector(label="before")
                    for _ in range(3):
                        circuit.append(gate, [0])
                    circuit.save_statevector(label="after")
                    data = self.run_circuit(circuit, threads)
                    # Use the actual approximate pre-gate state as oracle input:
                    # this tests phase exactly without a stochastic tolerance.
                    expected = Statevector(data["before"])
                    for _ in range(3):
                        expected = expected.evolve(gate)
                    np.testing.assert_allclose(data["after"], expected.data, atol=1e-10, rtol=0)

    def test_parallel_sampling_probabilities_unchanged(self):
        for threads in (1, 2):
            with self.subTest(threads=threads):
                circuit = QuantumCircuit(1, 1)
                circuit.t(0)
                circuit.sx(0)
                circuit.sxdg(0)
                circuit.rz(np.pi / 2, 0)
                circuit.measure(0, 0)
                result = (
                    AerSimulator(
                        method="extended_stabilizer",
                        max_parallel_threads=threads,
                        extended_stabilizer_parallel_threshold=1,
                        seed_simulator=117,
                    )
                    .run(circuit, shots=32)
                    .result()
                )
                self.assertTrue(result.success, result.status)
                self.assertEqual(result.get_counts(), {"0": 32})


if __name__ == "__main__":
    unittest.main()
