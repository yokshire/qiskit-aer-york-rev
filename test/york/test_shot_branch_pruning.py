# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regression for the iterator-owner fix in upstream Qiskit Aer PR #2489.
"""Pruning empty parameter branches must not corrupt the native shot vector."""

import subprocess
import sys
import textwrap
import unittest


class TestShotBranchPruning(unittest.TestCase):
    def test_sparse_parameter_branches(self):
        code = textwrap.dedent("""
            from qiskit import QuantumCircuit
            from qiskit.circuit import Parameter
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, pauli_error
            p = Parameter('p')
            circuit = QuantumCircuit(2)
            circuit.rx(p, 0)
            circuit.cx(0, 1)
            circuit.measure_all()
            noise = NoiseModel()
            noise.add_all_qubit_quantum_error(pauli_error([('X', .5), ('I', .5)]), ['rx'])
            for threads in (1, 2):
                backend = AerSimulator(method='statevector', max_parallel_threads=threads)
                for shots in (1, 2, 4):
                    for seed in range(4):
                        result = backend.run(
                            circuit, shots=shots, noise_model=noise,
                            parameter_binds=[{p: [0, 1, 2, 3, 4]}],
                            shot_branching_enable=True, runtime_parameter_bind_enable=True,
                            seed_simulator=seed,
                        ).result()
                        assert result.success, result.status
                        assert len(result.results) == 5
                        assert all(sum(c.values()) == shots for c in result.get_counts())
                        # Compare each bound circuit to the non-branching reference.
                        reference = backend.run(
                            circuit, shots=shots, noise_model=noise,
                            parameter_binds=[{p: [0, 1, 2, 3, 4]}],
                            shot_branching_enable=False, runtime_parameter_bind_enable=False,
                            seed_simulator=seed,
                        ).result()
                        assert reference.success, reference.status
                        assert result.get_counts() == reference.get_counts()
            """)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=90
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
