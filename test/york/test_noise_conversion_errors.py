# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regression coverage, adapted from upstream Qiskit Aer PR #2500.
"""Exercise installed-wheel error propagation, including native process failures."""

import subprocess
import sys
import textwrap
import unittest


class TestNoiseConversionErrors(unittest.TestCase):
    def test_original_error_serial_and_parallel(self):
        """An unsupported noise instruction must not corrupt the exception vector."""
        for method in ("statevector", "density_matrix"):
            for threads in (1, 2):
                with self.subTest(method=method, threads=threads):
                    code = textwrap.dedent(f"""
                        from qiskit import QuantumCircuit
                        from qiskit.circuit.library import Initialize
                        from qiskit_aer import AerSimulator
                        from qiskit_aer.noise import NoiseModel, QuantumError, amplitude_damping_error
                        initialize = QuantumCircuit(1)
                        initialize.append(Initialize([1, 0]), [0])
                        noise = NoiseModel()
                        noise.add_quantum_error(
                            QuantumError([(initialize, 0.5), (QuantumCircuit(1), 0.5)]), 'x', [0]
                        )
                        for qubit in range(1, 12):
                            noise.add_quantum_error(amplitude_damping_error(0.01 * qubit), 'x', [qubit])
                        circuit = QuantumCircuit(1)
                        circuit.x(0)
                        circuit.measure_all()
                        backend = AerSimulator(method={method!r}, max_parallel_threads={threads})
                        try:
                            backend.run(circuit, noise_model=noise).result()
                        except ValueError as error:
                            assert "invalid instruction 'initialize'" in str(error), str(error)
                        else:
                            raise AssertionError('original conversion error was lost')
                        """)
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", code],
                        capture_output=True,
                        text=True,
                        timeout=45,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
