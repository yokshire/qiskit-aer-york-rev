# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regressions for upstream Qiskit Aer PRs #2451 and #2489.
"""Schmidt truncation must retain a coefficient and read only live entries."""

import subprocess
import sys
import textwrap
import unittest


class TestMPSTruncation(unittest.TestCase):
    def test_threshold_and_discarded_weight(self):
        code = textwrap.dedent("""
            import re
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            circuit = QuantumCircuit(4)
            circuit.h(1)
            circuit.cx(1, 2)
            circuit.swap(0, 1)
            circuit.swap(2, 3)
            circuit.h(1)
            circuit.cx(1, 2)
            circuit.measure_all()
            for threshold, target in ((0, (2, 4, 2)), (.4, (2, 3, 2)), (.9, (1, 1, 1))):
                result = AerSimulator(
                    method='matrix_product_state',
                    matrix_product_state_truncation_threshold=threshold,
                    mps_log_data=True,
                ).run(circuit, shots=1).result()
                assert result.success, result.status
                matches = re.findall(
                    r'cx on qubits 1,2, BD=\\[(\\d+) (\\d+) (\\d+)\\]',
                    result.results[0].metadata['MPS_log_data'],
                )
                assert matches, result.results[0].metadata
                bonds = tuple(int(value) for value in matches[-1])
                assert bonds == target, (threshold, bonds, target)
            bell = QuantumCircuit(2)
            bell.h(0)
            bell.cx(0, 1)
            bell.save_statevector()
            result = AerSimulator(
                method='matrix_product_state', matrix_product_state_max_bond_dimension=1,
                mps_log_data=True,
            ).run(bell).result()
            assert result.success, result.status
            log = result.results[0].metadata['MPS_log_data']
            assert 'discarded_value=0.5' in log, log
            """)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=45
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
