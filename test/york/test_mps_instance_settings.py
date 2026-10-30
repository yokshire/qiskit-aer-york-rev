# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regression for the per-state settings fix in upstream Qiskit Aer PR #2298.
"""Interleaved native MPS states must retain their own truncation settings."""

import subprocess
import sys
import textwrap
import unittest


class TestMPSInstanceSettings(unittest.TestCase):
    def test_initializing_another_state_does_not_change_truncation(self):
        code = textwrap.dedent("""
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit_aer.quantum_info.states.aer_state import AerState
            for other_bond, other_threshold in ((1, 0), (4, .4)):
                state = AerState(
                    method='matrix_product_state', matrix_product_state_max_bond_dimension=4,
                    matrix_product_state_truncation_threshold=0,
                )
                state.allocate_qubits(4)
                state.initialize()
                other = AerState(
                    method='matrix_product_state', matrix_product_state_max_bond_dimension=other_bond,
                    matrix_product_state_truncation_threshold=other_threshold,
                )
                other.allocate_qubits(4)
                other.initialize()
                state.apply_h(0)
                state.apply_cx(0, 2)
                state.apply_h(1)
                state.apply_cx(1, 3)
                actual = Statevector(state.move_to_ndarray())
                circuit = QuantumCircuit(4)
                circuit.h(0)
                circuit.cx(0, 2)
                circuit.h(1)
                circuit.cx(1, 3)
                assert actual.equiv(Statevector(circuit)), (other_bond, other_threshold, actual)
            """)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=45
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
