# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regression for the per-state settings fix in upstream Qiskit Aer PR #2298.
"""Interleaved native MPS states must retain their own truncation settings."""

import subprocess
import sys
import textwrap
import unittest


class TestMPSInstanceSettings(unittest.TestCase):
    def run_case(self, code):
        result = subprocess.run(
            [sys.executable, "-I", "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_multiqubit_temporary_settings_and_moved_metadata(self):
        self.run_case("""
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import random_statevector, random_unitary
            from qiskit_aer import AerSimulator
            for kind in ('ccx', 'unitary', 'partial_initialize'):
                circuit = QuantumCircuit(4 if kind == 'partial_initialize' else 3)
                if kind == 'ccx':
                    circuit.h(0)
                    circuit.h(1)
                    circuit.ccx(0, 1, 2)
                elif kind == 'unitary':
                    circuit.unitary(random_unitary(8, seed=2298), [0, 1, 2])
                else:
                    circuit.initialize(random_statevector(8, seed=2298).data, [0, 1, 2])
                circuit.save_matrix_product_state()
                result = AerSimulator(
                    method='matrix_product_state', matrix_product_state_max_bond_dimension=1,
                    matrix_product_state_truncation_threshold=0, fusion_enable=False,
                ).run(circuit).result()
                assert result.success, result.status
                bonds = result.data()['matrix_product_state'][1]
                assert all(len(s) <= 1 for s in bonds), (kind, tuple(map(len, bonds)))
                # Moving the last saved state must clear tensors, not its configuration.
                assert result.results[0].metadata['matrix_product_state_max_bond_dimension'] == 1
            """)

    def test_copy_preserves_observable_truncation(self):
        self.run_case("""
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            operations = [
                (0, .27475436221522076, 2, 1), (2, .754749892784897, 1, 0),
                (4, 2.0437385549779874, 3, 4), (4, 2.8481864192797306, 4, 3),
                (4, .8047222249479843, 0, 4), (2, 1.3946815862919362, 2, 3),
                (3, 2.6753683734848623, 4, 1), (4, .5323093064752981, 1, 0),
                (1, .9754287862834801, 1, 2), (4, 2.02343909203761, 1, 3),
                (3, 2.2032050756819332, 2, 4), (2, 2.9429297192163797, 3, 1),
                (3, 2.14545269471236, 3, 1), (3, .8373067287690653, 3, 2),
                (3, 2.699101766269877, 3, 2),
            ]
            circuit = QuantumCircuit(5)
            for q, angle, a, b in operations:
                circuit.ry(angle, q)
                circuit.cx(a, b)
            circuit.save_density_matrix([4, 0, 2], label='rho')
            result = AerSimulator(
                method='matrix_product_state', matrix_product_state_max_bond_dimension=2,
                matrix_product_state_truncation_threshold=0, fusion_enable=False,
                max_parallel_threads=1,
            ).run(circuit).result()
            assert result.success, result.status
            # Reference captured from the original static-settings implementation
            # with only the separate SVD-bound/truncation repair applied (York #10).
            # The initial instance-settings port instead yields .0178728898523.
            observed = result.data()['rho'].data[6, 6].real
            assert abs(observed - .005791249905146464) < 1e-8, observed
            """)

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
