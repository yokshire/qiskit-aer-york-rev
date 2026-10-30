# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""York regression for a real CPU Thrust/chunked-statevector wheel."""

import unittest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


class TestThrustCPU(unittest.TestCase):
    def test_chunked_statevector(self):
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 3)
        circuit.ry(0.27, 1)
        circuit.cx(1, 2)
        target = Statevector(circuit)
        circuit.save_statevector()
        result = (
            AerSimulator(
                method="statevector",
                device="CPU",
                blocking_enable=True,
                blocking_qubits=2,
            )
            .run(circuit)
            .result()
        )
        self.assertTrue(result.success, result.status)
        metadata = result.results[0].metadata
        self.assertEqual(metadata["device"], "CPU")
        self.assertTrue(metadata["cacheblocking"]["enabled"])
        self.assertTrue(Statevector(result.get_statevector()).equiv(target))


if __name__ == "__main__":
    unittest.main()
