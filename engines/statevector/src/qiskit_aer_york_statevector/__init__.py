# Licensed under the Apache License, Version 2.0.
"""NumPy unitary evolution with little-endian Qiskit target ordering."""

import numpy as np


class StatevectorEngine:
    api_version = 1

    def create_backend(self, **options):
        from qiskit_aer_york_qiskit import StatevectorBackend

        return StatevectorBackend(self, engine="statevector", device="CPU", **options)

    def create_state(self, num_qubits, precision, options):
        if options:
            raise ValueError(f"CPU engine has no device selectors: {sorted(options)}")
        return _State(num_qubits, precision)

    def devices(self):
        return [{"device": "CPU", "name": "NumPy", "fp64": True}]


class _State:
    metadata = {"device": "CPU", "name": "NumPy", "runtime": "NumPy"}

    def __init__(self, num_qubits, precision):
        self.num_qubits = num_qubits
        self.dtype = np.complex128 if precision == "double" else np.complex64
        self.current = np.zeros(1 << num_qubits, dtype=self.dtype)
        self.current[0] = 1

    def apply(self, matrix, qubits):
        # Last tensor axis is bit 0; the first target is the local least significant bit.
        targets = [self.num_qubits - 1 - qubit for qubit in reversed(qubits)]
        rest = [axis for axis in range(self.num_qubits) if axis not in targets]
        order = targets + rest
        tensor = self.current.reshape((2,) * self.num_qubits).transpose(order)
        evolved = np.asarray(matrix, dtype=self.dtype) @ tensor.reshape(1 << len(qubits), -1)
        self.current = (
            evolved.reshape((2,) * self.num_qubits).transpose(np.argsort(order)).reshape(-1)
        )

    def to_host(self):
        return self.current.copy()

    def finish(self):
        pass
