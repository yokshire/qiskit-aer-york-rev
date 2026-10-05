# Licensed under the Apache License, Version 2.0.
"""Functional tests against independent Qiskit references, without native Aer."""

import importlib.util
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, random_unitary
from qiskit_aer_york import Simulator, SimulationError, load_plugin, plugins
from qiskit_aer_york_qiskit import save_statevector


class EngineTests(unittest.TestCase):
    def test_native_aer_is_absent(self):
        self.assertIsNone(importlib.util.find_spec("qiskit_aer"))

    def test_gpu_sdk_discovery_is_lazy(self):
        names = {info.name for info in plugins()}
        self.assertTrue({"statevector", "opencl", "cuda", "rocm"} <= names)
        for name in ("cupy", "pyopencl", "qiskit_aer"):
            self.assertNotIn(name, sys.modules)
        self.assertTrue({"qiskit", "data"} <= {info.name for info in plugins("integration")})

    def test_random_circuits_and_target_order(self):
        for precision, tolerance in (("double", 1e-10), ("single", 2e-5)):
            backend = Simulator("statevector", precision=precision)
            for seed in range(10):
                rng = np.random.default_rng(seed)
                circuit = QuantumCircuit(6, global_phase=0.37)
                for _ in range(30):
                    first, second = rng.choice(6, 2, replace=False)
                    circuit.u(*rng.uniform(-np.pi, np.pi, 3), int(first))
                    circuit.cx(int(first), int(second))
                circuit.unitary(random_unitary(8, seed=seed), [5, 0, 2])
                expected = Statevector.from_instruction(circuit).data
                save_statevector(circuit)
                result = backend.run(circuit).result()
                np.testing.assert_allclose(result.get_statevector().data, expected, atol=tolerance)
                self.assertEqual(result.results[0].metadata["device"], "CPU")
                self.assertEqual(result.results[0].metadata["gate_count"], 61)

    def test_sampling_parameter_binding_and_register_mapping(self):
        theta = Parameter("theta")
        circuit = QuantumCircuit(3, 3)
        circuit.ry(2 * theta, 2)
        circuit.measure(2, 0)
        backend = Simulator()
        result = backend.run(circuit, parameter_binds=[{theta: [0, np.pi / 2]}], shots=32).result()
        self.assertEqual(result.get_counts(0), {"000": 32})
        self.assertEqual(result.get_counts(1), {"001": 32})
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()
        a = backend.run(bell, shots=128, memory=True, seed_simulator=8).result()
        b = backend.run(bell, shots=128, memory=True, seed_simulator=8).result()
        self.assertEqual(set(a.get_counts()), {"00", "11"})
        self.assertEqual(a.get_memory(), b.get_memory())

    def test_qiskit_primitives(self):
        integration = load_plugin("qiskit", "integration")
        simulator = Simulator()
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        measured = bell.copy()
        measured.measure_all()
        sampled = integration.sampler(simulator).run([measured], shots=128).result()
        self.assertEqual(set(sampled[0].data.meas.get_counts()), {"00", "11"})
        estimated = integration.estimator(simulator).run([(bell, "ZZ")]).result()
        self.assertAlmostEqual(float(estimated[0].data.evs), 1)

    def test_repeated_matrices_and_host_snapshots_preserve_state(self):
        circuit = QuantumCircuit(2, 2, global_phase=0.3)
        for _ in range(50):
            circuit.h(0)
            circuit.cx(0, 1)
        expected = Statevector.from_instruction(circuit)
        save_statevector(circuit, "first")
        save_statevector(circuit, "second")
        circuit.measure([0, 1], [0, 1])
        result = Simulator().run(circuit, shots=32, seed_simulator=8).result()
        np.testing.assert_allclose(result.data()["first"], expected.data, atol=1e-12)
        np.testing.assert_allclose(result.data()["second"], expected.data, atol=1e-12)
        metadata = result.results[0].metadata
        self.assertEqual(metadata["matrix_validations"], 2)
        self.assertEqual(metadata["matrix_validation_hits"], 98)
        self.assertEqual(metadata["host_state_transfers"], 1)
        # A gate between saves must invalidate the exported host state.
        circuit = QuantumCircuit(1)
        save_statevector(circuit, "before")
        circuit.x(0)
        save_statevector(circuit, "after")
        result = Simulator().run(circuit).result()
        np.testing.assert_allclose(result.data()["before"], [1, 0])
        np.testing.assert_allclose(result.data()["after"], [0, 1])
        self.assertEqual(result.results[0].metadata["host_state_transfers"], 2)

    def test_equal_gate_names_with_different_definitions_do_not_share_matrices(self):
        first, second = QuantumCircuit(1), QuantumCircuit(1)
        first.x(0)
        second.h(0)
        circuit = QuantumCircuit(1)
        circuit.append(first.to_gate(label="same"), [0])
        circuit.append(second.to_gate(label="same"), [0])
        expected = Statevector.from_instruction(circuit).data
        save_statevector(circuit)
        np.testing.assert_allclose(Simulator().run(circuit).result().get_statevector(), expected)

    def test_invalid_options_and_unsupported_semantics(self):
        simulator = Simulator()
        circuit = QuantumCircuit(1)
        for options in ({"shots": 0}, {"max_memory_mb": -1}, {"precision": "half"}, {"typo": 1}):
            with self.assertRaises(SimulationError):
                simulator.run(circuit, **options)
        circuit.reset(0)
        with self.assertRaisesRegex(SimulationError, "reset"):
            simulator.run(circuit).result()
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)
        circuit.x(0)
        with self.assertRaisesRegex(SimulationError, "terminal"):
            simulator.run(circuit).result()
        with self.assertRaisesRegex(SimulationError, "max_memory_mb"):
            simulator.run(QuantumCircuit(20), max_memory_mb=1).result()


if __name__ == "__main__":
    unittest.main()
