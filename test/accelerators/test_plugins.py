# Licensed under the Apache License, Version 2.0.
"""Plugin contract tests. The host engine below is a test double, not GPU evidence."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli

from qiskit_aer import AerSimulator, AerError
from qiskit_aer.accelerators import available_accelerators, load_accelerator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2, EstimatorV2


class HostTestState:
    """Independent Qiskit reference engine used only by these tests."""

    metadata = {"device": "GPU", "name": "TEST DOUBLE", "runtime": "test"}

    def __init__(self, num_qubits):
        self.state = Statevector.from_int(0, 1 << num_qubits)

    def apply(self, matrix, qubits):
        self.state = self.state.evolve(matrix, qargs=list(qubits))

    def to_host(self):
        return self.state.data.copy()


class HostTestProvider:
    api_version = 1

    def devices(self):
        return [
            {
                "device": "GPU",
                "name": "TEST DOUBLE",
                "vendor": "test",
                "device_id": "0",
                "fp64": True,
            }
        ]

    def create_state(self, num_qubits, precision, options):
        return HostTestState(num_qubits)


def entry(name="test", provider=HostTestProvider):
    return SimpleNamespace(name=name, load=lambda: provider)


class PluginContractTests(unittest.TestCase):
    def setUp(self):
        self.discovery = patch("qiskit_aer.accelerators.entry_points", return_value=[entry()])
        self.discovery.start()
        self.addCleanup(self.discovery.stop)

    def backend(self, **kwargs):
        return AerSimulator(accelerator="test", **kwargs)

    def test_cpu_does_not_discover_plugins(self):
        with patch(
            "qiskit_aer.accelerators.entry_points", side_effect=AssertionError("GPU import")
        ):
            circuit = QuantumCircuit(1, 1)
            circuit.x(0)
            circuit.measure(0, 0)
            result = AerSimulator().run(circuit, shots=8).result()
            self.assertEqual(result.get_counts(), {"1": 8})
            self.assertEqual(result.results[0].metadata["device"], "CPU")

    def test_discovery_is_lazy(self):
        with patch(
            "qiskit_aer.accelerators.entry_points",
            return_value=[SimpleNamespace(name="broken", load=lambda: self.fail("SDK loaded"))],
        ):
            self.assertEqual(available_accelerators(), ["broken"])

    def test_missing_and_ambiguous_plugin(self):
        for entries in ([], [entry(), entry()]):
            with patch("qiskit_aer.accelerators.entry_points", return_value=entries):
                with self.assertRaises(AerError):
                    load_accelerator("test")

    def test_api_version_guard(self):
        with patch(
            "qiskit_aer.accelerators.entry_points",
            return_value=[entry(provider=lambda: SimpleNamespace(api_version=999))],
        ):
            with self.assertRaisesRegex(AerError, "API 1"):
                load_accelerator("test")

    def test_terminal_measurement_mapping_and_seed(self):
        circuit = QuantumCircuit(3, 3)
        circuit.x(0)
        circuit.h(2)
        circuit.measure(0, 2)
        circuit.measure(2, 0)
        backend = self.backend()
        first = backend.run(circuit, shots=100, memory=True, seed_simulator=42).result()
        second = backend.run(circuit, shots=100, memory=True, seed_simulator=42).result()
        self.assertEqual(first.get_memory(), second.get_memory())
        self.assertEqual(set(first.get_counts()), {"100", "101"})
        self.assertEqual(sum(first.get_counts().values()), 100)
        self.assertEqual(first.results[0].metadata["gpu_gate_count"], 2)

    def test_nonconsecutive_qubits_and_global_phase(self):
        circuit = QuantumCircuit(4, global_phase=0.37)
        circuit.h(3)
        circuit.ry(0.52, 0)
        circuit.cx(3, 1)
        circuit.swap(0, 2)
        reference = Statevector.from_instruction(circuit)
        circuit.save_statevector()
        result = self.backend().run(circuit).result()
        np.testing.assert_allclose(result.get_statevector().data, reference.data, atol=1e-12)

    def test_saved_observables_and_intermediate_state(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.save_statevector(label="before")
        circuit.cx(0, 1)
        circuit.save_expectation_value(Pauli("XX"), [0, 1], label="xx")
        circuit.save_expectation_value(Pauli("Z"), [1], label="z")
        data = self.backend().run(circuit).result().data()
        self.assertAlmostEqual(data["xx"], 1)
        self.assertAlmostEqual(data["z"], 0)
        np.testing.assert_allclose(data["before"].data, [2**-0.5, 2**-0.5, 0, 0])

    def test_parameter_batches(self):
        param = Parameter("theta")
        circuit = QuantumCircuit(1, 1)
        circuit.ry(2 * param, 0)
        circuit.measure(0, 0)
        result = (
            self.backend()
            .run(circuit, parameter_binds=[{param: [0, np.pi / 2]}], shots=16)
            .result()
        )
        self.assertEqual(result.get_counts(0), {"0": 16})
        self.assertEqual(result.get_counts(1), {"1": 16})

    def test_sampler_v2(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        result = (
            SamplerV2(options={"backend_options": {"accelerator": "test"}})
            .run([circuit], shots=128)
            .result()
        )
        self.assertEqual(set(result[0].data.meas.get_counts()), {"00", "11"})

    def test_estimator_v2(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        result = (
            EstimatorV2(options={"backend_options": {"accelerator": "test"}})
            .run([(circuit, "ZZ")])
            .result()
        )
        self.assertAlmostEqual(float(result[0].data.evs), 1)

    def test_reject_unsupported_options(self):
        circuit = QuantumCircuit(1)
        for options in (
            {"device": "CPU"},
            {"method": "density_matrix"},
            {"precision": "half"},
            {"cuStateVec_enable": True},
            {"shots": 0},
            {"shots": 1.2},
        ):
            with self.subTest(options=options), self.assertRaises(AerError):
                self.backend().run(circuit, **options)

    def test_reject_noise_and_nonterminal_measurements(self):
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)
        circuit.x(0)
        with self.assertRaisesRegex(AerError, "terminal"):
            self.backend().run(circuit).result()
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.1, 1), ["x"])
        with self.assertRaisesRegex(AerError, "noise"):
            self.backend(noise_model=noise).run(QuantumCircuit(1))

    def test_reject_reset_and_conditional_saves(self):
        for conditional in (False, True):
            circuit = QuantumCircuit(1)
            if conditional:
                circuit.save_statevector(conditional=True)
            else:
                circuit.reset(0)
            with self.assertRaises(AerError):
                self.backend().run(circuit).result()

    def test_no_cpu_fallback_from_plugin(self):
        class BadProvider(HostTestProvider):
            def create_state(self, num_qubits, precision, options):
                state = HostTestState(num_qubits)
                state.metadata = {"device": "CPU"}
                return state

        with patch(
            "qiskit_aer.accelerators.entry_points", return_value=[entry(provider=BadProvider)]
        ):
            with self.assertRaisesRegex(AerError, "GPU execution"):
                self.backend().run(QuantumCircuit(1)).result()

    def test_memory_limit_and_incomplete_binds(self):
        with self.assertRaisesRegex(AerError, "max_memory"):
            self.backend().run(QuantumCircuit(20), max_memory_mb=1).result()
        circuit = QuantumCircuit(1)
        circuit.rx(Parameter("theta"), 0)
        with self.assertRaisesRegex(AerError, "unbound"):
            self.backend().run(circuit).result()

    def test_temporary_selection_and_capabilities(self):
        backend = AerSimulator()
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.measure(0, 0)
        result = backend.run(circuit, accelerator="test", shots=8).result()
        self.assertEqual(result.get_counts(), {"1": 8})
        self.assertIsNone(backend.options.accelerator)
        self.assertEqual(backend.options.device, "CPU")
        self.assertEqual(self.backend().options.device, "GPU")
        self.assertEqual(self.backend().available_methods(), ["automatic", "statevector"])
        self.assertEqual(self.backend().available_devices(), ["GPU"])

    def test_capabilities_respect_selected_vendor_and_device(self):
        for selector in ({"vendor": "amd"}, {"device_id": "9"}):
            self.assertEqual(self.backend(accelerator_options=selector).available_devices(), [])


if __name__ == "__main__":
    unittest.main()
