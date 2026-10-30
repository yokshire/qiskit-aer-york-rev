# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regressions adapted from upstream Qiskit Aer PRs #2473 and #2490.
"""Sampler seed precedence and backend-error reporting on an installed wheel."""

import unittest

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, AerError
from qiskit_aer.primitives import SamplerV2


class TestSamplerV2Errors(unittest.TestCase):
    def test_seed_precedence(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.measure_all()
        for constructor, run_seed, expected in (
            (None, 7, 7),
            (42, 7, 42),
            (0, 7, 0),
            (42, None, 42),
        ):
            with self.subTest(constructor=constructor, run_seed=run_seed):
                options = {} if run_seed is None else {"seed_simulator": run_seed}
                sampler = SamplerV2(seed=constructor, options={"run_options": options})
                result = sampler.run([circuit], shots=200).result()
                reference = SamplerV2(seed=expected).run([circuit], shots=200).result()
                self.assertEqual(
                    result[0].data.meas.get_counts(), reference[0].data.meas.get_counts()
                )
                self.assertEqual(sampler.options.run_options, options)
        unseeded = SamplerV2().run([circuit], shots=8).result()
        self.assertEqual(unseeded[0].data.meas.num_shots, 8)

    def test_failed_experiment_preserves_backend_status(self):
        circuit = QuantumCircuit(1, 1)
        circuit.t(0)
        circuit.measure(0, 0)
        options = {"method": "stabilizer"}
        experiment = AerSimulator(**options).run(circuit, shots=8, memory=True).result().results[0]
        self.assertFalse(experiment.success)
        self.assertEqual(experiment.shots, 0)
        with self.assertRaises(AerError) as caught:
            SamplerV2(options={"backend_options": options}).run([circuit], shots=8).result()
        self.assertIn(experiment.status, str(caught.exception))

    def test_successful_unmeasured_circuit(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)
        result = SamplerV2().run([circuit], shots=8).result()
        self.assertEqual(result[0].metadata["shots"], 8)


if __name__ == "__main__":
    unittest.main()
