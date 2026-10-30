# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York physical-channel regression for the numerical fix in upstream Aer PR #2399.
"""A redundant physical operator basis must not fail CVXPY's numerical PSD test."""

import importlib.util
import unittest

import numpy as np
from qiskit.quantum_info import Chi, Kraus, random_unitary
from qiskit_aer.noise import amplitude_damping_error
from qiskit_aer.utils import approximate_quantum_error
from qiskit_aer.utils.noise_transformation import _transform_by_operator_list


@unittest.skipUnless(
    importlib.util.find_spec("cvxpy"), "optional CVXPY regression (CI Python 3.12)"
)
class TestNoiseTransformationPSD(unittest.TestCase):
    def test_redundant_unitary_basis(self):
        rng = np.random.default_rng(7)
        basis = [Kraus(random_unitary(2, seed=rng)) for _ in range(20)]
        vectors = np.array([Chi(op).data.ravel() for op in [Kraus(np.eye(2))] + basis])
        gram = (vectors.conj() @ vectors.T).real
        self.assertGreater(np.linalg.eigvalsh(gram).min(), -1e-10)
        target = amplitude_damping_error(0.2)
        probabilities = _transform_by_operator_list(basis, target)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertGreaterEqual(probabilities.min(), -1e-6)
        self.assertLessEqual(probabilities.sum(), 1 + 1e-6)
        approximation = approximate_quantum_error(target, operator_list=basis)
        self.assertTrue(approximation.to_quantumchannel().is_cptp())
        self.assertAlmostEqual(sum(approximation.probabilities), 1)


if __name__ == "__main__":
    unittest.main()
