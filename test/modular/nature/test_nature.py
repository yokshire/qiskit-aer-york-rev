# Licensed under the Apache License, Version 2.0.
"""Portable Nature behavior in an environment without any chemistry SDK or Aer."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qiskit import QuantumCircuit
from qiskit_aer_york import PluginError, Simulator, load_plugin, plugins
from qiskit_aer_york_qiskit import save_statevector
from qiskit_aer_york_data import IntegralBundle
import numpy as np

# One spatial orbital: h=-1.5, (11|11)=0.5, constant=0.4.
# Vacuum, one electron, and two electrons have analytic energies 0.4, -1.1, -2.1.
FCIDUMP = """&FCI NORB=1, NELEC=2, MS2=0,
 ORBSYM=1,
 ISYM=1,
&END
 0.5 1 1 1 1
 -1.5 1 1 0 0
 0.4 0 0 0 0
"""
RESPONSE = {"schema_version": 1, "format": "fcidump", "fcidump": FCIDUMP, "reference_energy": -2.1}


class NatureTests(unittest.TestCase):
    def test_absent_sdks_and_lazy_driver_discovery(self):
        for name in ("pyscf", "psi4", "qiskit_aer"):
            self.assertIsNone(importlib.util.find_spec(name), name)
            self.assertNotIn(name, sys.modules)
        self.assertIn("nature", {info.name for info in plugins("integration")})
        self.assertEqual({info.name for info in plugins("driver")}, {"pyscf"})
        driver = load_plugin("pyscf", "driver")
        self.assertEqual(driver.capabilities()["format"], "fcidump")
        self.assertNotIn("pyscf", sys.modules)

    def test_portable_integrals_and_actual_cpu_execution(self):
        nature = load_plugin("nature", "integration")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.fcidump"
            path.write_text(FCIDUMP, encoding="utf-8")
            problem = nature.from_fcidump(path)
        self.assertEqual(problem.num_particles, (1, 1))
        self.assertEqual(problem.num_spatial_orbitals, 1)
        operator = nature.qubit_operator(problem)
        simulator = Simulator("statevector")
        for occupied, expected in (((), 0.4), ((0,), -1.1), ((1,), -1.1), ((0, 1), -2.1)):
            circuit = QuantumCircuit(2)
            for qubit in occupied:
                circuit.x(qubit)
            save_statevector(circuit)
            vector = simulator.run(circuit).result().get_statevector()
            total = (
                float(vector.expectation_value(operator).real)
                + problem.hamiltonian.nuclear_repulsion_energy
            )
            self.assertAlmostEqual(total, expected, places=12)
        self.assertNotIn("pyscf", sys.modules)

    def test_driver_result_to_problem(self):
        nature = load_plugin("nature", "integration")
        fake = SimpleNamespace(run=lambda **options: RESPONSE)
        with patch("qiskit_aer_york_nature.load_plugin", return_value=fake):
            problem = nature.from_driver("external")
        self.assertAlmostEqual(problem.reference_energy, -2.1)
        self.assertEqual(problem.num_particles, (1, 1))
        fake.run = lambda **options: {"format": "unsupported"}
        with patch("qiskit_aer_york_nature.load_plugin", return_value=fake):
            with self.assertRaisesRegex(PluginError, "format"):
                nature.from_driver("external")

    def test_binary_integrals_match_analytic_energies_and_preserve_packed_views(self):
        nature = load_plugin("nature", "integration")
        data = IntegralBundle(
            [[-1.5]], [0.5], num_alpha=1, num_beta=1, nuclear_energy=0.4, reference_energy=-2.1
        )
        fake = SimpleNamespace(run_integrals=lambda **options: data)
        with patch("qiskit_aer_york_nature.load_plugin", return_value=fake):
            problem = nature.from_driver("external")
        alpha = problem.hamiltonian.electronic_integrals.alpha
        self.assertTrue(np.shares_memory(alpha["+-"].array, data.h1))
        self.assertTrue(np.shares_memory(alpha["++--"].array, data.eri_s8))
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        save_statevector(circuit)
        vector = Simulator().run(circuit).result().get_statevector()
        energy = vector.expectation_value(nature.qubit_operator(problem)).real + 0.4
        self.assertAlmostEqual(energy, -2.1)
        self.assertAlmostEqual(problem.reference_energy, -2.1)
        open_shell = IntegralBundle([[-1.5]], [0.5], num_alpha=1, num_beta=0, nuclear_energy=0.4)
        self.assertEqual(nature.from_integrals(open_shell).num_particles, (1, 0))

    def test_native_windows_rejected_before_process_or_sdk_load(self):
        driver = load_plugin("pyscf", "driver")
        with (
            patch("qiskit_aer_york_pyscf.platform.system", return_value="Windows"),
            patch("qiskit_aer_york_pyscf.subprocess.run") as run,
        ):
            self.assertFalse(driver.capabilities()["native_supported"])
            with self.assertRaisesRegex(PluginError, "native Windows"):
                driver.run(atom="H 0 0 0; H 0 0 0.735")
            run.assert_not_called()

    def test_wsl_request_is_data_and_runtime_is_explicit(self):
        driver = load_plugin("pyscf", "driver")
        atom = 'H 0 0 0; H 0 0 0.735; "$(data-only)"'
        reply = SimpleNamespace(returncode=0, stdout=json.dumps(RESPONSE), stderr="")
        with (
            patch("qiskit_aer_york_pyscf.platform.system", return_value="Windows"),
            patch("qiskit_aer_york_pyscf.shutil.which", return_value="wsl.exe"),
            patch("qiskit_aer_york_pyscf.subprocess.run", return_value=reply) as run,
        ):
            result = driver.run(
                atom=atom, runtime="wsl", python="/explicit/python", distribution="Ubuntu"
            )
        args, kwargs = run.call_args
        self.assertEqual(
            args[0][:5], ["wsl.exe", "--distribution", "Ubuntu", "--exec", "/explicit/python"]
        )
        self.assertFalse(kwargs["shell"])
        self.assertNotIn(atom, args[0])
        self.assertEqual(json.loads(kwargs["input"])["atom"], atom)
        self.assertEqual(result["runtime"], "wsl")

    def test_missing_wsl_configuration_and_invalid_inputs(self):
        driver = load_plugin("pyscf", "driver")
        with (
            patch("qiskit_aer_york_pyscf.platform.system", return_value="Windows"),
            patch("qiskit_aer_york_pyscf.shutil.which", return_value="wsl.exe"),
        ):
            with self.assertRaisesRegex(ValueError, "absolute Linux Python"):
                driver.run(atom="H 0 0 0", runtime="wsl")
        with (
            patch("qiskit_aer_york_pyscf.platform.system", return_value="Windows"),
            patch("qiskit_aer_york_pyscf.shutil.which", return_value=None),
        ):
            with self.assertRaisesRegex(PluginError, "WSL is unavailable"):
                driver.run(atom="H 0 0 0", runtime="wsl")
        for options in (
            {"spin": -1},
            {"threads": 0},
            {"timeout": float("nan")},
            {"unit": "unknown"},
        ):
            with self.assertRaises(ValueError):
                driver.run(atom="H 0 0 0", **options)

    def test_worker_timeout_and_failure_are_actionable(self):
        driver = load_plugin("pyscf", "driver")
        with patch("qiskit_aer_york_pyscf.platform.system", return_value="Linux"):
            with patch(
                "qiskit_aer_york_pyscf.subprocess.run",
                side_effect=subprocess.TimeoutExpired("python", 1),
            ):
                with self.assertRaisesRegex(PluginError, "exceeded"):
                    driver.run(atom="H 0 0 0", timeout=1)
            reply = SimpleNamespace(returncode=1, stdout="", stderr="No module named 'pyscf'")
            with patch("qiskit_aer_york_pyscf.subprocess.run", return_value=reply):
                with self.assertRaisesRegex(PluginError, "pyscf"):
                    driver.run(atom="H 0 0 0")
            reply.returncode, reply.stdout = 0, "not JSON"
            with patch("qiskit_aer_york_pyscf.subprocess.run", return_value=reply):
                with self.assertRaisesRegex(PluginError, "invalid JSON"):
                    driver.run(atom="H 0 0 0")
            reply.returncode, reply.stdout, reply.stderr = (
                1,
                "WSL distribution missing".encode("utf-16"),
                b"",
            )
            with patch("qiskit_aer_york_pyscf.subprocess.run", return_value=reply):
                with self.assertRaisesRegex(PluginError, "WSL distribution missing"):
                    driver.run(atom="H 0 0 0")


if __name__ == "__main__":
    unittest.main()
