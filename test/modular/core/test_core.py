# Licensed under the Apache License, Version 2.0.
"""Executed in a fresh environment containing only the core wheel."""

import importlib.util
import importlib.metadata as metadata
import sys
import unittest
from unittest.mock import Mock, patch

from qiskit_aer_york import (
    Simulator,
    Toolkit,
    MissingPluginError,
    PluginError,
    load_plugin,
    plugins,
)


class CoreTests(unittest.TestCase):
    def test_core_has_no_heavy_dependencies(self):
        for name in (
            "numpy",
            "scipy",
            "qiskit",
            "qiskit_nature",
            "pyscf",
            "psi4",
            "qiskit_aer",
            "cupy",
            "pyopencl",
        ):
            self.assertIsNone(importlib.util.find_spec(name), name)
            self.assertNotIn(name, sys.modules)
        requirements = metadata.requires("qiskit-aer-york-core") or []
        self.assertTrue(all("extra ==" in requirement for requirement in requirements))

    def test_empty_registry_and_missing_engine(self):
        self.assertEqual(plugins(), ())
        simulator = Simulator()
        with self.assertRaisesRegex(MissingPluginError, r"core\[statevector\]"):
            simulator.run("opaque input")

    def test_discovery_and_constructor_are_lazy(self):
        entry = Mock(name="metadata entry")
        entry.name = "custom"
        entry.dist = None
        with patch("qiskit_aer_york._entries", return_value=(entry,)):
            self.assertEqual(plugins()[0].name, "custom")
            simulator = Simulator("custom", shots=16)
            simulator.set_options(shots=32)
            entry.load.assert_not_called()
            backend = Mock()
            plugin = Mock(api_version=1)
            plugin.create_backend.return_value = backend
            entry.load.return_value = Mock(return_value=plugin)
            self.assertIs(simulator.backend, backend)
            self.assertIs(simulator.backend, backend)
            plugin.create_backend.assert_called_once_with(shots=32)
            simulator.run("opaque input", shots=4)
            backend.run.assert_called_once_with("opaque input", shots=4)
            simulator.set_options(shots=8)
            backend.set_options.assert_called_once_with(shots=8)

    def test_conflicting_and_incompatible_plugins(self):
        entry = Mock()
        entry.name = "bad"
        with patch("qiskit_aer_york._entries", return_value=(entry, entry)):
            with self.assertRaisesRegex(PluginError, "Multiple"):
                load_plugin("bad")
        entry.load.return_value = lambda: Mock(api_version=99)
        with patch("qiskit_aer_york._entries", return_value=(entry,)):
            with self.assertRaisesRegex(PluginError, "API 1"):
                load_plugin("bad")

    def test_loading_failure_preserves_cause(self):
        entry = Mock()
        entry.name = "missing-sdk"
        entry.load.side_effect = ImportError("SDK absent")
        with patch("qiskit_aer_york._entries", return_value=(entry,)):
            with self.assertRaisesRegex(PluginError, "SDK absent") as caught:
                load_plugin("missing-sdk")
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_unknown_kind_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unknown plugin kind"):
            plugins("invalid")

    def test_toolkit_reuses_only_explicit_plugins(self):
        toolkit = Toolkit()
        self.assertEqual(toolkit.inventory(), {"engine": (), "integration": (), "driver": ()})
        with patch("qiskit_aer_york.load_plugin") as load:
            toolkit.simulator("cuda")
            load.assert_not_called()
            first = toolkit.integration("nature")
            self.assertIs(toolkit.integration("nature"), first)
            load.assert_called_once_with("nature", "integration")
            toolkit.molecule("pyscf", atom="H 0 0 0")
            first.from_driver.assert_called_once_with("pyscf", atom="H 0 0 0")


if __name__ == "__main__":
    unittest.main()
