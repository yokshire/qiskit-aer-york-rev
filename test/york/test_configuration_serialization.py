# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regression for upstream Qiskit Aer PR #2236.
"""All existing Aer backends must serialize their open_pulse=False capability."""
import unittest

from qiskit_aer import AerSimulator, QasmSimulator, StatevectorSimulator, UnitarySimulator


class TestConfigurationSerialization(unittest.TestCase):
    def test_open_pulse_and_round_trip(self):
        for backend_class in (AerSimulator, QasmSimulator, StatevectorSimulator, UnitarySimulator):
            with self.subTest(backend=backend_class.__name__):
                config = backend_class().configuration()
                self.assertIs(config.open_pulse, False)
                serialized = config.to_dict()
                self.assertIs(serialized["open_pulse"], False)
                restored = type(config).from_dict(serialized)
                self.assertEqual(restored.to_dict(), serialized)


if __name__ == "__main__":
    unittest.main()
