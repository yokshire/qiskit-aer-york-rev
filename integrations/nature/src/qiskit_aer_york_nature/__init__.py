# Licensed under the Apache License, Version 2.0.
"""Portable molecular data and mapping; chemistry SDKs belong to driver plugins."""

from pathlib import Path
import tempfile

from qiskit_aer_york import PluginError, load_plugin


class NatureIntegration:
    api_version = 1

    def from_fcidump(self, path):
        """Read existing molecular integrals without invoking a chemistry driver."""
        from qiskit_nature.second_q.formats import fcidump_to_problem
        from qiskit_nature.second_q.formats.fcidump import FCIDump

        return fcidump_to_problem(FCIDump.from_file(Path(path)))

    def from_driver(self, name, **options):
        """Run one explicitly selected driver, then load its portable result locally."""
        driver = load_plugin(name, "driver")
        result = driver.run(**options)
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != 1
            or result.get("format") != "fcidump"
        ):
            raise PluginError(f"Driver {name!r} returned an unsupported integral format")
        text = result.get("fcidump")
        if not isinstance(text, str) or not text.strip():
            raise PluginError(f"Driver {name!r} returned no FCIDump data")
        # Close the file before the Nature parser opens it, including on Windows.
        with tempfile.TemporaryDirectory(prefix="york-nature-") as temporary:
            path = Path(temporary) / "integrals.fcidump"
            path.write_text(text, encoding="utf-8")
            problem = self.from_fcidump(path)
        problem.reference_energy = result.get("reference_energy")
        return problem

    def qubit_operator(self, problem, mapper=None):
        """Map the electronic Hamiltonian; nuclear energy stays in the problem constant."""
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        mapper = JordanWignerMapper() if mapper is None else mapper
        return mapper.map(problem.hamiltonian.second_q_op())

    def estimator(self, simulator, **options):
        return load_plugin("qiskit", "integration").estimator(simulator, **options)
