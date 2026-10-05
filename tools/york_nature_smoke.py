#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Strict real PySCF calculation followed by local Nature mapping and York simulation."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import tempfile

from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer_york import Simulator, load_plugin
from qiskit_aer_york_qiskit import save_statevector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=("native", "wsl"))
    parser.add_argument("--python", help="Explicit worker Python executable")
    parser.add_argument("--distribution", help="Explicit WSL distribution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if importlib.util.find_spec("qiskit_aer") is not None:
        raise RuntimeError("This check requires native Aer to be absent")
    if args.runtime == "wsl" and importlib.util.find_spec("pyscf") is not None:
        raise RuntimeError("WSL evidence requires PySCF to be absent from the Windows controller")
    driver = load_plugin("pyscf", "driver")
    result = driver.run(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto3g",
        runtime=args.runtime,
        python=args.python,
        distribution=args.distribution,
    )
    nature = load_plugin("nature", "integration")
    with tempfile.TemporaryDirectory(prefix="york-h2-") as temporary:
        path = Path(temporary) / "h2.fcidump"
        path.write_text(result["fcidump"], encoding="utf-8")
        problem = nature.from_fcidump(path)
    if problem.num_particles != (1, 1) or problem.num_spatial_orbitals != 2:
        raise AssertionError("Incorrect H2 electron/orbital data")
    mapper = JordanWignerMapper()
    operator = nature.qubit_operator(problem, mapper)
    circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    expected = Statevector.from_instruction(circuit)
    circuit = circuit.decompose()
    save_statevector(circuit)
    execution = Simulator("statevector").run(circuit).result()
    vector = execution.get_statevector()
    nuclear = float(problem.hamiltonian.nuclear_repulsion_energy)
    energy = float(vector.expectation_value(operator).real) + nuclear
    reference = float(result["reference_energy"])
    if abs(energy - reference) > 1e-9 or abs(reference - (-1.116998996754004)) > 1e-8:
        raise AssertionError(f"H2 energy mismatch: York={energy}, PySCF={reference}")
    if not vector.equiv(expected):
        raise AssertionError("Hartree-Fock circuit/reference state differs")
    evidence = {
        "passed": True,
        "controller_system": platform.system(),
        "runtime": args.runtime,
        "pyscf_version": result["pyscf_version"],
        "controller_has_pyscf": importlib.util.find_spec("pyscf") is not None,
        "native_aer_installed": False,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "reference_energy": reference,
        "simulated_energy": energy,
        "error": abs(energy - reference),
        "execution": execution.results[0].metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
