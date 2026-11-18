#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Build and validate plugin wheels beside an already installed York CPU wheel."""

import json
import subprocess
import sys
import tempfile
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    subprocess.run([sys.executable, "-I", *map(str, args)], check=True, cwd=cwd)


def main():
    output = ROOT / "artifacts" / "plugin-wheels"
    builder = runpy.run_path(str(ROOT / "tools/york_modular_ci.py"))
    wheels, evidence = builder["build_wheels"](output)
    run("-m", "pip", "install", *wheels)
    run("-m", "pip", "check")
    with tempfile.TemporaryDirectory(prefix="york-plugin-tests-") as outside_source:
        run(
            "-m",
            "unittest",
            "discover",
            "-s",
            ROOT / "test" / "accelerators",
            "-v",
            cwd=outside_source,
        )
        run(
            "-c",
            "from qiskit_aer_york import Simulator; from qiskit import QuantumCircuit; "
            "q=QuantumCircuit(1,1); q.x(0); q.measure(0,0); "
            "assert Simulator('native-aer').run(q,shots=16).result().get_counts()=={'1':16}",
            cwd=outside_source,
        )
    (ROOT / "artifacts" / "plugin-contract.json").write_text(
        json.dumps({"passed": True, "real_gpu_execution": False, "wheels": evidence}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
