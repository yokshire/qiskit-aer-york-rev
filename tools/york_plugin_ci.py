#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Build and validate plugin wheels beside an already installed York CPU wheel."""

import json
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    subprocess.run([sys.executable, "-I", *map(str, args)], check=True, cwd=cwd)


def main():
    output = ROOT / "artifacts" / "plugin-wheels"
    output.mkdir(parents=True, exist_ok=True)
    run(
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        output,
        ROOT / "accelerators" / "opencl",
        ROOT / "accelerators" / "cupy",
    )
    wheels = sorted(output.glob("*.whl"))
    if len(wheels) != 2:
        raise RuntimeError("Expected exactly two clean plugin wheels")
    evidence = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            if any(name.startswith("qiskit_aer/") for name in archive.namelist()):
                raise RuntimeError("Plugin wheel overwrites CPU package files")
            metadata_path = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
            requirements = metadata.get_all("Requires-Dist") or []
            if not any(req.startswith("qiskit-aer-york-rev") for req in requirements):
                raise RuntimeError("Plugin does not declare its CPU base dependency")
            evidence.append(
                {"name": metadata["Name"], "wheel": wheel.name, "requires_dist": requirements}
            )
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
    (ROOT / "artifacts" / "plugin-contract.json").write_text(
        json.dumps({"passed": True, "real_gpu_execution": False, "wheels": evidence}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
