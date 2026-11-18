#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Prove core-only installation and actual CPU execution without native Aer."""

import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "core",
    "data",
    "integrations/qiskit",
    "integrations/nature",
    "drivers/pyscf",
    "engines/statevector",
    "engines/native",
    "accelerators/opencl",
    "accelerators/cupy",
)


def run(python, *args, cwd=None):
    subprocess.run([str(python), "-I", *map(str, args)], check=True, cwd=cwd)


def build_wheels(output):
    output.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        output,
        *(ROOT / project for project in PROJECTS),
    )
    wheels = sorted(output.glob("*.whl"))
    if len(wheels) != len(PROJECTS):
        raise RuntimeError(f"Expected {len(PROJECTS)} clean independent wheels")
    evidence = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if any(name.startswith("qiskit_aer/") for name in names):
                raise RuntimeError("Plugin overwrites native Aer namespace")
            metadata = BytesParser().parsebytes(
                archive.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
            )
            requirements = metadata.get_all("Requires-Dist") or []
            distribution = metadata["Name"]
            if distribution == "qiskit-aer-york-core":
                if any("extra ==" not in requirement for requirement in requirements):
                    raise RuntimeError("Core has a mandatory runtime dependency")
                if any(name.endswith((".dll", ".so", ".pyd", ".dylib")) for name in names):
                    raise RuntimeError("Core contains native binaries")
            elif distribution != "qiskit-aer-york-native" and any(
                "qiskit-aer-york-rev" in requirement for requirement in requirements
            ):
                raise RuntimeError("Lightweight plugin depends on the full native bundle")
            if distribution == "qiskit-aer-york-nature" and any(
                "pyscf" in requirement.lower() or "psi4" in requirement.lower()
                for requirement in requirements
            ):
                raise RuntimeError("Nature integration depends on a chemistry SDK")
            if distribution == "qiskit-aer-york-pyscf" and any(
                "pyscf" in requirement.lower() and "extra ==" not in requirement
                for requirement in requirements
            ):
                raise RuntimeError("Driver installs PySCF without an explicit extra")
            evidence.append(
                {
                    "name": distribution,
                    "wheel": wheel.name,
                    "bytes": wheel.stat().st_size,
                    "requires_dist": requirements,
                }
            )
    return wheels, evidence


def env_python(directory):
    venv.EnvBuilder(with_pip=True).create(directory)
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=sys.platform)
    parser.add_argument("--qiskit", help="Pin the inventory's current stable Qiskit version")
    args = parser.parse_args()
    wheels, evidence = build_wheels(ROOT / "artifacts/modular-wheels")
    core = next(wheel for wheel in wheels if wheel.name.startswith("qiskit_aer_york_core-"))
    nature = [
        wheel
        for wheel in wheels
        if wheel.name.startswith(("qiskit_aer_york_nature-", "qiskit_aer_york_pyscf-"))
    ]
    light = [
        wheel
        for wheel in wheels
        if not wheel.name.startswith("qiskit_aer_york_native-") and wheel not in nature
    ]
    with tempfile.TemporaryDirectory(prefix="york-modular-") as temporary:
        outside = Path(temporary)
        bare = env_python(outside / "core-only")
        run(bare, "-m", "pip", "install", "--no-deps", core, cwd=outside)
        run(bare, "-m", "unittest", "discover", "-s", ROOT / "test/modular/core", "-v", cwd=outside)
        functional = env_python(outside / "light-engines")
        requirements = [f"qiskit=={args.qiskit}"] if args.qiskit else []
        run(functional, "-m", "pip", "install", *light, *requirements, cwd=outside)
        run(functional, "-m", "pip", "check", cwd=outside)
        run(
            functional,
            "-m",
            "unittest",
            "discover",
            "-s",
            ROOT / "test/modular/data",
            "-v",
            cwd=outside,
        )
        run(
            functional,
            "-m",
            "unittest",
            "discover",
            "-s",
            ROOT / "test/modular/engines",
            "-v",
            cwd=outside,
        )
        run(functional, "-m", "pip", "install", *nature, cwd=outside)
        run(functional, "-m", "pip", "check", cwd=outside)
        run(
            functional,
            "-m",
            "unittest",
            "discover",
            "-s",
            ROOT / "test/modular/nature",
            "-v",
            cwd=outside,
        )
        run(
            functional,
            ROOT / "tools/york_data_benchmark.py",
            "--output",
            ROOT / f"artifacts/data-{args.label}.json",
            cwd=outside,
        )
    report = {
        "nature_without_pyscf": True,
        "array_transport": True,
        "passed": True,
        "core_dependencies": [],
        "native_aer_installed": False,
        "real_gpu_execution": False,
        "label": args.label,
        "wheels": evidence,
    }
    (ROOT / f"artifacts/modular-{args.label}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
