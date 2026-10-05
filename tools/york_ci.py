# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Release inventory, wheel metadata checks, and deterministic York CI reports."""

import argparse
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PACKAGES = ("qiskit", "qiskit-ibm-runtime", "qiskit-aer", "qiskit-aer-gpu")
CUDA_REQUIREMENTS = {
    "nvidia-cuda-runtime-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "cuquantum-cu12",
}


def latest_stable(releases):
    """Select an actual, non-yanked stable release, not an older pip fallback."""
    candidates = []
    for text, files in releases.items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if not version.is_prerelease and not version.is_devrelease:
            if any(not entry.get("yanked", False) for entry in files):
                candidates.append(version)
    if not candidates:
        raise ValueError("No non-yanked stable release found")
    return str(max(candidates))


def fetch_json(url):
    """Use public PyPI metadata only; no credentials are read or transmitted."""
    request = urllib.request.Request(url, headers={"User-Agent": "qiskit-aer-york-rev-ci"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def inventory():
    """Resolve exact stable versions before creating the compatibility matrix."""
    packages = {}
    for name in PACKAGES:
        index = fetch_json(f"https://pypi.org/pypi/{name}/json")
        version = latest_stable(index["releases"])
        release = fetch_json(f"https://pypi.org/pypi/{name}/{version}/json")
        packages[name] = {
            "version": version,
            "requires_python": release["info"].get("requires_python"),
        }
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "upstream_gpu_version_matches_cpu": (
            packages["qiskit-aer"]["version"] == packages["qiskit-aer-gpu"]["version"]
        ),
    }


def check_wheel(wheel, gpu):
    """Inspect this distribution, not a wheel downloaded from upstream PyPI."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Expected exactly one distribution METADATA file")
        metadata = Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))
        source_version = archive.read("qiskit_aer/VERSION.txt").decode("utf-8").strip()
    expected_name = "qiskit-aer-york-rev-gpu" if gpu else "qiskit-aer-york-rev"
    if canonicalize_name(metadata["Name"]) != expected_name:
        raise ValueError(f"Wrong distribution identity: {metadata['Name']}")
    if Version(metadata["Version"]) != Version(source_version):
        raise ValueError("Wheel and import-package version metadata disagree")
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    names = {canonicalize_name(requirement.name) for requirement in requirements}
    if gpu and not CUDA_REQUIREMENTS.issubset(names):
        raise ValueError(f"Missing CUDA dependencies: {sorted(CUDA_REQUIREMENTS - names)}")
    accelerator_names = {
        name
        for name in names
        if name.startswith(("nvidia-", "cuquantum", "custatevec", "cutensornet", "rocm-"))
    }
    if not gpu and accelerator_names:
        raise ValueError(
            f"CPU wheel unexpectedly requires accelerator packages: {sorted(accelerator_names)}"
        )
    if "qiskit" not in names or "qiskit-aer" in names or "qiskit-aer-gpu" in names:
        raise ValueError("Incorrect Qiskit/Aer distribution dependencies")
    return {"name": metadata["Name"], "version": metadata["Version"], "cuda_dependencies": gpu}


def write_json(path, value):
    """Write generated evidence without modifying source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_report(directory, needs):
    """Report failed/skipped jobs even when no smoke JSON could be produced."""
    inventory_file = directory / "inventory.json"
    versions = (
        json.loads(inventory_file.read_text(encoding="utf-8")) if inventory_file.exists() else {}
    )
    checks = {name: detail.get("result", "unknown") for name, detail in needs.items()}
    required_jobs = {
        "inventory",
        "static-review",
        "modular-install",
        "nature-driver",
        "cpu-compatibility",
        "cpu-portability",
        "cuda-wheel",
    }
    passed = required_jobs.issubset(checks) and all(value == "success" for value in checks.values())
    smoke_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("smoke-*.json"))
    ]
    cpu_results = [result for result in smoke_results if result.get("device") == "CPU"]
    gpu_metadata_files = list(directory.rglob("gpu-wheel.json"))
    cpu_systems = {result.get("platform", {}).get("system") for result in cpu_results}
    if (
        not packages_complete(versions)
        or len(cpu_results) < 8
        or not {"Linux", "Windows", "Darwin"}.issubset(cpu_systems)
        or len(gpu_metadata_files) != 1
    ):
        passed = False
    modular_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("modular-*.json"))
    ]
    if {result.get("label") for result in modular_results} != {
        "ubuntu-24.04",
        "windows-2022",
        "macos-14",
    } or any(
        result.get("passed") is not True
        or result.get("core_dependencies") != []
        or result.get("native_aer_installed") is not False
        for result in modular_results
    ):
        passed = False
    nature_files = list(directory.rglob("nature-native.json"))
    nature_results = [json.loads(path.read_text(encoding="utf-8")) for path in nature_files]
    if (
        len(nature_results) != 1
        or nature_results[0].get("passed") is not True
        or nature_results[0].get("runtime") != "native"
        or nature_results[0].get("native_aer_installed") is not False
    ):
        passed = False
    for result in cpu_results:
        expected = versions.get("packages", {})
        for name in ("qiskit", "qiskit-ibm-runtime"):
            if result.get("versions", {}).get(name) != expected.get(name, {}).get("version"):
                passed = False
    if any(result.get("status") != "passed" for result in smoke_results):
        passed = False
    packages = versions.get("packages", {})
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "packages": packages,
                "checks": checks,
                "modular": modular_results,
                "nature": nature_results,
                "passed": passed,
                "source_commit": os.environ.get("GITHUB_SHA", "local"),
                "smoke": [
                    {
                        key: result.get(key)
                        for key in (
                            "python",
                            "platform",
                            "device",
                            "status",
                            "versions",
                            "tests",
                            "error",
                        )
                    }
                    for result in smoke_results
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    heading = (
        "PASS — dependency-free core, optional engines, CPU/OS compatibility and CUDA packaging"
        if passed
        else "FAIL — review required"
    )
    lines = [
        "# York compatibility review",
        "",
        heading,
        "",
        f"Tested source commit: `{os.environ.get('GITHUB_SHA', 'local')}`.",
        "",
        "**GPU execution: NOT TESTED by hosted CI.** A successful CUDA build/ELF check is not",
        "proof that Aer can execute on a physical GPU. These are targeted smoke checks,",
        "not an exhaustive Python/OS/Qiskit compatibility certification.",
        "",
        "## Latest stable release inventory",
        "",
        "| Distribution | Exact stable version | Requires Python |",
        "| --- | --- | --- |",
    ]
    for name, entry in packages.items():
        lines.append(
            f"| {name} | {entry['version']} | {entry.get('requires_python') or 'unspecified'} |"
        )
    if versions.get("upstream_gpu_version_matches_cpu") is False:
        lines += [
            "",
            "Upstream CPU and GPU PyPI versions differ; this is inventory information, not a York CI failure.",
        ]
    lines += ["", "## Job outcomes", "", "| Job | Outcome |", "| --- | --- |"]
    lines += [f"| {name} | {result} |" for name, result in checks.items()]
    lines += ["", "## Installed-wheel functional evidence", ""]
    for result in smoke_results:
        lines.append(
            f"- {result.get('platform', {}).get('system', 'unknown OS')} "
            f"Python {result.get('python', 'unknown')}: {result.get('status', 'unknown')}; "
            f"{len(result.get('tests', []))} completed checks; device {result.get('device', 'unknown')}."
        )
        if result.get("error"):
            lines.append(f"  Failure: {result['error']}")
    lines += ["", "## Lightweight install evidence", ""]
    for result in modular_results:
        lines.append(
            f"- {result.get('label')}: core has zero runtime dependencies; CPU and Nature run without native Aer/PySCF; passed={result.get('passed')}."
        )
    for result in nature_results:
        lines.append(
            f"- Native PySCF {result.get('pyscf_version')} H2 calculation and York CPU execution: passed={result.get('passed')}; energy error={result.get('error')}."
        )
    if not smoke_results:
        lines.append("No functional evidence was produced; inspect failed or skipped jobs.")
    lines += [
        "",
        "Static review means syntax/undefined-name checks, formatting of York tooling,",
        "and unit tests for the CI tools. It is not an AI review or a human approval.",
        "No changes are automatically committed, merged, released, or sent upstream.",
    ]
    return {"passed": passed, "fingerprint": fingerprint, "body": "\n".join(lines) + "\n"}


def packages_complete(versions):
    """A missing inventory must not turn a partial report into a pass."""
    return all(versions.get("packages", {}).get(name, {}).get("version") for name in PACKAGES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subcommands.add_parser("inventory")
    inventory_parser.add_argument("--output", type=Path, required=True)
    wheel_parser = subcommands.add_parser("check-wheel")
    wheel_parser.add_argument("wheel", type=Path)
    wheel_parser.add_argument("--gpu", action="store_true")
    report_parser = subcommands.add_parser("report")
    report_parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory()
        write_json(args.output, result)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as stream:
                stream.write(f"qiskit={result['packages']['qiskit']['version']}\n")
                stream.write(f"runtime={result['packages']['qiskit-ibm-runtime']['version']}\n")
        print(json.dumps(result, indent=2))
    elif args.command == "check-wheel":
        print(json.dumps(check_wheel(args.wheel, args.gpu), indent=2))
    else:
        result = make_report(args.artifacts, json.loads(os.environ.get("NEEDS_JSON", "{}")))
        write_json(args.artifacts / "review.json", result)
        (args.artifacts / "review.md").write_text(result["body"], encoding="utf-8")
        print(result["body"])
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write(result["body"])


if __name__ == "__main__":
    main()
