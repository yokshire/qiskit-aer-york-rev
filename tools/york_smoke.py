# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
"""Exercise an installed Aer wheel and Runtime local mode without IBM credentials."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def bell_counts(counts, shots=1024):
    """Reject wrong states, wrong shot totals, and gross sampling regressions."""
    if set(counts) != {"00", "11"} or sum(counts.values()) != shots:
        raise AssertionError(f"Incorrect Bell sampling: {counts}")
    if abs(counts["00"] / shots - 0.5) > 0.12:
        raise AssertionError(f"Unexpected Bell sampling probabilities: {counts}")


def smoke(device, evidence):
    """Fail on GPU unavailability/fallback rather than silently running on CPU."""
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
    import qiskit_aer
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

    evidence["versions"] = {
        name: importlib.metadata.version(name) for name in ("qiskit", "qiskit-ibm-runtime")
    }
    evidence["aer_module"] = qiskit_aer.__file__
    evidence["aer_version"] = qiskit_aer.__version__
    # Multiple distributions sharing qiskit_aer can overwrite each other's files.
    owners = importlib.metadata.packages_distributions().get("qiskit_aer", [])
    if len(owners) != 1:
        raise AssertionError(
            f"Expected one Aer distribution in isolated environment, found {owners}"
        )
    evidence["aer_distribution"] = owners[0]
    backend = AerSimulator(method="statevector", device=device, seed_simulator=42)
    if device not in backend.available_devices():
        raise AssertionError(f"Requested {device} unavailable: {backend.available_devices()}")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    measured = circuit.copy()
    measured.measure_all()
    for method in ("statevector", "density_matrix"):
        simulator = AerSimulator(method=method, device=device, seed_simulator=42)
        result = simulator.run(transpile(measured, simulator), shots=1024).result()
        if not result.success:
            raise AssertionError(f"{method} failed: {result}")
        if result.results[0].metadata.get("device") != device:
            raise AssertionError(f"{method} did not run on requested {device}")
        bell_counts(result.get_counts())
        evidence["tests"].append(f"{method} shots and device metadata")

    saved = circuit.copy()
    saved.save_statevector()
    result = backend.run(transpile(saved, backend)).result()
    if not result.success or not Statevector(result.get_statevector()).equiv(Statevector(circuit)):
        raise AssertionError("Saved statevector disagrees with ideal reference")
    if result.results[0].metadata.get("device") != device:
        raise AssertionError("Saved statevector used the wrong device")
    evidence["tests"].append("exact statevector")

    unitary_backend = AerSimulator(method="unitary", device=device)
    unitary_circuit = circuit.copy()
    unitary_circuit.save_unitary()
    result = unitary_backend.run(transpile(unitary_circuit, unitary_backend)).result()
    if not result.success:
        raise AssertionError("Unitary simulation failed")
    np.testing.assert_allclose(np.asarray(result.get_unitary()), Operator(circuit).data, atol=1e-7)
    if result.results[0].metadata.get("device") != device:
        raise AssertionError("Unitary simulation used the wrong device")
    evidence["tests"].append("exact unitary")

    sampler = AerSampler(
        default_shots=1024,
        seed=42,
        options={"backend_options": {"method": "statevector", "device": device}},
    )
    bell_counts(sampler.run([measured]).result()[0].data.meas.get_counts())
    evidence["tests"].append("Aer SamplerV2")
    isa_measured = transpile(measured, backend)
    bell_counts(
        RuntimeSampler(mode=backend)
        .run([isa_measured], shots=1024)
        .result()[0]
        .data.meas.get_counts()
    )
    evidence["tests"].append("Runtime SamplerV2 local mode")
    isa_circuit = transpile(circuit, backend)
    observable = SparsePauliOp("ZZ").apply_layout(isa_circuit.layout)
    expectation = (
        RuntimeEstimator(mode=backend).run([(isa_circuit, observable)]).result()[0].data.evs
    )
    np.testing.assert_allclose(expectation, 1.0, atol=0.1)
    evidence["tests"].append("Runtime EstimatorV2 local mode")

    # Runtime 0.50 introduced a separate client-side primitive API.
    from packaging.version import Version

    if Version(evidence["versions"]["qiskit-ibm-runtime"]) >= Version("0.50.0"):
        from qiskit_ibm_runtime.executor_sampler import Sampler as ClientSampler

        counts = (
            ClientSampler(mode=backend)
            .run([isa_measured], shots=1024)
            .result()[0]
            .data.meas.get_counts()
        )
        bell_counts(counts)
        evidence["tests"].append("Runtime client-side Sampler local mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        "python": platform.python_version(),
        "device": args.device,
        "tests": [],
        "status": "failed",
    }
    try:
        smoke(args.device, evidence)
        evidence["status"] = "passed"
    except Exception as exc:  # Keep failure evidence for the always-running report job.
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    sys.exit(0 if evidence["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
