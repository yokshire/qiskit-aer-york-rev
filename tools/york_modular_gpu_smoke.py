#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Strict real-GPU smoke evidence; missing hardware is a failure, never a skip."""

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, random_unitary
from qiskit_aer_york import Simulator, load_plugin
from qiskit_aer_york_qiskit import save_statevector


def main():
    if importlib.util.find_spec("qiskit_aer") is not None:
        raise RuntimeError("Run this evidence tool in an environment WITHOUT native Aer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", required=True, choices=["opencl", "cuda", "rocm"])
    parser.add_argument("--vendor", choices=["nvidia", "amd", "intel"])
    parser.add_argument("--device-id")
    parser.add_argument("--precision", default="double", choices=["double", "single"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selector = {}
    if args.vendor and args.accelerator == "opencl":
        selector["vendor"] = args.vendor
    if args.device_id is not None:
        selector["device_id"] = args.device_id
    options = {
        "engine_options": selector,
        "precision": args.precision,
    }
    backend = Simulator(args.accelerator, **options)
    tolerance = 2e-5 if args.precision == "single" else 1e-10
    checks = []
    maximum_error = 0.0
    metadata = None
    for seed in range(10):
        rng = np.random.default_rng(seed)
        circuit = QuantumCircuit(6, global_phase=0.37)
        for _ in range(30):
            first, second = rng.choice(6, 2, replace=False)
            circuit.u(*rng.uniform(-np.pi, np.pi, 3), int(first))
            circuit.cx(int(first), int(second))
        # Exercise the generic kernel with reversed, nonconsecutive target order.
        circuit.unitary(random_unitary(8, seed=seed), [5, 0, 2])
        expected = Statevector.from_instruction(circuit)
        save_statevector(circuit)
        result = backend.run(circuit).result()
        metadata = result.results[0].metadata
        if metadata["device"] != "GPU" or metadata["gate_count"] != 61:
            raise AssertionError(f"GPU execution evidence missing: {metadata}")
        if args.vendor and metadata["vendor"] != args.vendor:
            raise AssertionError("Executed on an unexpected GPU vendor")
        error = float(np.max(np.abs(result.get_statevector().data - expected.data)))
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            raise AssertionError(f"GPU/reference error {error} exceeds {tolerance}")
        checks.append(f"random-statevector-{seed}")

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    measured = bell.copy()
    measured.measure_all()
    first = backend.run(measured, shots=1024, memory=True, seed_simulator=91).result()
    second = backend.run(measured, shots=1024, memory=True, seed_simulator=91).result()
    assert set(first.get_counts()) == {"00", "11"}
    assert first.get_memory() == second.get_memory()
    checks += ["bell-sampling", "deterministic-seed"]
    integration = load_plugin("qiskit", "integration")
    sampler = integration.sampler(backend)
    sampled = sampler.run([measured], shots=128).result()
    assert set(sampled[0].data.meas.get_counts()) == {"00", "11"}
    checks.append("qiskit-backend-sampler-v2")
    estimated = integration.estimator(backend).run([(bell, "ZZ")]).result()
    assert abs(float(estimated[0].data.evs) - 1) < tolerance
    checks.append("qiskit-backend-estimator-v2")
    param = Parameter("theta")
    parameterized = QuantumCircuit(1, 1)
    parameterized.ry(2 * param, 0)
    parameterized.measure(0, 0)
    batched = backend.run(
        parameterized, parameter_binds=[{param: [0, np.pi / 2]}], shots=32
    ).result()
    assert batched.get_counts(0) == {"0": 32}
    assert batched.get_counts(1) == {"1": 32}
    checks.append("parameter-batch")
    # A real CPU execution in the same process demonstrates coexistence.
    cpu_result = Simulator("statevector").run(measured, shots=32).result()
    assert cpu_result.results[0].metadata["device"] == "CPU"
    checks.append("cpu-plugin-coexistence")
    evidence = {
        "passed": True,
        "native_aer_installed": False,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "accelerator": args.accelerator,
        "precision": args.precision,
        "checks": checks,
        "max_statevector_error": maximum_error,
        "device": metadata,
        "available_devices": backend.backend.state_provider.devices(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
