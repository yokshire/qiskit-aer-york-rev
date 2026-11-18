#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Reproducible packed-integral transport benchmark; synthetic data, no SCF timing."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time
import tracemalloc

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.operators.symmetric_two_body import S8Integrals
from qiskit_aer_york import Toolkit
from qiskit_aer_york_data import IntegralBundle
from qiskit_aer_york_qiskit import save_statevector


def median_seconds(function, repeats=3):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        times.append(time.perf_counter() - started)
    return statistics.median(times), value


def tracked_peak(function):
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    toolkit = Toolkit()
    n = 24
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(n, n))
    h1 = (raw + raw.T) / 2
    pairs = n * (n + 1) // 2
    eri = rng.normal(size=pairs * (pairs + 1) // 2)
    bundle = IntegralBundle(h1, eri, num_alpha=4, num_beta=4, nuclear_energy=0.4)
    binary = bundle.to_bytes()
    with tempfile.TemporaryDirectory(prefix="york-data-benchmark-") as temporary:
        path = Path(temporary) / "integrals.fcidump"
        FCIDump(num_electrons=8, hij=h1, hijkl=S8Integrals(eri), constant_energy=0.4).to_file(path)
        text = path.read_text(encoding="utf-8")

        def legacy():
            path.write_text(text, encoding="utf-8")
            return FCIDump.from_file(path)

        def direct():
            return IntegralBundle.from_bytes(binary)

        # Warm imports and file caches; timings include validation/checksum on the binary route.
        legacy()
        direct()
        text_seconds, parsed = median_seconds(legacy)
        binary_seconds, decoded = median_seconds(direct)
        text_peak, binary_peak = tracked_peak(legacy), tracked_peak(direct)
    np.testing.assert_allclose(parsed.hij, decoded.h1, atol=1e-14)
    np.testing.assert_allclose(parsed.hijkl.array, decoded.eri_s8, atol=1e-14)
    shared = np.shares_memory(decoded.eri_s8, np.frombuffer(binary, dtype=np.uint8))
    if not shared or len(binary) >= len(text.encode("utf-8")):
        raise AssertionError("Expected shared packed arrays and a smaller dense synthetic payload")

    circuit = QuantumCircuit(2)
    for _ in range(50):
        circuit.h(0)
        circuit.cx(0, 1)
    expected = Statevector.from_instruction(circuit)
    save_statevector(circuit, "first")
    save_statevector(circuit, "second")
    result = toolkit.simulator().run(circuit).result()
    np.testing.assert_allclose(result.data()["first"], expected.data, atol=1e-12)
    execution = result.results[0].metadata
    if execution["matrix_validations"] != 2 or execution["host_state_transfers"] != 1:
        raise AssertionError("Repeated gate/snapshot reuse was not exercised")
    evidence = {
        "passed": True,
        "platform_system": platform.system(),
        "workload": "synthetic-dense-s8",
        "num_orbitals": n,
        "packed_array_bytes": bundle.nbytes,
        "dense_eri_bytes": n**4 * 8,
        "binary_bytes": len(binary),
        "fcidump_bytes": len(text.encode("utf-8")),
        "payload_ratio": len(binary) / len(text.encode("utf-8")),
        "array_views_share_payload": bool(shared),
        "median_seconds": {
            "binary_decode_validate": binary_seconds,
            "fcidump_text_write_parse": text_seconds,
        },
        "python_tracked_peak_bytes": {
            "binary_decode_validate": binary_peak,
            "fcidump_text_write_parse": text_peak,
        },
        "execution": execution,
        "timing_scope": "Warm local decode; excludes SCF, IPC, file generation and operator mapping. "
        "Tracemalloc is not total process/GPU memory. No speed threshold is asserted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
