# Licensed under the Apache License, Version 2.0.
"""Qiskit job/result adapter for accelerator API 1 statevector engines.

Unitary evolution runs on the selected device. Terminal shot sampling and saved
observable evaluation run on the host. Unsupported semantics fail explicitly.
"""

import copy
import numbers
import platform
import time
import uuid

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit.result import Result

from qiskit_aer.aererror import AerError
from qiskit_aer.jobs import AerJob
from . import load_accelerator

SUPPORTED_OPTIONS = {
    "accelerator",
    "accelerator_options",
    "method",
    "device",
    "precision",
    "shots",
    "memory",
    "seed_simulator",
    "executor",
    "noise_model",
    "max_memory_mb",
}


def submit_job(backend, circuits, parameter_binds, options):
    """Validate options and submit a snapshot of the inputs as an AerJob."""
    unsupported = set(options) - SUPPORTED_OPTIONS
    if unsupported:
        raise AerError(f"GPU plugins do not support options: {sorted(unsupported)}")
    if options.get("method", "automatic") not in ("automatic", "statevector"):
        raise AerError("GPU plugin API 1 supports method='statevector' or 'automatic'")
    if options.get("device", "GPU") != "GPU":
        raise AerError("An accelerator plugin requires device='GPU'")
    if options.get("precision", "double") not in ("double", "single"):
        raise AerError("precision must be 'double' or 'single'")
    noise = options.get("noise_model")
    if noise is not None and not noise.is_ideal():
        raise AerError("GPU plugin API 1 does not implement noise models; use the CPU simulator")
    shots = options.get("shots", 1024)
    if not isinstance(shots, numbers.Integral) or isinstance(shots, bool) or shots < 1:
        raise AerError("shots must be a positive integer")
    circuits = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
    if not circuits or not all(isinstance(circuit, QuantumCircuit) for circuit in circuits):
        raise AerError("GPU plugins require a QuantumCircuit or a nonempty list of circuits")
    # Load before creating a job so missing SDKs/plugins are actionable immediately.
    provider = load_accelerator(options["accelerator"])
    job_options = {key: value for key, value in options.items() if key != "executor"}
    job_options = copy.deepcopy(job_options)

    def execute(job_circuits, binds, opts, job_id):
        return _execute(backend, provider, job_circuits, binds, opts, job_id)

    job = AerJob(
        backend,
        str(uuid.uuid4()),
        execute,
        circuits=[circuit.copy() for circuit in circuits],
        parameter_binds=copy.deepcopy(parameter_binds),
        run_options=job_options,
        executor=options.get("executor"),
    )
    job.submit()
    return job


def _bind_circuits(circuits, binds):
    if binds is None:
        binds = [{} for _ in circuits]
    if len(binds) != len(circuits):
        raise AerError("parameter_binds must contain one dictionary per circuit")
    for circuit, table in zip(circuits, binds):
        if not table:
            if circuit.parameters:
                raise AerError("Circuit contains unbound parameters")
            yield circuit
            continue
        if set(table) != set(circuit.parameters):
            raise AerError("parameter_binds must bind every circuit parameter exactly once")
        lengths = {len(values) for values in table.values()}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise AerError("Parameter value lists must have the same positive length")
        for index in range(next(iter(lengths))):
            yield circuit.assign_parameters(
                {param: values[index] for param, values in table.items()}
            )


def _plan(circuit):
    """Validate the whole circuit before any GPU allocation or execution."""
    plan = []
    measuring = False
    labels = set()
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
        clbits = tuple(circuit.find_bit(bit).index for bit in instruction.clbits)
        if getattr(operation, "condition", None) is not None:
            raise AerError("GPU plugins do not support classical conditions")
        if operation.name == "barrier":
            continue
        if operation.name == "measure":
            measuring = True
            plan.append(("measure", qubits, clbits))
            continue
        if measuring:
            raise AerError("GPU plugins require all measurements to be terminal")
        if operation.name in ("save_statevector", "save_expval"):
            allowed_subtypes = ("single",) if operation.name == "save_statevector" else ("average",)
            if getattr(operation, "_subtype", None) not in allowed_subtypes:
                raise AerError("GPU plugins do not support conditional or per-shot saves")
            if operation.label in labels:
                raise AerError(f"Duplicate save label: {operation.label}")
            labels.add(operation.label)
            if operation.name == "save_statevector" and qubits != tuple(range(circuit.num_qubits)):
                raise AerError("save_statevector must cover all qubits in circuit order")
            plan.append((operation.name, qubits, operation))
            continue
        if not isinstance(operation, Gate) or not 1 <= len(qubits) <= 4:
            raise AerError(
                f"GPU plugins do not support instruction {operation.name!r}; "
                "transpile unitary gates to ['u', 'cx']. Reset, noise and control flow need CPU Aer."
            )
        try:
            matrix = np.asarray(Operator(operation).data, dtype=np.complex128)
        except Exception as exc:
            raise AerError(f"Cannot convert gate {operation.name!r} to a unitary matrix") from exc
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix.conj().T @ matrix, np.eye(1 << len(qubits)), atol=1e-10, rtol=1e-10
        ):
            raise AerError(f"Instruction {operation.name!r} is not unitary")
        plan.append(("gate", qubits, matrix))
    return plan


def _execute(backend, provider, circuits, binds, options, job_id):
    started = time.perf_counter()
    bound = list(_bind_circuits(circuits, binds))
    plans = [_plan(circuit) for circuit in bound]
    rng = np.random.default_rng(options.get("seed_simulator"))
    shots = int(options.get("shots", 1024))
    precision = options.get("precision", "double")
    results = []
    for circuit, plan in zip(bound, plans):
        required = 2 * (1 << circuit.num_qubits) * (16 if precision == "double" else 8)
        max_memory = options.get("max_memory_mb")
        if max_memory is not None and required > max_memory * 1024**2:
            raise AerError("GPU statevector exceeds max_memory_mb")
        state = provider.create_state(
            circuit.num_qubits, precision, options.get("accelerator_options") or {}
        )
        data = {}
        measurements = []
        gates = 0
        phase = np.exp(1j * float(circuit.global_phase))
        for action, qubits, payload in plan:
            if action == "gate":
                state.apply(payload, qubits)
                gates += 1
            elif action == "measure":
                measurements.append((qubits[0], payload[0]))
            else:
                vector = Statevector(state.to_host() * phase)
                if action == "save_statevector":
                    data[payload.label] = vector
                else:
                    data[payload.label] = float(
                        sum(
                            coefficients[0]
                            * vector.expectation_value(Pauli(pauli), qargs=list(qubits)).real
                            for pauli, coefficients in payload.params
                        )
                    )
        if measurements:
            vector = state.to_host()
            probabilities = np.abs(vector.astype(np.complex128)) ** 2
            norm = probabilities.sum()
            tolerance = 1e-5 if precision == "single" else 1e-10
            if not np.isfinite(norm) or abs(norm - 1) > tolerance:
                raise AerError(f"GPU statevector lost normalization: {norm}")
            samples = rng.choice(len(vector), size=shots, p=probabilities / norm)
            memory = []
            counts = {}
            for sample in samples:
                value = 0
                for qubit, clbit in measurements:
                    value = (value & ~(1 << clbit)) | (((int(sample) >> qubit) & 1) << clbit)
                key = hex(value)
                memory.append(key)
                counts[key] = counts.get(key, 0) + 1
            data["counts"] = counts
            if options.get("memory"):
                data["memory"] = memory
        metadata = dict(state.metadata)
        if metadata.get("device") != "GPU":
            raise AerError("Accelerator plugin did not report GPU execution")
        metadata.update(
            {
                "accelerator": options["accelerator"],
                "accelerator_api": 1,
                "method": "statevector",
                "precision": precision,
                "gpu_gate_count": gates,
                "sampling_device": "CPU",
                "observable_device": "CPU",
                "platform_system": platform.system(),
            }
        )
        results.append(
            {
                "success": True,
                "status": "DONE",
                "shots": shots,
                "data": data,
                "header": {
                    "name": circuit.name,
                    "metadata": circuit.metadata,
                    "memory_slots": circuit.num_clbits,
                    "creg_sizes": [[register.name, register.size] for register in circuit.cregs],
                },
                "metadata": metadata,
            }
        )
    return Result.from_dict(
        {
            "backend_name": backend.name,
            "backend_version": backend.backend_version,
            "job_id": job_id,
            "qobj_id": None,
            "success": True,
            "results": results,
            "time_taken": time.perf_counter() - started,
            "metadata": {
                "accelerator": options["accelerator"],
                "device": "GPU",
                "platform_system": platform.system(),
            },
        }
    )
