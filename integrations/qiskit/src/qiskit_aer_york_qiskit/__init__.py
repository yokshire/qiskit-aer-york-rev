# Licensed under the Apache License, Version 2.0.
"""Qiskit integration, imported only after the user selects a Qiskit plugin."""

from concurrent.futures import ThreadPoolExecutor

from qiskit.circuit import Instruction, Measure, Parameter
from qiskit.circuit.library import CXGate, UGate
from qiskit.providers import BackendV2, JobV1, JobStatus, Options
from qiskit.transpiler import Target
from qiskit_aer_york import SimulationError

_EXECUTOR = ThreadPoolExecutor()


class SimulationJob(JobV1):
    def __init__(
        self, backend, job_id, fn, *, circuits, parameter_binds, run_options, executor=None
    ):
        super().__init__(backend, job_id)
        self._executor = executor or _EXECUTOR
        self._fn = fn
        self._args = (circuits, parameter_binds, run_options, job_id)
        self._future = None

    def submit(self):
        if self._future is not None:
            raise SimulationError("Job has already been submitted")
        self._future = self._executor.submit(self._fn, *self._args)

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    def status(self):
        if self._future is None:
            return JobStatus.INITIALIZING
        if self._future.cancelled():
            return JobStatus.CANCELLED
        if self._future.done():
            return JobStatus.ERROR if self._future.exception() else JobStatus.DONE
        return JobStatus.RUNNING if self._future.running() else JobStatus.QUEUED


class StatevectorBackend(BackendV2):
    """Ideal statevector BackendV2; the injected provider performs gate evolution."""

    def __init__(self, provider, *, engine, device, **options):
        super().__init__(name=f"york_{engine}", backend_version="0.1.0")
        self.state_provider = provider
        self.engine = engine
        self.expected_device = device
        self._target = Target(num_qubits=32)
        self._target.add_instruction(
            UGate(*(Parameter(name) for name in ("theta", "phi", "lambda"))), name="u"
        )
        self._target.add_instruction(CXGate(), name="cx")
        self._target.add_instruction(Measure(), name="measure")
        self.set_options(**options)

    @classmethod
    def _default_options(cls):
        return Options(
            method="statevector",
            precision="double",
            shots=1024,
            memory=False,
            seed_simulator=None,
            executor=None,
            noise_model=None,
            max_memory_mb=1024,
            engine_options=None,
        )

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    def run(self, circuits, parameter_binds=None, **options):
        from .statevector import submit_job

        merged = dict(self.options)
        merged.update(options)
        return submit_job(self, circuits, parameter_binds, merged)


class QiskitIntegration:
    api_version = 1

    def as_backend(self, simulator):
        backend = simulator.backend
        if not isinstance(backend, BackendV2):
            raise SimulationError("The selected engine does not provide a Qiskit BackendV2")
        return backend

    def sampler(self, simulator, **options):
        from qiskit.primitives import BackendSamplerV2

        return BackendSamplerV2(backend=self.as_backend(simulator), options=options)

    def estimator(self, simulator, **options):
        from qiskit.primitives import BackendEstimatorV2

        return BackendEstimatorV2(backend=self.as_backend(simulator), options=options)


def save_statevector(circuit, label="statevector"):
    """Append a save directive without importing or monkey-patching Aer."""
    instruction = Instruction("save_statevector", circuit.num_qubits, 0, [], label=label)
    instruction._subtype = "single"
    instruction._directive = True
    circuit.append(instruction, circuit.qubits)
    return circuit
