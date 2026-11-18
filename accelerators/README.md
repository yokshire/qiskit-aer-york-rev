# York accelerator plugins (API 1)

The general-purpose CPU wheel owns `qiskit_aer`. These small, independent wheels
own `qiskit_aer_york_opencl` and `qiskit_aer_york_cupy`. Installing or uninstalling
a plugin leaves the CPU wheel intact. They require York `0.17.2.post2.dev0` or
later; older CPU wheels do not implement the protocol. None are published yet.

## Installation and selection

First build/install the York CPU wheel, following the repository README. Use
the Python interpreter in that environment for every command below.

```text
python -m pip install ./accelerators/opencl
```

OpenCL uses the GPU driver's compute runtime, without a CUDA/ROCm toolkit or
Linux requirement. For NVIDIA CUDA 12 on native Windows or Linux:

```text
python -m pip install './accelerators/cupy[cuda12]'
```

For AMD ROCm/HIP, install a **CuPy HIP build matching the installed ROCm SDK**,
following the [CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html),
then `python -m pip install ./accelerators/cupy`. This wheel deliberately does
not install a CUDA dependency for AMD. A CuPy CUDA build and HIP build both own
the `cupy` namespace: use separate environments to test these two SDKs.
AMD's [HIP platform requirements](https://rocmdocs.amd.com/projects/HIP/en/latest/faq.html)
and CuPy's supported platform/build combinations determine whether a ROCm
installation is usable. Windows HIP SDK support alone does not establish CuPy
HIP support. AMD on native Windows can use the OpenCL plugin when its driver
exposes a supported GPU. We have not verified an AMD device yet.

```python
from qiskit_aer import AerSimulator
from qiskit_aer.accelerators import available_accelerators, accelerator_devices

print(available_accelerators())  # Installed names; no SDK import or GPU probe.
print(accelerator_devices("opencl"))  # Explicit runtime/device probe.

sim = AerSimulator(
    accelerator="opencl", accelerator_options={"vendor": "intel"}, precision="single"
)
# Other choices: accelerator='cuda' or 'rocm', accelerator_options={'device_id': '0'}.
# OpenCL device_id is '<platform-index>:<device-index>', reported by the probe.
job = sim.run(circuit, shots=1024, seed_simulator=123)
result = job.result()
print(result.get_counts())
print(result.results[0].metadata)
```

An explicit plugin implies GPU unless `device` was set, in which case it must
be `GPU`. Default `AerSimulator()` continues to use the CPU without probing any
GPU. Runtime selection also works with `sim.run(circuit, accelerator='opencl')`.
Aer SamplerV2/EstimatorV2 accept these selectors in `options['backend_options']`.

## Support and validation

| Path | Vendor | Platform | Validation |
| --- | --- | --- | --- |
| OpenCL | NVIDIA RTX 3060 | Native Windows | 16 real GPU checks; double precision |
| OpenCL | Intel UHD 770 | Native Windows | 16 real GPU checks; single precision |
| CUDA/CuPy | NVIDIA RTX 3060 | Native Windows | 16 real GPU checks; double precision |
| CUDA/CuPy | NVIDIA RTX 3060 | Linux under WSL2 | 16 real GPU checks; double precision |
| ROCm/HIP/CuPy | AMD | SDK/CuPy supported platforms | Implementation present; hardware validation outstanding |
| OpenCL | AMD | Driver-supported Windows/Linux | Implementation present; hardware validation outstanding |
| OpenCL | Other GPUs/macOS | Driver/runtime dependent | Not claimed as GPU-validated |

Hardware checks compare ten random circuits against an independent Qiskit
Statevector reference, exercise nonconsecutive/reversed target qubits and global
phase, sampling, deterministic seeds, parameter batches, both Aer V2 primitives,
and CPU coexistence. Native Windows local source tests reused the unchanged Aer
0.17.2 native CPU bindings; installed York-wheel checks run separately in CI.
The selected plugin performs the GPU evolution, independently of those bindings.
The Linux checks reused an existing York CPU native wheel and the new Python
adapter; hosted CI builds the complete new CPU wheel from source.
Results record device/vendor, driver/runtime, precision, GPU gate count, and OS.
Hosted CI checks packaging and the protocol with an explicit host test double.
It does not prove GPU execution.

```text
python -I tools/york_accelerator_smoke.py --accelerator opencl --vendor intel --precision single --output artifacts/gpu-intel.json
python -I tools/york_accelerator_smoke.py --accelerator cuda --vendor nvidia --output artifacts/gpu-cuda.json
python -I tools/york_accelerator_smoke.py --accelerator rocm --vendor amd --output artifacts/gpu-rocm.json
```

These checks fail for a missing GPU, wrong vendor, bad results, or SDK mismatch.
They never substitute CPU execution. A driver without FP64 must be explicitly
used with `precision='single'`; `double` fails rather than silently losing precision.

## Current feature boundary

This is an initial portable GPU statevector engine, not feature/performance
parity with the optimized native Aer controller. GPU kernels apply ideal unitary
gates with up to four target qubits in Qiskit's little-endian qubit order; larger
unitaries should be transpiled to `['u', 'cx']`. Terminal measurement mappings,
multiple classical registers, list-of-circuit execution, parameter batches,
global phase, `save_statevector` and averaged `save_expval` are supported.
Saved observables and terminal shot sampling run on the **host** and are recorded
as such. The statevector remains on the GPU during gate evolution.

Noise, reset, initialize, intermediate measurements, classical control flow,
conditional/per-shot saves, density matrices, tensor networks, MPI/multiple GPUs,
and native Aer-specific optimization options are rejected. Use default CPU Aer
for those features. No automatic CPU fallback or performance improvement is
claimed. `max_memory_mb` and device allocation limits bound state storage; host
sampling also needs memory for a copy of the statevector.

## Implementing another plugin

Register a provider factory in the `qiskit_aer_york.accelerators` entry-point
group under a unique name. The core loads only the explicitly chosen name and
requires `api_version = 1`, `devices()` and
`create_state(num_qubits, precision, accelerator_options)`.

The returned state implements `apply(matrix, qubits)` (complex square unitary
matrix, least-significant target first), `to_host()` (a copied NumPy complex
statevector), and `metadata` including `device='GPU'`, vendor, runtime, name,
driver, and device ID. It starts in |0...0>. SDK loading, actual GPU selection,
precision/memory checks, synchronization, and resource lifetime belong to the
provider. A provider must never label CPU/software execution as GPU execution.
An optional `finish()` synchronizes queued work without a host copy; otherwise
the adapter uses `to_host()` to synchronize before job completion. Circuit
buffers are released before allocating the next state in a batch.
The CPU distribution has no accelerator requirements and never imports SDKs
just to list installed entry points.

OpenCL portability is grounded in the vendor driver implementations listed by
[Khronos](https://www.khronos.org/conformance/adopters/conformant-products/opencl)
and [PyOpenCL's runtime installation guide](https://documen.tician.de/pyopencl/misc.html).
Individual hardware/driver combinations still need the smoke checks above.
