# York modular simulator

Install the dependency-free core first, then add only the engines and integrations
you need. Neither plugin discovery nor constructing a `Simulator` imports an
engine, initializes a GPU, downloads software, or installs another package.
The first `run()` or `.backend` access loads exactly the selected engine.

No York package is published to PyPI yet. The commands below work from this
checkout in a fresh Python 3.10+ virtual environment. Once released, the core's
optional extras provide equivalent package presets.

## Package boundaries

| Distribution / source directory | Installed purpose | Heavy dependencies |
| --- | --- | --- |
| `qiskit-aer-york-core` / `core` | Registry, lazy dispatch, API negotiation | None; Python standard library only |
| `qiskit-aer-york-qiskit` / `integrations/qiskit` | Qiskit circuits, BackendV2, jobs, results and primitive adapters | Qiskit and NumPy; Qiskit brings SciPy |
| `qiskit-aer-york-nature` / `integrations/nature` | Portable molecular data and qubit mapping | Qiskit Nature; no PySCF/Psi4 SDK |
| `qiskit-aer-york-pyscf` / `drivers/pyscf` | Native or Windows-to-WSL chemistry worker | None by default; PySCF belongs to the selected worker |
| `qiskit-aer-york-statevector` / `engines/statevector` | Ideal CPU statevector engine | Qiskit integration; no native Aer |
| `qiskit-aer-york-opencl` / `accelerators/opencl` | OpenCL GPU statevector engine | Qiskit integration and PyOpenCL; vendor driver |
| `qiskit-aer-york-cupy` / `accelerators/cupy` | CUDA or ROCm GPU statevector engine | Qiskit integration; separately selected CuPy SDK build |
| `qiskit-aer-york-native` / `engines/native` | Full native Aer methods/noise and existing Aer primitives | Explicitly installs the full `qiskit-aer-york-rev` bundle |

See [Nature/PySCF compatibility](NATURE_COMPATIBILITY.md) for portable integral
input and the explicit Windows-to-WSL worker. Chemistry runtimes never become
mandatory dependencies of the core or Nature's mapping integration.

The core wheel contains no C++ sources, native library, circuit framework or SDK.
Installing an optional integration can still bring its own substantial dependencies.
The native plugin is currently one complete Aer bundle: its C++ simulation methods
are not individually packaged. Splitting those native methods would require a
separate native ABI and build refactor. New engines and integrations can already
be delivered independently through the public plugin contract.

## Install incrementally

```text
python -m pip install ./core
```

At this point `from qiskit_aer_york import Simulator, plugins` works, and
`plugins()` is empty. Running an uninstalled engine raises an installation hint.
There is no implicit download or silent CPU fallback.

Add ideal CPU simulation:

```text
python -m pip install ./core ./integrations/qiskit ./engines/statevector
```

```python
from qiskit import QuantumCircuit
from qiskit_aer_york import Simulator, load_plugin, plugins

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

simulator = Simulator("statevector")
print(simulator.run(circuit, shots=1024).result().get_counts())
print(plugins())  # Reads metadata; does not initialize other engines.

qiskit = load_plugin("qiskit", "integration")
sampler = qiskit.sampler(simulator)
print(sampler.run([circuit], shots=128).result()[0].data.meas.get_counts())
```

Add cross-vendor GPU simulation, independently of the CPU engine:

```text
python -m pip install ./core ./integrations/qiskit ./accelerators/opencl
```

```python
simulator = Simulator(
    "opencl", engine_options={"vendor": "intel"}, precision="single"
)
# Other vendor selectors: "nvidia", "amd". FP64-capable devices can use "double".
print(simulator.run(circuit, shots=1024).result().get_counts())
```

For NVIDIA CUDA:

```text
python -m pip install ./core ./integrations/qiskit "./accelerators/cupy[cuda12]"
```

Select `Simulator("cuda", engine_options={"device_id": "0"})`. For AMD ROCm,
install `./accelerators/cupy` with the core/integration and a compatible CuPy HIP
build separately, then select `Simulator("rocm")`. A CUDA CuPy build is rejected
by the ROCm engine. See [GPU runtime and hardware limits](accelerators/README.md).

The Python packages are portable. Individual SDKs and vendor drivers determine
which GPU paths work on an OS; ROCm support on every Windows GPU is not promised.

To opt into the existing full native Aer bundle, first build/install the York
CPU wheel using `tools/york_build.py`, then install `./core ./engines/native` and
select `Simulator("native-aer", method="density_matrix")`. The native bundle owns
`qiskit_aer` and must not coexist with upstream `qiskit-aer` distributions.
Lightweight engines own separate namespaces and run without that bundle.

## Current lightweight engine capabilities

The CPU, OpenCL and CuPy engines share the optional Qiskit integration's ideal
statevector adapter. It supports unitary gates on up to four targets, barriers,
global phase, parameter batches, terminal measurements, reproducible host shot
sampling, and Qiskit BackendSamplerV2/BackendEstimatorV2. Transpile larger gates
to `u`/`cx` with the selected `.backend`. Reset, noise, control flow and intermediate
measurements require the native engine and fail explicitly on these engines.

Use `qiskit_aer_york_qiskit.save_statevector(circuit)` to append a save directive
without importing Aer. Observable and shot evaluation run on the CPU even when
gate evolution runs on the GPU. Results report the actual engine, device,
precision, gate count and driver details. `max_memory_mb` bounds the estimated
engine working buffers, not all Qiskit plans, saved results or SDK allocations.

## Third-party plugins

Publish a separate wheel with one or more entry points. Entry-point names must
be unique in their group. Conflicts and API version mismatches raise errors.

```toml
[project.entry-points."qiskit_aer_york.engines"]
my_engine = "my_package:Engine"

[project.entry-points."qiskit_aer_york.integrations"]
my_framework = "my_package:Integration"

[project.entry-points."qiskit_aer_york.drivers"]
my_chemistry_driver = "my_package:Driver"
```

An engine factory implements `api_version = 1` and `create_backend(**options)`;
the backend must implement `run(...)`. The engine defines its input/job/result
types. Qiskit-compatible engines return a `BackendV2`, allowing the optional
Qiskit integration to provide its primitives. An integration factory implements
`api_version = 1` and its framework-specific methods. The core contains no Qiskit
type checks. `load_plugin(name, kind="integration")` loads a selected integration.

CPU and GPU implementations can additionally reuse `StatevectorBackend` by
providing `create_state(num_qubits, precision, options)` and states implementing
`apply(matrix, qubits)`, `to_host()`, `finish()` and actual-device `metadata`.
This adapter belongs to the optional Qiskit package, keeping that protocol and
its matrix dependencies out of the core.

## Validation

`python tools/york_modular_ci.py` builds eight independent wheels, inspects their
metadata, installs the core alone in a clean environment, then installs the
light engines in another clean environment without native Aer. It tests CPU
evolution against independent Qiskit references, sampling, parameter/register
mapping and Qiskit primitives. Hosted CI runs this on Linux, Windows and macOS;
it also tests Nature without PySCF and does not claim physical GPU validation.

`tools/york_modular_gpu_smoke.py` requires native Aer to be absent and runs strict
physical-device reference tests. Install the CPU statevector plugin too for its
coexistence check. Missing hardware is an error. NVIDIA Windows CUDA/OpenCL and
Intel Windows OpenCL have local hardware evidence; AMD hardware remains untested.
