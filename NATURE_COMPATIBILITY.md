# Qiskit Nature and PySCF compatibility

PySCF [supports Linux, macOS and WSL; native Windows is unsupported](https://pyscf.org/user/install.html).
Nature's data/mapping layer and the external chemistry SDK have different
platform requirements. York separates them into two optional wheels:

| Package | Purpose | PySCF dependency |
| --- | --- | --- |
| `qiskit-aer-york-nature` / `integrations/nature` | Read FCIDump, build Nature problems, map to qubits | None |
| `qiskit-aer-york-pyscf` / `drivers/pyscf` | Dispatch an explicitly selected chemistry worker | None by default |

The core and CPU/GPU engines require neither package. Installing the driver on
Windows does not compile/import PySCF. Its `native` extra selects PySCF only on
Linux/macOS. Unsupported native execution fails before starting a process.
No runtime fallback, automatic install, WSL setup or substitute molecule occurs.
Psi4 and Gaussian are not installed or implicitly selected.

## Portable input without a chemistry runtime

No York packages are published yet. Install these local projects together:

```text
python -m pip install ./core ./integrations/qiskit ./integrations/nature ./engines/statevector
```

```python
from qiskit_aer_york import load_plugin

nature = load_plugin("nature", "integration")
problem = nature.from_fcidump("molecule.fcidump")
operator = nature.qubit_operator(problem)  # Jordan-Wigner by default.
```

FCIDump contains molecular integrals, not a cached quantum simulation result.
Generate it in a supported chemistry environment beforehand, then read it on
native Windows, Linux or macOS without PySCF or WSL. Nature uses its official
[FCIDump translator](https://github.com/qiskit-community/qiskit-nature/blob/main/qiskit_nature/second_q/formats/fcidump_translator.py).
The operator contains electronic energy; add the problem's
`hamiltonian.nuclear_repulsion_energy` once for total energy. These files do not
preserve all driver properties such as dipole integrals or the molecule object.
The current worker supports RHF/ROHF and real molecular-orbital integrals.

## Windows controller with an explicit WSL worker

Install `./core ./drivers/pyscf` together into the Windows environment. Separately
prepare a Python environment inside an existing WSL distribution with PySCF
installed, following its official installation instructions. Then select it:

```python
problem = nature.from_driver(
    "pyscf", atom="H 0 0 0; H 0 0 0.735", basis="sto3g",
    runtime="wsl", python="/home/your-user/venvs/chemistry/bin/python",
    distribution="Ubuntu", threads=1, timeout=300,
)
```

Only classical PySCF calculation runs in WSL. Geometry/options travel as JSON
stdin to `wsl.exe --exec` without shell interpolation; FCIDump/reference energy
return as JSON stdout. Nature, mapping and simulation continue in Windows.
The worker needs PySCF but no York/Qiskit/Nature installation. The controller
imports no Linux binaries and needs no cross-OS shared-file path. Worker errors,
malformed responses and timeouts are reported explicitly. WSL must be usable.

On native Linux/macOS, opt into the SDK using:

```text
python -m pip install ./core './drivers/pyscf[native]'
```

Use `nature.from_driver("pyscf", atom=..., runtime="native")`. This launches
the current Python; supply `python="/absolute/path/to/python"` for a separate
chemistry environment. Native is the default and never switches to WSL.
`capabilities()` reports controller runtime support, not SDK installation in an
arbitrary external Python environment.

## Simulate locally

```python
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer_york import Simulator
from qiskit_aer_york_qiskit import save_statevector

mapper = JordanWignerMapper()
operator = nature.qubit_operator(problem, mapper)
circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper).decompose()
save_statevector(circuit)
result = Simulator("statevector").run(circuit).result()
energy = result.get_statevector().expectation_value(operator).real
total_energy = energy + problem.hamiltonian.nuclear_repulsion_energy
```

This evaluates the Hartree-Fock reference state, not a correlated ground state.
`nature.estimator(simulator, **options)` also provides BackendEstimatorV2, with
sampling uncertainty, for quantum algorithms. GPU engines use the same facade
after explicit installation.

## Validation and boundaries

Portable tests use fresh Windows/Linux/macOS environments without PySCF, Psi4
or native Aer. They compare actual CPU simulation against analytic one-orbital
vacuum/one-electron/two-electron energies and test lazy discovery and worker errors.

`tools/york_nature_smoke.py` performs a real H2 STO-3G worker calculation, reads
its integrals locally, and compares York's circuit energy to PySCF's RHF energy.
Local Windows-to-WSL testing with PySCF 2.14.0 yielded identical
`-1.116998996754004` Hartree values while PySCF was absent from Windows.
CI also checks a native Linux worker. Portable macOS tests do not certify native
macOS PySCF execution; that path remains subject to upstream SDK support.

```text
python -I tools/york_nature_smoke.py --runtime wsl --python /absolute/linux/python --distribution Ubuntu --output artifacts/nature-wsl.json
python -I tools/york_nature_smoke.py --runtime native --output artifacts/nature-native.json
```

The old `test/benchmark/vqe_application.py` and Linux ASV files contain historical
Aqua/`qiskit.chemistry` benchmarks. Modular packages neither import nor install
them. They remain for upstream provenance; use the current integration above.
