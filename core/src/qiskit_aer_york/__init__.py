# Licensed under the Apache License, Version 2.0.
"""Lightweight plugin dispatch. Importing this package loads only the stdlib."""

from dataclasses import dataclass
from importlib import metadata
import threading

API_VERSION = 1
GROUPS = {
    "engine": "qiskit_aer_york.engines",
    "integration": "qiskit_aer_york.integrations",
}
INSTALL_HINTS = {
    "statevector": "qiskit-aer-york-core[statevector]",
    "native-aer": "qiskit-aer-york-core[native]",
    "opencl": "qiskit-aer-york-core[opencl]",
    "cuda": "qiskit-aer-york-core[cuda12]",
    "rocm": "qiskit-aer-york-core[rocm] (plus a compatible CuPy HIP build)",
    "qiskit": "qiskit-aer-york-core[qiskit]",
}


class PluginError(RuntimeError):
    """A requested plugin cannot be used."""


class MissingPluginError(PluginError):
    """The selected plugin is not installed."""


class SimulationError(RuntimeError):
    """The selected engine cannot execute the requested operation."""


@dataclass(frozen=True)
class PluginInfo:
    name: str
    kind: str
    distribution: str | None
    version: str | None


def _entries(kind):
    if kind not in GROUPS:
        raise ValueError(f"Unknown plugin kind {kind!r}; choose from {tuple(GROUPS)}")
    return tuple(metadata.entry_points(group=GROUPS[kind]))


def plugins(kind="engine"):
    """Inspect installed entry-point metadata without importing plugin code."""
    result = []
    for entry in _entries(kind):
        dist = entry.dist
        result.append(
            PluginInfo(
                entry.name,
                kind,
                dist.metadata.get("Name") if dist else None,
                dist.version if dist else None,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.name, item.distribution or "")))


def load_plugin(name, kind="engine"):
    """Load exactly the requested API-1 factory; never install or fall back."""
    matches = [entry for entry in _entries(kind) if entry.name == name]
    if not matches:
        hint = INSTALL_HINTS.get(name, f"a distribution providing {GROUPS[kind]}:{name}")
        raise MissingPluginError(f"Plugin {name!r} is not installed. Install {hint} explicitly.")
    if len(matches) != 1:
        raise PluginError(f"Multiple distributions provide plugin {name!r}; remove the ambiguity")
    try:
        factory = matches[0].load()
        plugin = factory()
    except Exception as exc:
        raise PluginError(f"Cannot load plugin {name!r}: {exc}") from exc
    if getattr(plugin, "api_version", None) != API_VERSION:
        raise PluginError(f"Plugin {name!r} must implement API {API_VERSION}")
    return plugin


class Simulator:
    """Lazy engine facade. The selected engine defines input, job and result types.

    Qiskit engine plugins accept QuantumCircuit and return Qiskit JobV1/Result.
    Other integrations can supply their own formats without changing the core.
    """

    def __init__(self, engine="statevector", **options):
        self.engine = engine
        self._options = dict(options)
        self._backend = None
        self._lock = threading.RLock()

    @property
    def backend(self):
        with self._lock:
            if self._backend is None:
                plugin = load_plugin(self.engine)
                create = getattr(plugin, "create_backend", None)
                if not callable(create):
                    raise PluginError(f"Engine {self.engine!r} has no create_backend()")
                try:
                    backend = create(**self._options)
                except Exception as exc:
                    raise PluginError(f"Cannot initialize engine {self.engine!r}: {exc}") from exc
                if not callable(getattr(backend, "run", None)):
                    raise PluginError(f"Engine {self.engine!r} returned a backend without run()")
                self._backend = backend
            return self._backend

    def run(self, *args, **options):
        return self.backend.run(*args, **options)

    def set_options(self, **options):
        with self._lock:
            if self._backend is not None:
                self._backend.set_options(**options)
            self._options.update(options)
        return self
