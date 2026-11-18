# Licensed under the Apache License, Version 2.0.
"""Lazy, versioned discovery of independently installed York GPU plugins.

Plugins own separate Python namespaces. Importing Aer never imports a GPU SDK.
"""

from importlib.metadata import entry_points

from qiskit_aer.aererror import AerError

API_VERSION = 1
ENTRY_POINT_GROUP = "qiskit_aer_york.accelerators"


def available_accelerators():
    """Return installed plugin names, without importing SDKs or probing hardware."""
    return sorted({entry.name for entry in entry_points(group=ENTRY_POINT_GROUP)})


def load_accelerator(name):
    """Load one explicitly selected plugin and validate its protocol version."""
    matches = [entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name == name]
    if len(matches) != 1:
        raise AerError(
            f"Accelerator {name!r} needs exactly one installed plugin; found {len(matches)}. "
            f"Installed plugins: {available_accelerators()}"
        )
    try:
        provider = matches[0].load()()
    except Exception as exc:
        raise AerError(f"Cannot load accelerator {name!r}: {exc}") from exc
    if getattr(provider, "api_version", None) != API_VERSION:
        raise AerError(
            f"Accelerator {name!r} does not implement York accelerator API {API_VERSION}"
        )
    for method in ("devices", "create_state"):
        if not callable(getattr(provider, method, None)):
            raise AerError(f"Accelerator {name!r} is missing {method}()")
    return provider


def accelerator_devices(name):
    """List actual GPU devices for one plugin; CPU devices are never substituted."""
    return load_accelerator(name).devices()


__all__ = ["API_VERSION", "available_accelerators", "accelerator_devices"]
