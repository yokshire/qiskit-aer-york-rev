# Licensed under the Apache License, Version 2.0.
"""Host-independent PySCF dispatch; importing this module never imports PySCF."""

import json
import math
import numbers
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from qiskit_aer_york import PluginError


def _decode_output(data):
    # Linux Python writes UTF-8; WSL's own Windows errors can be UTF-16LE.
    if isinstance(data, str):
        return data
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:128]:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


class PySCFDriver:
    api_version = 1

    def capabilities(self):
        """Runtime availability is separate from whether its worker has PySCF installed."""
        system = platform.system()
        return {
            "native_supported": system in ("Linux", "Darwin"),
            "wsl_available": system == "Windows" and shutil.which("wsl.exe") is not None,
            "format": "fcidump",
            "methods": ("RHF", "ROHF"),
        }

    def run(
        self,
        *,
        atom,
        basis="sto3g",
        charge=0,
        spin=0,
        threads=1,
        unit="Angstrom",
        runtime="native",
        python=None,
        distribution=None,
        timeout=300,
    ):
        """Compute integrals in the chosen worker; return JSON-compatible FCIDump data.

        Native Windows PySCF is unsupported. WSL requires an explicit Linux
        Python executable and distribution, with PySCF already installed there.
        The entire molecular request travels as JSON stdin, never shell code.
        """
        if not isinstance(atom, str) or not atom.strip():
            raise ValueError("atom must be a nonempty PySCF molecular geometry string")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("basis must be a nonempty named PySCF basis")
        for name, value in (("charge", charge), ("spin", spin), ("threads", threads)):
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if spin < 0:
            raise ValueError("spin must be nonnegative (number of unpaired electrons)")
        if threads < 1:
            raise ValueError("threads must be positive")
        if unit not in ("Angstrom", "Bohr"):
            raise ValueError("unit must be 'Angstrom' or 'Bohr'")
        if (
            not isinstance(timeout, numbers.Real)
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        source = Path(__file__).with_name("_worker.py").read_text(encoding="utf-8")
        if runtime == "native":
            if platform.system() not in ("Linux", "Darwin"):
                raise PluginError(
                    "PySCF does not support native Windows. Choose runtime='wsl' with an "
                    "explicit Linux Python/distribution, or use existing FCIDump data."
                )
            if distribution is not None:
                raise ValueError("distribution applies only to runtime='wsl'")
            command = [str(python or sys.executable), "-I", "-c", source]
        elif runtime == "wsl":
            if platform.system() != "Windows":
                raise PluginError("runtime='wsl' is a Windows controller option")
            wsl = shutil.which("wsl.exe")
            if wsl is None:
                raise PluginError("WSL is unavailable; configure WSL or load existing FCIDump data")
            if not isinstance(python, str) or not python.startswith("/"):
                raise ValueError("WSL requires an explicit absolute Linux Python executable")
            if (
                not isinstance(distribution, str)
                or not distribution.strip()
                or distribution.startswith("-")
            ):
                raise ValueError("WSL requires an explicit distribution name")
            command = [wsl, "--distribution", distribution, "--exec", python, "-I", "-c", source]
        else:
            raise ValueError("runtime must be 'native' or 'wsl'; no automatic fallback occurs")
        request = {
            "atom": atom,
            "basis": basis,
            "charge": int(charge),
            "spin": int(spin),
            "threads": int(threads),
            "unit": unit,
        }
        try:
            process = subprocess.run(
                command,
                input=json.dumps(request).encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PluginError(f"PySCF {runtime} worker exceeded {timeout} seconds") from exc
        except OSError as exc:
            raise PluginError(f"Cannot start PySCF {runtime} worker: {exc}") from exc
        stdout, stderr = _decode_output(process.stdout), _decode_output(process.stderr)
        if process.returncode:
            detail = stderr.strip()[-2000:] or stdout.strip()[-2000:]
            raise PluginError(f"PySCF {runtime} worker failed: {detail}")
        try:
            result = json.loads(stdout)
        except (TypeError, ValueError) as exc:
            raise PluginError(f"PySCF {runtime} worker returned invalid JSON") from exc
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != 1
            or result.get("format") != "fcidump"
        ):
            raise PluginError("PySCF worker returned an incompatible result schema")
        result["runtime"] = runtime
        return result
