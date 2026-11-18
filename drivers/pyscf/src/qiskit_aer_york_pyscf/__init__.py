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


def _native_supported(system):
    if system == "Windows":
        return sys.maxsize > 2**32 and platform.machine().lower() in ("amd64", "x86_64")
    return system in ("Linux", "Darwin")


class PySCFDriver:
    api_version = 1

    def capabilities(self):
        """Runtime availability is separate from whether its worker has PySCF installed."""
        system = platform.system()
        return {
            "native_supported": _native_supported(system),
            "wsl_available": system == "Windows" and shutil.which("wsl.exe") is not None,
            "format": "fcidump",
            "formats": ("fcidump", "integrals"),
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
        format="fcidump",
    ):
        """Compute integrals; choose legacy FCIDump or a packed IntegralBundle.

        Native mode uses the selected Python on Windows x64, Linux or macOS.
        Optional WSL mode requires an explicit Linux Python and distribution.
        The entire molecular request travels as JSON stdin, never shell code.
        """
        if format not in ("fcidump", "integrals"):
            raise ValueError("format must be 'fcidump' or 'integrals'")
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
        if format == "integrals":
            # Send the shared encoder with the worker. The remote environment
            # still needs only PySCF/NumPy, never York or Qiskit packages.
            try:
                from qiskit_aer_york_data import IntegralBundle, _wire
            except ImportError as exc:
                raise PluginError(
                    "Packed integrals require qiskit-aer-york-pyscf[data]; install it explicitly."
                ) from exc

            wire = Path(_wire.__file__).read_text(encoding="utf-8")
            prefix = "import types, sys\nyork_wire = types.ModuleType('york_wire')\n"
            prefix += f"exec({wire!r}, york_wire.__dict__)\nsys.modules['york_wire'] = york_wire\n"
            source = prefix + source
        if runtime == "native":
            if not _native_supported(platform.system()):
                raise PluginError(
                    "York's native PySCF driver requires Windows x64 with 64-bit Python, "
                    "Linux or macOS. Existing FCIDump data can still be imported."
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
            "format": format,
            "runtime": runtime,
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
        if process.returncode:
            stdout, stderr = _decode_output(process.stdout), _decode_output(process.stderr)
            detail = stderr.strip()[-2000:] or stdout.strip()[-2000:]
            if runtime == "native" and platform.system() == "Windows":
                detail += (
                    " Use an x64 Windows Python with the official PySCF 2.15.1+ "
                    "Windows wheel installed; York will not switch to WSL."
                )
            raise PluginError(f"PySCF {runtime} worker failed: {detail}")
        if format == "integrals":
            try:
                return IntegralBundle.from_bytes(process.stdout)
            except (TypeError, ValueError) as exc:
                raise PluginError(
                    f"PySCF {runtime} worker returned invalid integral data: {exc}"
                ) from exc
        stdout = _decode_output(process.stdout)
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

    def run_integrals(self, **options):
        """Optimized portable route; requires the separately installed data plugin."""
        if "format" in options:
            raise ValueError("run_integrals selects format='integrals'")
        return self.run(format="integrals", **options)
