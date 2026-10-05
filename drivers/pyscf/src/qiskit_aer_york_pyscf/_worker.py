# Licensed under the Apache License, Version 2.0.
"""Standalone worker, sent unchanged to an explicitly configured Python process."""

import contextlib
import json
import math
from pathlib import Path
import sys
import tempfile


def calculate(request):
    import pyscf
    from pyscf import gto, scf
    from pyscf.tools import fcidump

    molecule_options = dict(request)
    pyscf.lib.num_threads(molecule_options.pop("threads"))
    molecule = gto.M(**molecule_options, verbose=0)
    solver = scf.ROHF(molecule) if request["spin"] else scf.RHF(molecule)
    energy = float(solver.kernel())
    if not solver.converged or not math.isfinite(energy):
        raise RuntimeError("PySCF Hartree-Fock calculation did not converge")
    with tempfile.TemporaryDirectory(prefix="york-pyscf-") as temporary:
        path = Path(temporary) / "integrals.fcidump"
        fcidump.from_scf(solver, str(path))
        text = path.read_text(encoding="utf-8")
    return {
        "schema_version": 1,
        "format": "fcidump",
        "fcidump": text,
        "reference_energy": energy,
        "pyscf_version": pyscf.__version__,
        "method": "ROHF" if request["spin"] else "RHF",
        "input": request,
    }


def main():
    request = json.load(sys.stdin)
    # Keep Python-level SDK output out of the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        result = calculate(request)
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
