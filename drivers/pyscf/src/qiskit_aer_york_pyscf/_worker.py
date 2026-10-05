# Licensed under the Apache License, Version 2.0.
"""Standalone worker, sent unchanged to an explicitly configured Python process."""

import contextlib
import json
import math
from pathlib import Path
import sys
import tempfile


def calculate(request):
    import numpy as np
    import pyscf
    from pyscf import ao2mo, gto, scf
    from pyscf.tools import fcidump

    molecule_options = dict(request)
    output_format = molecule_options.pop("format", "fcidump")
    runtime = molecule_options.pop("runtime", "native")
    pyscf.lib.num_threads(molecule_options.pop("threads"))
    molecule = gto.M(**molecule_options, verbose=0)
    solver = scf.ROHF(molecule) if request["spin"] else scf.RHF(molecule)
    energy = float(solver.kernel())
    if not solver.converged or not math.isfinite(energy):
        raise RuntimeError("PySCF Hartree-Fock calculation did not converge")
    coefficients = solver.mo_coeff
    if np.iscomplexobj(coefficients):
        raise ValueError("Restricted integral format requires real molecular orbitals")
    n = coefficients.shape[1]
    h1 = coefficients.T @ solver.get_hcore() @ coefficients
    source = molecule if solver._eri is None else solver._eri
    eri = ao2mo.restore(8, ao2mo.full(source, coefficients), n)
    nuclear = float(solver.energy_nuc())
    if output_format == "integrals":
        metadata = {
            "kind": "restricted-molecular-integrals",
            "basis": "MO",
            "units": "hartree",
            "index_order": "chemist",
            "eri_layout": "s8",
            "num_alpha": int(molecule.nelec[0]),
            "num_beta": int(molecule.nelec[1]),
            "nuclear_energy": nuclear,
            "reference_energy": energy,
            "provenance": {
                "pyscf_version": pyscf.__version__,
                "runtime": runtime,
                "method": "ROHF" if request["spin"] else "RHF",
                "input": request,
            },
        }
        return metadata, {"h1": h1, "eri_s8": eri}
    with tempfile.TemporaryDirectory(prefix="york-pyscf-") as temporary:
        path = Path(temporary) / "integrals.fcidump"
        # from_scf writes MS2=0, including for ROHF. Preserve actual spin here.
        fcidump.from_integrals(str(path), h1, eri, n, molecule.nelec, nuclear, ms=molecule.spin)
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
    # Keep Python-level SDK output out of both data protocols.
    with contextlib.redirect_stdout(sys.stderr):
        result = calculate(request)
    if request.get("format") == "integrals":
        from york_wire import write_frame

        write_frame(sys.stdout.buffer, *result)
        sys.stdout.buffer.flush()
    else:
        json.dump(result, sys.stdout, allow_nan=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
