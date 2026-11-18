# Licensed under the Apache License, Version 2.0.
"""Portable array transport and restricted, real molecular integrals in MO basis."""

import io
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ._wire import read_frame, write_frame


class IntegralBundle:
    """Immutable frame-backed h1 and eightfold packed ERI, in Hartree/chemist order.

    Public construction snapshots caller-owned arrays once. Decoding immutable
    bytes or a read-only memory map instead keeps views of the original buffer.
    The frame defines orbital/electron/spin conventions explicitly; no heuristic
    index-order detection or dense n**4 tensor allocation is needed by Nature.
    """

    def __init__(
        self,
        h1,
        eri_s8,
        *,
        num_alpha,
        num_beta,
        nuclear_energy,
        reference_energy=None,
        provenance=None,
    ):
        h1 = np.asarray(h1)
        eri_s8 = np.asarray(eri_s8)
        if h1.dtype.kind == "c" or eri_s8.dtype.kind == "c":
            raise ValueError("Restricted integral format requires real arrays")
        metadata = {
            "kind": "restricted-molecular-integrals",
            "basis": "MO",
            "units": "hartree",
            "index_order": "chemist",
            "eri_layout": "s8",
            "num_alpha": num_alpha,
            "num_beta": num_beta,
            "nuclear_energy": nuclear_energy,
            "reference_energy": reference_energy,
            "provenance": {} if provenance is None else provenance,
        }
        self._initialize(
            metadata, {"h1": np.asarray(h1, dtype="<f8"), "eri_s8": np.asarray(eri_s8, dtype="<f8")}
        )
        # Frame bytes own the snapshot, and cannot be made writable by a caller.
        snapshot = self.to_bytes()
        self._initialize(*read_frame(snapshot))

    def _initialize(self, metadata, arrays):
        for key, expected in (
            ("kind", "restricted-molecular-integrals"),
            ("basis", "MO"),
            ("units", "hartree"),
            ("index_order", "chemist"),
            ("eri_layout", "s8"),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"Unsupported integral convention: {key}")
        if set(arrays) != {"h1", "eri_s8"}:
            raise ValueError("Integral frame must contain exactly h1 and eri_s8")
        h1, eri = arrays["h1"], arrays["eri_s8"]
        if h1.ndim != 2 or h1.shape[0] < 1 or h1.shape[0] != h1.shape[1]:
            raise ValueError("h1 must be a nonempty square array")
        n = h1.shape[0]
        pairs = n * (n + 1) // 2
        if eri.shape != (pairs * (pairs + 1) // 2,):
            raise ValueError("eri_s8 has the wrong packed length")
        if any(array.dtype.str != "<f8" for array in arrays.values()):
            raise ValueError("Integral arrays must use little-endian float64")
        for array in arrays.values():
            flat = array.reshape(-1)
            for offset in range(0, flat.size, 131072):
                if not np.isfinite(flat[offset : offset + 131072]).all():
                    raise ValueError("Integral arrays must be finite")
        if not np.allclose(h1, h1.T, atol=1e-12, rtol=1e-12):
            raise ValueError("h1 must be symmetric")
        alpha, beta = metadata.get("num_alpha"), metadata.get("num_beta")
        if (
            any(type(value) is not int or not 0 <= value <= n for value in (alpha, beta))
            or alpha < beta
        ):
            raise ValueError("Invalid alpha/beta electron counts")
        for key in ("nuclear_energy", "reference_energy"):
            value = metadata.get(key)
            if key == "reference_energy" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be finite")
        if not isinstance(metadata.get("provenance"), dict):
            raise ValueError("provenance must be a JSON object")
        # Nested metadata is returned as a fresh JSON snapshot to prevent mutation.
        self._metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
        self._arrays = MappingProxyType(dict(arrays))

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def h1(self):
        return self._arrays["h1"]

    @property
    def eri_s8(self):
        return self._arrays["eri_s8"]

    @property
    def num_orbitals(self):
        return self.h1.shape[0]

    @property
    def nbytes(self):
        return sum(array.nbytes for array in self._arrays.values())

    def write(self, stream):
        write_frame(stream, self.metadata, self._arrays)

    def to_bytes(self):
        stream = io.BytesIO()
        self.write(stream)
        return stream.getvalue()

    @classmethod
    def from_bytes(cls, buffer):
        if not isinstance(buffer, bytes) and not (
            isinstance(buffer, np.memmap) and buffer.mode == "r"
        ):
            buffer = bytes(buffer)
        instance = cls.__new__(cls)
        instance._initialize(*read_frame(buffer))
        return instance

    def save(self, path):
        """Save for explicit reuse without rerunning the external driver."""
        with Path(path).open("wb") as stream:
            self.write(stream)

    @classmethod
    def load(cls, path, *, mmap=False):
        """Mapped arrays own the file mapping; release all views before replacing files."""
        buffer = np.memmap(path, mode="r", dtype=np.uint8) if mmap else Path(path).read_bytes()
        return cls.from_bytes(buffer)


class DataIntegration:
    api_version = 1
    IntegralBundle = IntegralBundle
    read_frame = staticmethod(read_frame)
    write_frame = staticmethod(write_frame)
