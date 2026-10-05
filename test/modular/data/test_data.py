# Licensed under the Apache License, Version 2.0.
"""Portable array ownership, damaged payloads and explicit scientific conventions."""

import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from qiskit_aer_york_data import IntegralBundle, read_frame, write_frame


def bundle(h1=None, eri=None, **options):
    return IntegralBundle(
        np.array([[-1.5]]) if h1 is None else h1,
        np.array([0.5]) if eri is None else eri,
        num_alpha=1,
        num_beta=1,
        nuclear_energy=0.4,
        **options,
    )


class DataTests(unittest.TestCase):
    def test_snapshot_and_decode_are_immutable(self):
        h1 = np.array([[-1.5]])
        source = bundle(h1)
        h1[0, 0] = 999
        self.assertEqual(source.h1[0, 0], -1.5)
        data = source.to_bytes()
        decoded = IntegralBundle.from_bytes(data)
        self.assertFalse(decoded.h1.flags.owndata)
        self.assertFalse(decoded.h1.flags.writeable)
        self.assertTrue(np.shares_memory(decoded.h1, np.frombuffer(data, dtype=np.uint8)))
        with self.assertRaises(ValueError):
            decoded.h1.setflags(write=True)
        mutable = bytearray(data)
        snapshot = IntegralBundle.from_bytes(memoryview(mutable).toreadonly())
        mutable[-64:] = bytes(64)
        self.assertEqual(snapshot.eri_s8[0], 0.5)
        metadata = decoded.metadata
        metadata["provenance"]["changed"] = True
        self.assertEqual(decoded.metadata["provenance"], {})

    def test_memory_mapped_reuse_without_driver_or_dense_eri(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "molecule.york"
            bundle().save(path)
            decoded = IntegralBundle.load(path, mmap=True)
            self.assertEqual(decoded.eri_s8.shape, (1,))
            self.assertFalse(decoded.eri_s8.flags.writeable)
            self.assertEqual(decoded.nbytes, 16)
            del decoded

    def test_corrupt_and_truncated_frames_fail(self):
        data = bundle().to_bytes()
        for damaged in (data[:10], data[:-1], data + b"x", b"other" + data[5:]):
            with self.assertRaises(ValueError):
                IntegralBundle.from_bytes(damaged)
        damaged = bytearray(data)
        damaged[-64] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            IntegralBundle.from_bytes(damaged)

    def test_shape_spin_units_and_finite_values_are_validated(self):
        for h1, eri in (
            (np.zeros((2, 3)), [1]),
            ([[1]], [1, 2]),
            ([[np.nan]], [1]),
            ([[1, 2], [3, 1]], np.zeros(6)),
        ):
            with self.assertRaises(ValueError):
                bundle(h1, eri)
        with self.assertRaises(ValueError):
            IntegralBundle([[1]], [1], num_alpha=0, num_beta=1, nuclear_energy=0)
        stream = io.BytesIO()
        source = bundle()
        metadata = source.metadata
        metadata["index_order"] = "physicist"
        write_frame(stream, metadata, {"h1": source.h1, "eri_s8": source.eri_s8})
        with self.assertRaisesRegex(ValueError, "index_order"):
            IntegralBundle.from_bytes(stream.getvalue())

    def test_generic_frame_preserves_dtype_shape_and_endianness(self):
        value = np.arange(12, dtype=">f8").reshape(3, 4)[:, ::2]
        stream = io.BytesIO()
        write_frame(
            stream,
            {"units": "example"},
            {"noncontiguous": value, "complex": np.array([1 + 2j], dtype=np.complex64)},
        )
        metadata, arrays = read_frame(stream.getvalue())
        np.testing.assert_array_equal(arrays["noncontiguous"], value)
        self.assertEqual(arrays["complex"].dtype, np.dtype("<c8"))
        self.assertEqual(metadata["units"], "example")
        with self.assertRaises(ValueError):
            write_frame(io.BytesIO(), {}, {"objects": np.array([object()])})

    def test_generic_empty_scalar_and_mutable_payloads(self):
        stream = io.BytesIO()
        write_frame(stream, {}, {"empty": np.empty((0, 2)), "scalar": np.array(3.5)})
        mutable = bytearray(stream.getvalue())
        _, arrays = read_frame(memoryview(mutable).toreadonly())
        self.assertEqual(arrays["empty"].shape, (0, 2))
        self.assertEqual(arrays["scalar"].shape, ())
        mutable[-64:] = bytes(64)
        self.assertEqual(float(arrays["scalar"]), 3.5)


if __name__ == "__main__":
    unittest.main()
