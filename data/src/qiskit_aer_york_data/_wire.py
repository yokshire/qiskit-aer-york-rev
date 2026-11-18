# Licensed under the Apache License, Version 2.0.
"""Array wire format, also injected into isolated chemistry workers unchanged.

No York, Qiskit or Nature imports: the worker only needs its existing NumPy.
Buffers are contiguous little-endian, with 64-byte aligned offsets. Headers
contain JSON and SHA-256 digests, never Python objects or executable metadata.
"""

import hashlib
import json
import math
import struct

MAGIC = b"YORKARR1"
PREFIX = struct.Struct("<8sQ")
DTYPES = frozenset(("<f4", "<f8", "<c8", "<c16", "<i4", "<i8", "|u1"))
MAX_HEADER = 65536


def write_frame(stream, metadata, arrays):
    """Write directly to a binary stream without joining all array payloads."""
    import numpy as np

    blocks = []
    descriptors = []
    offset = 0
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Array names must be nonempty strings")
        array = np.asarray(value)
        dtype = array.dtype.newbyteorder("<")
        if dtype.str not in DTYPES:
            raise ValueError(f"Unsupported array dtype {dtype}")
        array = np.asarray(array, dtype=dtype, order="C")
        block = memoryview(array.reshape(-1)).cast("B")
        descriptors.append(
            {
                "name": name,
                "dtype": dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(block),
                "sha256": hashlib.sha256(block).hexdigest(),
            }
        )
        padding = (-len(block)) % 64
        blocks.append((block, padding))
        offset += len(block) + padding
    header = json.dumps(
        {"schema_version": 1, "metadata": metadata, "arrays": descriptors},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-(PREFIX.size + len(header)) % 64)
    if len(header) > MAX_HEADER:
        raise ValueError("Array frame header is too large")
    stream.write(PREFIX.pack(MAGIC, len(header)))
    stream.write(header)
    for block, padding in blocks:
        stream.write(block)
        stream.write(b"\0" * padding)


def read_frame(buffer):
    """Validate bounds/checksums and expose read-only views into the input buffer."""
    import numpy as np

    view = memoryview(buffer).cast("B")
    # A read-only view of a mutable bytearray is not an immutable snapshot.
    if not isinstance(buffer, bytes) and not (isinstance(buffer, np.memmap) and buffer.mode == "r"):
        view = memoryview(bytes(view))
    if len(view) < PREFIX.size:
        raise ValueError("Truncated array frame")
    magic, length = PREFIX.unpack_from(view)
    if magic != MAGIC or not 0 < length <= MAX_HEADER or PREFIX.size + length > len(view):
        raise ValueError("Invalid array frame header")
    try:
        header = json.loads(bytes(view[PREFIX.size : PREFIX.size + length]))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid array frame JSON") from exc
    if (
        not isinstance(header, dict)
        or header.get("schema_version") != 1
        or not isinstance(header.get("metadata"), dict)
        or not isinstance(header.get("arrays"), list)
    ):
        raise ValueError("Incompatible array frame schema")
    base = PREFIX.size + length
    if base % 64:
        raise ValueError("Array frame payload must be aligned")
    arrays = {}
    expected_offset = 0
    for descriptor in header["arrays"]:
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid array descriptor")
        name, dtype, shape = (descriptor.get(key) for key in ("name", "dtype", "shape"))
        offset, size = (descriptor.get(key) for key in ("offset", "nbytes"))
        if (
            not isinstance(name, str)
            or not name
            or name in arrays
            or not isinstance(dtype, str)
            or dtype not in DTYPES
            or not isinstance(shape, list)
            or len(shape) > 8
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or type(offset) is not int
            or offset != expected_offset
            or type(size) is not int
            or size < 0
        ):
            raise ValueError("Invalid array descriptor")
        if size != math.prod(shape) * np.dtype(dtype).itemsize or base + offset + size > len(view):
            raise ValueError("Array shape/size exceeds frame bounds")
        block = view[base + offset : base + offset + size]
        if hashlib.sha256(block).hexdigest() != descriptor.get("sha256"):
            raise ValueError("Array checksum mismatch")
        arrays[name] = np.frombuffer(block, dtype=dtype).reshape(shape)
        expected_offset += size + (-size % 64)
    if base + expected_offset != len(view):
        raise ValueError("Truncated or trailing array frame data")
    return header["metadata"], arrays
