# Licensed under the Apache License, Version 2.0.
"""Explicit CuPy CUDA/HIP providers sharing a portable GPU unitary kernel."""

import numpy as np
from collections import OrderedDict

KERNEL = r"""
typedef {real} real_t;
struct complex_t {{ real_t x, y; }};
extern "C" __global__ void apply_matrix(const complex_t *input, complex_t *output,
                                       const complex_t *matrix, const int *qubits,
                                       const int width, const unsigned long long size) {{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    unsigned long long base = index;
    int row = 0;
    for (int q = 0; q < width; ++q) {{
        row |= ((index >> qubits[q]) & 1ULL) << q;
        base &= ~(1ULL << qubits[q]);
    }}
    complex_t sum = {{0, 0}};
    for (int column = 0; column < (1 << width); ++column) {{
        unsigned long long source = base;
        for (int q = 0; q < width; ++q)
            source |= ((unsigned long long)((column >> q) & 1)) << qubits[q];
        complex_t a = matrix[row * (1 << width) + column], b = input[source];
        sum.x += a.x * b.x - a.y * b.y;
        sum.y += a.x * b.y + a.y * b.x;
    }}
    output[index] = sum;
}}
"""


class _Provider:
    api_version = 1
    hip = False

    def __init__(self):
        import cupy

        self.cp = cupy
        if bool(cupy.cuda.runtime.is_hip) != self.hip:
            expected = "ROCm/HIP" if self.hip else "CUDA"
            raise RuntimeError(
                f"This plugin requires a CuPy {expected} build; installed build differs"
            )

    def devices(self):
        runtime = self.cp.cuda.runtime
        devices = []
        for index in range(runtime.getDeviceCount()):
            properties = runtime.getDeviceProperties(index)
            name = properties["name"]
            devices.append(
                {
                    "device_id": str(index),
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "device": "GPU",
                    "vendor": "amd" if self.hip else "nvidia",
                    "runtime": "ROCm/HIP" if self.hip else "CUDA",
                    "fp64": True,
                    "driver_version": str(runtime.driverGetVersion()),
                    "global_memory_bytes": int(properties["totalGlobalMem"]),
                }
            )
        return devices

    def create_state(self, num_qubits, precision, options):
        unknown = set(options) - {"device_id"}
        if unknown:
            raise ValueError(f"Unknown CuPy device options: {sorted(unknown)}")
        devices = self.devices()
        selected = str(options.get("device_id", "0"))
        matches = [device for device in devices if device["device_id"] == selected]
        if not matches:
            raise RuntimeError(f"No GPU with device_id={selected}; CPU fallback is disabled")
        return _State(self.cp, matches[0], num_qubits, precision)


class CUDAProvider(_Provider):
    """NVIDIA CUDA provider, including native Windows."""


class ROCmProvider(_Provider):
    """AMD HIP provider, requiring a matching CuPy ROCm build and supported SDK."""

    hip = True


class CUDAEngine:
    api_version = 1
    provider = CUDAProvider
    engine = "cuda"

    def create_backend(self, **options):
        from qiskit_aer_york_qiskit import StatevectorBackend

        return StatevectorBackend(self.provider(), engine=self.engine, device="GPU", **options)


class ROCmEngine(CUDAEngine):
    provider = ROCmProvider
    engine = "rocm"


class _State:
    def __init__(self, cp, metadata, num_qubits, precision):
        self.cp = cp
        self.metadata = metadata
        self.device = cp.cuda.Device(int(metadata["device_id"]))
        self.dtype = np.complex128 if precision == "double" else np.complex64
        self.size = 1 << num_qubits
        self._matrices = OrderedDict()
        self.metadata = dict(metadata, matrix_uploads=0, matrix_upload_hits=0)
        with self.device:
            free, _ = cp.cuda.runtime.memGetInfo()
            free += cp.get_default_memory_pool().free_bytes()
            required = 2 * self.size * np.dtype(self.dtype).itemsize
            if required + 4112 > free:
                raise MemoryError("Statevector exceeds selected GPU free memory")
            # At most 32 small gate/target buffers, further limited by free memory.
            self._cache_limit = min(32, max(1, (free - required) // 4112))
            initial = np.zeros(self.size, dtype=self.dtype)
            initial[0] = 1
            self.current = cp.asarray(initial)
            self.spare = cp.empty_like(self.current)
            self.kernel = cp.RawKernel(
                KERNEL.format(real="double" if precision == "double" else "float"), "apply_matrix"
            )

    def apply(self, matrix, qubits):
        with self.device:
            host = np.ascontiguousarray(matrix, dtype=self.dtype)
            key = (tuple(qubits), host.shape, host.tobytes())
            if key in self._matrices:
                matrix, targets = self._matrices.pop(key)
                self.metadata["matrix_upload_hits"] += 1
            else:
                if len(self._matrices) >= self._cache_limit:
                    self._matrices.popitem(last=False)
                matrix = self.cp.asarray(host, dtype=self.dtype)
                targets = self.cp.asarray(qubits, dtype=self.cp.int32)
                self.metadata["matrix_uploads"] += 1
            self._matrices[key] = (matrix, targets)
            self.kernel(
                ((self.size + 127) // 128,),
                (128,),
                (
                    self.current,
                    self.spare,
                    matrix,
                    targets,
                    np.int32(len(qubits)),
                    np.uint64(self.size),
                ),
            )
            self.current, self.spare = self.spare, self.current

    def to_host(self):
        with self.device:
            return self.cp.asnumpy(self.current)

    def finish(self):
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()
