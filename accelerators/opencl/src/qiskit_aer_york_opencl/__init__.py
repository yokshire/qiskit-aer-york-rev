# Licensed under the Apache License, Version 2.0.
"""Native Windows/Linux OpenCL execution on NVIDIA, AMD and Intel GPUs."""

import numpy as np

KERNEL = r"""
{extension}
typedef {real}2 complex_t;
__kernel void apply_matrix(__global const complex_t *input,
                           __global complex_t *output,
                           __global const complex_t *matrix,
                           __global const int *qubits,
                           const int width, const ulong size) {{
    ulong index = get_global_id(0);
    if (index >= size) return;
    ulong base = index;
    int row = 0;
    for (int q = 0; q < width; ++q) {{
        row |= ((index >> qubits[q]) & 1UL) << q;
        base &= ~(1UL << qubits[q]);
    }}
    complex_t sum = (complex_t)(0, 0);
    for (int column = 0; column < (1 << width); ++column) {{
        ulong source = base;
        for (int q = 0; q < width; ++q)
            source |= ((ulong)((column >> q) & 1)) << qubits[q];
        complex_t a = matrix[row * (1 << width) + column], b = input[source];
        sum += (complex_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }}
    output[index] = sum;
}}
"""


def _vendor(raw):
    raw = raw.lower()
    if "nvidia" in raw:
        return "nvidia"
    if "amd" in raw or "advanced micro" in raw:
        return "amd"
    if "intel" in raw:
        return "intel"
    return raw


class OpenCLProvider:
    """Accelerator API 1 provider. Only GPU devices are exposed."""

    api_version = 1

    def __init__(self):
        import pyopencl

        self.cl = pyopencl

    def _devices(self):
        devices = []
        for platform_index, platform in enumerate(self.cl.get_platforms()):
            for device_index, device in enumerate(platform.get_devices()):
                if device.type & self.cl.device_type.GPU:
                    devices.append((f"{platform_index}:{device_index}", device))
        return devices

    def devices(self):
        """Return stable-in-process IDs, driver details and precision capabilities."""
        return [self._info(device_id, device) for device_id, device in self._devices()]

    @staticmethod
    def _info(device_id, device):
        return {
            "device_id": device_id,
            "name": device.name.strip(),
            "vendor": _vendor(device.vendor),
            "device": "GPU",
            "runtime": "OpenCL",
            "driver_version": device.driver_version,
            "fp64": bool(device.double_fp_config),
            "global_memory_bytes": int(device.global_mem_size),
        }

    def create_state(self, num_qubits, precision, options):
        unknown = set(options) - {"vendor", "device_id"}
        if unknown:
            raise ValueError(f"Unknown OpenCL device options: {sorted(unknown)}")
        devices = self._devices()
        vendor = options.get("vendor")
        if vendor is not None:
            devices = [(key, dev) for key, dev in devices if _vendor(dev.vendor) == vendor.lower()]
        device_id = options.get("device_id")
        if device_id is not None:
            devices = [(key, dev) for key, dev in devices if key == str(device_id)]
        if not devices:
            raise RuntimeError(f"No OpenCL GPU matches {options}; CPU fallback is disabled")
        if precision == "double":
            devices = [(key, dev) for key, dev in devices if dev.double_fp_config]
            if not devices:
                raise RuntimeError("Selected GPU lacks FP64; explicitly choose precision='single'")
        device_id, device = devices[0]
        return _State(self.cl, device, self._info(device_id, device), num_qubits, precision)


class _State:
    def __init__(self, cl, device, metadata, num_qubits, precision):
        self.cl = cl
        self.metadata = metadata
        self.dtype = np.complex128 if precision == "double" else np.complex64
        self.size = 1 << num_qubits
        size_bytes = self.size * np.dtype(self.dtype).itemsize
        if size_bytes > device.max_mem_alloc_size or 2 * size_bytes + 4096 > device.global_mem_size:
            raise MemoryError("Statevector exceeds selected GPU memory limits")
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        extension = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" if precision == "double" else ""
        source = KERNEL.format(
            extension=extension, real="double" if precision == "double" else "float"
        )
        program = cl.Program(self.context, source).build()
        self.kernel = cl.Kernel(program, "apply_matrix")
        flags = cl.mem_flags
        initial = np.zeros(self.size, dtype=self.dtype)
        initial[0] = 1
        self.current = cl.Buffer(
            self.context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=initial
        )
        self.spare = cl.Buffer(self.context, flags.READ_WRITE, size=size_bytes)

    def apply(self, matrix, qubits):
        cl = self.cl
        matrix = np.ascontiguousarray(matrix, dtype=self.dtype)
        qubits_array = np.asarray(qubits, dtype=np.int32)
        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        matrix_buffer = cl.Buffer(self.context, flags, hostbuf=matrix)
        qubits_buffer = cl.Buffer(self.context, flags, hostbuf=qubits_array)
        self.kernel(
            self.queue,
            (self.size,),
            None,
            self.current,
            self.spare,
            matrix_buffer,
            qubits_buffer,
            np.int32(len(qubits)),
            np.uint64(self.size),
        )
        self.current, self.spare = self.spare, self.current

    def to_host(self):
        output = np.empty(self.size, dtype=self.dtype)
        self.cl.enqueue_copy(self.queue, output, self.current).wait()
        return output
