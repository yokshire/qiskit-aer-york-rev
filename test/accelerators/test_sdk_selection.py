# Licensed under the Apache License, Version 2.0.
"""Vendor/runtime guards without a required vendor SDK or physical device."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qiskit_aer_york_cupy import CUDAProvider, ROCmProvider
from qiskit_aer_york_opencl import OpenCLProvider


class SDKSelectionTests(unittest.TestCase):
    def test_cuda_and_hip_builds_are_not_interchangeable(self):
        for hip in (False, True):
            cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=hip)))
            with patch.dict(sys.modules, {"cupy": cupy}):
                matching = ROCmProvider if hip else CUDAProvider
                differing = CUDAProvider if hip else ROCmProvider
                self.assertEqual(matching().hip, hip)
                with self.assertRaisesRegex(RuntimeError, "build"):
                    differing()

    def provider(self):
        cpu = SimpleNamespace(type=2, vendor="Intel", name="CPU", double_fp_config=63)
        gpu = SimpleNamespace(
            type=4,
            vendor="Intel(R) Corporation",
            name="Intel GPU",
            double_fp_config=0,
            driver_version="test-driver",
            global_mem_size=1024,
        )
        platform = SimpleNamespace(get_devices=lambda: [cpu, gpu])
        provider = OpenCLProvider.__new__(OpenCLProvider)
        provider.cl = SimpleNamespace(
            get_platforms=lambda: [platform], device_type=SimpleNamespace(GPU=4)
        )
        return provider

    def test_opencl_exposes_gpu_only(self):
        devices = self.provider().devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["device_id"], "0:1")
        self.assertEqual(devices[0]["vendor"], "intel")
        self.assertFalse(devices[0]["fp64"])

    def test_missing_vendor_and_device_fail(self):
        for selector in ({"vendor": "amd"}, {"device_id": "9:9"}):
            with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                self.provider().create_state(2, "single", selector)

    def test_fp64_does_not_silently_downgrade(self):
        with self.assertRaisesRegex(RuntimeError, "FP64"):
            self.provider().create_state(2, "double", {"vendor": "intel"})

    def test_unknown_selector_fails(self):
        with self.assertRaises(ValueError):
            self.provider().create_state(2, "single", {"typo": 3})


if __name__ == "__main__":
    unittest.main()
