# Licensed under the Apache License, Version 2.0.
"""The full native bundle is installed only when this plugin is selected."""


class NativeAerEngine:
    api_version = 1

    def create_backend(self, **options):
        from qiskit_aer import AerSimulator

        return AerSimulator(**options)
