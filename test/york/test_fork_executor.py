# Licensed under the Apache License, Version 2.0; see LICENSE.txt.
# York regression adapted from the fork fix in upstream Qiskit Aer PR #2486.
"""A warmed default executor must work in a forked child without changing the parent."""

import os
import subprocess
import sys
import textwrap
import unittest
from concurrent.futures import ThreadPoolExecutor

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.jobs import AerJob


class TestForkExecutor(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_child_gets_fresh_executor(self):
        code = textwrap.dedent("""
            import os
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            from qiskit_aer.jobs import utils
            circuit = QuantumCircuit(1, 1)
            circuit.x(0)
            circuit.measure(0, 0)
            backend = AerSimulator(max_parallel_threads=1)
            assert backend.run(circuit, shots=8).result(timeout=10).success
            parent_executor = utils.DEFAULT_EXECUTOR
            pid = os.fork()
            if pid == 0:
                try:
                    result = backend.run(circuit, shots=8).result(timeout=8)
                    assert result.success and result.get_counts() == {'1': 8}
                    assert utils.DEFAULT_EXECUTOR is not parent_executor
                except BaseException as error:
                    print(type(error).__name__, str(error), flush=True)
                    os._exit(1)
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0, status
            assert utils.DEFAULT_EXECUTOR is parent_executor
            assert backend.run(circuit, shots=8).result(timeout=10).get_counts() == {'1': 8}
            """)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=40
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_custom_executor_is_preserved(self):
        with ThreadPoolExecutor(1) as executor:
            job = AerJob(AerSimulator(), "custom-executor", lambda *args: 8, executor=executor)
            self.assertIs(job._executor, executor)
            job.submit()
            self.assertEqual(job.result(timeout=10), 8)


if __name__ == "__main__":
    unittest.main()
