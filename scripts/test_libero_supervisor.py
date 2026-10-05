"""Check cross-process supervisor exclusion without starting models or servers."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_libero_protocols as runner


class TestSupervisorLock(unittest.TestCase):
    def test_second_process_is_excluded_and_lock_releases(self):
        program = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_libero_protocols as runner
runner.ART = Path(sys.argv[2])
try:
    lock = runner.acquire_supervisor_lock('evaluation')
except RuntimeError:
    sys.exit(43)
lock.close()
"""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "ART", Path(directory)),
        ):
            args = [
                sys.executable,
                "-c",
                program,
                str(Path(__file__).parent),
                directory,
            ]
            with runner.acquire_supervisor_lock("evaluation"):
                self.assertEqual(subprocess.run(args, check=False).returncode, 43)
            self.assertEqual(subprocess.run(args, check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
