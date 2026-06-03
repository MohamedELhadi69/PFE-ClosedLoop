import os
import subprocess
import sys
from pathlib import Path


EXPECTED_PYTHON = (3, 13)
BOOTSTRAP_ENV_VAR = "PFE_RUNTIME_BOOTSTRAPPED"
RUNTIME_ENV_VAR = "PFE_PYTHON_RUNTIME"
RUNTIME_CANDIDATES = [
    os.environ.get(RUNTIME_ENV_VAR),
    r"C:\Program Files\PostgreSQL\17\pgAdmin 4\python\python.exe",
]


def ensure_repo_python():
    if sys.version_info[:2] == EXPECTED_PYTHON:
        return

    if os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        expected = ".".join(str(value) for value in EXPECTED_PYTHON)
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            f"This project expects Python {expected} because .python_packages contains native wheels "
            f"for that version, but the current interpreter is {current} at {sys.executable}."
        )

    for candidate in RUNTIME_CANDIDATES:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            continue

        env = os.environ.copy()
        env[BOOTSTRAP_ENV_VAR] = "1"
        argv = [str(candidate_path), *sys.argv]
        if os.name == "nt":
            completed = subprocess.run(argv, env=env)
            raise SystemExit(completed.returncode)
        os.execve(str(candidate_path), argv, env)

    expected = ".".join(str(value) for value in EXPECTED_PYTHON)
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    raise RuntimeError(
        f"This project expects Python {expected}, but the current interpreter is {current} at "
        f"{sys.executable}. Set {RUNTIME_ENV_VAR} to a Python {expected} executable or install "
        f"matching dependencies for the current interpreter."
    )
