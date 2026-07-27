#!/usr/bin/env python3
"""Capture a reproducibility-focused hardware and software environment record.

The output deliberately omits host names, network addresses, device UUIDs,
serial numbers, environment secrets, and process listings.  It is suitable for
archiving alongside a public experiment release.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import locale
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "oats-runtime-environment-1.0.0"
PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "psutil",
    "pytest",
    "setuptools",
    "wheel",
    "pip",
)
ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "LANG",
    "LC_ALL",
    "TZ",
)


def run_command(command: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": list(command),
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def numpy_configuration() -> str | None:
    try:
        import numpy as np
    except ImportError:
        return None
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue().strip()


def cpu_affinity_count() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return len(getter(0))
    except OSError:
        return None


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def capture() -> dict[str, Any]:
    gpu_query = (
        "name,driver_version,memory.total,compute_cap,pstate,power.limit"
    )
    commands = {
        "lscpu_json": run_command(("lscpu", "--json")),
        "kernel": run_command(("uname", "-srvmo")),
        "libc": run_command(("ldd", "--version")),
        "gcc": run_command(("gcc", "--version")),
        "nvidia_smi": run_command(
            (
                "nvidia-smi",
                f"--query-gpu={gpu_query}",
                "--format=csv,noheader,nounits",
            )
        ),
        "cuda_compiler": run_command(("nvcc", "--version")),
        "python_pip_freeze": run_command(
            (sys.executable, "-m", "pip", "freeze", "--all")
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_policy": {
            "omitted": [
                "host name",
                "network address",
                "device UUID and serial number",
                "environment secrets",
                "process listing",
            ]
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_build": platform.python_build(),
            "python_compiler": platform.python_compiler(),
            "python_executable": sys.executable,
            "locale": locale.getlocale(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "affinity_count": cpu_affinity_count(),
            "processor": platform.processor(),
        },
        "memory": {
            "proc_meminfo": read_text(Path("/proc/meminfo")),
        },
        "container": {
            "proc_1_cgroup": read_text(Path("/proc/1/cgroup")),
        },
        "software": {
            "selected_package_versions": package_versions(),
            "numpy_configuration": numpy_configuration(),
        },
        "execution_environment": {
            key: os.environ.get(key) for key in ENVIRONMENT_ALLOWLIST
        },
        "command_snapshots": commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this file; omit to write to stdout.",
    )
    args = parser.parse_args()
    payload = json.dumps(
        capture(), indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
