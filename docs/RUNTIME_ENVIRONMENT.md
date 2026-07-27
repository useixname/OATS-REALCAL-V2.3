# Formal runtime environment archive

This document records the environment that produced the repaired, frozen
830-cell REAL-CAL evaluation. The machine-readable record is
[`environment/formal_runtime_environment.json`](../environment/formal_runtime_environment.json);
the exact direct runtime package pins are in
[`environment/formal-requirements-lock.txt`](../environment/formal-requirements-lock.txt).

## Formal execution identity

| Item | Archived value |
|---|---|
| Run | `formal-realcal-trustfix-1.0.0` |
| Matrix | 830 completed cells, 0 invalid cells |
| Seeds | `20260715` through `20260724` |
| Worker processes | 25 |
| Exact formal-source commit | `2cc2ac64c98937bac9d7ac0ad3d94f0e5548126a` |
| Evidence-manifest SHA-256 | `70ec46458909f969d73e2ae3ae4eba3993a5f8a386503c8bfdc25ea5b4444d7a` |

The source hashes stored in the frozen evidence manifest were independently
checked against the exact formal-source commit before the later queue-replay
and published-baseline code was added.

## Hardware allocation

| Component | Archived value | Evidence class |
|---|---|---|
| CPU allocation | 25 cgroup quota cores; 128 logical CPUs visible | Remote preflight |
| CPU model | Intel Xeon Gold 6459C | Provider allocation declared by the job owner; no frozen `lscpu` record |
| Memory | 96,636,764,160 bytes (90 GiB) | Remote cgroup preflight |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB | Formal authorization |
| GPU use | None | Formal execution policy and CPU/SciPy code path |
| Free storage at preflight | 46,016,823,296 bytes | Remote preflight |

The formal path was CPU/SciPy-bound. Each of `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` was set to
`1`, while the driver used 25 worker processes. The requested negative nice
level was not applied; the launch log records a permission denial.

## Software stack

| Component | Frozen value |
|---|---|
| Platform | Linux 5.15.0-78-generic, x86-64, glibc 2.35 |
| Python | 3.10.12, GCC 11.4.0 build |
| Interpreter | `/usr/bin/python3` |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |

NumPy and SciPy are the direct third-party imports on the formal execution
path. Other imports are from the Python standard library or this repository.
The generic capture utility
[`scripts/capture_runtime_environment.py`](../scripts/capture_runtime_environment.py)
is included so future reruns record the broader system state before execution.

## Evidence boundary

This is a reconstruction from records frozen during the run, not a later
snapshot of a different machine. The original host was unavailable when the
archive was completed, so a full `pip freeze`, `/etc/os-release`, complete
`lscpu` topology and microcode, GPU driver/CUDA versions, NumPy BLAS build
configuration, and filesystem layout cannot be recovered. These omissions are
listed machine-readably rather than being silently inferred.

Accordingly, the paper's timing results remain component bottleneck
diagnostics. The archive does not make method timings with different cell
mixtures a controlled cross-method speed comparison.
