# OATS REAL-CAL

This repository is the public research-code release for **OATS: Outcome-Anchored
Trust and Settlement for Online Value-Aware Crowdsensed Data Trading**. It
contains the mechanism implementation, REAL-CAL data-construction pipeline,
formal experiment driver, repaired trust-feedback contract, calendar-time
feedback queue, published Oasis baseline adapter, analysis code, and tests.

Repository: <https://github.com/useixname/OATS-REALCAL>

## What is included

- `src/oats_v2/`: OATS mechanism, ledgers, settlement, trust, screening,
  calibration, REAL-CAL generation, experiment runner, and LP comparator.
- `src/oats_external/`: the isolated Oasis adapter used by the paper's
  published-baseline extension.
- `scripts/`: data-profile construction, trace generation, formal preflight,
  formal execution, delay replay, published-baseline execution, audit
  generation, environment capture, and result analysis.
- `tests/`: implementation, property, oracle, data, experiment, calibration,
  robustness, and REAL-CAL tests.
- `docs/`: the minimal data-layout and execution instructions required to use
  the released code.
- `environment/`: the frozen formal runtime record and exact direct package
  pins.

The raw third-party datasets and generated REAL-CAL trace files are not
redistributed. They remain subject to their original providers' terms and are
approximately 6.3 GB in the authors' working copy. See
[`docs/DATA.md`](docs/DATA.md) for the expected local layout and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full workflow.
Figure-rendering code, figure source tables, manuscript sources, frozen raw
results, internal audit packages, and later experimental branches are outside
the scope of this code release.

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,external]"
python -m pytest -q
```

## Frozen identity

Commit `2cc2ac64c98937bac9d7ac0ad3d94f0e5548126a` preserves the exact
trust-repaired formal source identified by the 830-cell evidence manifest.
Later commits add the calendar-time delayed-feedback replay and the frozen
Oasis published-baseline adapter without rewriting that history. The runtime
allocation, software versions, source hashes, and explicit archival gaps are
recorded in [`docs/RUNTIME_ENVIRONMENT.md`](docs/RUNTIME_ENVIRONMENT.md).

Earlier development runs, plotting code, private data, credentials, and
unrelated experimental branches are intentionally excluded.

## License and data terms

The source code is released under the MIT License. Third-party datasets are not
covered by that license; obtain them from their original providers and comply
with their respective terms.

## Citation

Please cite the paper and the repository metadata in [`CITATION.cff`](CITATION.cff).

