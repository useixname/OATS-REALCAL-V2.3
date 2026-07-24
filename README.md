Exit code: 0
Wall time: 3.5 seconds
Output:
# OATS REAL-CAL-V2.3

This repository is the public research-code release for **OATS: Outcome-Anchored
Trust and Settlement for Online Value-Aware Crowdsensed Data Trading**. It
contains the mechanism implementation, REAL-CAL data-construction pipeline,
formal experiment driver, analysis code, and tests associated with run
`formal-realcal-v2.3-1.0.0`.

Repository: <https://github.com/useixname/OATS-REALCAL-V2.3>

## What is included

- `src/oats_v2/`: OATS mechanism, ledgers, settlement, trust, screening,
  calibration, REAL-CAL generation, experiment runner, and LP comparator.
- `scripts/`: data-profile construction, trace generation, V2.3 preflight,
  formal execution, audit generation, and result analysis.
- `tests/`: implementation, property, oracle, data, experiment, calibration,
  robustness, and REAL-CAL tests.
- `docs/`: the minimal data-layout and execution instructions required to use
  the released code.

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
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Frozen identity

The released copies of `scripts/run_formal_r4.py` and
`scripts/analyze_realcal_v2.py` are the scripts identified by the manuscript's
frozen V2.3 evidence record.

Earlier V2.x pilots and later experimental branches are intentionally excluded:
this repository is scoped to the V2.3 evidence used by the manuscript.

## License and data terms

The source code is released under the MIT License. Third-party datasets are not
covered by that license; obtain them from their original providers and comply
with their respective terms.

## Citation

Please cite the paper and the repository metadata in [`CITATION.cff`](CITATION.cff).

