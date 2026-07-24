# REAL-CAL reproducibility workflow

This workflow reconstructs the formal evaluation in a fresh output directory.
Do not use it to overwrite an existing frozen result tree.

## 1. Environment

Python 3.11 is recommended:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Prepare the third-party data and traces as described in [`DATA.md`](DATA.md).

## 2. Tests and preflight

```bash
python -m pytest -q
python scripts/preflight_realcal.py
```

The preflight is fail-closed: a failed acceptance condition blocks the formal
matrix.

## 3. Formal online run

`--workers 0` uses all detected CPU cores. Reduce this value if memory pressure
or a shared machine requires a lower process count.

```bash
python scripts/run_formal_r4.py \
  --matrix v2 \
  --data-root data/REAL-CAL \
  --trace-hashes data/REAL-CAL/trace_hashes_realcal.json \
  --output-root results/reproduce_realcal \
  --run-version formal-realcal \
  --seeds 20260715 20260716 20260717 20260718 20260719 \
          20260720 20260721 20260722 20260723 20260724 \
  --workers 0
```

The runner is resumable by default. It writes checkpoints and raw results only
under the selected output root.

## 4. Realized-cost LP post-pass

Refresh the LP comparator fields without changing the completed online
decisions:

```bash
python scripts/run_formal_r4.py \
  --matrix v2 \
  --data-root data/REAL-CAL \
  --trace-hashes data/REAL-CAL/trace_hashes_realcal.json \
  --output-root results/reproduce_realcal \
  --run-version formal-realcal \
  --seeds 20260715 20260716 20260717 20260718 20260719 \
          20260720 20260721 20260722 20260723 20260724 \
  --workers 0 \
  --lp-refresh
```

## 5. Aggregate analysis

```bash
python scripts/analyze_realcal.py \
  --results results/reproduce_realcal \
  --out reports/reproduce_realcal
```

The manuscript uses paired seeds as the sampling unit. Primary contrasts use
paired differences and 10,000 paired-bootstrap resamples with bootstrap seed
20260715. Descriptive grids and curves are not multiplicity-controlled tests.

