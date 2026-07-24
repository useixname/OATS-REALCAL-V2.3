#!/usr/bin/env python3
"""REAL-CAL economics smoke: run a few E1 cells on one seed and print the
headline economics (net value, payments split, effort, screening, AUC)."""

from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oats_v2.experiments.formal_runner import simulate_cell
from src.oats_v2.experiments.lp_comparator import LPComparatorCache
from src.oats_v2.experiments.run_matrix import build_run_matrix_v2
from src.oats_v2.experiments.trace_loader import load_contracts, load_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "REAL-CAL")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["V2-FULL", "B-NODUAL", "B-MYOPIC", "B-RANDOM", "B-COST", "B-TRUST", "B-QUALITY"],
    )
    parser.add_argument("--budget", default="0.25")
    parser.add_argument("--family", default="E1_OVERALL")
    args = parser.parse_args()

    matrix = build_run_matrix_v2()
    cells = [
        c
        for c in matrix
        if c.seed == args.seed
        and c.family == args.family
        and str(c.budget_ratio) == args.budget
        and c.method_id in args.methods
    ]
    print("cells:", [c.method_id for c in cells], flush=True)
    trace = load_trace(
        args.seed,
        args.data_root,
        None,
        verify_hashes=False,
        load_eligibility_index=False,
        load_contracts_flag=False,
    )
    load_contracts(trace, load_gammas=frozenset({Decimal("0.3")}))
    for cell in cells:
        t0 = time.perf_counter()
        r = simulate_cell(cell, trace, LPComparatorCache(), compute_lp=False)
        dt = time.perf_counter() - t0
        print(
            f"{cell.method_id:10s} {dt:6.1f}s inv={r.invariant_status}"
            f" net={float(r.platform_net_value):12.1f}"
            f" gross={float(r.gross_external_value):10.1f}"
            f" paid={float(r.total_paid):10.1f}"
            f" (base={float(r.base_paid):.1f} score={float(r.score_paid):.1f})"
            f" purchased={r.purchased_count}"
            f" effort_hist={r.effort_histogram}"
            f" mrr={r.screening.mrr} soft={r.screening.soft_pass_rate}"
            f" auc={r.trust.auc}"
            f" mc_rho={r.mc_correlation} topk={r.mc_top_k_overlap}"
            f" rej={r.rejection_counts}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

