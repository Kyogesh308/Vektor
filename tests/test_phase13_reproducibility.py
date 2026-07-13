"""
benchmarks/check_monotonicity.py
-----------------------------------
Validates expected monotonic relationships in benchmark CSV output.
A violation is either a real finding (rare) or a bug (common) — this
script forces you to look at every violation before it goes in the report.
"""

import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def check_monotonic_increasing(rows: list[dict], group_key: str,
                                sort_key: str, value_key: str,
                                tolerance: float = 0.0) -> list[str]:
    """Group rows by group_key, sort by sort_key, check value_key increases."""
    violations = []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    for group_name, group_rows in groups.items():
        sorted_rows = sorted(group_rows, key=lambda r: float(r[sort_key]))
        values = [float(r[value_key]) for r in sorted_rows]
        for i in range(1, len(values)):
            if values[i] < values[i-1] - tolerance:
                violations.append(
                    f"[{group_name}] {value_key} decreased from "
                    f"{values[i-1]:.4f} at {sort_key}={sorted_rows[i-1][sort_key]} "
                    f"to {values[i]:.4f} at {sort_key}={sorted_rows[i][sort_key]}"
                )
    return violations


def check_monotonic_decreasing(rows: list[dict], group_key: str,
                                sort_key: str, value_key: str,
                                tolerance: float = 0.0) -> list[str]:
    violations = []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    for group_name, group_rows in groups.items():
        sorted_rows = sorted(group_rows, key=lambda r: float(r[sort_key]))
        values = [float(r[value_key]) for r in sorted_rows]
        for i in range(1, len(values)):
            if values[i] > values[i-1] + tolerance:
                violations.append(
                    f"[{group_name}] {value_key} increased from "
                    f"{values[i-1]:.4f} at {sort_key}={sorted_rows[i-1][sort_key]} "
                    f"to {values[i]:.4f} at {sort_key}={sorted_rows[i][sort_key]}"
                )
    return violations


def main() -> None:
    all_violations = []

    # Exp 1: recall must increase with ef, per dataset
    exp1 = load_csv(RESULTS_DIR / "exp1_recall_vs_ef.csv")
    v = check_monotonic_increasing(exp1, "dataset", "ef", "recall_at_10", tolerance=0.01)
    all_violations.extend([f"Exp1 (recall vs ef): {x}" for x in v])

    # Exp 2: QPS must decrease with ef, per dataset
    exp2 = load_csv(RESULTS_DIR / "exp2_qps_vs_ef.csv")
    v = check_monotonic_decreasing(exp2, "dataset", "ef", "median_qps", tolerance=5.0)
    all_violations.extend([f"Exp2 (QPS vs ef): {x}" for x in v])

    # Exp 3: build time must increase with dataset size
    exp3 = load_csv(RESULTS_DIR / "exp3_build_time_vs_size.csv")
    v = check_monotonic_increasing(exp3, "dataset", "n_vectors", "median_build_time_s")
    all_violations.extend([f"Exp3 (build time vs size): {x}" for x in v])

    # Exp 4: memory must increase with dataset size
    exp4 = load_csv(RESULTS_DIR / "exp4_memory_vs_size.csv")
    v = check_monotonic_increasing(exp4, "dataset", "n_vectors", "measured_graph_mb")
    all_violations.extend([f"Exp4 (memory vs size): {x}" for x in v])

    # Exp 5: recall must increase with M
    exp5 = load_csv(RESULTS_DIR / "exp5_recall_vs_m.csv")
    v = check_monotonic_increasing(exp5, "dataset", "M", "recall_at_10", tolerance=0.01)
    all_violations.extend([f"Exp5 (recall vs M): {x}" for x in v])

    # Exp 6: speedup must exceed 1.0 for every metric
    exp6 = load_csv(RESULTS_DIR / "exp6_distance_speedup.csv")
    for row in exp6:
        if float(row["speedup"]) <= 1.0:
            all_violations.append(
                f"Exp6 (distance speedup): {row['metric']} speedup "
                f"{row['speedup']} <= 1.0 — pure-Python function may be "
                f"accidentally using NumPy"
            )

    if all_violations:
        print(f"\n{'='*70}\nMONOTONICITY VIOLATIONS FOUND: {len(all_violations)}\n{'='*70}")
        for v in all_violations:
            print(f"  - {v}")
        print("\nInvestigate each before writing results into the report.")
        sys.exit(1)
    else:
        print("All monotonicity checks passed. Results are ready for the report.")


if __name__ == "__main__":
    main()