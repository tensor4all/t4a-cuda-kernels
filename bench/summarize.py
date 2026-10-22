#!/usr/bin/env python3
"""Paired summary of the frozen A100 performance experiment.

Reads the raw `nsys` exports and harness `BENCH_META` lines written by
`bench/run-paired.sh` and prints, per case and kernel:

  * the per-instance device time (nsys `cuda_gpu_kern_sum` "Med (ns)") for the
    candidate and the baseline, median over the process repetitions, with the
    min-max spread of each side,
  * the candidate/baseline ratio,
  * validity (a case is invalid when either side's max/min over the repetitions
    exceeds 1.15, the predeclared rule) and the regression flag (valid and the
    candidate median exceeds 1.10 x the baseline median),
  * for composite cases the per-iteration total device time.

Three rules are enforced rather than assumed:

1. **Declared cases only.** The case list is the campaign's declared list (from
   `environment.txt`, else `--cases`), never the set of stdout files that happen
   to exist. A declared case with a missing repetition is an incomplete
   campaign, not a smaller sample.
2. **Explicit kernel pairing.** The two sides name the same kernels differently
   (`t4a_cuda_add` versus CubeCL's mangled `add_float_f_f32_...`), and sorting by
   device time pairs them by cost, which mis-pairs exactly the composite cases
   (RRQR above all). `PAIRING` below declares, per case, which baseline symbol
   corresponds to which `t4a_cuda_*` entry; every observed kernel must be
   covered by exactly one declared row and the two sides' per-iteration instance
   counts must agree. A missing or unpaired kernel, or a count mismatch, makes
   the case **invalid-pairing**; nothing is ever reordered silently.
3. **Complete campaigns only.** Every declared case x repetition x side must have
   its `.stdout`, `.kern.csv` and `.api.csv`, and every repetition must carry a
   `BENCH_META` line. Otherwise the summariser prints `CAMPAIGN INCOMPLETE` with
   the missing artifacts and exits non-zero without any tables, so no verdict can
   be derived from fewer repetitions than declared.

Only the Python standard library is used. Raw exports stay outside the
repository, next to this script's output.
"""

import argparse
import csv
import os
import statistics
import sys

ITERS = 100
VALIDITY_SPREAD = 1.15
REGRESSION_RATIO = 1.10

FAMILY_ORDER = [
    "elementwise",
    "fusion",
    "prepare",
    "layout",
    "reduce",
    "indexing",
    "linalg",
]

# The frozen case list of `docs/worklogs/2026-10-21-complete-port-perf.md` §4.1.
DECLARED_CASES = [
    "elementwise-add-f32-1000",
    "elementwise-add-f32-1000000",
    "elementwise-add-f32-100000000",
    "elementwise-exp-f32-10000000",
    "elementwise-mul-c64-10000000",
    "elementwise-div-c64-10000000",
    "fusion-chain-f32-1000",
    "fusion-chain-f32-10000000",
    "fusion-add-mul-c64-10000000",
    "fusion-two-live-outs-f32-10000000",
    "prepare-fused-cold-compile",
    "layout-copy-f32-4096sq",
    "layout-transpose-f32-4096sq",
    "layout-transpose-f32-4093x4099",
    "layout-transpose-f32-4099x4093",
    "layout-transpose-offset-f32-4096sq",
    "layout-transpose-batched-f32-64x256x256",
    "layout-transpose-32x8-p1-v1-f32-4096sq",
    "layout-transpose-16x8-p1-v2-f32-4096sq",
    "layout-materialize-negative-stride-f32-4096sq",
    "layout-slice-f32-4096sq-set1",
    "layout-slice-f32-4096sq-set2",
    "layout-slice-f32-4096sq-set3",
    "layout-slice-f32-4096sq-set4",
    "layout-pad-f32-4096sq-set1",
    "layout-pad-f32-4096sq-set2",
    "layout-pad-f32-4096sq-set3",
    "layout-pad-f32-4096sq-set4",
    "reduce-sum-f32-axis-511",
    "reduce-sum-f32-axis-512",
    "reduce-sum-f32-axis-513",
    "reduce-sum-f32-axis-600",
    "reduce-sum-f32-4096sq-axis0",
    "reduce-sum-f32-4096sq-axis1",
    "reduce-sum-f64-4096sq-axis0",
    "reduce-sum-c64-4096sq-axis0",
    "reduce-max-i32-4096sq-axis0",
    "reduce-sum-f32-full-1000000",
    "reduce-sum-f32-full-10000000",
    "indexing-gather-f32-1024x256-table1000",
    "indexing-gather-f32-1024x256-table100000",
    "indexing-gather-f32-1024x256-table1000000",
    "indexing-scatter-add-f32-1e6-into-16",
    "indexing-scatter-add-c64-1e6-into-16",
    "linalg-rrqr-256x256-f64",
    "linalg-rrqr-256x256-c64",
    "linalg-rrqr-1024x256-f64",
    "linalg-rrqr-1024x256-c64",
    "linalg-lu-apply-pivots-1024-f64",
    "linalg-qr-apply-phase-1024x256-f64",
    "linalg-qr-apply-phase-1024x256-c64",
    "linalg-householder-explicit-v-1024x256-f64",
    "linalg-householder-explicit-v-1024x256-c64",
]

# --- Declared kernel correspondence ----------------------------------------
#
# One row per (baseline kernel symbol prefix, candidate entry). The baseline's
# CubeCL kernels carry a mangled suffix (`add_float_f_f32_a70f4883`,
# `column_norms_c64_6973f0d6`), so a row matches by prefix; the longest matching
# prefix wins, which keeps `reduce_sum_float_chunked*` from being read as
# `reduce_sum_float*`. A candidate entry is our own ABI symbol and matches
# exactly. `None` on the baseline side declares a candidate-side helper that the
# baseline arm has no equivalent of.
#
# Instance counts: for every candidate entry, the observed per-iteration instance
# count must equal the sum, over the declared rows mapping to it, of the
# corresponding baseline kernels' per-iteration instance count. A reduction
# therefore declares three baseline "combine" rows (`*_plane`, `*_float_t_`,
# `*_complex`) of which exactly one is present per case, next to its `chunked`
# pass-1 row.

REDUCE_ROWS = [
    ("reduce_sum_complex_chunked", "t4a_cuda_reduce_partials"),
    ("reduce_sum_float_chunked", "t4a_cuda_reduce_partials"),
    ("reduce_max_int_chunked", "t4a_cuda_reduce_partials"),
    ("reduce_sum_complex", "t4a_cuda_reduce"),
    ("reduce_sum_float_plane", "t4a_cuda_reduce"),
    ("reduce_sum_float", "t4a_cuda_reduce"),
    ("reduce_max_int", "t4a_cuda_reduce"),
]

QR_APPLY_PHASE_ROWS = [
    ("qr_phase", "t4a_cuda_qr_phase"),
    ("qr_apply_phase", "t4a_cuda_qr_apply_phase"),
]

RRQR_ROWS = [
    ("initialize_permutation", "t4a_cuda_initialize_permutation"),
    ("column_norms", "t4a_cuda_column_norms"),
    ("select_pivot", "t4a_cuda_select_pivot"),
    ("swap_columns", "t4a_cuda_swap_columns"),
    ("reflector", "t4a_cuda_reflector"),
    ("apply_reflector", "t4a_cuda_apply_reflector"),
    ("extract_r", "t4a_cuda_extract_r"),
    ("initialize_q", "t4a_cuda_identity_shifted"),
    ("prefix_rank", "t4a_cuda_prefix_rank"),
    # Revision 8 requires `apply_reflector`'s packed reflector storage and its
    # target to be disjoint allocations, so the candidate arm copies each step's
    # reflector vector out of the work matrix into its own buffer (one `copy_exact`
    # of `m - step` contiguous elements per step with a trailing block). The
    # baseline arm applies the reflector in place and has no equivalent kernel, so
    # this row is declared a candidate-side helper rather than being paired with a
    # baseline kernel.
    (None, "t4a_cuda_copy_exact"),
]

PAIRING = {}
for _case in DECLARED_CASES:
    if _case.startswith("elementwise-add"):
        PAIRING[_case] = [("add_float", "t4a_cuda_add")]
    elif _case.startswith("elementwise-exp"):
        PAIRING[_case] = [("exp_float", "t4a_cuda_unary")]
    elif _case.startswith("elementwise-mul"):
        PAIRING[_case] = [("mul_complex", "t4a_cuda_mul")]
    elif _case.startswith("elementwise-div"):
        # Revision 8 keeps `t4a_cuda_div` for F32/F64 only; the complex route is
        # `t4a_cuda_binary`'s `DIV_COMPLEX` tag (the scale-robust form the
        # baseline's `div_complex_parts` kernel implements).
        PAIRING[_case] = [("div_complex_parts", "t4a_cuda_binary")]
    elif _case.startswith("fusion-") or _case.startswith("prepare-"):
        PAIRING[_case] = [("tenferro_fused_elementwise", "t4a_fused_elementwise")]
    elif _case == "layout-copy-f32-4096sq":
        PAIRING[_case] = [("materialize_strided_kernel", "t4a_cuda_copy_exact")]
    elif _case.startswith("layout-materialize-negative-stride"):
        PAIRING[_case] = [("materialize_strided_kernel", "t4a_cuda_materialize_strided")]
    elif _case.startswith("layout-transpose"):
        PAIRING[_case] = [("tiled_transpose_kernel", "t4a_cuda_tiled_transpose")]
    elif _case.startswith("layout-slice"):
        PAIRING[_case] = [("slice_kernel", "t4a_cuda_slice")]
    elif _case.startswith("layout-pad"):
        PAIRING[_case] = [("pad_kernel", "t4a_cuda_pad")]
    elif _case.startswith("reduce-"):
        PAIRING[_case] = REDUCE_ROWS
    elif _case.startswith("indexing-gather"):
        PAIRING[_case] = [("gather_kernel", "t4a_cuda_gather")]
    elif _case.startswith("indexing-scatter-add"):
        PAIRING[_case] = [
            ("scatter_copy_kernel", "t4a_cuda_copy_exact"),
            ("scatter_float_kernel", "t4a_cuda_scatter_add"),
            ("scatter_complex_kernel", "t4a_cuda_scatter_add"),
        ]
    elif _case.startswith("linalg-rrqr"):
        PAIRING[_case] = RRQR_ROWS
    elif _case.startswith("linalg-lu-apply-pivots"):
        PAIRING[_case] = [("lu_apply_pivots", "t4a_cuda_lu_apply_pivots")]
    elif _case.startswith("linalg-qr-apply-phase"):
        PAIRING[_case] = QR_APPLY_PHASE_ROWS
    elif _case.startswith("linalg-householder-explicit-v"):
        PAIRING[_case] = [("householder_explicit_v", "t4a_cuda_householder_explicit_v")]
    else:  # pragma: no cover - a case without a declared correspondence
        PAIRING[_case] = []


# --- Campaign metadata ------------------------------------------------------


def parse_environment(root):
    """`environment.txt` fields written by `bench/run-paired.sh`."""
    fields = {}
    path = os.path.join(root, "environment.txt")
    if not os.path.exists(path):
        return fields
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key, _, value = line.rstrip("\n").partition("=")
            if key:
                fields[key] = value
    return fields


def parse_meta(path):
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("BENCH_META "):
                for field in line.split(None, 1)[1].strip().split(" "):
                    key, _, value = field.partition("=")
                    meta[key] = value
            elif line.startswith("BENCH_PREPARE "):
                for field in line.split(None, 1)[1].strip().split(" "):
                    key, _, value = field.partition("=")
                    meta["prepare_" + key] = value
    return meta


def parse_kern_sum(path):
    """Return {kernel: row} from a `cuda_gpu_kern_sum` CSV export.

    `nsys` emits one row per kernel symbol, so a name normally appears once. If a
    report does split one symbol over several rows, they are aggregated (instances
    and total time summed, per-instance time derived from them) instead of being
    dropped or counted as several samples.
    """
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    start = content.find("Time (%)")
    if start < 0:
        return rows
    reader = csv.reader(content[start:].splitlines())
    header = None
    for record in reader:
        if not record:
            continue
        if header is None:
            header = [field.strip() for field in record]
            continue
        if len(record) != len(header):
            continue
        entry = dict(zip(header, (field.strip() for field in record)))
        name = entry.get("Kernel Name") or entry.get("Name")
        if not name:
            continue
        try:
            med = float(entry.get("Med (ns)", "0") or 0)
            total = float(entry.get("Total Time (ns)", "0") or 0)
            instances = float(entry.get("Instances", "0") or 0)
        except ValueError:
            continue
        row = rows.setdefault(
            name, {"med": 0.0, "total": 0.0, "instances": 0.0, "rows": 0}
        )
        row["total"] += total
        row["instances"] += instances
        row["med"] += med
        row["rows"] += 1
    for row in rows.values():
        # A single row keeps nsys' per-instance median; several rows for one
        # symbol are aggregated into their per-instance mean.
        if row["rows"] > 1:
            row["med"] = (
                row["total"] / row["instances"] if row["instances"] > 0 else float("nan")
            )
    return rows


def median(values):
    return statistics.median(values) if values else float("nan")


def spread(values):
    if not values:
        return (float("nan"), float("nan"))
    return (min(values), max(values))


def fmt_ms(ns):
    if ns != ns:
        return "-"
    return "%.1f" % (ns / 1e3)


# --- Completeness -----------------------------------------------------------


def missing_artifacts(root, cases, sides, reps):
    """Every declared case x side x repetition artifact the summariser needs."""
    missing = []
    for side in sides:
        for case in cases:
            for rep in range(1, reps + 1):
                base = os.path.join(root, side, "%s.rep%d" % (case, rep))
                for suffix in (".stdout", ".kern.csv", ".api.csv"):
                    if not os.path.exists(base + suffix):
                        missing.append(os.path.relpath(base + suffix, root))
                        continue
                    if suffix == ".stdout" and not parse_meta(base + suffix):
                        missing.append(
                            os.path.relpath(base + suffix, root) + " (no BENCH_META)"
                        )
    return missing


# --- Side summaries ---------------------------------------------------------


def summarise_side(root, side, case, reps):
    kernels = {}
    metas = []
    complete = True
    for rep in range(1, reps + 1):
        base = os.path.join(root, side, "%s.rep%d" % (case, rep))
        metas.append(parse_meta(base + ".stdout"))
        for name, entry in parse_kern_sum(base + ".kern.csv").items():
            kernels.setdefault(name, {})[rep] = entry
    for name, per_rep in kernels.items():
        if len(per_rep) != reps:
            complete = False
    return kernels, metas, complete


def instance_rate(per_rep, reps):
    """Per-iteration instance count over the whole campaign, from `instances`."""
    total = sum(entry["instances"] for entry in per_rep.values())
    return total / max(reps * ITERS, 1)


def row_flags(base_values, cand_values):
    base_median = median(base_values)
    cand_median = median(cand_values)
    base_min, base_max = spread(base_values)
    cand_min, cand_max = spread(cand_values)
    validity = float("nan")
    if base_min > 0 and cand_min > 0:
        validity = max(base_max / base_min, cand_max / cand_min)
    valid = validity == validity and validity <= VALIDITY_SPREAD
    ratio = cand_median / base_median if base_median > 0 else float("nan")
    regressed = bool(valid and ratio == ratio and ratio > REGRESSION_RATIO)
    return {
        "base_median": base_median,
        "cand_median": cand_median,
        "base_min": base_min,
        "base_max": base_max,
        "cand_min": cand_min,
        "cand_max": cand_max,
        "validity": validity,
        "valid": valid,
        "ratio": ratio,
        "regressed": regressed,
    }


def api_launch_median(root, side, case, reps):
    values = []
    for rep in range(1, reps + 1):
        path = os.path.join(root, side, "%s.rep%d.api.csv" % (case, rep))
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        start = content.find("Time (%)")
        if start < 0:
            continue
        reader = csv.reader(content[start:].splitlines())
        header = None
        for record in reader:
            if not record:
                continue
            if header is None:
                header = [field.strip() for field in record]
                continue
            if len(record) != len(header):
                continue
            entry = dict(zip(header, (field.strip() for field in record)))
            if entry.get("Name") != "cuLaunchKernel":
                continue
            try:
                values.append(float(entry.get("Med (ns)", "0") or 0))
            except ValueError:
                continue
    return median(values)


# --- Pairing ----------------------------------------------------------------


def pair_case(case, base_kernels, cand_kernels, reps):
    """Pair the two sides through the declared table; return rows + problems.

    Baseline kernels are matched to declared rows by their longest matching
    symbol prefix; candidate entries are matched by exact ABI symbol. Per
    candidate entry the *sum* of the declared baseline rows' per-iteration
    instance counts must equal the candidate's own count: a reduction declares
    its `chunked` pass-1 row and every possible combine row, of which exactly one
    is present in a given case.
    """
    rows = PAIRING.get(case, [])
    problems = []

    assigned = {}
    for name in base_kernels:
        best = None
        for index, (prefix, _) in enumerate(rows):
            if prefix is not None and name.startswith(prefix):
                if best is None or len(prefix) > len(rows[best][0]):
                    best = index
        if best is None:
            problems.append("baseline kernel `%s` has no declared correspondence" % name)
        else:
            assigned.setdefault(best, []).append(name)

    groups = {}
    for index, (_, candidate) in enumerate(rows):
        groups.setdefault(candidate, []).append(index)

    for name in cand_kernels:
        if name not in groups:
            problems.append("candidate kernel `%s` has no declared correspondence" % name)

    def values_of(per_rep):
        return [entry["med"] for entry in per_rep.values()]

    kernel_rows = []
    for candidate, indices in groups.items():
        helper = all(rows[index][0] is None for index in indices)
        present = [
            (index, name)
            for index in indices
            for name in assigned.get(index, [])
        ]
        cand_per_rep = cand_kernels.get(candidate, {})
        cand_rate = instance_rate(cand_per_rep, reps) if cand_per_rep else 0.0
        base_rate = sum(instance_rate(base_kernels[name], reps) for _, name in present)
        if not helper:
            if abs(base_rate - cand_rate) > 1e-6:
                problems.append(
                    "%s: %s vs %s submit %.3f/iter against %.3f/iter"
                    % (
                        case,
                        ", ".join(sorted({name for _, name in present})) or "<absent>",
                        candidate,
                        base_rate,
                        cand_rate,
                    )
                )
            elif not present:
                problems.append(
                    "%s: neither side submits the declared %s row" % (case, candidate)
                )
        elif not cand_per_rep:
            problems.append("%s: declared helper %s is missing" % (case, candidate))

        if cand_per_rep and not present and helper:
            # A declared candidate-side helper has no baseline counterpart, so it
            # has no ratio and cannot be a regression; it still contributes its
            # device time to the candidate total.
            flags = row_flags([], values_of(cand_per_rep))
            flags["valid"] = True
            flags["regressed"] = False
            kernel_rows.append(
                (
                    "- -> %s (candidate-side helper)" % candidate,
                    flags,
                    0.0,
                    cand_rate,
                )
            )
            continue
        for index, name in present:
            if not cand_per_rep:
                problems.append(
                    "%s: baseline kernel `%s` has no candidate submission" % (case, name)
                )
                continue
            kernel_rows.append(
                (
                    "%s -> %s%s"
                    % (name, candidate, " (candidate-side helper)" if helper else ""),
                    row_flags(values_of(base_kernels[name]), values_of(cand_per_rep)),
                    instance_rate(base_kernels[name], reps),
                    cand_rate,
                )
            )
    return kernel_rows, problems


# --- Report -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="directory with the raw nsys exports")
    parser.add_argument("--reps", type=int, default=None, help="process repetitions per side")
    parser.add_argument("--cases", default=None, help="comma-separated declared case list")
    parser.add_argument(
        "--sides",
        default="baseline,candidate",
        help="comma-separated sides the campaign must contain",
    )
    args = parser.parse_args()

    environment = parse_environment(args.root)
    if args.cases:
        cases = [case for case in args.cases.split(",") if case]
    elif environment.get("cases"):
        cases = [case for case in environment["cases"].split(",") if case]
    elif environment:
        cases = []
        print("CAMPAIGN INCOMPLETE: environment.txt declares no case list")
        return 1
    else:
        cases = DECLARED_CASES
    sides = [side for side in args.sides.split(",") if side]
    reps = args.reps
    if reps is None:
        reps = int(environment.get("reps", "0") or 0) or 0
    if reps < 1:
        print("CAMPAIGN INCOMPLETE: no repetition count (needs --reps or environment.txt)")
        return 1

    unknown = [case for case in cases if case not in PAIRING]
    if unknown:
        print("CAMPAIGN INCOMPLETE: cases without a declared correspondence: %s"
              % ", ".join(unknown))
        return 1

    missing = missing_artifacts(args.root, cases, sides, reps)
    if missing:
        print("CAMPAIGN INCOMPLETE: %d declared artifact(s) are missing" % len(missing))
        for path in missing:
            print("  missing: %s" % path)
        print("")
        print(
            "No verdict is reported: every declared case x repetition x side is "
            "required, and a smaller sample is not a valid pair."
        )
        return 1

    order = {case: index for index, case in enumerate(cases)}
    cases = sorted(
        cases,
        key=lambda case: (
            FAMILY_ORDER.index(case.split("-")[0])
            if case.split("-")[0] in FAMILY_ORDER
            else len(FAMILY_ORDER),
            order[case],
        ),
    )

    lines = []
    lines.append("# Paired A100 performance evidence (candidate vs CubeCL baseline)")
    lines.append("")
    lines.append(
        "Device times are nsys `cuda_gpu_kern_sum` per-instance medians, median over "
        "%d process repetitions; the measured region is delimited by "
        "`cuProfilerStart`/`cuProfilerStop` and holds %d iterations." % (reps, ITERS)
    )
    lines.append("")
    lines.append(
        "Validity rule: a case is invalid when either side's max/min over the "
        "repetitions exceeds %.2f. Regression: a valid case whose candidate median "
        "exceeds %.2fx the baseline median. Kernels are paired through the declared "
        "per-case correspondence table, and both sides' per-iteration instance counts "
        "must agree; a mismatch makes the case `invalid-pairing`." % (
            VALIDITY_SPREAD,
            REGRESSION_RATIO,
        )
    )
    if environment.get("gpu"):
        lines.append("")
        lines.append(
            "Environment: %s, driver %s, %s."
            % (
                environment.get("gpu", "?"),
                environment.get("driver", "?"),
                environment.get("date", "?"),
            )
        )
    lines.append("")

    per_case = []
    for case in cases:
        base_kernels, base_metas, base_complete = summarise_side(
            args.root, "baseline", case, reps
        )
        cand_kernels, cand_metas, cand_complete = summarise_side(
            args.root, "candidate", case, reps
        )
        kernel_rows, problems = pair_case(case, base_kernels, cand_kernels, reps)
        if not base_complete or not cand_complete:
            problems.append("a kernel is missing from at least one repetition")
        if not kernel_rows:
            problems.append("no kernel could be paired")
        per_case.append(
            {
                "case": case,
                "kernel_rows": kernel_rows,
                "problems": problems,
                "base_meta": base_metas[0] if base_metas else {},
                "cand_meta": cand_metas[0] if cand_metas else {},
            }
        )

    def total(entry, side_index):
        value = 0.0
        for _, flags, base_rate, cand_rate in entry["kernel_rows"]:
            rate = base_rate if side_index == 0 else cand_rate
            med = flags["base_median"] if side_index == 0 else flags["cand_median"]
            if med == med:
                value += med * rate
        return value

    lines.append("## Per-case totals")
    lines.append("")
    lines.append(
        "| Case | Baseline total (us/iter) | Candidate total (us/iter) | Ratio | "
        "Launches (base/cand) | Baseline kernels | Candidate kernels | Validity | Verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in per_case:
        base_total = total(entry, 0)
        cand_total = total(entry, 1)
        case_ratio = cand_total / base_total if base_total > 0 else float("nan")
        valid = all(flags["valid"] for _, flags, _, _ in entry["kernel_rows"])
        regressed = any(flags["regressed"] for _, flags, _, _ in entry["kernel_rows"])
        if entry["problems"]:
            verdict = "invalid-pairing"
        elif not valid:
            verdict = "inconclusive"
        elif regressed:
            verdict = "REGRESSION"
        else:
            verdict = "ok"
        validity = max(
            [flags["validity"] for _, flags, _, _ in entry["kernel_rows"]]
            or [float("nan")]
        )
        lines.append(
            "| `%s` | %s | %s | %s | %s/%s | %s | %s | %s | %s |"
            % (
                entry["case"],
                fmt_ms(base_total),
                fmt_ms(cand_total),
                "%.2f" % case_ratio if case_ratio == case_ratio else "-",
                entry["base_meta"].get("launches_per_iter", "-"),
                entry["cand_meta"].get("launches_per_iter", "-"),
                entry["base_meta"].get("kernels", "-"),
                entry["cand_meta"].get("kernels", "-"),
                "%.3f" % validity if validity == validity else "-",
                verdict,
            )
        )

    pairing_problems = [
        (entry["case"], problem)
        for entry in per_case
        for problem in entry["problems"]
    ]
    if pairing_problems:
        lines.append("")
        lines.append("## Pairing failures (these cases carry no verdict)")
        lines.append("")
        for case, problem in pairing_problems:
            lines.append("- `%s`: %s" % (case, problem))

    lines.append("")
    lines.append("## Per-kernel device time")
    lines.append("")
    lines.append(
        "| Case | Pinned correspondence | Baseline median (us) | Baseline min-max | "
        "Candidate median (us) | Candidate min-max | Ratio | Validity | Verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in per_case:
        for name, flags, _, _ in entry["kernel_rows"]:
            if not flags["valid"]:
                verdict = "inconclusive"
            elif flags["regressed"]:
                verdict = "REGRESSION"
            else:
                verdict = "ok"
            lines.append(
                "| `%s` | `%s` | %s | %s-%s | %s | %s-%s | %s | %s | %s |"
                % (
                    entry["case"],
                    name,
                    fmt_ms(flags["base_median"]),
                    fmt_ms(flags["base_min"]),
                    fmt_ms(flags["base_max"]),
                    fmt_ms(flags["cand_median"]),
                    fmt_ms(flags["cand_min"]),
                    fmt_ms(flags["cand_max"]),
                    "%.2f" % flags["ratio"] if flags["ratio"] == flags["ratio"] else "-",
                    "%.3f" % flags["validity"] if flags["validity"] == flags["validity"] else "-",
                    verdict,
                )
            )

    lines.append("")
    lines.append("## Region contents reported by the harnesses")
    lines.append("")
    lines.append(
        "| Case | Side | Launches/iter | Host waits/iter | Transfer bytes/iter | "
        "Allocations/iter | Zero fills/iter | Scratch bytes | Check |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in per_case:
        for side, meta in (("baseline", entry["base_meta"]), ("candidate", entry["cand_meta"])):
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    entry["case"],
                    side,
                    meta.get("launches_per_iter", "-"),
                    meta.get("host_waits_per_iter", "-"),
                    meta.get("transfer_bytes_per_iter", "-"),
                    meta.get("device_allocations_per_iter", "-"),
                    meta.get("zero_fills_per_iter", "-"),
                    meta.get("scratch_bytes", "-"),
                    meta.get("check", "-"),
                )
            )

    lines.append("")
    lines.append("## Host submission cost (not compared, from `cuda_api_sum`)")
    lines.append("")
    lines.append("| Case | Baseline cuLaunchKernel med (ns) | Candidate cuLaunchKernel med (ns) |")
    lines.append("| --- | --- | --- |")
    for entry in per_case:
        base_ns = api_launch_median(args.root, "baseline", entry["case"], reps)
        cand_ns = api_launch_median(args.root, "candidate", entry["case"], reps)
        lines.append(
            "| `%s` | %s | %s |"
            % (
                entry["case"],
                "%.0f" % base_ns if base_ns == base_ns else "-",
                "%.0f" % cand_ns if cand_ns == cand_ns else "-",
            )
        )

    prepare_cases = [
        entry
        for entry in per_case
        if "prepare_cold_ms" in entry["cand_meta"] or "prepare_cold_ms" in entry["base_meta"]
    ]
    if prepare_cases:
        lines.append("")
        lines.append("## Prepare path (cold compile) and warm launch")
        lines.append("")
        lines.append(
            "| Case | Side | Cold (ms) | Cold compile (ms) | Cold load (ms) | Warm (ms) | "
            "Compile attempts | Module loads | Emits | Cache hits/misses |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in prepare_cases:
            for side, meta in (
                ("baseline", entry["base_meta"]),
                ("candidate", entry["cand_meta"]),
            ):
                if not meta:
                    continue
                lines.append(
                    "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s/%s |"
                    % (
                        entry["case"],
                        side,
                        meta.get("prepare_cold_ms", meta.get("prepare_cold_compile_ms", "-")),
                        meta.get("prepare_cold_compile_ms", "-"),
                        meta.get("prepare_cold_load_ms", "-"),
                        meta.get("prepare_warm_ms", "-"),
                        meta.get("prepare_compile_attempts", "-"),
                        meta.get("prepare_module_loads", "-"),
                        meta.get("prepare_emits", "-"),
                        meta.get("prepare_cache_hits", "-"),
                        meta.get("prepare_cache_misses", "-"),
                    )
                )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
