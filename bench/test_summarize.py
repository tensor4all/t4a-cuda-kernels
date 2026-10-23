#!/usr/bin/env python3
"""CPU-only fixtures for `bench/summarize.py`.

Each test builds a tiny synthetic campaign directory (the same artifact layout
`bench/run-paired.sh` writes: `environment.txt`, one `.stdout` with a `BENCH_META`
line, one `cuda_gpu_kern_sum` CSV and one `cuda_api_sum` CSV per side per
repetition), runs the summariser against it and asserts on its exit status and
output. No GPU, no `nsys` and no Rust build are involved.

Run with `python3 bench/test_summarize.py -v`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ITERS = 100
HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIZE = os.path.join(HERE, "summarize.py")

KERN_HEADER = (
    '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns),'
    '"Max (ns)","StdDev (ns)","Kernel Name"\n'
).replace('"Min (ns),', '"Min (ns)",')
API_HEADER = (
    '"Time (%)","Total Time (ns)","Num Calls","Average (ns)","Med (ns)",'
    '"Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
)

# The three cases the fixtures use, with their declared correspondence:
#   elementwise-add  single kernel, single instance
#   reduce            split reduction (chunked pass 1 + combine pass 2)
#   linalg-rrqr      the nine-kernel primitives plus the revision-8 disjointness copy
ADD = "elementwise-add-f32-1000"
REDUCE = "reduce-sum-f32-axis-513"
RRQR = "linalg-rrqr-256x256-f64"
SCATTER = "indexing-scatter-add-f32-1e6-into-16"


def write_side(root, side, case, rep, kernels, check, launches, case_name=None):
    """One side's artifacts for one case repetition.

    `kernels` is a list of `(name, per_instance_median_ns, instances_per_iteration)`.
    The instance count is rounded up so a fractional rate stays representable.
    """
    directory = os.path.join(root, side)
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, "%s.rep%d" % (case, rep))
    declared = case if case_name is None else case_name
    with open(base + ".stdout", "w", encoding="utf-8") as handle:
        handle.write(
            "BENCH_META case=%s kernels=%s launches_per_iter=%d scratch_bytes=0 "
            "host_waits_per_iter=0 transfer_bytes_per_iter=0 "
            "device_allocations_per_iter=0 zero_fills_per_iter=0 "
            "boundary_barriers=1 input_digest=0x0000000000000001 check=%s\n"
            % (declared, "|".join(name for name, _, _ in kernels), launches, check)
        )
    with open(base + ".kern.csv", "w", encoding="utf-8") as handle:
        handle.write(KERN_HEADER)
        for name, median, rate in kernels:
            instances = int(round(rate * ITERS))
            if instances == 0:
                continue
            handle.write(
                '"50.0","%d","%d","%d","%d","%d","%d","1.0","%s"\n'
                % (median * instances, instances, median, median, median, median, name)
            )
    with open(base + ".api.csv", "w", encoding="utf-8") as handle:
        handle.write(API_HEADER)
        handle.write(
            '"100.0","1000","1","3000","3000","3000","3000","1.0","cuLaunchKernel"\n'
        )


RRQR_BASELINE = [
    ("initialize_permutation_6e6b4b9f", 1800, 1),
    ("column_norms_real_f_f64_4986bb3e", 8000, 256),
    ("select_pivot_f_f64_d6a46476", 3500, 256),
    ("swap_columns_e_f64_facdca52", 2500, 256),
    ("reflector_real_f_f64_a1a75538", 10400, 256),
    ("apply_reflector_real_f_f64_4d4454e3", 6900, 511),
    ("extract_r_e_f64_8aadcce7", 2600, 1),
    ("initialize_q_e_f64_ac734ce7", 2200, 1),
    ("prefix_rank_real_f_f64_abe5ff93", 3300, 1),
]
RRQR_CANDIDATE = [
    ("t4a_cuda_initialize_permutation", 1800, 1),
    ("t4a_cuda_column_norms", 8300, 256),
    ("t4a_cuda_select_pivot", 4400, 256),
    ("t4a_cuda_swap_columns", 2900, 256),
    ("t4a_cuda_reflector", 11500, 256),
    ("t4a_cuda_copy_exact", 1400, 255),
    ("t4a_cuda_apply_reflector", 7100, 511),
    ("t4a_cuda_extract_r", 3200, 1),
    ("t4a_cuda_identity_shifted", 2700, 1),
    ("t4a_cuda_prefix_rank", 3700, 1),
]
ADD_BASELINE = [("add_float_f_f32_a70f4883", 1900, 1)]
ADD_CANDIDATE = [("t4a_cuda_add", 2000, 1)]
REDUCE_BASELINE = [
    ("reduce_sum_float_chunked_t_f32_b1cf76a9", 9500, 1),
    ("reduce_sum_float_t_f32_73677391", 2700, 1),
]
REDUCE_CANDIDATE = [
    ("t4a_cuda_reduce_partials", 11800, 1),
    ("t4a_cuda_reduce", 3500, 1),
]


def rrqr_launches(kernels):
    return int(sum(rate for _, _, rate in kernels))


class Campaign:
    """A synthetic campaign directory with one helper to fill in a case."""

    def __init__(self, cases, reps=2):
        self.root = tempfile.mkdtemp(prefix="summarize-fixture-")
        self.cases = cases
        self.reps = reps
        with open(os.path.join(self.root, "environment.txt"), "w", encoding="utf-8") as handle:
            handle.write("out=%s\nreps=%d\ncases=%s\n" % (self.root, reps, ",".join(cases)))

    def fill(self, case, baseline, candidate, check="ok", check_baseline=None, **kwargs):
        for rep in range(1, self.reps + 1):
            write_side(
                self.root,
                "baseline",
                case,
                rep,
                baseline,
                check if check_baseline is None else check_baseline,
                rrqr_launches(baseline),
                **kwargs
            )
            write_side(
                self.root,
                "candidate",
                case,
                rep,
                candidate,
                check,
                rrqr_launches(candidate),
                **kwargs
            )
        return self

    def per_rep(self, case, baseline_by_rep, candidate_by_rep, check="ok"):
        """Fill a case whose per-repetition kernel lists differ."""
        for rep in range(1, self.reps + 1):
            baseline = baseline_by_rep[rep - 1]
            candidate = candidate_by_rep[rep - 1]
            write_side(
                self.root,
                "baseline",
                case,
                rep,
                baseline,
                check,
                rrqr_launches(baseline),
            )
            write_side(
                self.root,
                "candidate",
                case,
                rep,
                candidate,
                check,
                rrqr_launches(candidate),
            )
        return self

    def run(self):
        return subprocess.run(
            [sys.executable, SUMMARIZE, "--root", self.root],
            capture_output=True,
            text=True,
        )

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class SummarizeFixtures(unittest.TestCase):
    def setUp(self):
        self.campaigns = []

    def tearDown(self):
        for campaign in self.campaigns:
            campaign.cleanup()

    def campaign(self, cases, reps=2):
        campaign = Campaign(cases, reps)
        self.campaigns.append(campaign)
        return campaign

    def verdict_line(self, output, case):
        for line in output.splitlines():
            if line.startswith("| `%s` |" % case):
                return line
        self.fail("no per-case row for %s in:\n%s" % (case, output))

    def test_complete_campaign_has_no_pairing_failure(self):
        campaign = self.campaign([ADD, REDUCE, RRQR])
        campaign.fill(ADD, ADD_BASELINE, ADD_CANDIDATE)
        campaign.fill(REDUCE, REDUCE_BASELINE, REDUCE_CANDIDATE)
        campaign.fill(RRQR, RRQR_BASELINE, RRQR_CANDIDATE)
        result = campaign.run()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("CAMPAIGN INVALID", result.stdout)
        self.assertNotIn("## Pairing failures", result.stdout)
        for case in (ADD, REDUCE, RRQR):
            self.assertNotIn("invalid-pairing", self.verdict_line(result.stdout, case))
        # The reduction's pass-1/pass-2 pair and the candidate-only reflector copy
        # are both declared, so they appear as paired rows and as a helper row.
        self.assertIn("reduce_sum_float_chunked_t_f32_b1cf76a9 -> t4a_cuda_reduce_partials",
                      result.stdout)
        self.assertIn("t4a_cuda_copy_exact (candidate-side helper)", result.stdout)
        self.assertIn(ADD, self.verdict_line(result.stdout, ADD))

    def test_missing_required_row_invalidates_the_case(self):
        campaign = self.campaign([RRQR], reps=1)
        baseline = [k for k in RRQR_BASELINE if not k[0].startswith("extract_r")]
        candidate = [k for k in RRQR_CANDIDATE if k[0] != "t4a_cuda_extract_r"]
        campaign.fill(RRQR, baseline, candidate)
        result = campaign.run()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("invalid-pairing", self.verdict_line(result.stdout, RRQR))
        self.assertIn("`extract_r` -> t4a_cuda_extract_r is not submitted by both arms",
                      result.stdout)

    def test_reduction_alternatives_may_be_absent(self):
        # The direct (non-split) reduction submits only the plane combine kernel on
        # the baseline side and only `t4a_cuda_reduce` on the candidate side; the
        # `chunked` and unit-combine rows are declared alternatives and may vanish.
        campaign = self.campaign([REDUCE], reps=1)
        campaign.fill(
            REDUCE,
            [("reduce_sum_float_plane_t_f32_48736aac", 8100, 1)],
            [("t4a_cuda_reduce", 9700, 1)],
        )
        result = campaign.run()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("## Pairing failures", result.stdout)
        self.assertNotIn("invalid-pairing", self.verdict_line(result.stdout, REDUCE))
        self.assertIn("reduce_sum_float_plane_t_f32_48736aac -> t4a_cuda_reduce",
                      result.stdout)

    def test_scatter_dtype_alternatives_may_be_absent(self):
        # Only the float update kernel of the two declared scatter alternatives runs
        # in the F32 case; the copy kernel is always present.
        campaign = self.campaign([SCATTER], reps=1)
        campaign.fill(
            SCATTER,
            [("scatter_copy_kernel_e_f32_a2208266", 1800, 1),
             ("scatter_float_kernel_e_f32_i_i32_7acf9087", 210900, 1)],
            [("t4a_cuda_copy_exact", 2000, 1), ("t4a_cuda_scatter_add", 213900, 1)],
        )
        result = campaign.run()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("## Pairing failures", result.stdout)
        self.assertNotIn("invalid-pairing", self.verdict_line(result.stdout, SCATTER))

    def test_counts_cannot_cancel_across_repetitions(self):
        # Same per-repetition totals and the same aggregate per kernel, but
        # repetitions 1 and 2 ran different workloads.
        campaign = self.campaign([REDUCE], reps=3)
        baseline_by_rep = [
            [("reduce_sum_float_chunked_t_f32_b1cf76a9", 9500, 3),
             ("reduce_sum_float_t_f32_73677391", 2700, 1)],
            [("reduce_sum_float_chunked_t_f32_b1cf76a9", 9500, 1),
             ("reduce_sum_float_t_f32_73677391", 2700, 3)],
            [("reduce_sum_float_chunked_t_f32_b1cf76a9", 9500, 2),
             ("reduce_sum_float_t_f32_73677391", 2700, 2)],
        ]
        candidate_by_rep = [
            [("t4a_cuda_reduce_partials", 11800, 1), ("t4a_cuda_reduce", 3500, 3)],
            [("t4a_cuda_reduce_partials", 11800, 3), ("t4a_cuda_reduce", 3500, 1)],
            [("t4a_cuda_reduce_partials", 11800, 2), ("t4a_cuda_reduce", 3500, 2)],
        ]
        campaign.per_rep(REDUCE, baseline_by_rep, candidate_by_rep)
        result = campaign.run()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("invalid-pairing", self.verdict_line(result.stdout, REDUCE))
        self.assertIn("repetition 1 submits", result.stdout)

    def test_failed_numerical_check_invalidates_the_campaign(self):
        campaign = self.campaign([REDUCE], reps=1)
        campaign.fill(REDUCE, REDUCE_BASELINE, REDUCE_CANDIDATE,
                      check="ok", check_baseline="FAIL")
        result = campaign.run()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("CAMPAIGN INVALID", result.stdout)
        self.assertIn("numerical check is `FAIL`", result.stdout)
        self.assertNotIn("## Per-case totals", result.stdout)

    def test_case_identity_must_match(self):
        campaign = self.campaign([ADD], reps=1)
        campaign.fill(ADD, ADD_BASELINE, ADD_CANDIDATE, case_name="elementwise-add-f32-1e6")
        result = campaign.run()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("BENCH_META declares case=elementwise-add-f32-1e6", result.stdout)

    def test_declared_launch_count_must_match_the_capture(self):
        campaign = self.campaign([RRQR], reps=1)
        # Declare the revision-6 launch count for a capture that holds the
        # revision-8 sequence (the disjointness copy included).
        campaign.fill(RRQR, RRQR_BASELINE, RRQR_CANDIDATE)
        path = os.path.join(campaign.root, "candidate", "%s.rep1.stdout" % RRQR)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("launches_per_iter=1794", "launches_per_iter=1539"))
        result = campaign.run()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("CAMPAIGN INVALID", result.stdout)
        self.assertIn("against 1539 declared", result.stdout)


if __name__ == "__main__":
    unittest.main()
