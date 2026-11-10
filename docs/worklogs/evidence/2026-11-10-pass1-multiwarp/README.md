# Evidence: split pass 1 in multi-warp blocks (#30 item 3)

A100 80GB on primerose, CUDA 12.6. Arms are `bench_kernels` built from frozen worktrees:
`ab7395c` (base), `4ea91e6` (#29), `a029e80` (the first version, with a 64-bit divide) and
`f21234e`.

- `nsys-final2.md`: device µs per launch for base, #29 and `f21234e` (3 rounds,
  `scripts/nsys-sweep.sh` and `scripts/nsys-table.py`).
- `nsys-final-a029e80-divide.md`: the same sweep with `a029e80`, showing the 64-bit-divide
  regression on `4096sq-axis0`.
- `nsys-nocap-experiment.md`: `f21234e` against a build without the cube limit (not
  committed).
- `campaign/`: paired host campaign, base against `f21234e`, using
  `../2026-11-09-combine-load-ahead/campaign.sh` and `table.py`.
- `golden-v2-on-f21234e.log`: 1332 cases, partials 0 diffs, finals 0 diffs apart from the 48
  listed #29 NaN-payload cases.
- `golden-v2-on-4ea91e6.log`: #29 against the v2 golden before the payload list existed. It
  fails only on those 48 unlisted payload-only finals, which is where the list comes from.
- `mutants-a029e80/`, `mutants-f21234e/`: `scripts/mutants.py` runs (anchors updated for the 32-bit mapping before the `f21234e` run). Each mutant breaks bit-identity and the golden
  test catches it.
- `sass-diff-vs-4ea91e6.log`: `scripts/sass-diff.sh 4ea91e6`.
- `ncu-attempt.log`: `ERR_NVGPUCTRPERM` (`RmProfilingAdminOnly: 1`); no counters could be
  read.
- `probe/`: the clock64/smid substitute for ncu (`driver.cu`, `reduce_probe.cu`,
  `analyze.py`) and its output.
