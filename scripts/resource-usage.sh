#!/usr/bin/env bash
# Static resource usage for every device entry in cuda/src/*.cu.
#
# Compiles each source to a cubin with `nvcc -cubin -arch=sm_80 -Xptxas -v`
# and cross-checks the numbers with `cuobjdump --dump-resource-usage`. Prints
# one line per `extern "C"` entry:
#
#   source entry registers shared stack local spill_stores spill_loads local_memory_ops
#
# `shared` is the statically declared shared memory; kernels that take dynamic
# shared memory (`t4a_cuda_tiled_transpose`) report 0 here and receive it at
# launch time. `local` is the per-thread local memory that ptxas would spill
# into; spill stores/loads come from ptxas. `local_memory_ops` is the number of
# static SASS local-memory instructions (`LDL`/`STL`, every width, with or without
# a guard predicate such as `@P0`) in the entry's body: the stack frame is only a
# size, this column shows whether the code actually touches it per element (a
# dynamically indexed by-value layout POD copies the POD into the stack frame and
# reads it back with `LDL`). It is a static instruction count, not a dynamic
# per-element count; predicated `LDL`/`STL` are included, so the diagnostic cannot
# silently undercount a guarded local access.
#
# Usage: scripts/resource-usage.sh [-a ARCH]
# Exit status is non-zero when a source fails to compile.

set -euo pipefail

arch="sm_80"
while getopts "a:" option; do
    case "$option" in
    a) arch="$OPTARG" ;;
    *)
        echo "usage: $0 [-a arch]" >&2
        exit 2
        ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

printf '%s\n' "source entry registers shared stack local spill_stores spill_loads local_memory_ops"

status=0
for source in cuda/src/*.cu; do
    name="$(basename "$source" .cu)"
    cubin="$tmp/$name.cubin"

    if ! nvcc -std=c++17 -cubin "-arch=$arch" -I cuda/include -Xptxas -v \
        -o "$cubin" "$source" >/dev/null 2>"$tmp/$name.ptxas"; then
        echo "nvcc failed for $source" >&2
        cat "$tmp/$name.ptxas" >&2
        status=1
        continue
    fi
    cuobjdump --dump-resource-usage "$cubin" >"$tmp/$name.cuobjdump"
    cuobjdump -sass "$cubin" >"$tmp/$name.sass"

    # ptxas has the spill counts, cuobjdump has the authoritative
    # REG/SHARED/STACK/LOCAL columns; join them per entry, and count the
    # local-memory instructions of each entry in the SASS dump.
    python3 - "$name" "$tmp/$name.ptxas" "$tmp/$name.cuobjdump" "$tmp/$name.sass" <<'PY'
import re
import sys

name, ptxas_path, cuobjdump_path, sass_path = sys.argv[1:5]

ptxas = {}
entry = None
with open(ptxas_path) as handle:
    lines = handle.readlines()
for index, line in enumerate(lines):
    match = re.search(r"Compiling entry function '([^']+)'", line)
    if match:
        entry = match.group(1)
        ptxas.setdefault(entry, {"spill_stores": "-", "spill_loads": "-", "registers": "-"})
    elif entry and "bytes stack frame" in line:
        spills = re.findall(r"(\d+) bytes spill (stores|loads)", line)
        for count, kind in spills:
            ptxas[entry][f"spill_{kind}"] = count
    elif entry and "Used" in line and "registers" in line:
        match = re.search(r"Used (\d+) registers", line)
        if match:
            ptxas[entry]["registers"] = match.group(1)

usage = {}
current = None
with open(cuobjdump_path) as handle:
    for line in handle:
        stripped = line.strip()
        if stripped.startswith("Function "):
            current = stripped[len("Function "):].rstrip(":")
            usage.setdefault(current, {})
        elif current and stripped.startswith("REG:"):
            for field in stripped.split():
                key, _, value = field.partition(":")
                if key in {"REG", "SHARED", "STACK", "LOCAL"}:
                    usage[current][key] = value

# SASS local-memory instructions per entry: LDL/STL (any width or variant
# suffix, e.g. LDL.64/STL.128/STL.STRONG), with or without a leading guard
# predicate (`@P0 LDL.64 ...`, `@!P3 STL.64 ...`). Shared memory (LDS/STS) is
# not local. The column is a *static* instruction count, not a dynamic count.
local_ops = {}
current = None
# An instruction line is `/*<hex>*/ [predicate] MNEMONIC operands ;`.
mnemonic = re.compile(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9._]*)")
local_pattern = re.compile(r"(?:LDL|STL)(?:\.\w+)*")
with open(sass_path) as handle:
    for line in handle:
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match:
            current = match.group(1).rstrip(":")
            local_ops.setdefault(current, 0)
            continue
        match = mnemonic.match(line)
        if current and match and local_pattern.fullmatch(match.group(1)):
            local_ops[current] += 1

order = list(ptxas.keys()) + [key for key in usage if key not in ptxas]
for key in order:
    ptx = ptxas.get(key, {})
    use = usage.get(key, {})
    print(
        "\t".join(
            [
                name,
                key,
                use.get("REG", ptx.get("registers", "-")),
                use.get("SHARED", "-"),
                use.get("STACK", "-"),
                use.get("LOCAL", "-"),
                ptx.get("spill_stores", "-"),
                ptx.get("spill_loads", "-"),
                str(local_ops.get(key, "-")),
            ]
        )
    )
PY
done

exit "$status"
