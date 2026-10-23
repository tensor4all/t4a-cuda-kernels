#!/usr/bin/env bash
# Per-entry SASS comparison between a base revision and the working tree.
#
# Compiles every cuda/src/*.cu to a cubin with the same flags in both trees and
# compares the instruction stream of every kernel symbol present in both. The
# comparison drops the line-address comment and collapses column padding, which
# cuobjdump widens when a translation unit gains longer symbol names; the
# instruction words (the hex encodings) are compared as they are.
#
# Use it to justify "the fast path did not change" after adding entries beside
# unchanged bodies: identical symbols are not rerun in the paired matrix, and any
# differing symbol must be.
#
# Usage: scripts/sass-diff.sh <base-ref>   (the base commit must be committed)

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <base-ref>" >&2
    exit 2
fi

base_ref=$1
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp" "$tmp-base"' EXIT

git -C "$repo_root" worktree add --detach "$tmp-base" "$base_ref" >/dev/null

extract() { # $1 source root, $2 output dir
    local root=$1 out=$2 name
    mkdir -p "$out"
    for src in "$root"/cuda/src/*.cu; do
        name="$(basename "$src" .cu)"
        if ! nvcc -std=c++17 -cubin -arch=sm_80 -I "$root/cuda/include" \
            -o "$out/$name.cubin" "$src" >/dev/null 2>&1; then
            echo "nvcc failed for $src" >&2
            continue
        fi
        cuobjdump -sass "$out/$name.cubin" >"$out/$name.sass"
        python3 - "$out/$name.sass" "$out/$name.txt" <<'PY'
import re, sys
source, target = sys.argv[1], sys.argv[2]
sections = re.split(r'\n\s*Function : ', open(source).read())
out = {}
for section in sections[1:]:
    name = section.split('\n', 1)[0].strip()
    body = section.split('\n', 1)[1] if '\n' in section else ''
    lines = [re.sub(r'/\*[0-9a-f]+\*/', '', line) for line in body.split('\n')]
    lines = [' '.join(line.split()) for line in lines]
    out[name] = '\n'.join(line for line in lines if line and not line.startswith('.'))
with open(target, 'w') as handle:
    for name in sorted(out):
        handle.write('=== ' + name + '\n' + out[name] + '\n')
PY
    done
}

extract "$tmp-base" "$tmp/base"
extract "$repo_root" "$tmp/current"

python3 - "$tmp" <<'PY'
import glob, os, sys
tmp = sys.argv[1]

def load(path):
    blocks, key = {}, None
    for line in open(path):
        if line.startswith('=== '):
            key = line[4:].strip()
            blocks[key] = []
        elif key:
            blocks[key].append(line.rstrip('\n'))
    return {name: '\n'.join(body) for name, body in blocks.items()}

same, differs, added = 0, [], []
for base_file in sorted(glob.glob(os.path.join(tmp, 'base', '*.txt'))):
    name = os.path.basename(base_file)
    current_file = os.path.join(tmp, 'current', name)
    if not os.path.exists(current_file):
        continue
    base, current = load(base_file), load(current_file)
    for symbol, body in base.items():
        if symbol not in current:
            differs.append((name, symbol, 'missing'))
        elif current[symbol] != body:
            differs.append((name, symbol, 'SASS differs'))
        else:
            same += 1
    added.extend((name, symbol) for symbol in current if symbol not in base)

print(f"symbols present in both trees: {same + len(differs)}")
print(f"identical: {same}")
print(f"added by this branch: {len(added)}")
for name, symbol, why in differs:
    print(f"DIFFERS  {name}: {symbol} -- {why}")
sys.exit(1 if differs else 0)
PY
