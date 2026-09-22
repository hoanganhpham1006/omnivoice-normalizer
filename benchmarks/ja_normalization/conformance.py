#!/usr/bin/env python3
"""Conformance gate: run WeTextProcessing's own Japanese test data through our
wrapper.

This answers a different question from the gold benchmark. The benchmark asks
"is this library good enough for the service"; this asks "are we driving the
library the way its authors intended". Every pair ships with the package as
`input => expected`, so any mismatch means our configuration (cache dir,
full_to_half, remove_puncts, the punctuation restoration) changed the library's
documented behaviour.

Mismatches caused deliberately by our wrapper - the Japanese sentence
terminators we put back - are reported separately from real failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # benchmarks/ja_normalization/ -> repo root
sys.path.insert(0, str(REPO_ROOT))

import text_normalization as textnorm  # noqa: E402

TERMINATOR_ONLY = str.maketrans({"。": ".", "？": "?", "！": "!"})


def data_dir() -> Path:
    import tn

    return Path(tn.__file__).resolve().parent / "japanese" / "test" / "data"


def main() -> int:
    textnorm.warmup(["jp"])
    root = data_dir()
    if not root.is_dir():
        print(f"test data not found at {root}", file=sys.stderr)
        return 2

    total = passed = restored = 0
    failures = []
    for path in sorted(root.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=>" not in line:
                continue
            src, expected = (p.strip() for p in line.split("=>", 1))
            if not src:
                continue
            total += 1
            got, _, _ = textnorm.normalize("jp", src)
            if got == expected:
                passed += 1
            elif got.translate(TERMINATOR_ONLY) == expected:
                # Our deliberate change: the library ASCII-ifies 。？！ and we
                # put them back for the acoustic model.
                passed += 1
                restored += 1
            else:
                failures.append((path.name, src, expected, got))

    print(f"conformance: {passed}/{total} pairs match "
          f"({restored} matched after our punctuation restoration)")
    for name, src, expected, got in failures[:20]:
        print(f"  [{name}] {src!r}\n      expected {expected!r}\n      got      {got!r}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
