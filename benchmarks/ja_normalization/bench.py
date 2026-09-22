#!/usr/bin/env python3
"""Before/after benchmark for the Japanese normalization layer.

    baseline    what the service does today - the text goes through untouched
    library     tn.japanese.Normalizer called directly
    normalized  what ships: text_normalization.normalize("jp", ...)

Run inside the `omnivoice-nlp` env.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # benchmarks/bench_core.py
sys.path.insert(0, str(HERE.parents[1]))      # repo root: text_normalization.py

import ja_reading  # noqa: E402
from bench_core import benchmark  # noqa: E402


def build_adapters() -> dict:
    import text_normalization as textnorm

    textnorm.warmup(["jp"])
    # `tn` here is WeTextProcessing's package, not our module.
    from tn.japanese.normalizer import Normalizer as JaNormalizer

    library = JaNormalizer(
        cache_dir=textnorm.CACHE_DIR,
        overwrite_cache=False,
        remove_puncts=False,
        full_to_half=True,
        transliterate=False,
        remove_interjections=False,
        tag_oov=False,
    )
    return {
        "baseline": lambda text: text,
        "library": library.normalize,
        "normalized": lambda text: textnorm.normalize("jp", text)[0],
    }


if __name__ == "__main__":
    import logging

    logging.disable(logging.INFO)
    benchmark("jp", ja_reading, build_adapters(), HERE)
