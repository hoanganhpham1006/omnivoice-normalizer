"""Language-agnostic benchmark runner and scorer for the TTS normalization layer.

The question this answers is "what does turning normalization on change?", so it
runs named adapters over the same cases and reports them side by side:

    baseline    what the service does today - nothing
    library     the upstream normalizer called directly
    normalized  what actually ships: text_normalization.normalize()

The gap between `library` and `normalized` is the value of the wrapper (tag
masking, punctuation/spacing restoration, safety valves).

A per-language ``reading`` module supplies the scoring primitives, because what
counts as an unspeakable character differs by script: ``canon_tokens``,
``leaked_chars``, ``match_any``, ``wer``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = min(len(sorted_values) - 1, int(round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[k]


def run_adapter(fn: Callable[[str], str], cases: List[dict], label: str,
                out_path: Path, progress_every: int = 2500) -> dict:
    """Run one adapter over one track, one sentence per call, recording output
    and per-sentence latency. Never aborts: an adapter that raises is recorded
    as an error and the original text is kept."""
    rows: List[dict] = []
    lat: List[float] = []
    errors = 0

    for idx, case in enumerate(cases):
        text = case["text"]
        t0 = time.perf_counter()
        try:
            out = fn(text)
            err = None
        except Exception as exc:  # noqa: BLE001
            out, err = text, f"{type(exc).__name__}: {exc}"
            errors += 1
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        rows.append({"id": case["id"], "out": out, "ms": round(ms, 4), "error": err})
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"  {label}: {idx + 1}/{len(cases)}", flush=True)

    write_jsonl(out_path, rows)

    srt = sorted(lat)
    total_s = sum(lat) / 1000
    total_chars = sum(len(c["text"]) for c in cases)
    return {
        "n": len(cases),
        "errors": errors,
        "mean_ms": statistics.fmean(lat) if lat else 0.0,
        "median_ms": _percentile(srt, 50),
        "p90_ms": _percentile(srt, 90),
        "p95_ms": _percentile(srt, 95),
        "p99_ms": _percentile(srt, 99),
        "max_ms": max(lat) if lat else 0.0,
        "total_s": total_s,
        "chars_per_s": total_chars / total_s if total_s else 0.0,
        "sentences_per_s": len(cases) / total_s if total_s else 0.0,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_gold(reading, cases: List[dict], outputs: Dict[str, dict]) -> dict:
    """Score a gold track.

    metric="reading"       an accepted spoken form appears in the output
    metric="leak"          the NSW's unspeakable characters are gone AND the raw
                           NSW is no longer sitting there verbatim
    metric="all_readings"  every scored slot of a multi-NSW sentence matched
    """
    per_cat: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"acc": [], "strict": [], "leak": [], "wer": [], "ms": []}
    )
    examples: Dict[str, List[dict]] = defaultdict(list)

    for case in cases:
        rec = outputs.get(case["id"])
        if rec is None:
            continue
        hyp = rec["out"]
        bucket = per_cat[case["category"]]
        bucket["ms"].append(rec["ms"])
        bucket["leak"].append(bool(reading.leaked_chars(hyp)))

        metric = case.get("metric", "reading")
        if metric == "reading":
            ok = reading.match_any(hyp, case["readings"], lenient=True)
            bucket["acc"].append(ok)
            bucket["strict"].append(
                reading.match_any(hyp, case["readings"], lenient=False))
            bucket["wer"].append(reading.wer(case["gold_text"], hyp))
        elif metric == "leak":
            nsw_bad = set(reading.leaked_chars(case["nsw"]))
            verbatim = case["nsw"].lower() in hyp.lower()
            ok = not (nsw_bad & set(hyp)) and not verbatim
            bucket["acc"].append(ok)
            bucket["strict"].append(ok)
        else:
            hits = [reading.match_any(hyp, [r], lenient=True)
                    for r in case["readings"]]
            ok = all(hits) if hits else False
            bucket["acc"].append(ok)
            bucket["strict"].append(ok)

        if not ok and len(examples[case["category"]]) < 8:
            examples[case["category"]].append({
                "nsw": case["nsw"],
                "text": case["text"],
                "expected": case["readings"][:2],
                "got": hyp,
            })

    summary = {}
    for cat, b in sorted(per_cat.items()):
        summary[cat] = {
            "n": len(b["acc"]),
            "accuracy": statistics.fmean(b["acc"]) if b["acc"] else 0.0,
            "strict_accuracy": statistics.fmean(b["strict"]) if b["strict"] else 0.0,
            "leak_rate": statistics.fmean(b["leak"]) if b["leak"] else 0.0,
            "wer": statistics.fmean(b["wer"]) if b["wer"] else None,
            "mean_ms": statistics.fmean(b["ms"]) if b["ms"] else 0.0,
        }

    flat = lambda key: [v for b in per_cat.values() for v in b[key]]  # noqa: E731
    acc, strict, leak, wer_vals = flat("acc"), flat("strict"), flat("leak"), flat("wer")
    return {
        "per_category": summary,
        "overall": {
            "n": len(acc),
            "accuracy": statistics.fmean(acc) if acc else 0.0,
            "strict_accuracy": statistics.fmean(strict) if strict else 0.0,
            "leak_rate": statistics.fmean(leak) if leak else 0.0,
            "wer": statistics.fmean(wer_vals) if wer_vals else 0.0,
        },
        "examples": examples,
    }


def score_real(reading, cases: List[dict], outputs: Dict[str, dict]) -> dict:
    """Score the reference-free track: residual unspeakable characters, crashes."""
    leaks, digit_leaks, errors, changed = [], [], [], []
    leaked_counter: Dict[str, int] = defaultdict(int)

    for case in cases:
        rec = outputs.get(case["id"])
        if rec is None:
            continue
        hyp = rec["out"]
        errors.append(rec["error"] is not None)
        changed.append(hyp != case["text"])
        bad = reading.leaked_chars(hyp)
        leaks.append(bool(bad))
        digit_leaks.append(any(c.isdigit() for c in bad))
        for c in bad:
            leaked_counter[c] += 1

    mean = lambda xs: statistics.fmean(xs) if xs else 0.0  # noqa: E731
    return {
        "n": len(leaks),
        "leak_rate": mean(leaks),
        "digit_leak_rate": mean(digit_leaks),
        "error_rate": mean(errors),
        "changed_rate": mean(changed),
        "top_leaked_chars": sorted(leaked_counter.items(), key=lambda kv: -kv[1])[:15],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(x) -> str:
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def print_report(lang: str, report: dict, names: List[str]) -> None:
    gold_n = report[names[0]]["gold"]["overall"]["n"]
    print(f"\n{'=' * 78}\n{lang.upper()} — gold track ({gold_n} cases)\n{'=' * 78}")
    print(f"{'metric':<26}" + "".join(f"{n:>16}" for n in names))
    for key, label in [("accuracy", "accuracy"),
                       ("strict_accuracy", "strict accuracy"),
                       ("leak_rate", "unspeakable-char leak"),
                       ("wer", "WER vs gold")]:
        print(f"{label:<26}" + "".join(
            f"{_pct(report[n]['gold']['overall'][key]):>16}" for n in names))
    for key, label in [("mean_ms", "mean ms/sentence"), ("median_ms", "p50 ms"),
                       ("p95_ms", "p95 ms"), ("p99_ms", "p99 ms"),
                       ("errors", "exceptions")]:
        row = f"{label:<26}"
        for n in names:
            v = report[n]["timing"]["gold"].get(key, 0)
            row += f"{v:>15.2f} " if isinstance(v, float) else f"{v:>15} "
        print(row)

    print(f"\n{lang.upper()} — per-category accuracy")
    cats = sorted(report[names[0]]["gold"]["per_category"])
    print(f"{'category':<22}{'n':>6}" + "".join(f"{n:>16}" for n in names))
    for cat in cats:
        row = f"{cat:<22}{report[names[0]]['gold']['per_category'][cat]['n']:>6}"
        for n in names:
            c = report[n]["gold"]["per_category"].get(cat)
            row += f"{_pct(c['accuracy'] if c else None):>16}"
        print(row)

    if report[names[0]].get("real"):
        real_n = report[names[0]]["real"]["n"]
        print(f"\n{lang.upper()} — real-text track ({real_n} sentences)")
        print(f"{'metric':<26}" + "".join(f"{n:>16}" for n in names))
        for key, label in [("leak_rate", "unspeakable-char leak"),
                           ("digit_leak_rate", "digit leak"),
                           ("changed_rate", "sentences changed"),
                           ("error_rate", "crash rate")]:
            print(f"{label:<26}" + "".join(
                f"{_pct(report[n]['real'].get(key)):>16}" for n in names))
        for key, label in [("mean_ms", "mean ms/sentence"), ("p95_ms", "p95 ms")]:
            row = f"{label:<26}"
            for n in names:
                row += f"{report[n]['timing']['real'].get(key, 0):>15.2f} "
            print(row)


def benchmark(lang: str, reading, adapters: Dict[str, Callable[[str], str]],
              here: Path, tracks=("gold", "real")) -> dict:
    """Run every adapter over every track, score, print and persist."""
    data, results = here / "data", here / "results"
    loaded = {t: read_jsonl(data / f"{t}.jsonl")
              for t in tracks if (data / f"{t}.jsonl").exists()}

    report: dict = {}
    for name, fn in adapters.items():
        print(f"\n=== {lang}/{name} ===", flush=True)
        entry: dict = {"timing": {}}
        for track, cases in loaded.items():
            print(f"  running {track}: {len(cases)} cases", flush=True)
            entry["timing"][track] = run_adapter(
                fn, cases, f"{name}/{track}", results / f"{name}_{track}.jsonl")
            outputs = {r["id"]: r for r in
                       read_jsonl(results / f"{name}_{track}.jsonl")}
            entry[track] = (score_gold(reading, cases, outputs) if track == "gold"
                            else score_real(reading, cases, outputs))
            t = entry["timing"][track]
            print(f"  {track}: mean={t['mean_ms']:.2f}ms p95={t['p95_ms']:.2f}ms "
                  f"errors={t['errors']}", flush=True)
        report[name] = entry

    results.mkdir(parents=True, exist_ok=True)
    (results / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(lang, report, list(adapters))
    print(f"\nFull report -> {results / 'report.json'}")
    return report
