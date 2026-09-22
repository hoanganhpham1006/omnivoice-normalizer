# omnivoice-normalizer

A standalone text-normalization layer for Japanese and Vietnamese TTS,
extracted from OmniVoice's HTTP TTS service. Turns written-form text (digits,
dates, money, phone numbers, decimals) into the spoken form a TTS acoustic
model can actually pronounce, before the request reaches the model.

```python
import text_normalization as tn

out, elapsed_ms, applied = tn.normalize("jp", "1,500,000,000円を予定しています。")
# out == "ジューゴオクエンを予定しています。"   (15億円 = "juugo-oku-en", not a literal comma)

out, elapsed_ms, applied = tn.normalize("vn", "Cuộc họp ngày 25/12/2023.")
# out == "Cuộc họp ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi ba."
```

## Why this exists

An acoustic model has no pronunciation for `"1,500,000,000"`, `"25/12/2023"`,
or `"090-1234-5678"` — those are written-form conventions, not words. Left
untouched, a duration estimator that sizes output audio from the written text
also budgets it too short for what a speaker would actually say. This module
sits in front of the model and rewrites exactly the spans that need it,
leaving everything else untouched.

| Language | Library | Approach |
|---|---|---|
| `vn` | [`soe-vinorm`](https://pypi.org/project/soe-vinorm/) | CRF non-standard-word tagger + rule expanders |
| `jp` | [`WeTextProcessing`](https://pypi.org/project/WeTextProcessing/) (`tn.japanese`) + [`pyopenjtalk-plus`](https://pypi.org/project/pyopenjtalk-plus/) | pynini WFST grammar → kanji numerals, then only the *rewritten* fragments are re-read as kana via a morphological dictionary |

Every import is lazy and every failure is non-fatal: if a library is missing
or raises, the original text comes back with `applied=False` and the request
still gets audio. Nothing here imports `torch` — it runs fine in a small
CPU-only environment separate from the acoustic model's own dependency tree
(which is exactly why this was split out).

## Install

```bash
pip install -r requirements.txt
```

`soe-vinorm` pins `numpy<2`; if your acoustic-model environment needs numpy
2.x, keep this normalizer in its own virtualenv/conda env and call it as a
separate process or service, the way the original TTS router does.

## API

```python
import text_normalization as tn

tn.normalize(lang: str, text: str) -> tuple[str, float, bool]
```
Returns `(text, elapsed_ms, applied)`. `lang` is `"jp"` or `"vn"`
(`tn.SUPPORTED_LANGS`). Never raises — any internal failure falls back to the
original text with `applied=False`.

```python
tn.warmup(langs=tn.SUPPORTED_LANGS) -> dict[str, bool]
```
Loads every normalizer up front (builds/loads the Japanese FSTs, touches the
Vietnamese CRF model) so the first real call doesn't pay that cost. Returns
which languages loaded successfully.

```python
tn.available() -> dict
```
Status snapshot for a health check: whether the layer is enabled, the LRU
cache's hit/miss counts, and per-language `{library, version, ready, error}`.

## Configuration

All via environment variables, read once at import time:

| Variable | Default | Effect |
|---|---|---|
| `OMNIVOICE_NORMALIZE` | on | `"0"`/`"false"`/`"no"` disables the whole layer |
| `OMNIVOICE_NORMALIZE_JP` | on | disable just the Japanese path |
| `OMNIVOICE_NORMALIZE_VN` | on | disable just the Vietnamese path |
| `OMNIVOICE_NORM_CACHE_DIR` | `/root/.cache/omnivoice-norm` | where the Japanese FSTs are built/cached |
| `OMNIVOICE_NORM_CACHE_SIZE` | `2048` | LRU entries for `(lang, text)` results |
| `OMNIVOICE_NORM_MAX_CHARS` | `4000` | inputs longer than this pass through untouched (WFST normalization is superlinear on pathological input) |
| `OMNIVOICE_NORM_MAX_GROWTH` | `25` | safety valve — output more than this many times longer than input falls back to the original text |
| `OMNIVOICE_JA_TRANSLITERATE` | off | `"1"` renders Latin words as katakana |

A per-call escape hatch also exists at the caller level: pass `normalize=False`
through to a router built on this module to bypass normalization for one
request (see the original `router.py` this was extracted from, if you need
that pattern).

## Japanese specifics

The Japanese path does more than digit-to-kanji conversion:

- **Counter suffixes** (万/億/兆/月/年/時/人/本/…) already present in the
  source text are folded into the same reading as the digits they attach to
  (`"25万円"` → correctly read as one unit — `"万"` isn't left as a stray,
  unread kanji character next to a kana number), and irregular readings
  resolve correctly (4月 = *shigatsu*, not *yongatsu*; 3本 = *sanbon*, not
  *sanhon*; 一兆 = *itchō* with the euphonic contraction).
- **Large numbers (≥1億)** are grouped correctly in 万/億/兆 units — the
  underlying WFST grammar (`tn.japanese`'s Cardinal, which Money/Date/
  Fraction/Math/Measure all embed) only groups correctly up to 8 digits;
  above that it fails unpredictably (digit-by-digit, or a literal Western
  comma/zero-run leaking into what's supposed to be pure kanji). This module
  detects comma-grouped numbers ≥100,000,000 and converts them itself,
  before the broken grammar ever sees them.
- **Phone numbers and serial/ID-shaped digit strings** are left exactly as
  written (never spelled out), both hyphenated (`090-1234-5678`) and bare
  (`09012345678`, or any other 9+-digit run with no comma grouping) — a
  written phone number measures better heard-correctly than any spelled-out
  alternative.
- **Capital letters attached to a number** get their letter-name reading
  together with the digits (`4R` → ヨンアール, `A4` → エイヨン, `H2O` →
  エイチニオー, `B5判` → ビーゴ判). The reading is `pyopenjtalk`'s own
  dictionary entry for the letter, which its kana formatter otherwise drops.
  Only a run of one to four capitals directly next to a rewritten digit run is
  read this way; acronyms and Latin words anywhere else (`（EU）`, `USBメモリ`,
  `Windows11`) are left exactly as written, and unit symbols (`4V`, `10KB`)
  are still read as units.

**Deliberately out of scope** (documented, not silent bugs):
- Lexicalised English-number forms keep a Japanese number reading: `F1` →
  エフイチ (not エフワン), `3D` → サンディー, `5G` → ゴジー. A per-deployment
  override table is the right place for these, not the generic rule.
- Floor-plan codes: `4LDK` is read as four litres plus the letters
  (`ヨンリットルディーケイ`) because the library's measure grammar consumes `4L`
  before this module sees it.
- `pyopenjtalk` occasionally misreads a kanji numeral once a letter is glued
  to it: `13M` → ジューソーエム (十三 taken as the Osaka place name), `M18` →
  エムワンハチ. Measured at 6 of 400 cases in the benchmark's `alnum` category.
- A decimal tail on a ≥1億 amount (`"1,500,000,000.5"`) is left completely
  untouched rather than partially converted.
- Numbers ≥京 (10¹⁶) are left untouched — `pyopenjtalk` was empirically found
  to mis-segment a reading at that magnitude as a proper name rather than a
  number.
- A leading `-`/`−` sign and full-width digit input aren't recognized by the
  large-number converter.

See the extensive comments in `text_normalization.py` for the empirical
evidence behind each of these (what the broken grammar actually outputs,
what a correct reading was verified to sound like via `pyopenjtalk`).

## Testing

```bash
python -m unittest discover -s tests -v
```

## Benchmarks

`benchmarks/ja_normalization/` is the evaluation harness the Japanese fixes
above were built and verified against:

```bash
cd benchmarks/ja_normalization
python build_dataset.py          # regenerate data/gold.jsonl + data/real.jsonl
python bench.py                  # baseline vs. raw library vs. this module, scored against gold.jsonl
python conformance.py            # runs WeTextProcessing's own test pairs through this wrapper
```

`data/gold.jsonl` (10,400 synthetic carrier sentences across 26 categories,
with readings generated independently by `ja_reading.py`) and `data/real.jsonl`
(real-world sentences for reference-free metrics) ship pre-built.

> **Note:** `conformance.py` compares this module's output against
> `tn.japanese`'s own raw test pairs verbatim. That comparison predates the
> counter-suffix and kana-reading work above — this module now *deliberately*
> diverges from the raw library's kanji output in the ways described above
> (converting rewritten fragments to kana, protecting phone/serial numbers,
> correcting large numbers), so most of today's "mismatches" are that
> intentional divergence, not regressions. Read failures with that in mind
> rather than treating the pass count as a correctness gate.
