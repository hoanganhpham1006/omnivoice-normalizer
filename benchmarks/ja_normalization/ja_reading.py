"""Japanese spoken-form reading rules + variant-tolerant scoring helpers.

The Japanese TTS front-end convention (and WeTextProcessing's output) is kanji
numerals, not kana: ``1,115,000`` is spoken as ``百十一万五千``. This module
generates those readings independently of the library under test, so the
benchmark measures the library rather than echoing it back.

Scoring folds conventions that are all correct Japanese:

* ``〇`` / ``零`` / ``ゼロ`` for zero
* a dropped leading ``一`` before ``十 / 百 / 千`` (``一千万`` == ``千万``)
* phone-number separators ``の`` / ``ー`` / ``-``

Structurally different readings (a year as a cardinal ``二千二十三年`` vs as
digits ``二〇二三年``; a long digit string read as a quantity vs digit-by-digit)
are emitted as alternative accepted readings instead, because folding those
would hide real errors.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Iterable, List

DIGITS = ["〇", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
GROUP_SCALES = ["", "万", "億", "兆", "京"]


# ---------------------------------------------------------------------------
# Number reading
# ---------------------------------------------------------------------------


def _read_group4(n: int) -> str:
    """Read 1..9999 as kanji. A leading 一 is dropped before 十/百/千, which is
    the standard written form (千百十八, not 一千一百一十八)."""
    assert 0 <= n < 10000
    if n == 0:
        return ""
    out = []
    thousands, rest = divmod(n, 1000)
    hundreds, rest = divmod(rest, 100)
    tens, units = divmod(rest, 10)
    if thousands:
        out.append("千" if thousands == 1 else f"{DIGITS[thousands]}千")
    if hundreds:
        out.append("百" if hundreds == 1 else f"{DIGITS[hundreds]}百")
    if tens:
        out.append("十" if tens == 1 else f"{DIGITS[tens]}十")
    if units:
        out.append(DIGITS[units])
    return "".join(out)


def read_int(n: int) -> str:
    """Read a non-negative integer as kanji numerals."""
    if n == 0:
        return "〇"
    groups: List[int] = []
    while n > 0:
        n, g = divmod(n, 10000)
        groups.append(g)
    parts = []
    for idx in range(len(groups) - 1, -1, -1):
        if groups[idx] == 0:
            continue
        # 一 is kept before 万/億/兆 (一万), unlike before 十/百/千.
        chunk = _read_group4(groups[idx])
        if idx > 0 and groups[idx] == 1:
            chunk = "一"
        parts.append(chunk + GROUP_SCALES[idx])
    return "".join(parts)


def read_digits(s: str) -> str:
    """Read a digit string one character at a time (IDs, phone numbers, years)."""
    return "".join(DIGITS[int(c)] for c in s if c.isdigit())


def read_decimal(int_part: int, frac_digits: str) -> List[str]:
    """Accepted readings for a decimal. The fractional tail is read
    digit-by-digit in Japanese (九点九九九九九)."""
    return [f"{read_int(int_part)}点{read_digits(frac_digits)}"]


def read_year(y: int) -> List[str]:
    """Both the cardinal and the digit-string year readings are standard."""
    return [read_int(y), read_digits(str(y))]


def read_number_or_digits(n: int) -> List[str]:
    """For long bare digit strings, a quantity reading and a digit-by-digit
    reading are both defensible without context."""
    readings = [read_int(n)]
    digits = read_digits(str(n))
    if digits not in readings:
        readings.append(digits)
    return readings


# ---------------------------------------------------------------------------
# Canonicalization for scoring
# ---------------------------------------------------------------------------

CHAR_EQUIV = {
    "零": "〇",
    "ゼ": "",  # ゼロ -> 〇 handled by the string pass below
    "ロ": "",
}

_ZERO_WORDS = [("ゼロ", "〇"), ("零", "〇"), ("れい", "〇")]
# pyopenjtalk reads a bare letter name シー (C) as スィー when the katakana
# stands alone in a gold reading, but as シー when the pipeline has glued it to
# a number, so "シー八" and "シーハチ" disagree in pronunciation space for no
# reason. Same letter; folded like the zero spellings above.
_KANA_EQUIV = [("スィ", "シ")]
# の / ー / - between digit groups are all read as the same pause.
_SEPARATORS = "のー-‐−–—ｰ"
_DROP_ONE_RE = re.compile(r"一(?=[十百千])")
_PUNCT_RE = re.compile(r"[^\w]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def canon(text: str) -> str:
    """Fold the conventions that are all equally correct, then drop spacing and
    punctuation (Japanese is written without spaces, so this is a plain
    character stream)."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _ZERO_WORDS + _KANA_EQUIV:
        text = text.replace(src, dst)
    text = _DROP_ONE_RE.sub("", text)
    for ch in _SEPARATORS:
        text = text.replace(ch, "")
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub("", text)
    return text.lower()


def canon_tokens(text: str) -> List[str]:
    """Characters are the comparison unit for Japanese, so `wer` below is a
    character error rate."""
    return list(canon(text))


def despace(text: str) -> str:
    return canon(text)


def contains_reading(hypothesis: str, reading: str) -> bool:
    r = canon(reading)
    return bool(r) and r in canon(hypothesis)


contains_reading_lenient = contains_reading


@functools.lru_cache(maxsize=4096)
def to_kana(text: str) -> str:
    """Katakana pronunciation via pyopenjtalk. Empty string if unavailable."""
    try:
        import pyopenjtalk

        out = pyopenjtalk.g2p(text, kana=True)
        return out if isinstance(out, str) else ""
    except Exception:  # noqa: BLE001
        return ""


@functools.lru_cache(maxsize=1)
def _wetext():
    """WeTextProcessing on its own, for canonicalizing an ASR transcript back to
    written form (digits -> kanji numerals) when comparing against gold."""
    from tn.japanese.normalizer import Normalizer

    return Normalizer(cache_dir="/root/.cache/omnivoice-norm", overwrite_cache=False,
                      remove_puncts=False, full_to_half=True, transliterate=False,
                      remove_interjections=False, tag_oov=False)


def to_written_form(text: str) -> str:
    try:
        return _wetext().normalize(text)
    except Exception:  # noqa: BLE001
        return text


def match_any(hypothesis: str, readings: Iterable[str], lenient: bool = True) -> bool:
    """Accept a match in written form or in pronunciation.

    The gold readings are kanji numerals, but the shipped pipeline emits a kana
    reading, so a written-form comparison alone would score every correct kana
    output as wrong. Both sides are therefore also compared as pronunciations,
    which is the level the two representations actually agree on.
    """
    readings = list(readings)
    if any(contains_reading(hypothesis, r) for r in readings):
        return True
    spoken = to_kana(hypothesis)
    if not spoken:
        return False
    spoken = canon(spoken)
    return any(canon(to_kana(r)) in spoken for r in readings if to_kana(r))


# ---------------------------------------------------------------------------
# Residual unspeakable material
# ---------------------------------------------------------------------------

# Japanese punctuation is speakable prosody, not a leak. Latin letters are not a
# leak either — Japanese text legitimately contains them and the model reads
# them — so they are reported separately by the caller if wanted.
_ALLOWED_PUNCT = set(
    " \t\n.,。、，．・「」『』（）()｛｝[]【】〈〉《》!?！？:：;；'\"“”‘’…―ー－-‥　"
)
_LETTER_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)


def leaked_chars(text: str) -> List[str]:
    """Characters a Japanese TTS front-end should never hand to the model:
    digits (half or full width), maths/currency/unit symbols, emoji."""
    bad: List[str] = []
    for ch in text:
        if ch.isdigit() or ch in "０１２３４５６７８９":
            bad.append(ch)
            continue
        if ch in _ALLOWED_PUNCT:
            continue
        if _LETTER_RE.match(ch):
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            continue
        bad.append(ch)
    return bad


def has_leak(text: str) -> bool:
    return bool(leaked_chars(text))


def latin_char_rate(text: str) -> float:
    """Share of Latin characters — reported alongside leaks, not as a leak."""
    if not text:
        return 0.0
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    return latin / len(text)


# ---------------------------------------------------------------------------
# Character error rate (named `wer` for interface parity with vi_reading)
# ---------------------------------------------------------------------------


def wer(ref: str, hyp: str) -> float:
    """Character error rate, measured on pronunciations.

    The pipeline may emit kanji numerals or a kana reading depending on the
    fragment, so comparing written forms character-by-character would score a
    correct kana reading as ~100% wrong. Both sides are mapped to their
    pronunciation first, which is the representation they genuinely share.
    """
    ref_k, hyp_k = to_kana(ref), to_kana(hyp)
    if ref_k and hyp_k:
        ref, hyp = ref_k, hyp_k
    r = canon_tokens(ref)
    h = canon_tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[len(h)] / len(r)
