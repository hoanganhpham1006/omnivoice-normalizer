"""Text normalization for the OmniVoice TTS router.

The router forwards request text to a single-language backend, which passes it
straight to ``OmniVoice.generate()``. Without normalization the acoustic model
receives literal digits, dates, currency signs and emoji — characters it has no
pronunciation for — and ``utils/duration.py::RuleDurationEstimator`` also sizes
the output audio from that written form, so the utterance is budgeted too short
for the words a speaker would actually say.

This module turns written form into spoken form before the request leaves the
router:

    vn  soe-vinorm        CRF non-standard-word tagger + rule expanders
    jp  WeTextProcessing  pynini WFST grammars (tn.japanese), kanji-numeral output

It deliberately imports no torch. The router runs in its own CPU-only conda env
(``omnivoice-nlp``) because soe-vinorm pins ``numpy<2`` and
``huggingface-hub<1.0``, while the GPU env needs numpy 2.x and hub 1.x for
transformers. Keeping the two apart means the TTS stack's dependency tree is
never touched by a front-end library.

Every import is lazy and every failure is non-fatal: if a library is missing or
raises, the original text is returned and the request still gets audio.

Env knobs:
    OMNIVOICE_NORMALIZE         "0" disables the whole layer (default on)
    OMNIVOICE_NORM_CACHE_DIR    where the Japanese FSTs are cached/built
    OMNIVOICE_NORM_CACHE_SIZE   LRU entries for (lang, text) results
    OMNIVOICE_NORM_MAX_CHARS    inputs longer than this are passed through
    OMNIVOICE_JA_TRANSLITERATE  "1" renders latin words as katakana
"""

from __future__ import annotations

import functools
import logging
import os
import re
import time
from typing import Callable, Dict, Optional, Tuple

log = logging.getLogger("omnivoice-normalize")

SUPPORTED_LANGS = ("jp", "vn")

def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


ENABLED = _flag("OMNIVOICE_NORMALIZE")
# Per-language switches, because the two endpoints do not benefit equally.
# Vietnamese improves on every measure. Japanese is mixed: the checkpoint
# already reads Arabic numerals well and misreads some kanji-numeral forms, so
# whether this is a win depends on the text a deployment actually sends. See
# benchmarks/ja_normalization/README.md for the per-category evidence.
LANG_ENABLED = {
    "vn": _flag("OMNIVOICE_NORMALIZE_VN"),
    "jp": _flag("OMNIVOICE_NORMALIZE_JP"),
}
CACHE_DIR = os.environ.get(
    "OMNIVOICE_NORM_CACHE_DIR", "/root/.cache/omnivoice-norm"
)
CACHE_SIZE = int(os.environ.get("OMNIVOICE_NORM_CACHE_SIZE", "2048"))
# Inputs beyond this are forwarded untouched: WFST normalization is superlinear
# on pathological input and a TTS request should never hang on the front end.
MAX_CHARS = int(os.environ.get("OMNIVOICE_NORM_MAX_CHARS", "4000"))
# Runaway guard. Spelling out a long digit string legitimately multiplies
# length, so this is deliberately loose — it only catches a broken grammar.
MAX_GROWTH = float(os.environ.get("OMNIVOICE_NORM_MAX_GROWTH", "25"))
JA_TRANSLITERATE = os.environ.get("OMNIVOICE_JA_TRANSLITERATE", "0").strip() in (
    "1",
    "true",
    "yes",
)


# ---------------------------------------------------------------------------
# Tag preservation
# ---------------------------------------------------------------------------

# OmniVoice parses bracketed spans out of the input text: non-verbal tags such
# as [laughter] / [sigh], and pronunciation overrides such as the CMU phone
# sequence [B EY1 S]. A normalizer would happily rewrite those into nonsense, so
# each bracketed span is cut out and the surrounding text normalized around it.
_TAG_RE = re.compile(r"\[[^\[\]]*\]")
_EDGE_WS_RE = re.compile(r"^(\s*)(.*?)(\s*)$", re.DOTALL)


def _normalize_segments(text: str, fn: Callable[[str], str],
                       protect: "re.Pattern[str]" = None) -> str:
    """Apply ``fn`` to every non-tag segment, leaving [tags] byte-identical.

    Splitting beats placeholder substitution here: a placeholder has to survive
    a tokenizer that was never told about it, whereas a split simply never shows
    the tag to the normalizer.
    """
    protect = protect or _TAG_RE
    tags = protect.findall(text)
    if not tags:
        return _normalize_segment(text, fn)
    # findall returns group contents when the pattern has groups; rebuild the
    # literal spans from finditer so protected text is restored byte-for-byte.
    tags = [m.group(0) for m in protect.finditer(text)]
    parts = protect.split(text)
    if len(parts) != len(tags) + 1:  # capturing groups leaked into split()
        parts = [text[a:b] for a, b in
                 zip([0] + [m.end() for m in protect.finditer(text)],
                     [m.start() for m in protect.finditer(text)] + [len(text)])]
    out = []
    for i, segment in enumerate(parts):
        out.append(_normalize_segment(segment, fn))
        if i < len(tags):
            out.append(tags[i])
    return "".join(out)


def _normalize_segment(segment: str, fn: Callable[[str], str]) -> str:
    """Normalize one segment, preserving its own leading/trailing whitespace.

    soe-vinorm returns ``" ".join(tokens)`` and therefore eats edge whitespace;
    without this the segments either collide or lose a space when rejoined
    around a tag.
    """
    if not segment.strip():
        return segment
    lead, core, trail = _EDGE_WS_RE.match(segment).groups()
    return f"{lead}{fn(core)}{trail}"


# ---------------------------------------------------------------------------
# Japanese punctuation restoration
# ---------------------------------------------------------------------------

# tn.japanese's postprocessor rewrites the Japanese sentence terminators to
# their ASCII forms on *every* sentence, including ones with nothing to
# normalize -- `full_to_half` does not gate it. The jp backend is tuned on
# Japanese punctuation and OmniVoice's own `_combine_text` folds only full-width
# parentheses, so the terminators are put back. Restoration is deliberately
# narrow: only a terminator sitting directly after kana/kanji, plus a trailing
# one the caller actually wrote full-width. "example.com" and a decimal point
# are therefore left alone.
_JA_CHARS = "\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f"
_JA_TERMINATORS = {".": "\u3002", "?": "\uff1f", "!": "\uff01"}
_JA_HAS_CHARS_RE = re.compile(f"[{_JA_CHARS}]")
_JA_AFTER_KANA_RE = re.compile(f"(?<=[{_JA_CHARS}])([.?!])")
_JA_TRAILING_RE = re.compile(r"([.?!])(\s*)$")
_JA_ORIGINAL_TRAILING_RE = re.compile(r"([\u3002\uff1f\uff01])\s*$")


def _ja_restore_punctuation(original: str, out: str) -> str:
    if not _JA_HAS_CHARS_RE.search(original):
        return out
    out = _JA_AFTER_KANA_RE.sub(lambda m: _JA_TERMINATORS[m.group(1)], out)
    tail = _JA_ORIGINAL_TRAILING_RE.search(original)
    if tail:
        out = _JA_TRAILING_RE.sub(lambda m: tail.group(1) + m.group(2), out)
    return out


# ---------------------------------------------------------------------------
# Japanese kana readings
# ---------------------------------------------------------------------------

# WeTextProcessing emits kanji numerals, which is correct written Japanese but
# measured worse end-to-end: the checkpoint reads Arabic numerals fluently and
# misreads kanji-numeral forms (juu-nana-seiki came out as garbage). Kana
# removes the ambiguity entirely - there is nothing left to resolve.
#
# The reading has to come from a morphological dictionary, not a rewriter,
# because Japanese number readings depend on the counter that follows and are
# irregular (1-pon vs 3-bon vs 6-ppon; hitori/futari/san-nin; hatsuka).
# pyopenjtalk (Open JTalk + unidic) supplies that.
#
# Only the fragments normalization actually rewrote get a reading - the rest of
# the sentence stays ordinary Japanese, the way furigana annotates just the
# word that needs it. Reading the whole utterance in kana measured worse (63%
# of sentences heard correctly, against 81% for this approach): all-kana text
# is itself unusual input, and it throws away the word boundaries and kanji
# semantics the model relies on.
#
# The same dictionary also owns the reading of a Latin letter (Ｒ -> アール,
# pos 記号/アルファベット), which is what a digit+letter token like "4R" needs:
# the model was heard saying "Four アール" for it. pyopenjtalk's own
# g2p(kana=True) formatter throws that reading away - it substitutes the
# surface string for every 記号 token - so _to_kana walks run_frontend()
# itself and keeps the pronunciation for alphabet symbols only. Everything
# else is formatted exactly as g2p would (verified identical on 8,720 real
# fragments; see docs/research/ja-letter-readings.md).


@functools.lru_cache(maxsize=1)
def _openjtalk():
    import pyopenjtalk

    pyopenjtalk.g2p("\u521d\u671f\u5316", kana=True)  # warm the dictionary
    return pyopenjtalk


def _to_kana(text: str) -> str:
    """Katakana reading of one fragment. Returns it unchanged on failure.

    Mirrors ``pyopenjtalk.g2p(text, kana=True)`` - which is itself just this
    walk over ``run_frontend()`` - except that an alphabet symbol keeps its
    dictionary pronunciation instead of its surface form (see above).
    """
    if not text or not text.strip():
        return text
    try:
        parts = []
        for feature in _openjtalk().run_frontend(text):
            if feature["pos"] == "記号" and feature.get("pos_group1") != "アルファベット":
                parts.append(feature["string"])  # punctuation etc.: as written
            else:
                parts.append(feature["pron"])
        # g2p strips this too: the dictionary marks a devoiced mora with it
        # (X -> エック’ス).
        kana = "".join(parts).replace("’", "")
    except Exception as exc:  # noqa: BLE001
        log.warning("kana reading failed for %r (%s); keeping original", text[:40], exc)
        return text
    return kana if kana.strip() else text


# A source sentence almost always spells the counter/scale word itself
# ("25万円", "12月", "3時30分") - only the digit run differs from what
# WeTextProcessing emits, so the counter kanji is untouched and, by the diff
# below, would otherwise be left dangling next to the kana reading of the
# digits it belongs to. That defeats the reason pyopenjtalk is used at all:
# the irregular reading (1-pon vs 3-bon vs 6-ppon; hitori/futari; hatsuka;
# 4-gatsu is "shigatsu" not "yongatsu") lives on the counter, not the digits,
# and pyopenjtalk can only resolve it when both are handed over together. So
# a counter suffix immediately following a rewritten digit run is pulled into
# its kana span even though the counter itself never changed. Longest match
# first, so "ヶ月" is matched whole instead of leaving a stray "ヶ".
_JA_COUNTER_SUFFIXES = (
    # Number-scale words: part of the numeral's value, not a separate word.
    "兆", "億", "万", "千", "百",
    # Calendar / duration counters.
    "ヶ月", "か月", "カ月", "箇月", "月",
    "年間", "年生", "年",
    "週間", "週",
    "日間", "日",
    "時間", "時",
    "分間", "分",
    "秒間", "秒",
    # Common classifier counters.
    "人", "名", "個", "台", "本", "冊", "枚", "匹", "頭", "羽", "杯",
    "回", "度", "歳", "才", "円",
)
_JA_COUNTER_RE = re.compile(
    "|".join(re.escape(s) for s in sorted(_JA_COUNTER_SUFFIXES, key=len, reverse=True))
)

# The same applies to a short run of capital letters on either side of a
# rewritten digit run: "4R", "A4", "H2O", "B5判", "ISO9001" are one token, and
# the dictionary reads digit and letter together only if it sees both (the
# digit alone comes back as ヨン and the letter is never rewritten at all).
# Capped at four letters and uppercase-only, so a Latin word next to a number
# (Windows11, COVID19, Type2, iPhone 15) is never touched.
#
# Crucially the fold is gated on the neighbouring rewrite having replaced
# *digits* in the source, not on there being a rewrite at all: tn.japanese's
# full_to_half also rewrites （ ） and other width variants, and folding next
# to those reads every parenthesised acronym (（GDP）, （EU）) as letter names.
# Measured on the real-text benchmark track: 102 sentences changed without
# the gate, 41 with it, all 41 being genuine digit+letter tokens.
_JA_LETTER_RUN_RE = re.compile(r"[A-Z]{1,4}(?![A-Za-z])")          # after the digits
_JA_LEADING_LETTERS_RE = re.compile(r"(?<![A-Za-z])[A-Z]{1,4}$")    # before the digits


def _ja_kana_spans(original: str, normalized: str) -> str:
    """Give a reading only to the fragments normalization rewrote.

    Everything the normalizer left alone is text the model already handles, so
    it is passed through untouched; only the rewritten fragments - a date, a
    money amount, a unit - are replaced by their katakana reading. A counter
    suffix right after a rewritten digit run is folded into that reading too,
    as is a short capital-letter run on either side of one; see
    ``_JA_COUNTER_SUFFIXES`` and ``_JA_LETTER_RUN_RE`` above for why.
    """
    import difflib

    opcodes = list(
        difflib.SequenceMatcher(None, original, normalized, autojunk=False).get_opcodes()
    )
    # Which rewrites replaced digits in the source - the only ones allowed to
    # pull neighbouring letters into their span.
    rewrote_digits = [
        tag != "equal" and any(c.isdigit() for c in original[i1:i2])
        for tag, i1, i2, _j1, _j2 in opcodes
    ]
    out = []
    lead = ""  # capital letters held back from the preceding equal run
    i = 0
    while i < len(opcodes):
        tag, _i1, _i2, j1, j2 = opcodes[i]
        chunk = normalized[j1:j2]
        if tag == "equal":
            m = None
            if i + 1 < len(opcodes) and rewrote_digits[i + 1]:
                # Bounded search on the whole string, so the lookbehind sees
                # the character before this run rather than the chunk edge.
                m = _JA_LEADING_LETTERS_RE.search(normalized, j1, j2)
            if m:
                out.append(normalized[j1:m.start()])
                lead = m.group(0)
            else:
                out.append(chunk)
                lead = ""
            i += 1
            continue

        extra = ""
        if i + 1 < len(opcodes) and opcodes[i + 1][0] == "equal":
            _, ni1, ni2, nj1, nj2 = opcodes[i + 1]
            m = _JA_COUNTER_RE.match(normalized, nj1, nj2)
            if m is None and rewrote_digits[i]:
                # No endpos here: a lookahead cannot see past it, so the run
                # would look word-final at the chunk edge even mid-word.
                m = _JA_LETTER_RUN_RE.match(normalized, nj1)
                if m and m.end() > nj2:
                    m = None
            if m:
                extra = m.group(0)
                # Shrink the following equal run so its already-emitted prefix
                # isn't repeated when the loop reaches it next iteration.
                opcodes[i + 1] = ("equal", ni1, ni2, nj1 + len(extra), nj2)

        out.append(_to_kana(lead + chunk + extra))
        lead = ""
        i += 1
    return "".join(out)





# ---------------------------------------------------------------------------
# Japanese large numbers
# ---------------------------------------------------------------------------

# tn.japanese's Cardinal grammar (shared by Money/Date/Fraction/Math/Measure)
# only correctly groups digits into 万(10**4)/億(10**8) units up to 8 digits.
# At 9+ digits it fails unpredictably depending on comma formatting: bare
# digits fall back to reading one character at a time ("100000000" ->
# "一〇〇〇〇〇〇〇〇" instead of "一億"), while comma-grouped input leaks a
# literal Western comma or a run of "〇" zeros into what's supposed to be
# pure kanji ("1,500,000,000" -> "一,五億"). That corruption then survives
# straight through the kana-reading step below, since there's nothing left
# to fix once the grammar itself has already produced the wrong characters.
# So a large number is converted to correct kanji ourselves, before
# tn.japanese ever sees it, rather than trying to repair its output after.
#
# The grouping algorithm (GROUP_SCALES / _read_group4 / read_int, renamed
# here to this file's _ja_ convention) is ported from the gold-reference
# generator at benchmarks/ja_normalization/ja_reading.py:28,36-74 - that
# module already implements this correctly, as what the benchmark scores
# *against*, but was never wired into the shipped normalization path.
_JA_GROUP_SCALES = ("", "万", "億", "兆", "京")
_JA_DIGITS = "〇一二三四五六七八九"


def _ja_group4_to_kanji(n: int) -> str:
    """Read 1..9999 as kanji. A leading 一 is dropped before 十/百/千, which
    is the standard written form (千百十八, not 一千一百一十八)."""
    assert 0 <= n < 10000
    if n == 0:
        return ""
    out = []
    thousands, rest = divmod(n, 1000)
    hundreds, rest = divmod(rest, 100)
    tens, units = divmod(rest, 10)
    if thousands:
        out.append("千" if thousands == 1 else f"{_JA_DIGITS[thousands]}千")
    if hundreds:
        out.append("百" if hundreds == 1 else f"{_JA_DIGITS[hundreds]}百")
    if tens:
        out.append("十" if tens == 1 else f"{_JA_DIGITS[tens]}十")
    if units:
        out.append(_JA_DIGITS[units])
    return "".join(out)


def _ja_int_to_kanji(n: int) -> str:
    """Read a non-negative integer as kanji numerals, grouped every 4 digits
    (万/億/兆/京), not every 3 (thousand) - see module docstring context."""
    if n == 0:
        return "〇"
    groups = []
    while n > 0:
        n, g = divmod(n, 10000)
        groups.append(g)
    parts = []
    for idx in range(len(groups) - 1, -1, -1):
        if groups[idx] == 0:
            continue
        # 一 is kept before 万/億/兆/京 (一万), unlike before 十/百/千.
        chunk = _ja_group4_to_kanji(groups[idx])
        if idx > 0 and groups[idx] == 1:
            chunk = "一"
        parts.append(chunk + _JA_GROUP_SCALES[idx])
    return "".join(parts)


# tn.japanese's own failure boundary (see above): below this, the existing
# grammar already groups correctly and is left untouched.
_JA_LARGE_NUMBER_THRESHOLD = 100_000_000  # 1億
# pyopenjtalk/UniDic was empirically observed to mis-segment a 京-scale
# (10**16) reading as a proper name rather than a number+counter (一京円 ->
# "イチキョーマドカ", not an "en" reading) - the algorithm above supports up
# to 京 for fidelity with the ported reference, but production stops short
# of ever actually using that range.
_JA_LARGE_NUMBER_CEILING = 10 ** 16  # 京
# Comma-grouped only, not bare digits: a genuine quantity is overwhelmingly
# written with Western comma grouping; a bare, unformatted 9+-digit run is,
# per this codebase's own benchmark convention (see build_dataset.py's
# gen_serial), conventionally an ID rather than a quantity, and is instead
# left untouched by _JA_PROTECTED_RE below. The optional decimal tail is
# matched (but never converted, see _ja_expand_large_numbers) purely so it
# can't be split off and left dangling next to a kanji number - that would
# otherwise confuse _ja_restore_punctuation, which has only ever had to
# handle a "." following Arabic digits before.
_JA_COMMA_NUMBER_RE = re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)")


def _ja_expand_large_numbers(text: str) -> str:
    """Rewrite comma-grouped numbers >= 1億 as correct kanji, in place."""

    def repl(m: "re.Match[str]") -> str:
        raw = m.group(0)
        if "." in raw:
            # A decimal remainder on a large amount is deliberately out of
            # scope: leave the whole match (integer + decimal) untouched
            # rather than convert only the integer part and strand the tail.
            return raw
        value = int(raw.replace(",", ""))
        if not (_JA_LARGE_NUMBER_THRESHOLD <= value < _JA_LARGE_NUMBER_CEILING):
            return raw
        return _ja_int_to_kanji(value)

    return _JA_COMMA_NUMBER_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Japanese phone numbers
# ---------------------------------------------------------------------------

# A phone number is left exactly as written - every spelled-out alternative
# measured worse than the written form (8% of sentences heard correctly for
# a kana spelling, against 58% for the digits), so the span is protected
# from the grammar instead of rewritten. This also covers bare (unhyphenated)
# phone numbers and serial/ID-shaped digit strings: a 9+-digit run with no
# comma grouping is, per this codebase's own benchmark convention (see
# build_dataset.py's gen_serial), conventionally an identifier rather than a
# quantity - and it's also exactly where tn.japanese's own bare-digit
# Cardinal grammar starts breaking (see "Japanese large numbers" below), so
# leaving it untouched is doubly justified.
_JA_PROTECTED_RE = re.compile(
    r"\[[^\[\]]*\]|(?<![0-9\uff10-\uff19])"
    r"0\d{1,4}(?:[-\uff0d\u2212]\d{1,4}){1,2}(?![0-9\uff10-\uff19])"
    r"|(?<![0-9,\uff10-\uff19])\d{9,}(?![0-9,\uff10-\uff19])"
)


# ---------------------------------------------------------------------------
# Residual emoji
# ---------------------------------------------------------------------------

# Anything still matching after normalization is, by definition, a pictograph
# the normalizer chose not to expand, and no TTS model has a pronunciation for
# it. soe-vinorm already drops these for Vietnamese; tn.japanese passes them
# straight through, so stripping here keeps the two endpoints consistent.
# Deliberately narrow: emoji and pictographs only, not every symbol - dropping
# a unit sign like the degree symbol would silently change meaning, so those
# stay visible as a known gap instead.
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"   # emoticons, pictographs, transport, symbols
    "\U0001f1e6-\U0001f1ff"   # regional indicators (flags)
    "\u2600-\u27bf"           # misc symbols and dingbats
    "\u2b00-\u2bff"           # extra arrows and shapes
    "\ufe00-\ufe0f"           # variation selectors
    "\u200d"                   # zero-width joiner
    "]+"
)
_EMOJI_GAP_RE = re.compile(r"[ \t]{2,}")


def _strip_emoji(text: str) -> str:
    if not _EMOJI_RE.search(text):
        return text
    return _EMOJI_GAP_RE.sub(" ", _EMOJI_RE.sub("", text)).strip()


# ---------------------------------------------------------------------------
# Vietnamese spacing restoration
# ---------------------------------------------------------------------------

# soe-vinorm tokenizes and returns " ".join(tokens), which detaches punctuation
# from the preceding word ("troi dep ." instead of "troi dep."). Left alone that
# would alter every Vietnamese sentence, including ones with nothing to
# normalize, so the spacing convention of written Vietnamese is restored.
_VI_SPACE_BEFORE_RE = re.compile(r"\s+([,.;:!?%\u2026\)\]\}])")
_VI_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _vi_restore_spacing(out: str) -> str:
    out = _VI_SPACE_BEFORE_RE.sub(r"\1", out)
    out = _VI_SPACE_AFTER_OPEN_RE.sub(r"\1", out)
    return _MULTI_SPACE_RE.sub(" ", out)


# ---------------------------------------------------------------------------
# Per-language backends
# ---------------------------------------------------------------------------


class _Backend:
    """One language's normalizer: lazily constructed, failure recorded."""

    lang = ""
    library = ""

    def __init__(self) -> None:
        self._fn: Optional[Callable[[str], str]] = None
        self.version = "unknown"
        self.error: Optional[str] = None
        self._loaded = False

    def _build(self) -> Callable[[str], str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def load(self) -> bool:
        if self._loaded:
            return self._fn is not None
        self._loaded = True
        t0 = time.perf_counter()
        try:
            self._fn = self._build()
            log.info(
                "%s normalizer ready (%s %s) in %.2fs",
                self.lang, self.library, self.version, time.perf_counter() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._fn = None
            log.warning(
                "%s normalizer unavailable (%s): %s — text will pass through",
                self.lang, self.library, self.error,
            )
        return self._fn is not None

    def preprocess(self, text: str) -> str:
        return text

    def protected_spans(self):
        """Pattern for spans the normalizer must not see. Tags by default."""
        return _TAG_RE

    def postprocess(self, original: str, out: str) -> str:
        return out

    def __call__(self, text: str) -> str:
        if not self.load():
            return text
        assert self._fn is not None
        # preprocess() runs per-segment, inside the same protected-span
        # boundary as _fn itself - not once over the whole raw text - so a
        # rewrite step (e.g. the Japanese large-number expansion) can never
        # reach inside a protected [tag] or phone number. postprocess sees
        # the whole rejoined string so rules like trailing punctuation
        # behave correctly.
        fn = lambda seg: self._fn(self.preprocess(seg))  # noqa: E731
        return self.postprocess(
            text, _normalize_segments(text, fn, self.protected_spans()))

    def status(self) -> dict:
        return {
            "library": self.library,
            "version": self.version,
            "ready": self._fn is not None,
            "error": self.error,
            "loaded": self._loaded,
        }


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


class _VietnameseBackend(_Backend):
    lang = "vn"
    library = "soe-vinorm"

    def _build(self) -> Callable[[str], str]:
        from soe_vinorm import SoeNormalizer

        self.version = _dist_version("soe-vinorm")
        normalizer = SoeNormalizer()
        # Touch the model once so the CRF weights and dictionaries are resident
        # before the first real request rather than on it.
        normalizer.normalize("Đơn hàng 150.000 đồng giao ngày 25/12/2023.")
        return normalizer.normalize

    def postprocess(self, original: str, out: str) -> str:
        return _strip_emoji(_vi_restore_spacing(out))


class _JapaneseBackend(_Backend):
    lang = "jp"
    library = "WeTextProcessing"

    def _build(self) -> Callable[[str], str]:
        from tn.japanese.normalizer import Normalizer as JaNormalizer

        self.version = _dist_version("WeTextProcessing")
        os.makedirs(CACHE_DIR, exist_ok=True)
        # cache_dir must be set explicitly: the library defaults to
        # files("tn"), i.e. it writes compiled FSTs into site-packages.
        normalizer = JaNormalizer(
            cache_dir=CACHE_DIR,
            overwrite_cache=False,
            # Punctuation drives the model's prosody, so it stays in.
            remove_puncts=False,
            full_to_half=True,
            transliterate=JA_TRANSLITERATE,
            remove_interjections=False,
            tag_oov=False,
        )
        normalizer.normalize("2023年11月14日、150,000円を10km/hで。")
        return normalizer.normalize

    def preprocess(self, text: str) -> str:
        return _ja_expand_large_numbers(text)

    def protected_spans(self):
        return _JA_PROTECTED_RE

    def postprocess(self, original: str, out: str) -> str:
        out = _ja_restore_punctuation(original, out)
        out = _ja_kana_spans(original, out)
        return _strip_emoji(_ja_restore_punctuation(original, out))


_BACKENDS: Dict[str, _Backend] = {
    "vn": _VietnameseBackend(),
    "jp": _JapaneseBackend(),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=CACHE_SIZE)
def _normalize_cached(lang: str, text: str) -> str:
    backend = _BACKENDS[lang]
    out = backend(text)

    # Safety valve. A front-end library must never be able to break a request,
    # so anything implausible falls back to what the caller sent.
    if not out or not out.strip():
        log.warning("%s normalizer returned empty output; using original", lang)
        return text
    if len(out) > MAX_GROWTH * max(len(text), 1):
        log.warning(
            "%s normalizer grew text %dx (%d -> %d chars); using original",
            lang, len(out) // max(len(text), 1), len(text), len(out),
        )
        return text
    return out


def normalize(lang: str, text: str) -> Tuple[str, float, bool]:
    """Return ``(text, elapsed_ms, applied)`` for one request.

    Never raises: any failure yields the original text with ``applied=False``.
    """
    if (not ENABLED or not LANG_ENABLED.get(lang, True)
            or lang not in _BACKENDS or not text or not text.strip()):
        return text, 0.0, False
    if len(text) > MAX_CHARS:
        log.warning(
            "text of %d chars exceeds OMNIVOICE_NORM_MAX_CHARS=%d; passing through",
            len(text), MAX_CHARS,
        )
        return text, 0.0, False

    t0 = time.perf_counter()
    try:
        out = _normalize_cached(lang, text)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "%s normalization failed (%s: %s); using original text",
            lang, type(exc).__name__, exc,
        )
        return text, (time.perf_counter() - t0) * 1000, False
    return out, (time.perf_counter() - t0) * 1000, out != text


def warmup(langs=SUPPORTED_LANGS) -> Dict[str, bool]:
    """Load every normalizer up front. Used by the launcher pre-flight and at
    router startup so the first request never pays the FST build."""
    if not ENABLED:
        log.info("normalization disabled via OMNIVOICE_NORMALIZE=0")
        return {lang: False for lang in langs}
    out = {}
    for lang in langs:
        if lang not in _BACKENDS:
            continue
        if not LANG_ENABLED.get(lang, True):
            log.info("normalization disabled for %s via OMNIVOICE_NORMALIZE_%s=0",
                     lang, lang.upper())
            out[lang] = False
            continue
        out[lang] = _BACKENDS[lang].load()
    return out


def available() -> dict:
    """Normalizer status for /health."""
    return {
        "enabled": ENABLED,
        "cache_dir": CACHE_DIR,
        "cache": {
            "size": CACHE_SIZE,
            **{
                k: v
                for k, v in _normalize_cached.cache_info()._asdict().items()
                if k in ("hits", "misses", "currsize")
            },
        },
        "languages": {
            lang: {**b.status(), "enabled": LANG_ENABLED.get(lang, True)}
            for lang, b in _BACKENDS.items()
        },
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ok = warmup()
    print("loaded:", ok)
    if not all(ok.values()):
        print("status:", available(), file=sys.stderr)
        sys.exit(1)
    for lang, sample in (
        ("vn", "Đơn hàng 150.000 VNĐ giao ngày 25/12/2023 lúc 14:30 [laughter]."),
        ("jp", "2023/11/14に150,000円を10km/hで、090-1234-5678まで [laughter]。"),
    ):
        out, ms, applied = normalize(lang, sample)
        print(f"[{lang}] {ms:6.2f}ms applied={applied}\n  in : {sample}\n  out: {out}")
