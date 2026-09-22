"""Regression tests for the normalization layer (text_normalization.py).

Run inside an env with WeTextProcessing, pyopenjtalk-plus and soe-vinorm
installed (see the top-level README):

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import text_normalization  # noqa: E402


class NormalizerBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        loaded = text_normalization.warmup()
        if not all(loaded.values()):
            raise unittest.SkipTest(f"normalizers unavailable: {loaded}")

    def test_vietnamese_expands_non_standard_words(self) -> None:
        out, _, applied = text_normalization.normalize(
            "vn", "Cuộc họp ngày 25/12/2023."
        )
        self.assertTrue(applied)
        self.assertIn("hai mươi lăm tháng mười hai", out)
        self.assertNotIn("25", out)

    def test_japanese_expands_non_standard_words(self) -> None:
        out, _, applied = text_normalization.normalize(
            "jp", "2023/11/14に150,000円を支払いました。"
        )
        self.assertTrue(applied)
        self.assertNotIn("150,000", out)
        self.assertNotIn("2023/11/14", out)
        # The rewritten fragments come back as a kana reading.
        self.assertIn("ジューゴマン", out)
        self.assertIn("ニセンニジューサンネン", out)

    def test_japanese_large_numbers_use_the_man_unit_throughout(self) -> None:
        """25万 is 250,000 in units of 万 (10,000), read "nijuugo-man" - not
        digit-by-digit and not left with a stray kanji "万" dangling off the
        kana digits just because the source text already spelled it out."""
        for text in ("25万円です。", "250000円です。"):
            with self.subTest(text=text):
                out, _, applied = text_normalization.normalize("jp", text)
                self.assertTrue(applied)
                self.assertIn("ニジューゴマン", out)
                self.assertNotIn("万", out)

    def test_japanese_large_numbers_reach_the_oku_unit(self) -> None:
        """tn.japanese's own Cardinal grammar only groups correctly up to 8
        digits; at 1億 (100,000,000) and above it fails unpredictably
        (digit-by-digit, or a literal comma/zero-run leaking into the
        output), so this codebase converts numbers at or above that
        boundary itself. Covers the user-reported example (1,500,000,000)."""
        for text, reading in (
            ("100,000,000円です。", "イチオク"),
            ("1,500,000,000円です。", "ジューゴオク"),
        ):
            with self.subTest(text=text):
                out, _, applied = text_normalization.normalize("jp", text)
                self.assertTrue(applied)
                self.assertIn(reading, out)
                self.assertNotIn(",", out)

    def test_japanese_large_numbers_reach_the_cho_unit(self) -> None:
        """1兆 (10**12) must also group correctly, including the
        euphonic contraction on 一兆 (イッチョー,
        not イチチョー)."""
        out, _, applied = text_normalization.normalize(
            "jp", "予算は1,000,000,000,000円を予定しています。"
        )
        self.assertTrue(applied)
        self.assertIn("イッチョー", out)
        self.assertNotIn(",", out)

    def test_japanese_numbers_below_the_oku_boundary_are_unaffected(self) -> None:
        """8-digit numbers are already grouped correctly by tn.japanese
        today; the new large-number path must not touch them."""
        for text in ("99,999,999円です。", "99999999円です。"):
            with self.subTest(text=text):
                out, _, applied = text_normalization.normalize("jp", text)
                self.assertTrue(applied)
                self.assertNotIn(",", out)

    def test_japanese_bare_long_digit_runs_stay_literal(self) -> None:
        """A bare (no-comma) 9+-digit run is conventionally an identifier,
        not a quantity - it must stay exactly as written, not be read as a
        giant cardinal number and not fall back to a spelled-out reading."""
        text = "整理番号は123456789012です。"
        out, _, applied = text_normalization.normalize("jp", text)
        self.assertIn("123456789012", out)
        self.assertFalse(applied)

    def test_japanese_hyphen_less_phone_numbers_are_left_as_written(self) -> None:
        """Extends the phone-number protection below to unhyphenated
        numbers, which previously fell through to the same broken
        large-number path as any other bare digit run."""
        out, _, _ = text_normalization.normalize(
            "jp", "電話は09012345678までお願いします。"
        )
        self.assertIn("09012345678", out)

    def test_japanese_bracketed_large_number_is_not_mistaken_for_a_quantity(self) -> None:
        """A comma-grouped number inside a [tag] must survive byte-identical,
        same as any other tag content - the large-number rewrite must not
        reach inside a protected span."""
        text = "彼は [pron 1,500,000,000] と言った。"
        out, _, _ = text_normalization.normalize("jp", text)
        self.assertIn("[pron 1,500,000,000]", out)

    def test_japanese_large_number_with_decimal_tail_is_left_alone(self) -> None:
        """A decimal remainder on a >=1-oku amount is a deliberate non-goal
        for _ja_expand_large_numbers: it must leave the whole match (integer
        + decimal) untouched rather than convert only the integer part and
        strand the decimal next to a kanji character - that adjacency is
        what confuses sentence-terminator restoration into inserting a
        spurious extra full stop. (tn.japanese's own, separate, pre-existing
        handling of the untouched comma+decimal text is unaffected either
        way - the guarantee here is no *new* corruption from this fix.)"""
        self.assertEqual(
            text_normalization._ja_expand_large_numbers("1,500,000,000.5ドルです。"),
            "1,500,000,000.5ドルです。",
        )
        out, _, _ = text_normalization.normalize("jp", "残高は1,500,000,000.5ドルです。")
        self.assertEqual(out.count("。"), 1)

    def test_japanese_numbers_at_or_above_kyo_are_left_alone(self) -> None:
        """pyopenjtalk cannot reliably read a 10**16-scale (京) reading as a
        number (empirically it mis-segments it as a proper name), so
        _ja_expand_large_numbers deliberately stops short of that magnitude
        even though the underlying algorithm supports it. Checked directly
        against the converter rather than through the full pipeline, because
        tn.japanese's own (separate, out-of-scope) handling of an untouched
        10**16-scale comma number is not itself byte-identical passthrough,
        which would otherwise make this assertion about the wrong thing."""
        text = "10,000,000,000,000,000円"
        self.assertEqual(text_normalization._ja_expand_large_numbers(text), text)
        # One order of magnitude below the ceiling still converts normally.
        self.assertNotEqual(
            text_normalization._ja_expand_large_numbers("1,000,000,000,000,000円"),
            "1,000,000,000,000,000円",
        )

    def test_japanese_month_counter_uses_gatsu_not_getsu(self) -> None:
        """N月 (a calendar month) is always read "-gatsu", with irregular
        readings for 4/7/9 - never the generic "-getsu" reading, which only
        applies to the ヶ月/か月 duration counter."""
        for text, reading in (
            ("1月に行きます。", "イチガツ"),
            ("4月に行きます。", "シガツ"),
            ("7月に行きます。", "シチガツ"),
            ("9月に行きます。", "クガツ"),
            ("12月に行きます。", "ジューニガツ"),
        ):
            with self.subTest(text=text):
                out, _, applied = text_normalization.normalize("jp", text)
                self.assertTrue(applied)
                self.assertIn(reading, out)
                self.assertNotIn("ゲツ", out)
                self.assertNotIn("月", out)

    def test_japanese_phone_numbers_are_left_as_written(self) -> None:
        """The model reads a written phone number better than any spelled-out
        form, so that span is protected from the grammar."""
        out, _, _ = text_normalization.normalize("jp", "電話は090-1234-5678です。")
        self.assertIn("090-1234-5678", out)

    def test_plain_sentences_are_left_alone(self) -> None:
        """A sentence with nothing to normalize must come back byte-identical,
        so enabling the layer cannot change prosody on ordinary traffic."""
        for lang, text in (
            ("vn", "Xin chào, hôm nay trời đẹp."),
            ("jp", "こんにちは、世界。"),
        ):
            with self.subTest(lang=lang):
                out, _, applied = text_normalization.normalize(lang, text)
                self.assertEqual(out, text)
                self.assertFalse(applied)

    def test_bracketed_spans_survive(self) -> None:
        """Non-verbal tags and CMU pronunciation hints are parsed by the model
        out of the text; a normalizer must not rewrite them."""
        for lang, text, tag in (
            ("vn", "Anh ấy cười [laughter] rồi nói 5 câu.", "[laughter]"),
            ("jp", "彼は [sigh] と言って3回うなずいた。", "[sigh]"),
            ("vn", "Từ [B EY1 S] có 2 âm tiết.", "[B EY1 S]"),
        ):
            with self.subTest(text=text):
                out, _, _ = text_normalization.normalize(lang, text)
                self.assertIn(tag, out)

    def test_failure_falls_back_to_the_original_text(self) -> None:
        """A broken normalizer must never fail a TTS request."""
        class _Exploding:
            def __call__(self, text: str) -> str:
                raise RuntimeError("boom")

        original = text_normalization._BACKENDS["vn"]
        text_normalization._BACKENDS["vn"] = _Exploding()
        text_normalization._normalize_cached.cache_clear()
        self.addCleanup(text_normalization._normalize_cached.cache_clear)
        self.addCleanup(
            text_normalization._BACKENDS.__setitem__, "vn", original)

        text = "Giá 150.000 đồng."
        out, _, applied = text_normalization.normalize("vn", text)
        self.assertEqual(out, text)
        self.assertFalse(applied)

    def test_oversized_input_passes_through(self) -> None:
        text = "một " * (text_normalization.MAX_CHARS)
        out, _, applied = text_normalization.normalize("vn", text)
        self.assertEqual(out, text)
        self.assertFalse(applied)

    def test_health_payload_reports_both_languages(self) -> None:
        status = text_normalization.available()
        self.assertTrue(status["enabled"])
        self.assertEqual(sorted(status["languages"]), ["jp", "vn"])
        for lang in ("jp", "vn"):
            self.assertTrue(status["languages"][lang]["ready"])


if __name__ == "__main__":
    unittest.main()
