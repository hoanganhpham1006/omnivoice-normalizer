# Reading digit+letter tokens (`4R` → ヨンアール) in the Japanese normalizer

Research note, 2026-09-22. Question: what is the proper way to make
`text_normalization.normalize("jp", "4R")` produce `ヨンアール` instead of today's
`ヨンR`, without breaking units (`4V`, `10kg`), floor plans (`4LDK`) or ordinary
Latin text. Every claim below is traced to library source, a package data file,
or an official page; empirical outputs come from the pinned libraries
(WeTextProcessing 1.2.0, pyopenjtalk-plus 0.4.1.post9) run through this repo's
pipeline.

This is the repo's first research note; `docs/research/` was created for it.

## Summary

- **The letter reading already exists.** pyopenjtalk-plus's bundled dictionary
  (a customised mecab-naist-jdic) tags `Ｒ` as `記号/アルファベット` with
  pronunciation `アール`, and every letter A–Z has such an entry. The reading is
  lost only in the last step: `pyopenjtalk.g2p(..., kana=True)` substitutes the
  surface string for **every** token whose part of speech is `記号`, and
  alphabet letters are a `記号` subclass. The phoneme path (`kana=False`) reads
  `4R` correctly as `y o N a a r u`. This repo's `_to_kana` uses the kana path,
  so `ヨンR` is a pyopenjtalk *output-formatting* artefact, not a missing
  reading.
- **WeTextProcessing cannot be the fix.** Its `transliterate=True` flag does
  produce `四アール`, but the single-letter rows in its 6,791-line table are
  corpus artefacts (`A→アンペア`, `V→ゴ`, `X→テン`, `i→イチ`, `H→エッチ`) and the
  flag's tagger weight beats the cardinal and measure grammars, so `A4` becomes
  `アンペア四` and `5G` becomes `五ギガ`. Its measure grammar independently eats
  `4L` in `4LDK` as four litres (flag or no flag).
- **Recommended fix (wrapper-only, no new dependency):** build the kana reading
  from `pyopenjtalk.run_frontend()` and keep `pron` for `記号/アルファベット`
  tokens, and let `_ja_kana_spans` fold a short uppercase letter run on either
  side of a rewritten digit into the span so the digit and the letter reach
  pyopenjtalk together. Add one narrow pre-rule for the floor-plan collision
  (`\dS?L?DK`). Leave `OMNIVOICE_JA_TRANSLITERATE` off. Prototyped against the
  live pipeline: `4R→ヨンアール`, `A4サイズ→エイヨンサイズ`, `4LDK→ヨンエルディーケイ`,
  `5G→ゴジー`, `4V→ヨンボルト`, `10kg→ジュッキログラム`, with all existing
  regression sentences unchanged.
- **`4R` is genuinely ambiguous** (letter name アール; real-estate `1R` is read
  ワンルーム; JRA race cards use `1R` for 第1レース; `a`/アール is also the legal
  100 m² area unit). The letter-name reading is the only one derivable from the
  text alone and is what every surveyed TTS front-end defaults to; domain
  readings belong in an optional override table, not the default rule.

## Findings

### 1. WeTextProcessing `tn.japanese`

**Where the transliteration table lives and how it is applied.**
`tn/japanese/rules/transliteration.py:30-31` loads
`tn/japanese/data/pyopenjtalk/transliteration.tsv` with `pynini.string_file`
and wraps it as a tagger; the file has 6,791 tab-separated `surface\treading`
rows (`wc -l`). The Normalizer only unions it into the tagger when
`transliterate=True`, at weight **1.04** (`tn/japanese/normalizer.py:82-83`).
The other weights (`normalizer.py:70-79`) are cardinal 1.06, char 100, date
1.02, fraction 1.05, math 90, measure 1.05, money 1.05, sport 1.06, time 1.05,
whitelist 1.03. pynini's `shortestpath` picks the lowest total weight
(`tn/processor.py:118-126`), so a transliteration match outranks cardinal and
measure matches over the same span.

**Single-letter rows in the table** (`awk 'length($1)==1'` over the TSV):

| key | reading | key | reading |
|---|---|---|---|
| `a` | アール | `A` | アンペア |
| `F` | エフ | `g` | グラム |
| `G` | ギガ | `H` | エッチ |
| `i` / `I` | イチ | `l` | リットル |
| `M` | メガ | `P` | ピー |
| `R` | アール | `s` | セカンド |
| `t` | トン | `v` / `V` | ゴ |
| `w` / `W` | ワット | `x` | ジュー |
| `X` | テン | | |

Only 21 of 52 letters are present, several as Roman numerals (`v→ゴ`, `x→ジュー`,
`X→テン`) or units (`A→アンペア`, `l→リットル`). This is why the flag turns
`A4サイズ` into `アンペア四サイズ` (the `A→アンペア` row, not the measure grammar) and
`5G` into `五ギガ`. Multi-letter rows are more consistent but not uniform:
`ABC→エイビーシー` yet `ABM→エービーエム`; `3D→スリーディー` (line 35),
`4WD→ヨンダブリューディー` (42), `DK→ディーケイ` (1551), `F1→エフワン` (1941),
`LDK→エルディーケイ` (3476), `USB→ユーエスビー` (6343). The upstream README documents
the flag only as a CLI example (`python -m tn --language ja --transliterate`);
it does not say how the table was built
(https://github.com/wenet-e2e/WeTextProcessing, README).

**Where `4L→四リットル` comes from.** `tn/japanese/rules/measure.py:32-43`:
`units_en.tsv | units_ja.tsv` is appended directly to a cardinal
(`measure = prefix.ques + number + rmspace + units`, line 43). `units_en.tsv`
contains bare single letters `L` (line 32), `N` (63), `J` (69), `W` (74),
`A` (78), `V` (79) plus `KB/MB/GB/TB` (98-101). For `4LDK` the tagger's
cheapest path is measure `4L` (1.05) + two `char` tokens (100 each) rather than
cardinal `4` (1.06) + three `char` tokens, so `4L` is consumed as a volume with
or without transliteration. Empirically: `4LDK → 四リットルDK`,
`4V → 四ボルト`, `4W → 四ワット`, `4R → 四R` (R is not a unit, so cardinal + char).

**Digit immediately followed by a letter, in general.** Cardinal
(`rules/cardinal.py`) has no letter context; it tags the digit run and the
letter falls through to `Char`, which is an identity pass-through
(`rules/char.py:28-29`). The only letter-aware rules are Measure (units above),
Money (`USD1001→千一アメリカドル`), Time (`3:40pm`), Sport (`ACミラン3:2`) and
Whitelist.

**Whitelist / override mechanism.** `rules/whitelist.py:30-31` loads
`data/default/whitelist.tsv` (weight 1.03, beats everything except date). The
shipped file has 15 rows, e.g. `R-18→R十八`, `M1→Mワン`, `P2P→P to P`. A
whitelist row for `LDK` *would* beat the `4L` measure path (1.03 + 1.06 vs
1.05 + 200), but the path is hard-coded to the package data directory
(`tn/utils.get_abs_path`) and the compiled FST is cached, so using it means
patching site-packages or subclassing `Normalizer.build_tagger_and_verbalizer`
and rebuilding the FST. `blacklist.tsv` is empty and only feeds the
`remove_interjections` postprocessor.

### 2. pyopenjtalk-plus / Open JTalk

**Dictionary.** pyopenjtalk-plus replaces the stock `open_jtalk_dic_utf_8-1.11`
with a wheel-bundled custom dictionary based on mecab-naist-jdic plus
jpreprocess/naist-jdic fixes, split into `naist-jdic.csv`, `unidic-csj.csv`,
`fillers.csv`, `symbols.csv`, `rare_syllables.csv`, `heteronyms.csv`
(README lines 33-41, https://github.com/tsukumijima/pyopenjtalk-plus). The
installed copy is `pyopenjtalk/dictionary/` (NAIST licence in `COPYING`), used
via `OPEN_JTALK_DICT_DIR` (`pyopenjtalk/__init__.py:57-60`).

**ASCII is folded to full width before lookup.** Open JTalk's `text2mecab`
rewrites each ASCII letter to its full-width form
(`src/text2mecab/text2mecab_rule_utf_8.h:108` `"A","Ａ"`, `:125` `"R","Ｒ"`,
`:140` `"a","ａ"`, in r9y9/open_jtalk), so the lexicon only needs full-width
entries.

**The lexicon has every letter, with a reading.** `run_mecab("Ｒ")` returns
`Ｒ,記号,アルファベット,*,*,*,*,Ｒ,アール,アール,1/3,*`. Running each of A–Z
through `run_frontend` gives `pos=記号, pos_group1=アルファベット` and these
prons:

| A | B | C | D | E | F | G | H | I | J | K | L | M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| エイ | ビー | シー | ディー | イー | エフ | ジー | エイチ | アイ | ジェイ | ケイ | エル | エム |

| N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| エヌ | オー | ピー | キュー | アール | エス | ティー | ユー | ブイ | ダブリュー | エック’ス | ワイ | ゼット |

Lowercase letters map to the same prons. Open JTalk also carries a fallback
table for letters that reach `njd_set_pronunciation` without a reading
(`src/njd_set_pronunciation/njd_set_pronunciation_rule_utf_8.h:390-416`:
`Ａ→エイ`, `Ｈ→エイチ`, `Ｊ→ジェー`, `Ｋ→ケー`, `Ｒ→アール`, `Ｖ→ブイ`,
`Ｘ→エックス`, `Ｚ→ズィー`).

**Why `g2p(kana=True)` still prints `Ｒ`.** `pyopenjtalk/__init__.py:379-386`:

```python
for n in njd_features:
    if n["pos"] == "記号":
        p = n["string"]
    else:
        p = n["pron"]
```

Every `記号` token (punctuation *and* alphabet) is emitted as its surface
string; the only post-processing is stripping `’`. The phoneme branch
(`kana=False`, lines 371-377) uses `extract_phonemes` and does read the letter:
`g2p("4R") == "y o N a a r u"`. `normalize_text()` does not touch letters
(`normalize_text("4R") == "4R"`). Nothing in pyopenjtalk-plus adds an
alphabet/English feature; Sudachi is used only for homograph readings
(`use_sudachi_kanji_yomi`, `__init__.py:313-335`), and the ONNX model only for
「何」.

**User dictionary.** `mecab_dict_index(csv, out_dic)` compiles a naist-jdic
compatible CSV (`__init__.py:2071-2095`; README example
`ＧＮＵ,,,1,名詞,一般,*,*,*,*,ＧＮＵ,グヌー,グヌー,2/3,*`) and
`update_global_jtalk_with_user_dict(paths | [UserDictionaryEntry])` swaps the
global instance (`:2098-2169`; `unset_user_dict` at `:2172`). It is not needed
for letters, because the readings are already in the system dictionary; it
would only matter for domain words (e.g. registering `１Ｒ` as ワンルーム). Note
that it replaces the process-global instance, so it must run once at warm-up.

**Empirical g2p (kana=True) on the requested inputs.**

| input | output | note |
|---|---|---|
| `4アール` | ヨンアール | 4 read ヨン before a loanword |
| `1アール` | イチアール | |
| `7アール` | ナナアール | |
| `9アール` | キューアール | |
| `4エー` | ヨンエー | |
| `USB` | ユーエスビー | `ＵＳＢ` is a noun entry, not 記号 |
| `ユーエスビー` | ユーエスビー | |
| `1R` | イチＲ | 記号 → surface string |
| `4LDK` | ヨンエルディーケイ | `ＬＤＫ` is a `名詞/一般` entry |
| `3Dプリンター` | スリーディープリンター | `３Ｄ` is a noun entry (lexicalised English number) |

### 3. How other Japanese TTS front-ends read Latin letters

- **VOICEVOX ENGINE** (`voicevox_engine/tts_pipeline/njd_feature_processor.py:89-110`)
  runs `pyopenjtalk.run_frontend`, then `make_label` → full-context labels, i.e.
  the *phoneme* path, so dictionary alphabet prons are used as-is. Its
  `katakana_english.py` only intervenes for tokens whose reading is unknown
  (`_is_unknown_reading_word`: `pos == "フィラー" and chain_rule == "*"`), and
  its policy is explicit: a single letter is never converted (the dictionary
  handles it), an all-uppercase run is read letter-by-letter, a mixed-case word
  goes through the `kanalizer` English→kana model
  (`_should_convert_english_to_katakana`, `convert_english_to_katakana`).
  Its letter table `ojt_alphabet_kana_mapping` (copied, per its comment, from
  Open JTalk's `njd_set_pronunciation_rule_utf_8.h`): A エー, H エイチ, J ジェー,
  K ケー, V ブイ, X エックス, Z ズィー.
- **Style-Bert-VITS2** (`style_bert_vits2/nlp/japanese/g2p.py:94-160`,
  `text_to_sep_kata`) takes `parts["pron"]` for every `run_frontend` token
  (line 121) and only falls back to the surface for punctuation-only tokens
  (139-152), so `R` becomes アール there. Its `normalizer.py` has no letter
  table; it just NFKC-folds letters to half width (line 112).
- **Bert-VITS2** (`text/japanese.py:443-480`, `_ALPHASYMBOL_YOMI`; applied
  per character by `japanese_convert_alpha_symbols_to_words`, 521-522) uses a
  static table: a エー, h エイチ, j ジェー, k ケー, v ブイ, x エックス, z ゼット.

**Canonical letter names and where sources disagree.** All four sources agree
on B–G, I, L–U, W, Y. They differ on:

| letter | naist-jdic (pyopenjtalk-plus) | Open JTalk rule / VOICEVOX | Bert-VITS2 |
|---|---|---|---|
| A | エイ | エイ (r9y9 header) / エー (VOICEVOX) | エー |
| H | エイチ | エイチ | エイチ |
| J | ジェイ | ジェー | ジェー |
| K | ケイ | ケー | ケー |
| V | ブイ | ブイ | ブイ |
| Z | ゼット | ズィー | ゼット |

`エイ/エー`, `ジェイ/ジェー`, `ケイ/ケー` are the same phoneme sequence for TTS
(`e i` vs `e e` differs only in vowel quality); `ゼット/ズィー` is a real
British/American split, and ゼット is the majority convention in the three
Japanese-authored sources. Using the dictionary's own prons keeps the wrapper
consistent with how pyopenjtalk reads the same letters inside longer tokens
(`ＬＤＫ→エルディーケイ`, `ＡＢＣ→エイビーシー`).

### 4. What `4R` can mean

- **Letter name.** アール, the default in every surveyed engine and the only
  reading derivable from the string alone.
- **Real-estate floor plans.** at home's guide gives the readings
  `1R＝ワンルーム`, `1K＝ワンケー`, `1DK＝ワンディーケー`, `1LDK＝ワンエルディーケー`, with
  R = Room, and cites 首都圏不動産公正取引協議会 size rules for DK/LDK
  (https://www.athome.co.jp/contents/manual/howto-choose/floor/difference/);
  Winslink titles its article "1DK（ワンディーケー）" and reads 1R as ワンルーム
  (https://www.winslink.co.jp/article/look/1dk.php). So in this domain the
  digit is read in English (ワン) and `R` is expanded to ルーム, not アール.
- **Racing.** JRA's own results guide shows the card header `東京 1R` alongside
  `第1レース` (https://www.jra.go.jp/JRADB/mikata/result.html); `4R` = 第4レース,
  read ヨンレース.
- **Area unit.** 計量単位令 (平成四年政令第三百五十七号) lists アール in its
  units table, row 七 「土地の面積の計量」, defined as 「平方メートルの百倍」, with
  ヘクタール as 「アールの百倍」 (https://laws.e-gov.go.jp/law/404CO0000000357,
  read via the e-Gov law API). Its symbol is lowercase `a`, so `4a`
  (ヨンアール) is valid text.
- **Digit reading before a loanword.** The dictionary reads 4/7/9 as ヨン/ナナ/
  キュー in this position (`4アール→ヨンアール`, `7アール→ナナアール`,
  `9アール→キューアール`), matching the alternatives listed on MEXT's number
  sheet (https://tsunagarujp.mext.go.jp/assets/download/numbers_ver202404.pdf).
  Lexicalised English-number forms exist and are irregular: `3D→スリーディー`
  (naist-jdic noun), `F1→エフワン` and `3D→スリーディー` (WeText table), 5G is
  commonly ファイブジー, `4K` is ヨンケー. No rule predicts these; they are
  dictionary facts.

**Conclusion on ambiguity.** For a general TTS service the default must be the
letter-name reading with the Japanese digit (`4R→ヨンアール`, `1R→イチアール`).
Domain readings (ワンルーム, ヨンレース) require context the normalizer does not
have and should be an optional override table (env- or per-deployment
configured) applied before the generic rule.

### 5. Options, evaluated on the live pipeline

Prototype scripts (scratchpad only, repo untouched) monkey-patched
`text_normalization` and ran the requested sentences. Full table below.

**(a) `OMNIVOICE_JA_TRANSLITERATE=1`.** Fixes `4R` but breaks `A4サイズ`
(`アンペアヨンサイズ`), `5G` (`ゴギガ`), `4X` (`ヨンテン`), `4H` (`ヨンエッチ`), and
also rewrites every Latin word it knows (`USBメモリ→ユーエスビーメモリ`,
`iPhone→アイフォン`) even where the model reads the original fine. Not
recommended.

**(b) Wrapper pre-rule (letter → katakana before WeText).** Regex-replace an
uppercase letter adjacent to a digit with its name, skipping the six letters
that are units in `units_en.tsv` (A, V, W, L, N, J). Works for the target
cases and for `A4`, but it has to hard-code a letter table and a unit
exclusion list that duplicates WeText's, it leaves lowercase letters alone
(`4a→ヨンa`), and multi-letter runs need their own handling (`4WD→ヨンワットD`).
The transformed text also changes what `_ja_kana_spans` diffs against, which
is harmless today but couples the two stages.

**(c) Post-rule in `_ja_kana_spans` + dictionary reading.** Two changes:
1. Replace the body of `_to_kana` with a `run_frontend` walk that uses
   `n["pron"]` unless `n["pos"] == "記号" and n["pos_group1"] != "アルファベット"`,
   stripping `’` as `g2p` does. This is what Style-Bert-VITS2 does and is
   equivalent to VOICEVOX's phoneme path.
2. In `_ja_kana_spans`, extend the "extra" fold (`text_normalization.py:257-264`)
   so a short uppercase run is pulled into the kana span from the *following*
   equal chunk (add `|[A-Z]{1,4}(?![A-Za-z])` to `_JA_COUNTER_RE`) and from the
   *preceding* equal chunk (hold back a trailing `(?<![A-Za-z])[A-Z]{1,4}$` and
   prepend it to the rewritten chunk). The digit and letter then reach
   pyopenjtalk together, which is also what makes `4→ヨン` (not シ) and
   `H2O→エイチニオー` come out right.
   Measure units are untouched because WeText has already verbalised them
   before the diff (`4V→ヨンボルト`, `10KB→ジュッキロバイト`). Restricting the fold
   to uppercase runs of ≤4 letters mirrors VOICEVOX's "all-caps = letter
   names" rule and leaves `Windows11`, `Type2`, `COVID19` exactly as today.

**(c′) Floor-plan pre-rule.** `4LDK` still needs help because the measure
grammar consumes `4L` before anything downstream can see it. A one-line
`preprocess` rule `(?<=\d)(S?L?DK)(?![A-Za-z0-9])` → dictionary reading
(`ＬＤＫ→エルディーケイ`, `ＤＫ→ディーケイ`) gives `4LDK→ヨンエルディーケイ`,
`1LDK→イチエルディーケイ`, `2DK→ニディーケイ`. This is the only place a WeText
collision has to be pre-empted.

**(d) pyopenjtalk user dictionary.** Unnecessary for letters (already in the
system dictionary) and cannot fix the `g2p(kana=True)` formatting. Keep in
reserve for domain words (`１Ｒ→ワンルーム`) if a deployment wants them.

**(e) WeText whitelist.** Would fix `4LDK` and could add `4R→四アール`, but
requires editing package data or subclassing and rebuilding the cached FST, and
would still emit kanji that (c) must read anyway. Not worth the coupling.

**Letter/unit collisions to keep guarded.** Single uppercase letters that
`units_en.tsv` treats as units: A, V, W, L, N, J (and lowercase g, m, s, t, h,
in). Under (c) these are never folded because WeText has already verbalised them
into kana-readable words; the fold only sees letters WeText left raw. The two
known exceptions are floor plans (`LDK/DK`, handled by c′) and lexicalised
English-number forms (`3D`, `5G`, `F1`), which come out as サンディー / ゴジー /
エフイチ under (c). If those matter, add a small explicit table in `preprocess`
rather than widening the generic rule.

## Empirical results

Inputs run through `text_normalization.normalize("jp", …)` in each
configuration. "default" is today's code, "tl" is option (a),
"(c)+(c′)" is the recommended prototype (fold limited to `[A-Z]{1,4}`).

| input | default | tl (option a) | (c)+(c′) |
|---|---|---|---|
| 4R | ヨンR | ヨンアール | **ヨンアール** |
| 部屋は4Rです。 | 部屋はヨンRです。 | 部屋はヨンアールです。 | **部屋はヨンアールです。** |
| 1R | イチR | イチアール | イチアール |
| A4サイズ | Aヨンサイズ | アンペアヨンサイズ | **エイヨンサイズ** |
| 4LDK | ヨンリットルDK | ヨンリットルディーケイ | **ヨンエルディーケイ** |
| 2DK | ニDK | ニディーケイ | ニディーケイ |
| USBメモリ | USBメモリ | ユーエスビーメモリ | USBメモリ |
| 3Dプリンター | サンDプリンター | スリーディープリンター | サンディープリンター |
| 5G | ゴG | ゴギガ | ゴジー |
| iPhone 15 | iPhone ジューゴ | アイフォン ジューゴ | iPhone ジューゴ |
| 100km | ヒャッキロメートル | ヒャッキロメートル | ヒャッキロメートル |
| 10kg | ジュッキログラム | ジュッキログラム | ジュッキログラム |
| 3:40pm | ゴゴサンジヨンジュップン | ゴゴサンジヨンジュップン | ゴゴサンジヨンジュップン |
| 4K | ヨンK | ヨンK | ヨンケイ |
| 4V | ヨンボルト | ヨンボルト | ヨンボルト |
| 4W | ヨンワット | ヨンワット | ヨンワット |
| 4X | ヨンX | ヨンテン | ヨンエックス |
| 4H | ヨンH | ヨンエッチ | ヨンエイチ |
| 4a | ヨンa | ヨンアール | ヨンa |
| B4 | Bヨン | Bヨン | ビーヨン |
| F1 | Fイチ | エフワン | エフイチ |
| 12R | ジューニR | ジューニアール | ジューニアール |
| 4WD | ヨンワットD | (not run) | ヨンワットディー |
| 10KB | ジュッキロバイト | (not run) | ジュッキロバイト |
| H2O | HニO | (not run) | エイチニオー |
| PS5 / MP3 | PSゴ / MPサン | (not run) | ピーエスゴ / エムピーサン |
| Windows11 / COVID19 | Windowsジューイチ / COVIDジュック | (not run) | unchanged |

Regression sentences from `tests/test_normalization.py` under (c)+(c′) were
byte-identical to today's output for dates, money, counters, months, phone
numbers, tags and the large-number path (`2023/11/14に150,000円…`,
`25万円です。`, `4月に行きます。`, `3本ください。`, `090-1234-5678`, `[sigh]`,
`1,500,000,000円`). Three WeText conformance pairs change, all in the expected
direction: `R-18 → アールジューハチ` (was `Rジューハチ`), `P2P → ピーツーピー`
(was `ピーニピー`), `abc+5 → エービーシープラスゴ` (was `abcプラスゴ`).

Two pre-existing defects surfaced while testing and are out of scope here:
`ACミラン3:2` reads 対 as ツイ (`サンツイニ`) in every mode, and `COVID19`
gives `ジュック` for 十九.

## Recommendation

1. **Implement (c): read alphabet symbols from the dictionary.** Replace the
   `g2p(kana=True)` call in `_to_kana` (`text_normalization.py:195`) with a
   `run_frontend` loop that keeps `pron` for `記号/アルファベット` and the
   surface for other `記号`, stripping `’`. Then let `_ja_kana_spans` fold a
   short uppercase run (`[A-Z]{1,4}`, not part of a longer Latin word) on
   either side of a rewritten digit into the kana span. This is the smallest
   change, uses readings the dictionary already owns, matches what VOICEVOX and
   Style-Bert-VITS2 do, and leaves units and ordinary Latin words untouched.
2. **Add the floor-plan pre-rule (c′)** in `_JapaneseBackend.preprocess` next
   to `_ja_expand_large_numbers`, because `4L` is otherwise consumed as litres
   before the diff stage can help.
3. **Keep `OMNIVOICE_JA_TRANSLITERATE` off** and document why (single-letter
   table rows are unit/Roman-numeral artefacts that outrank the grammars).
4. **Do not encode domain readings in the default.** `1R→ワンルーム`,
   `4R→ヨンレース`, `5G→ファイブジー`, `F1→エフワン` are correct only in context;
   if a deployment needs them, add an explicit surface→reading table applied in
   `preprocess`, or register them in a pyopenjtalk user dictionary at warm-up.
5. **Tests to add** (all verified in the prototype): `4R→ヨンアール` inside a
   sentence, `A4サイズ→エイヨンサイズ`, `4LDK→ヨンエルディーケイ`, `4V→ヨンボルト`
   and `10KB→ジュッキロバイト` unchanged, `USBメモリ` and `Windows11` unchanged, and
   the existing suite green.
6. **A/B on the model before shipping**, as with every other rule in this repo:
   the benchmark README shows the checkpoint sometimes reads raw Latin text
   well, so confirm that `ヨンアール` is heard better than `ヨンR` on the actual
   voice rather than assuming it.

## Addendum: benchmark review of the recommendation (2026-09-22)

The recommended prototype and a narrowed variant were run through the repo's
own harness (`benchmarks/ja_normalization`, 10,000 gold cases + 3,000 real
Wikipedia sentences) against today's code.

| | gold outputs changed | real outputs changed | gold accuracy / leak / WER |
|---|---:|---:|---|
| prototype as written above (fold next to *any* rewritten chunk) | 0 / 10,000 | **102 / 3,000** | unchanged |
| narrowed fold (only next to a rewritten **digit** run) | 0 / 10,000 | **41 / 3,000** | unchanged |

**Why the prototype over-fired.** `tn.japanese` runs with `full_to_half=True`,
which rewrites every full-width ASCII character, so `（` → `(` and `）` → `)`
show up as rewritten chunks in the `_ja_kana_spans` diff (pyopenjtalk then
re-widens them, which is why the shipped output still shows full-width
parentheses). The fold therefore treated every all-caps acronym touching a
parenthesis as "adjacent to a rewrite": `（GDP）→（ジーディーピー）`,
`（EU）→（イーユー）`, `（NATO）→（ナトー）`, `（IFA）→（イフエイ）`, `（BZÖ）→（ビーズÖ）`,
and the same acronym read two ways in one sentence (`（JIS）はJIS` →
`（ジス）はJIS`). None of those inputs contain a digit. 61 of the 102 changed
sentences were this class.

**Narrowed rule.** Fold the letter run only when the neighbouring rewritten
chunk's *original* span contains a digit (`any(c.isdigit() for c in
original[i1:i2])`), on both the leading and trailing side. All 41 remaining
changes are digit-adjacent tokens: `H2O→エイチニオー`, `F1→エフイチ` (×6),
`TPP11→ティーピーピージューイチ` (×5), `GPT-3→ジーピーティーマイナスサン`,
`4GL→ヨンジーエル`, `B5判→ビーゴ判`, `3D→サンディー`, `45RPM→ヨンジューゴアールピーエム`,
`G20→ジーニジュー`, `MI6→エムアイロク`, `A3出口→エイサン出口`, `USドル→ユーエスドル`,
`FOXP2→エフオーエックスピーニ`, `U+204A→ユープラス…` (already garbage today).
The gold set contains no digit+letter category, so it can neither reward nor
penalise this rule; a gold category for alphanumeric tokens should be added
alongside the change.

**Part 1 equivalence.** Over the 8,720 unique fragments today's pipeline hands
to `_to_kana` across both tracks, the `run_frontend` reading differs from
`g2p(kana=True)` in exactly 2, both containing a Latin letter (`x五`:
`ｘゴ`→`エックスゴ`). Zero differences on fragments without Latin letters.

**Revised recommendation.** Keep part 1 as is. Implement part 2 with the
digit-adjacency condition, not the "any rewrite" condition. Treat the
floor-plan pre-rule (c′) as an optional domain patch: it is a workaround for a
WeText measure-grammar collision, touches no benchmark sentence, and is only
worth carrying if real-estate text is in the traffic.
