#!/usr/bin/env python3
"""Build the Japanese text-normalization benchmark.

Track A - ``data/gold.jsonl``: 10,000 Japanese carrier sentences (25 categories
x 400), each containing one non-standard word whose accepted spoken form is
known by construction. Readings are generated from ``ja_reading.py``, which
implements the kanji-numeral rules independently of the library under test.

Track B - ``data/real.jsonl``: real Japanese Wikipedia sentences containing
NSWs, for the reference-free metrics (residual unspeakable characters, crash
rate, latency at realistic lengths).

Usage:  python build_dataset.py [--per-category 400] [--real 3000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List

from ja_reading import (
    read_decimal,
    read_digits,
    read_int,
    read_year,
)

HERE = Path(__file__).resolve().parent
RNG = random.Random(20260917)

CARRIERS: Dict[str, List[str]] = {
    "generic": [
        "報告書には{}と記載されています。",
        "担当者は{}だと述べました。",
        "資料の末尾に{}とあります。",
        "統計によると{}でした。",
        "調査の結果は{}となりました。",
    ],
    "money": [
        "価格は{}です。",
        "昨年の売上は{}に達しました。",
        "予算は{}を予定しています。",
        "追加費用として{}がかかります。",
    ],
    "date": [
        "会議は{}に開催されます。",
        "契約は{}から有効です。",
        "商品は{}にお届けします。",
        "発表は{}に行われました。",
    ],
    "time": [
        "出発は{}の予定です。",
        "式典は{}に始まります。",
        "店舗は{}から営業します。",
        "点検は{}に実施します。",
    ],
    "measure": [
        "重さは約{}です。",
        "残りの距離は{}です。",
        "気温は{}まで上がりました。",
        "最高速度は{}に達します。",
    ],
    "phone": [
        "お問い合わせは{}までご連絡ください。",
        "受付番号は{}です。",
        "担当窓口は{}になります。",
    ],
    "math": [
        "テストでは{}を計算します。",
        "黒板に{}と書きました。",
        "答えは{}になります。",
    ],
    "web": [
        "詳細は{}をご覧ください。",
        "応募は{}から受け付けます。",
    ],
    "emoji": [
        "投稿の末尾に{}が付いていました。",
        "返信は{}だけでした。",
    ],
}


def carrier(kind: str) -> str:
    return RNG.choice(CARRIERS.get(kind, CARRIERS["generic"]))


def make_case(cid: str, category: str, nsw: str, readings: List[str],
              kind: str = "generic", metric: str = "reading") -> dict:
    tpl = carrier(kind)
    return {
        "id": cid,
        "category": category,
        "nsw": nsw,
        "text": tpl.format(nsw),
        "readings": readings,
        "gold_text": tpl.format(readings[0] if readings else nsw),
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_cardinal_small(i: int) -> dict:
    n = RNG.randint(0, 999)
    return make_case(f"cs{i}", "cardinal_small", str(n), [read_int(n)])


def _clean_large() -> int:
    """Up to 8 digits, so the cardinal reading is unambiguous. Longer bare digit
    strings are an ID, not a quantity, and live in the serial category."""
    n = RNG.choice([
        RNG.randint(1000, 9999),
        RNG.randint(1, 99) * 10000,
        RNG.randint(10000, 99999999),
        RNG.randint(1, 999) * 100000,
    ])
    return n


def gen_cardinal_large(i: int) -> dict:
    n = _clean_large()
    return make_case(f"cl{i}", "cardinal_large", str(n), [read_int(n)])


def gen_cardinal_grouped(i: int) -> dict:
    n = _clean_large()
    return make_case(f"cg{i}", "cardinal_grouped", f"{n:,}", [read_int(n)])


_FULLWIDTH = str.maketrans("0123456789", "０１２３４５６７８９")


def gen_fullwidth_digits(i: int) -> dict:
    n = RNG.randint(1, 99999)
    return make_case(f"fw{i}", "fullwidth_digits", str(n).translate(_FULLWIDTH),
                     [read_int(n)])


def gen_decimal(i: int) -> dict:
    ip = RNG.randint(0, 999)
    fd = "".join(str(RNG.randint(0, 9)) for _ in range(RNG.choice([1, 2, 2, 3])))
    return make_case(f"dec{i}", "decimal", f"{ip}.{fd}", read_decimal(ip, fd))


def gen_negative(i: int) -> dict:
    n = RNG.randint(1, 9999)
    r = read_int(n)
    return make_case(f"neg{i}", "negative", f"-{n}",
                     [f"マイナス{r}", f"ひく{r}"])


def gen_fraction(i: int) -> dict:
    num, den = RNG.randint(1, 9), RNG.randint(2, 100)
    return make_case(f"frac{i}", "fraction", f"{num}/{den}",
                     [f"{read_int(den)}分の{read_int(num)}"])


def gen_percent(i: int) -> dict:
    if RNG.random() < 0.3:
        ip, fd = RNG.randint(0, 100), str(RNG.randint(1, 9))
        raw = f"{ip}.{fd}%"
        base = read_decimal(ip, fd)
    else:
        n = RNG.randint(0, 100)
        raw, base = f"{n}%", [read_int(n)]
    return make_case(f"pct{i}", "percent", raw,
                     [f"{b}パーセント" for b in base])


def gen_money_jpy(i: int) -> dict:
    n = RNG.choice([RNG.randint(1, 999) * 100, RNG.randint(1, 99) * 10000,
                    RNG.randint(100, 99999)])
    style = RNG.random()
    r = read_int(n)
    if style < 0.45:
        raw = f"{n:,}円"
    elif style < 0.75:
        raw = f"{n}円"
    else:
        raw = f"￥{n:,}"
    return make_case(f"jpy{i}", "money_jpy", raw, [f"{r}円"], kind="money")


def gen_money_foreign(i: int) -> dict:
    n = RNG.choice([RNG.randint(1, 9999), RNG.randint(1, 99) * 1000])
    r = read_int(n)
    style = RNG.random()
    if style < 0.4:
        raw, readings = f"${n:,}", [f"{r}ドル", f"{r}アメリカドル"]
    elif style < 0.7:
        raw, readings = f"USD{n}", [f"{r}アメリカドル", f"{r}ドル"]
    else:
        raw, readings = f"HKD{n}", [f"{r}香港ドル", f"{r}ドル"]
    return make_case(f"fx{i}", "money_foreign", raw, readings, kind="money")


def gen_date_full(i: int) -> dict:
    y, m, d = RNG.randint(1950, 2035), RNG.randint(1, 12), RNG.randint(1, 28)
    sep = RNG.choice(["/", "-", "."])
    pad = RNG.random() < 0.5
    ms, ds = (f"{m:02d}", f"{d:02d}") if pad else (str(m), str(d))
    raw = f"{y}{sep}{ms}{sep}{ds}"
    return make_case(f"date{i}", "date_full", raw,
                     [f"{yr}年{read_int(m)}月{read_int(d)}日"
                      for yr in read_year(y)], kind="date")


def gen_date_partial(i: int) -> dict:
    if RNG.random() < 0.5:
        m, d = RNG.randint(1, 12), RNG.randint(1, 28)
        raw = f"{m:02d}/{d:02d}"
        readings = [f"{read_int(m)}月{read_int(d)}日"]
    else:
        y, m = RNG.randint(1950, 2035), RNG.randint(1, 12)
        raw = RNG.choice([f"{y}/{m:02d}", f"{m:02d}/{y}"])
        readings = [f"{yr}年{read_int(m)}月" for yr in read_year(y)]
    return make_case(f"dp{i}", "date_partial", raw, readings, kind="date")


def gen_time(i: int) -> dict:
    h, mi = RNG.randint(0, 23), RNG.randint(0, 59)
    style = RNG.random()
    if style < 0.5:
        raw = f"{h}:{mi:02d}"
        readings = [f"{read_int(h)}時{read_int(mi)}分" if mi
                    else f"{read_int(h)}時"]
    elif style < 0.75:
        h12 = RNG.randint(1, 12)
        raw = f"{h12}:{mi:02d}pm"
        readings = [f"午後{read_int(h12)}時{read_int(mi)}分" if mi
                    else f"午後{read_int(h12)}時"]
    else:
        h12 = RNG.randint(1, 12)
        raw = f"{h12}:{mi:02d}am"
        readings = [f"午前{read_int(h12)}時{read_int(mi)}分" if mi
                    else f"午前{read_int(h12)}時"]
    return make_case(f"time{i}", "time", raw, readings, kind="time")


def gen_range(i: int) -> dict:
    style = RNG.random()
    if style < 0.35:
        a, b = sorted(RNG.sample(range(1, 100), 2))
        unit = RNG.choice(["年", "月", "日", "人"])
        raw = f"{a}-{b}{unit}"
        readings = [f"{read_int(a)}から{read_int(b)}{unit}"]
    elif style < 0.7:
        a, b = sorted(RNG.sample(range(1, 200), 2))
        raw = f"{a}~{b}"
        readings = [f"{read_int(a)}から{read_int(b)}"]
    else:
        h1, h2 = sorted(RNG.sample(range(0, 23), 2))
        m1, m2 = RNG.randint(0, 59), RNG.randint(0, 59)
        raw = f"{h1}:{m1:02d}-{h2}:{m2:02d}"
        readings = [f"{read_int(h1)}時{read_int(m1)}分から"
                    f"{read_int(h2)}時{read_int(m2)}分"]
    return make_case(f"rng{i}", "range", raw, readings)


def gen_phone_mobile(i: int) -> dict:
    body = "".join(str(RNG.randint(0, 9)) for _ in range(8))
    digits = "090" + body
    raw = RNG.choice([f"090-{body[:4]}-{body[4:]}", f"090-{body}",
                      f"080-{body[:4]}-{body[4:]}"])
    digits = raw.replace("-", "")
    return make_case(f"tel{i}", "phone_mobile", raw, [read_digits(digits)],
                     kind="phone")


def gen_phone_landline(i: int) -> dict:
    area = RNG.choice(["02", "03", "06", "045", "052"])
    body = "".join(str(RNG.randint(0, 9)) for _ in range(8 - (len(area) - 2)))
    raw = f"{area}-{body[:4]}-{body[4:]}"
    return make_case(f"tell{i}", "phone_landline", raw,
                     [read_digits(raw.replace("-", ""))], kind="phone")


UNITS = [
    ("km/h", ["キロメートル毎時", "キロメートル毎時間"]),
    ("m/s", ["メートル毎秒"]),
    ("kg", ["キログラム"]),
    ("g", ["グラム"]),
    ("mg", ["ミリグラム"]),
    ("km", ["キロメートル"]),
    ("m", ["メートル"]),
    ("cm", ["センチメートル"]),
    ("mm", ["ミリメートル"]),
    ("m²", ["平方メートル"]),
    ("ml", ["ミリリットル"]),
    ("l", ["リットル"]),
    ("℃", ["摂氏", "度"]),
    ("kW", ["キロワット"]),
    ("MB", ["メガバイト"]),
    ("GB", ["ギガバイト"]),
    ("Hz", ["ヘルツ"]),
]


def gen_measure(i: int) -> dict:
    unit, unit_readings = RNG.choice(UNITS)
    n = RNG.choice([RNG.randint(1, 999), RNG.randint(1, 50) * 100])
    raw = f"{n}{unit}"
    return make_case(f"meas{i}", "measure", raw,
                     [f"{read_int(n)}{u}" for u in unit_readings],
                     kind="measure")


COUNTERS = ["人", "部", "匹", "本", "冊", "台", "枚", "個", "回", "件", "軒", "階"]


def gen_counter(i: int) -> dict:
    n = RNG.randint(1, 999)
    c = RNG.choice(COUNTERS)
    return make_case(f"cnt{i}", "counter", f"{n}{c}", [f"{read_int(n)}{c}"])


def gen_ordinal(i: int) -> dict:
    n = RNG.randint(1, 99)
    style = RNG.random()
    if style < 0.5:
        raw, readings = f"第{n}回", [f"第{read_int(n)}回"]
    elif style < 0.8:
        raw, readings = f"{n}位", [f"{read_int(n)}位"]
    else:
        raw, readings = f"第{n}章", [f"第{read_int(n)}章"]
    return make_case(f"ord{i}", "ordinal", raw, readings)


def gen_math(i: int) -> dict:
    a, b = RNG.randint(1, 99), RNG.randint(1, 99)
    ra, rb = read_int(a), read_int(b)
    op = RNG.choice(["+", "-", "×", "÷", "=", ">", "<", "≥", "±"])
    if op == "+":
        raw, readings = f"{a}+{b}", [f"{ra}プラス{rb}"]
    elif op == "-":
        raw, readings = f"{a}-{b}", [f"{ra}マイナス{rb}"]
    elif op == "×":
        raw, readings = f"{a}×{b}", [f"{ra}カケル{rb}", f"{ra}かける{rb}"]
    elif op == "÷":
        raw, readings = f"{a}÷{b}", [f"{ra}ワル{rb}", f"{ra}わる{rb}"]
    elif op == "=":
        raw, readings = f"{a}={b}", [f"{ra}イコール{rb}"]
    elif op == ">":
        raw, readings = f"{a}>{b}", [f"{ra}大なり{rb}"]
    elif op == "<":
        raw, readings = f"{a}<{b}", [f"{ra}小なり{rb}"]
    elif op == "≥":
        raw, readings = f"{a}≥{b}", [f"{ra}大なりイコール{rb}"]
    else:
        raw, readings = f"±{a}", [f"プラスマイナス{ra}"]
    return make_case(f"math{i}", "math", raw, readings, kind="math")


SYMBOLS = ["&", "@", "#", "±", "°", "§", "©", "®", "™", "№", "≤", "≥", "√",
           "∞", "→", "×", "÷", "~", "%", "$"]


def gen_symbol(i: int) -> dict:
    return make_case(f"sym{i}", "symbol", RNG.choice(SYMBOLS), [],
                     metric="leak")


EMOJI = ["😀", "😂", "❤️", "👍", "✅", "❌", "⭐", "🔥", "🎉", "🙏", "💰",
         "📞", "✈️", "☀️", "🇯🇵"]


def gen_emoji(i: int) -> dict:
    return make_case(f"emo{i}", "emoji", RNG.choice(EMOJI), [], kind="emoji",
                     metric="leak")


def gen_score(i: int) -> dict:
    a, b = RNG.randint(0, 9), RNG.randint(0, 9)
    sep = RNG.choice([":", "-"])
    return make_case(f"sco{i}", "score", f"{a}{sep}{b}",
                     [f"{read_int(a)}対{read_int(b)}"])


def gen_serial(i: int) -> dict:
    style = RNG.random()
    if style < 0.35:
        n = RNG.randint(1000, 9999)
        raw, readings = f"No.{n}", [f"No.{read_digits(str(n))}"]
    elif style < 0.7:
        n = RNG.randint(100, 9999)
        raw, readings = f"{n}号室", [f"{read_digits(str(n))}号室",
                                     f"{read_int(n)}号室"]
    else:
        n = "".join(str(RNG.randint(0, 9)) for _ in range(RNG.randint(9, 12)))
        raw, readings = n, [read_digits(n)]
    return make_case(f"ser{i}", "serial", raw, readings)


MIXED = [
    "{d}に{money}円の商品を{cnt}個購入しました。",
    "{time}から会場は{pct}%の入りで、気温は{t}℃でした。",
    "お問い合わせは{tel}まで、受付は{time}までです。",
    "{d}の売上は{money}円、前年比{pct}%の増加です。",
]


def gen_mixed(i: int) -> dict:
    y, m, d = RNG.randint(2015, 2035), RNG.randint(1, 12), RNG.randint(1, 28)
    money = RNG.randint(1, 99) * 10000
    pct = RNG.randint(1, 99)
    cnt = RNG.randint(1, 99)
    h, mi = RNG.randint(0, 23), RNG.randint(0, 59)
    t = RNG.randint(1, 40)
    tel = "090-" + "".join(str(RNG.randint(0, 9)) for _ in range(4)) + "-" + \
          "".join(str(RNG.randint(0, 9)) for _ in range(4))
    tpl = RNG.choice(MIXED)
    text = tpl.format(d=f"{y}/{m:02d}/{d:02d}", money=f"{money:,}", pct=pct,
                      cnt=cnt, time=f"{h}:{mi:02d}", t=t, tel=tel)
    readings: List[str] = []
    if "{d}" in tpl:
        readings.append(f"{read_int(y)}年{read_int(m)}月{read_int(d)}日")
    if "{money}" in tpl:
        readings.append(read_int(money))
    if "{pct}" in tpl:
        readings.append(f"{read_int(pct)}パーセント")
    if "{cnt}" in tpl:
        readings.append(f"{read_int(cnt)}個")
    return {
        "id": f"mix{i}", "category": "mixed", "nsw": text, "text": text,
        "readings": readings, "gold_text": text, "metric": "all_readings",
    }


GENERATORS: Dict[str, Callable[[int], dict]] = {
    "cardinal_small": gen_cardinal_small,
    "cardinal_large": gen_cardinal_large,
    "cardinal_grouped": gen_cardinal_grouped,
    "fullwidth_digits": gen_fullwidth_digits,
    "decimal": gen_decimal,
    "negative": gen_negative,
    "fraction": gen_fraction,
    "percent": gen_percent,
    "money_jpy": gen_money_jpy,
    "money_foreign": gen_money_foreign,
    "date_full": gen_date_full,
    "date_partial": gen_date_partial,
    "time": gen_time,
    "range": gen_range,
    "phone_mobile": gen_phone_mobile,
    "phone_landline": gen_phone_landline,
    "measure": gen_measure,
    "counter": gen_counter,
    "ordinal": gen_ordinal,
    "math": gen_math,
    "symbol": gen_symbol,
    "emoji": gen_emoji,
    "score": gen_score,
    "serial": gen_serial,
    "mixed": gen_mixed,
}


# ---------------------------------------------------------------------------
# Track B: real Japanese sentences
# ---------------------------------------------------------------------------

NSW_RE = re.compile(r"[0-9０-９]|[%％€$¥￥&@#+×÷°±/~〜]")
SENT_SPLIT = re.compile(r"(?<=[。！？])")


def harvest_real(n_target: int) -> List[dict]:
    out: List[dict] = []
    seen = set()
    offset = 0
    base = ("https://datasets-server.huggingface.co/rows?dataset="
            + urllib.parse.quote("wikimedia/wikipedia", safe="")
            + "&config=20231101.ja&split=train")
    while len(out) < n_target and offset < 20000:
        url = f"{base}&offset={offset}&length=100"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bench"})
            data = json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as exc:  # noqa: BLE001
            print(f"  datasets-server error at offset {offset}: {exc}")
            break
        rows = data.get("rows", [])
        if not rows:
            break
        for row in rows:
            for para in row["row"].get("text", "").split("\n"):
                for sent in SENT_SPLIT.split(para):
                    sent = sent.strip()
                    if not (20 <= len(sent) <= 200) or not NSW_RE.search(sent):
                        continue
                    key = sent[:60]
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"id": f"real{len(out)}",
                                "category": "real_wikipedia", "text": sent})
                    if len(out) >= n_target:
                        return out
        offset += 100
        print(f"  harvested {len(out)}/{n_target} (offset {offset})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=400)
    ap.add_argument("--real", type=int, default=3000)
    args = ap.parse_args()

    gold = [fn(i) for _, fn in GENERATORS.items()
            for i in range(args.per_category)]
    RNG.shuffle(gold)
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    with (HERE / "data" / "gold.jsonl").open("w", encoding="utf-8") as f:
        for case in gold:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(gold)} gold cases ({len(GENERATORS)} categories)")

    if args.real:
        print(f"Harvesting {args.real} real Japanese sentences ...")
        real = harvest_real(args.real)
        with (HERE / "data" / "real.jsonl").open("w", encoding="utf-8") as f:
            for case in real:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"Wrote {len(real)} real sentences")


if __name__ == "__main__":
    main()
