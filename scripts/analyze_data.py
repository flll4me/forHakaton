"""Первичный статистический анализ датасета Support Copilot.

Читает tickets_train.jsonl.gz и tickets_valid.jsonl, пишет агрегированный отчёт
в reports/data_analysis.md. Отчёт содержит только агрегаты (счётчики,
распределения), без текстов обращений, чтобы его можно было хранить в репозитории.

Запуск: python scripts/analyze_data.py
"""
from __future__ import annotations

import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "handout"
OUT = ROOT / "reports" / "data_analysis.md"


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def top(counter: collections.Counter, n: int = 20) -> list[tuple]:
    return counter.most_common(n)


def analyze(path: Path) -> dict:
    c = collections.defaultdict(collections.Counter)
    lens = collections.defaultdict(list)
    n = 0
    keys = collections.Counter()
    for t in read_jsonl(path):
        n += 1
        keys.update(t.keys())
        kind = t.get("kind")
        c["kind"][kind] += 1
        c["lang"][t.get("lang")] += 1
        c["region"][t.get("region")] += 1
        c["server_lang"][t.get("server_lang")] += 1
        c["lang_by_region"][(t.get("region"), t.get("lang"))] += 1
        c[f"verdict_{kind}"][t.get("verdict")] += 1
        c["status"][t.get("status")] += 1
        c["violation"][t.get("violation")] += 1
        c["punishment_present"][bool(t.get("punishment"))] += 1
        c["rule_refs_raw"][(t.get("rule_refs_raw") or "").strip().lower()[:20]] += 1
        c["images_count"][min(int(t.get("images_count") or 0), 5)] += 1
        c["category"][t.get("category")] += 1
        c["priority"][t.get("priority")] += 1
        roles = tuple(m.get("role") for m in t.get("messages") or [])
        c["roles"][roles] += 1
        player = " ".join(m.get("text") or "" for m in t.get("messages") or [] if m.get("role") == "player")
        lens[kind].append(len(player))
        if not player.strip():
            c["empty_player"][kind] += 1
        if any(ch in player for ch in "ќґѕєјўї"):
            c["cp1251_kk_chars"][t.get("lang")] += 1
    return {"n": n, "c": c, "lens": lens, "keys": keys}


def fmt_table(rows, headers) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def section(name: str, res: dict) -> str:
    c, n = res["c"], res["n"]
    parts = [f"## {name}\n", f"Записей: **{n}**\n", "Поля: " + ", ".join(f"`{k}`" for k in sorted(res["keys"])) + "\n"]
    for key in ["kind", "lang", "region", "server_lang", "status", "verdict_complaint", "verdict_report",
                "punishment_present", "images_count", "category", "priority", "empty_player", "cp1251_kk_chars"]:
        rows = [(k, v, f"{100 * v / n:.1f}%") for k, v in top(c[key], 25)]
        parts.append(f"### {key}\n\n" + fmt_table(rows, ["значение", "кол-во", "доля"]) + "\n")
    parts.append("### lang × region (top 25)\n\n" + fmt_table(
        [(f"{r}/{l}", v) for (r, l), v in top(c["lang_by_region"], 25)], ["region/lang", "кол-во"]) + "\n")
    parts.append("### violation (top 40)\n\n" + fmt_table(top(c["violation"], 40), ["violation", "кол-во"]) + "\n")
    parts.append("### rule_refs_raw, нормализовано (top 30)\n\n" + fmt_table(
        [(repr(k), v) for k, v in top(c["rule_refs_raw"], 30)], ["rule_refs_raw", "кол-во"]) + "\n")
    parts.append("### Последовательности ролей (top 10)\n\n" + fmt_table(
        [(" → ".join(map(str, k)), v) for k, v in top(c["roles"], 10)], ["roles", "кол-во"]) + "\n")
    rows = []
    for kind, ls in res["lens"].items():
        ls = sorted(ls)
        rows.append((kind, len(ls), statistics.median(ls), ls[int(0.9 * (len(ls) - 1))], ls[-1]))
    parts.append("### Длина текста игрока (символы)\n\n" + fmt_table(rows, ["kind", "n", "median", "p90", "max"]) + "\n")
    return "\n".join(parts)


def main() -> int:
    files = [("tickets_valid.jsonl", RAW / "tickets_valid.jsonl"), ("tickets_train.jsonl.gz", RAW / "tickets_train.jsonl.gz")]
    md = ["# Анализ данных Support Copilot\n",
          "Сгенерировано `scripts/analyze_data.py`. Только агрегаты, без текстов обращений.\n"]
    for name, p in files:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        md.append(section(name, analyze(p)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
