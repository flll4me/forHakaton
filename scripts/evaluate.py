"""Оценка качества на валидации: python scripts/evaluate.py

Пишет:
  reports/evaluation_report.md     — метрики (без текстов обращений, можно хранить в репозитории);
  reports/evaluation_examples.md   — примеры обработки с фрагментами текстов (в .gitignore, сдаётся отдельно);
  data/processed/draft_rating_sample.jsonl — выборка черновиков ru/kk/uz для ручной оценки 1–5.

Источники «истины»:
  - control v1 (200, data/processed/control_labels.jsonl) — ручная разметка разработчиком; правила
    дорабатывались по ошибкам на ней → оценка ОПТИМИСТИЧНАЯ;
  - test (100, data/processed/test_labels.jsonl) — ручная разметка разработчиком, на ней ничего не
    настраивалось → основная оценка;
  - weak labels valid — согласие со слабой разметкой (верхняя граница, не истина);
  - violation из ответа админа — проверка релевантности цитат на реальных решениях.
"""
from __future__ import annotations

import collections
import json
import random
import statistics
import sys
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.knowledge_base.rules_index import parse_rule_refs, rules_by_id  # noqa: E402
from app.models.schemas import TriageRequest  # noqa: E402
from app.services.classifier import priority_for  # noqa: E402
from app.services.llm_provider import estimate_tokens  # noqa: E402
from app.services.pipeline import TriagePipeline  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HAIKU_IN, HAIKU_OUT = 0.10, 0.50  # $/MTok, claude-haiku-5-5 (прайс Anthropic, кэш 2026-10-06)


def load_labels(name: str) -> dict:
    p = PROC / name
    return {json.loads(l)["ticket_id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()} if p.exists() else {}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(rows, headers) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main() -> int:
    setup_logging("WARNING")
    tickets = [json.loads(l) for l in (get_settings().data_dir / "tickets_valid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    weak = load_labels("valid_weak.jsonl")
    ctl, tst = load_labels("control_labels.jsonl"), load_labels("test_labels.jsonl")

    pipeline = TriagePipeline()
    order = sorted(range(len(tickets)), key=lambda i: (tickets[i].get("created_at") or "9999", i))
    out: dict[int, tuple] = {}
    lat = []
    for i in order:
        t = tickets[i]
        t0 = time.perf_counter()
        resp, dbg = pipeline.triage(TriageRequest(**t))
        lat.append((time.perf_counter() - t0) * 1000)
        out[i] = (resp, dbg)

    # HTTP-латентность (полный цикл FastAPI) на 200 запросах
    from fastapi.testclient import TestClient

    from app.main import app
    app.state.pipeline = pipeline
    http_lat = []
    with TestClient(app) as client:
        for t in random.Random(1).sample(tickets, 200):
            body = {"kind": t["kind"], "messages": [{"role": m["role"], "text": m["text"]} for m in t["messages"]],
                    "rule_refs_raw": t.get("rule_refs_raw"), "reported_nick": t.get("reported_nick"), "images_count": t.get("images_count")}
            t0 = time.perf_counter()
            r = client.post("/triage", json=body)
            http_lat.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200

    md = ["# Отчёт об оценке Grand Mobile Support Copilot\n",
          f"Сгенерировано `scripts/evaluate.py` {time.strftime('%Y-%m-%d %H:%M')}. Только агрегаты, без текстов обращений.\n",
          "## Данные и методика\n",
          f"- Валидация: `tickets_valid.jsonl`, {len(tickets)} обращений ({sum(t['kind'] == 'complaint' for t in tickets)} complaint, "
          f"{sum(t['kind'] == 'report' for t in tickets)} report). Категорий/приоритетов в данных нет.",
          "- Обучение: только `tickets_train.jsonl.gz` со слабой разметкой (`reports/weak_supervision.md`). Валидация в обучение не входила.",
          "- **Test (основная оценка)**: 100 случайных обращений valid, размечены вручную разработчиком (Claude) по taxonomy.json "
          "после завершения настройки; на них ничего не подбиралось.",
          "- **Control v1**: 200 обращений valid (стратифицированы по языку), размечены вручную до обучения; по ошибкам на них "
          "дорабатывались правила → оценка оптимистична, приводится для полноты и для разбивки по языкам.",
          "- Ручная разметка — не экспертная разметка Grand Mobile. Accuracy/F1 ниже — оценка, а не результат на истинных метках.",
          "- Приоритет «истины» выводится из ручной категории по критериям taxonomy.json (та же функция, что в сервисе), "
          "поэтому macro-F1 приоритета отражает в основном ошибки категории и доказательств.\n"]

    def gold_eval(labels: dict, name: str) -> list[str]:
        idx = [i for i, t in enumerate(tickets) if t["ticket_id"] in labels]
        if not idx:
            return [f"_{name}: разметка отсутствует_\n"]
        gc = [labels[tickets[i]["ticket_id"]]["category"] for i in idx]
        pc = [out[i][0].category for i in idx]
        gp, pp = [], []
        for i in idx:
            lab = labels[tickets[i]["ticket_id"]]
            facts = out[i][1].extraction
            gp.append(lab.get("priority_override") or priority_for(lab["category"], facts["has_evidence"], out[i][1].normalized_text).priority)
            pp.append(out[i][0].priority)
        gl = [labels[tickets[i]["ticket_id"]]["lang"] for i in idx]
        pl = [out[i][0].lang for i in idx]
        amb = [labels[tickets[i]["ticket_id"]].get("ambiguous") for i in idx]
        clear = [k for k, a in enumerate(amb) if not a]
        res = [f"### {name} (n={len(idx)})\n",
               table([
                   ["Accuracy категории", f"{accuracy_score(gc, pc):.3f}"],
                   ["Accuracy категории без неоднозначных", f"{accuracy_score([gc[k] for k in clear], [pc[k] for k in clear]):.3f} (n={len(clear)})"],
                   ["Macro-F1 категории", f"{f1_score(gc, pc, average='macro', zero_division=0):.3f}"],
                   ["Macro-F1 приоритета", f"{f1_score(gp, pp, average='macro', zero_division=0):.3f}"],
                   ["Accuracy приоритета", f"{accuracy_score(gp, pp):.3f}"],
                   ["Accuracy языка", f"{accuracy_score(gl, pl):.3f}"],
               ], ["метрика", "значение"]), ""]
        by_lang = collections.defaultdict(list)
        for k, i in enumerate(idx):
            by_lang[gl[k]].append(gc[k] == pc[k])
        res.append(table([[l, len(v), f"{sum(v) / len(v):.3f}"] for l, v in sorted(by_lang.items())], ["язык (ручной)", "n", "accuracy категории"]))
        errs = collections.Counter((g, p) for g, p in zip(gc, pc) if g != p)
        res.append("\nЧастые ошибки (истина → прогноз):\n")
        res.append(table([[g, p, n] for (g, p), n in errs.most_common(8)], ["истина", "прогноз", "n"]) + "\n")
        return res

    md.append("## Классификация\n")
    md += gold_eval(tst, "Test — независимая выборка")
    md += gold_eval(ctl, "Control v1 — выборка, использованная при настройке правил")
    w_idx = [i for i, t in enumerate(tickets) if weak.get(t["ticket_id"], {}).get("weak_label")]
    if w_idx:
        agree = sum(out[i][0].category == weak[tickets[i]["ticket_id"]]["weak_label"] for i in w_idx) / len(w_idx)
        md.append(f"Согласие со слабой разметкой на valid: {pct(agree)} (n={len(w_idx)}; слабые метки частично получены теми же правилами — верхняя граница).\n")

    # подбор порога уверенности на control v1
    if ctl:
        idx = [i for i, t in enumerate(tickets) if t["ticket_id"] in ctl]
        rows = []
        for thr in (0.4, 0.5, 0.55, 0.6, 0.7, 0.8):
            keep = [i for i in idx if out[i][1].category_confidence >= thr]
            acc = sum(out[i][0].category == ctl[tickets[i]["ticket_id"]]["category"] for i in keep) / max(1, len(keep))
            rows.append([thr, pct(len(keep) / len(idx)), f"{acc:.3f}", pct(1 - len(keep) / len(idx))])
        md.append("### Порог уверенности категории (control v1)\n")
        md.append(table(rows, ["порог", "доля выше порога", "accuracy выше порога", "эскалация из-за порога"]))
        md.append(f"\nТекущий порог MIN_CATEGORY_CONFIDENCE={get_settings().min_category_confidence}.\n")

    # цитирование
    index = rules_by_id()
    cited_total = invented = invalid_check = with_cite = 0
    agree_n = agree_ok = 0
    for i, t in enumerate(tickets):
        resp, dbg = out[i]
        cited = resp.draft_reply.cited
        cited_total += len(cited)
        with_cite += bool(cited)
        invented += sum(c not in index for c in cited)
        invalid_check += not dbg.citation_check.get("valid", True)
        admin_refs = [r for r in parse_rule_refs(t.get("violation") or "") + parse_rule_refs(t.get("punishment") or "") if r in index]
        main_cited = [c for c in cited if not c.startswith("ЖБ")]
        if t["kind"] == "complaint" and t.get("verdict") in {"approved", "other", "already_punished"} and admin_refs and main_cited:
            agree_n += 1
            agree_ok += bool(set(admin_refs) & set(main_cited))
    md.append("## Цитирование правил (отсутствие галлюцинаций)\n")
    md.append(table([
        ["Черновиков с цитатами", f"{with_cite} из {len(tickets)}"],
        ["Всего цитат", cited_total],
        ["Несуществующих пунктов в цитатах", invented],
        ["Ответов, не прошедших validate_citations (в выдаче)", invalid_check],
        ["Citation validity (существует + найден RAG + соответствует категории)", pct(1 - invalid_check / len(tickets))],
        ["Совпадение основной цитаты с пунктом, который применил админ (violation)", f"{pct(agree_ok / max(1, agree_n))} (n={agree_n})"],
    ], ["метрика", "значение"]) + "\n")

    # эскалация
    md.append("## Передача человеку (needs_human)\n")
    nh = [out[i][0].needs_human for i in range(len(tickets))]
    md.append(f"Общая доля: **{pct(sum(nh) / len(nh))}**.\n")
    by_cat = collections.defaultdict(list)
    for i in range(len(tickets)):
        by_cat[out[i][0].category].append(out[i][0].needs_human)
    md.append(table([[c, len(v), pct(sum(v) / len(v))] for c, v in sorted(by_cat.items())], ["категория (прогноз)", "n", "needs_human"]))
    reasons = collections.Counter(r.split(":")[0].split("(")[0].strip() for i in range(len(tickets)) for r in out[i][0].reasons)
    md.append("\nПричины:\n")
    md.append(table([[r, n] for r, n in reasons.most_common(15)], ["причина", "n"]) + "\n")
    by_lang = collections.defaultdict(list)
    for i in range(len(tickets)):
        by_lang[out[i][0].lang].append(out[i][0].needs_human)
    md.append(table([[l, len(v), pct(sum(v) / len(v))] for l, v in sorted(by_lang.items())], ["язык (прогноз)", "n", "needs_human"]) + "\n")

    # распределения
    md.append("## Распределения прогнозов на valid\n")
    md.append(table([[k, v] for k, v in collections.Counter(out[i][0].category for i in out).most_common()], ["категория", "n"]))
    md.append("")
    md.append(table([[k, v] for k, v in sorted(collections.Counter(out[i][0].priority for i in out).items())], ["приоритет", "n"]))
    md.append("")
    md.append(table([[k, v] for k, v in collections.Counter(out[i][0].lang for i in out).most_common()], ["язык", "n"]))
    md.append("")
    md.append(table([[k, v] for k, v in collections.Counter(out[i][0].draft_reply.lang for i in out).most_common()], ["язык черновика", "n"]) + "\n")

    # дедупликация
    st = collections.Counter(out[i][1].duplicate_status for i in out)
    spam = sum(out[i][1].spam for i in out)
    dup_verdict = [i for i, t in enumerate(tickets) if t.get("verdict") == "duplicate"]
    caught = sum(out[i][1].duplicate_status in {"duplicate", "related"} for i in dup_verdict)
    md.append("## Спам и повторы\n")
    md.append(table([["duplicate", st["duplicate"]], ["related", st["related"]], ["new_issue", st["new_issue"]], ["спам (без содержания)", spam],
                     ["is_spam_or_duplicate=true", sum(out[i][0].is_spam_or_duplicate for i in out)],
                     ["verdict=duplicate у админа", len(dup_verdict)],
                     ["из них найдено как duplicate/related", caught]], ["показатель", "n"]))
    md.append("\nОграничение: valid — выборка отдельных обращений, серии одного игрока в ней редки; админский verdict=duplicate "
              "обычно относится к жалобе, поданной раньше (возможно, вне окна часа и вне выборки).\n")

    # стоимость и латентность
    ls, hs = sorted(lat), sorted(http_lat)
    q = lambda a, p: a[min(len(a) - 1, int(p * (len(a) - 1)))]  # noqa: E731
    complex_n = sum(pipeline.generator.is_complex(tickets[i]["kind"], out[i][0].category, out[i][1].normalized_text) for i in out)
    est_in = statistics.mean(estimate_tokens(out[i][1].normalized_text[:1200]) + 700 for i in out)  # + системный промпт и 3 правила
    est_cost = (est_in * HAIKU_IN + 150 * HAIKU_OUT) / 1e6
    md.append("## Стоимость и латентность\n")
    md.append(table([
        ["Режим генерации в замере", f"LLM_PROVIDER={get_settings().llm_provider} (шаблоны + локальные модели)"],
        ["Фактическая стоимость на обращение", f"${sum(out[i][1].llm_cost_usd for i in out) / len(out):.6f} (весь инференс локальный, CPU)"],
        ["Обращений, где включился бы LLM (сложные)", f"{complex_n} ({pct(complex_n / len(out))})"],
        ["Оценка стоимости LLM-вызова (claude-haiku-5-5, ~%d вх./150 вых. ток.)" % est_in, f"${est_cost:.5f} за вызов"],
        ["Оценка средней стоимости на обращение с LLM", f"${est_cost * complex_n / len(out):.5f} (лимит $0.002)"],
        ["Пайплайн p50 / p95 / max, мс", f"{q(ls, .5):.1f} / {q(ls, .95):.1f} / {ls[-1]:.1f}"],
        ["HTTP POST /triage p50 / p95, мс (TestClient, n=200)", f"{q(hs, .5):.1f} / {q(hs, .95):.1f}"],
        ["Время загрузки сервиса, с", pipeline.load_s],
    ], ["показатель", "значение"]))
    md.append("\nЛатентность измерена на CPU ноутбука (Windows 10, Python 3.14). Оценка стоимости LLM — расчётная (API-ключ в "
              "среде разработки отсутствовал), токены оценены как символы/3; при включении LLM добавится сетевая задержка "
              "(таймаут 3 с, без ретраев, затем шаблон).\n")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")

    # примеры и выборка для ручной оценки черновиков
    rng = random.Random(5)
    ex = ["# Примеры обработки (содержит фрагменты данных кейса — не публиковать)\n"]
    test_ids = [i for i, t in enumerate(tickets) if t["ticket_id"] in tst]
    good = [i for i in test_ids if out[i][0].category == tst[tickets[i]["ticket_id"]]["category"]]
    bad = [i for i in test_ids if out[i][0].category != tst[tickets[i]["ticket_id"]]["category"]]
    for title, ids in [("Успешные", rng.sample(good, min(6, len(good)))), ("Неуспешные", bad[:8])]:
        ex.append(f"## {title}\n")
        for i in ids:
            t, (resp, dbg) = tickets[i], out[i]
            gold = tst[t["ticket_id"]]["category"]
            ex.append(f"- **{t['kind']}**, текст: «{dbg.normalized_text[:90]}»  \n  истина: `{gold}` → прогноз: `{resp.category}` "
                      f"({dbg.category_source}, conf {dbg.category_confidence}), приоритет {resp.priority}, lang {resp.lang}, "
                      f"needs_human={resp.needs_human}  \n  черновик: {resp.draft_reply.text}  \n  cited: {resp.draft_reply.cited}; причины: {resp.reasons}\n")
    (REPORTS / "evaluation_examples.md").write_text("\n".join(ex), encoding="utf-8")

    # отдельный генератор: состав выборки не зависит от того, какие обращения test классифицированы верно
    rng = random.Random(7)
    sample = []
    for lang, n in (("ru", 15), ("kk", 10), ("uz", 15)):
        ids = [i for i in out if out[i][0].draft_reply.lang == lang]
        for i in rng.sample(ids, min(n, len(ids))):
            t, (resp, dbg) = tickets[i], out[i]
            admin = " ".join(m["text"] for m in t["messages"] if m["role"] == "admin")
            sample.append({"ticket_id": t["ticket_id"], "lang": lang, "kind": t["kind"], "player": dbg.normalized_text[:300],
                           "category": resp.category, "draft": resp.draft_reply.text, "cited": resp.draft_reply.cited, "admin_reply": admin[:300]})
    with (PROC / "draft_rating_sample.jsonl").open("w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("written reports/evaluation_report.md, reports/evaluation_examples.md, data/processed/draft_rating_sample.jsonl", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
