"""Обучение и сравнение классификаторов категорий на слабой разметке.

Подходы (по ТЗ):
  A. TF-IDF (слова 1-2 + символьные n-граммы 2-5) + LogisticRegression
  B. TF-IDF (те же признаки) + LinearSVC (+ калибровка вероятностей)
  C. Мультиязычные эмбеддинги (sentence-transformers) + LogisticRegression

Оценка:
  - holdout 10% train со слабыми метками, разбиение по reporter_hash (без утечки игроков);
  - ручная контрольная выборка (200 обращений из valid, data/processed/control_labels.jsonl):
    ML отдельно и гибрид «ML + правила», как в сервисе.
Победитель (по accuracy гибрида на контрольной выборке, при равенстве — macro-F1 holdout)
сохраняется в models/classifier.joblib, таблица — в reports/model_comparison.md.

Запуск: python scripts/train_classifier.py [--approaches A,B,C] [--max-train-c 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services import embeddings  # noqa: E402
from app.services.classifier import MODEL_FILE, CategoryClassifier, model_input  # noqa: E402
from app.services.normalizer import normalize_ticket  # noqa: E402

PROC = ROOT / "data" / "processed"
SEED = 42


def tfidf_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
    ])


def build(approach: str) -> Pipeline:
    if approach == "A":
        return Pipeline([("features", tfidf_features()),
                         ("clf", LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced"))])
    if approach == "B":
        return Pipeline([("features", tfidf_features()),
                         ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3))])
    if approach == "C":
        return Pipeline([("features", embeddings.EmbeddingFeatures(get_settings().embedding_model)),
                         ("clf", LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))])
    raise ValueError(approach)


def load_weak(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_control() -> list[dict]:
    labels = {json.loads(l)["ticket_id"]: json.loads(l) for l in (PROC / "control_labels.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = []
    for line in (get_settings().data_dir / "tickets_valid.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t["ticket_id"] in labels:
            n = normalize_ticket(t["messages"], t.get("region"), t.get("server_lang"))
            rows.append({"kind": t["kind"], "text": n.clean_text, "rule_refs_raw": t.get("rule_refs_raw") or "",
                         "gold": labels[t["ticket_id"]]["category"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approaches", default="A,B,C")
    ap.add_argument("--max-train-c", type=int, default=20000, help="подвыборка для подхода C (CPU-эмбеддинги)")
    args = ap.parse_args()

    data = [r for r in load_weak(PROC / "train_weak.jsonl") if r["weak_label"]]
    control = load_control()
    print(f"weakly labeled train: {len(data)}, control: {len(control)}", file=sys.stderr)

    results, fitted = [], {}
    for approach in args.approaches.split(","):
        if approach == "C" and not embeddings.available():
            results.append({"approach": "C", "status": "пропущен: модель эмбеддингов недоступна"})
            continue
        models, holdout_pred, holdout_gold, t0 = {}, [], [], time.time()
        for kind in ("complaint", "report"):
            rows = [r for r in data if r["kind"] == kind]
            X = np.array([model_input(kind, r["text"], r["rule_refs_raw"]) for r in rows], dtype=object)
            y = np.array([r["weak_label"] for r in rows])
            groups = np.array([r.get("reporter_hash") or r["ticket_id"] for r in rows])
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(X, y, groups))
            if approach == "C" and len(tr) > args.max_train_c:
                tr = np.random.default_rng(SEED).choice(tr, args.max_train_c, replace=False)
            pipe = build(approach).fit(X[tr], y[tr])
            holdout_pred += list(pipe.predict(X[te])); holdout_gold += list(y[te])
            models[kind] = pipe
        train_s = time.time() - t0

        clf = CategoryClassifier.__new__(CategoryClassifier)
        clf.models, clf.meta, clf.model_path = models, {"approach": approach}, None
        t1 = time.time()
        hybrid = [clf.predict(r["kind"], r["text"], r["rule_refs_raw"]).category for r in control]
        per_ticket_ms = 1000 * (time.time() - t1) / len(control)
        ml_only = [max(p := clf._ml(r["kind"], model_input(r["kind"], r["text"], r["rule_refs_raw"])), key=p.get) for r in control]
        gold = [r["gold"] for r in control]
        res = {
            "approach": approach, "status": "ok", "train_s": round(train_s, 1), "ms_per_ticket": round(per_ticket_ms, 2),
            "holdout_acc": accuracy_score(holdout_gold, holdout_pred), "holdout_macro_f1": f1_score(holdout_gold, holdout_pred, average="macro"),
            "control_acc_ml": accuracy_score(gold, ml_only), "control_acc_hybrid": accuracy_score(gold, hybrid),
            "control_macro_f1_hybrid": f1_score(gold, hybrid, average="macro"),
        }
        results.append(res); fitted[approach] = models
        print(json.dumps(res, ensure_ascii=False), file=sys.stderr)

    ok = [r for r in results if r["status"] == "ok"]
    best = max(ok, key=lambda r: (round(r["control_acc_hybrid"], 3), r["holdout_macro_f1"]))
    out = get_settings().models_dir / MODEL_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted[best["approach"]], "meta": {"approach": best["approach"], "trained_on": "train_weak",
                                                             "created": time.strftime("%Y-%m-%d %H:%M")}}, out)

    lines = ["# Сравнение классификаторов категорий\n",
             "Обучение на слабой разметке train (`scripts/prepare_dataset.py`). Holdout — 10% train по группам reporter_hash, "
             "метки слабые. Контроль — 200 обращений valid, размеченных вручную разработчиком (не экспертами Grand Mobile).\n",
             "| Подход | holdout acc (weak) | holdout macro-F1 (weak) | control acc ML | control acc ML+правила | control macro-F1 ML+правила | обучение, с | мс/тикет |",
             "|---|---|---|---|---|---|---|---|"]
    names = {"A": "A. TF-IDF + LogReg", "B": "B. TF-IDF + LinearSVC", "C": "C. Эмбеддинги + LogReg"}
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {names[r['approach']]} | {r['status']} | | | | | | |")
            continue
        lines.append(f"| {names[r['approach']]} | {r['holdout_acc']:.3f} | {r['holdout_macro_f1']:.3f} | {r['control_acc_ml']:.3f} | "
                     f"{r['control_acc_hybrid']:.3f} | {r['control_macro_f1_hybrid']:.3f} | {r['train_s']} | {r['ms_per_ticket']} |")
    lines.append(f"\nВыбран подход **{names[best['approach']]}**, сохранён в `models/{MODEL_FILE}`.")
    lines.append("\nОграничения: holdout оценивает согласие со слабыми метками (верхняя граница, т.к. метки частично получены "
                 "теми же правилами по тексту); контрольная выборка мала (n=200), 95% ДИ для accuracy ≈ ±6 п.п.")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"best={best['approach']} saved to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
