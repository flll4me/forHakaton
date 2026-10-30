"""Интерфейс администратора (Streamlit): очередь, карточка обращения, Send / Edit / Escalate, feedback.

Запуск: streamlit run dashboard/app.py  (демо: DEMO=1, см. DEMO.md)
Send в автономном MVP ничего не отправляет в игру (API игры нет): ответ и действие
сохраняются локально в SQLite (FEEDBACK_DB) как обучающий сигнал.

Источник очереди: «Демо-сценарий» (demo/demo_tickets.jsonl — обращения, написанные вручную, без данных
кейса) или любой JSONL-файл. Вкладка «О системе» показывает метрики только из отчётов reports/*.md.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import safety  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.knowledge_base.rules_index import category_codes, rules_by_id  # noqa: E402
from app.models.schemas import TriageRequest  # noqa: E402
from app.services.feedback import FeedbackStore  # noqa: E402
from app.services.pipeline import TriagePipeline  # noqa: E402

PRIORITY_ORDER = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
DEMO_FILE = ROOT / "demo" / "demo_tickets.jsonl"
# безусловные причины эскалации (ТЗ + taxonomy.json) — выделяются красным
MANDATORY_REASONS = (safety.R_PAYMENT, safety.R_REFUND, safety.R_BAN_APPEAL, safety.R_HACK, safety.R_THREAT,
                     safety.R_POSSIBLE_APPEAL, safety.R_ALWAYS_CATEGORY.split("{")[0])


def highlight_citations(text: str, cited: list[str]) -> str:
    """Черновик с подсветкой процитированных пунктов (текст экранируется)."""
    out = html.escape(text)
    for rid in sorted(cited, key=len, reverse=True):
        out = re.sub(rf"(?<![\w.]){re.escape(html.escape(rid))}(?![\w.]|\.\d)", lambda m: f"<mark>{m.group(0)}</mark>", out)
    return out


def report_section(name: str, heading: str) -> str | None:
    """Раздел markdown-отчёта от заголовка до следующего заголовка того же или более высокого уровня."""
    p = ROOT / "reports" / name
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#") and heading in line:
            level = len(line) - len(line.lstrip("#"))
            body = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("#") and len(nxt) - len(nxt.lstrip("#")) <= level:
                    break
                body.append(nxt)
            return "\n".join(body).strip().replace("$", r"\$")  # иначе Streamlit рендерит $…$ как формулу
    return None
st.set_page_config(page_title="Support Copilot", layout="wide")


@st.cache_resource(show_spinner="Загрузка моделей и индекса правил…")
def get_pipeline() -> TriagePipeline:
    return TriagePipeline()


@st.cache_resource
def get_store() -> FeedbackStore:
    return FeedbackStore()


@st.cache_data(show_spinner="Триаж очереди…")
def load_queue(path: str, limit: int) -> list[dict]:
    p = get_pipeline()
    p.deduplicator.reset()
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    rows.sort(key=lambda t: t.get("created_at") or "")
    out = []
    for t in rows:
        resp, dbg = p.triage(TriageRequest(**t))
        player = "\n".join(m.get("text") or "" for m in t.get("messages", []) if m.get("role", "player") == "player")
        accused = "\n".join(m.get("text") or "" for m in t.get("messages", []) if m.get("role") == "reported_player")
        out.append({"ticket": t, "player": player, "accused": accused, "resp": resp.model_dump(), "debug": dbg.model_dump()})
    return out


store = get_store()
st.sidebar.title("Support Copilot")
valid_path = get_settings().data_dir / "tickets_valid.jsonl"
demo_default = os.getenv("DEMO", "").lower() in {"1", "true", "yes"} or not valid_path.exists()
source = st.sidebar.radio("Источник очереди", ["Демо-сценарий", "Файл JSONL"], index=0 if demo_default else 1)
if source == "Демо-сценарий":
    path, limit = str(DEMO_FILE), 1000
    st.sidebar.caption("16 обращений, написанных вручную для показа (не данные кейса).")
else:
    path = st.sidebar.text_input("Файл обращений (JSONL)", str(valid_path))
    limit = st.sidebar.slider("Сколько обращений загрузить", 20, 1000, 150, step=10)
only_human = st.sidebar.checkbox("Только needs_human", False)
hide_done = st.sidebar.checkbox("Скрыть обработанные", source != "Демо-сценарий")
llm_status = get_pipeline().status()
st.sidebar.caption(f"Генерация: {'LLM ' + str(llm_status['llm_model']) if llm_status['llm_provider'] != 'none' else 'шаблоны'}"
                   + (f" ({llm_status['llm_disabled_reason']})" if llm_status.get("llm_disabled_reason") else "")
                   + f" · классификатор: {llm_status['classifier']} · RAG dense: {llm_status['rag_dense']}")

tab_queue, tab_new, tab_fb, tab_about = st.tabs(["Очередь", "Новое обращение", "Feedback", "О системе"])

with tab_new:
    kind = st.radio("Тип", ["complaint", "report"], horizontal=True)
    text = st.text_area("Текст игрока", "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП")
    c1, c2, c3 = st.columns(3)
    att = c1.number_input("Вложений", 0, 20, 2)
    nick = c2.text_input("reported_nick", "")
    region = c3.selectbox("Сервер", ["cis", "kz", "uz"])
    if st.button("Обработать"):
        req = TriageRequest(kind=kind, region=region, reported_nick=nick or None,
                            messages=[{"role": "player", "text": text, "attachments": att}])
        resp, dbg = get_pipeline().triage(req)
        st.json(resp.model_dump())
        with st.expander("Диагностика"):
            st.json(dbg.model_dump())

with tab_queue:
    if not Path(path).exists():
        st.warning("Файл не найден. Распакуйте данные кейса в data/raw/handout или укажите путь.")
        st.stop()
    queue = load_queue(path, limit)
    status = store.status_by_ticket()
    table = []
    for k, item in enumerate(queue):
        r, t = item["resp"], item["ticket"]
        tid = t.get("ticket_id") or f"row{k}"
        table.append({"#": k, "ID": tid[:12], **({"сценарий": t["demo_title"]} if t.get("demo_title") else {}),
                      "kind": t["kind"], "категория": r["category"], "приоритет": r["priority"],
                      "язык": r["lang"], "статус": status.get(tid, "new"), "needs_human": r["needs_human"],
                      "confidence": r["confidence"], "дубль/спам": r["is_spam_or_duplicate"]})
    df = pd.DataFrame(table)
    if only_human:
        df = df[df["needs_human"]]
    if hide_done:
        df = df[df["статус"] == "new"]
    df = df.sort_values(by=["приоритет", "confidence"], key=lambda s: s.map(PRIORITY_ORDER) if s.name == "приоритет" else s)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("В очереди", len(df))
    c2.metric("needs_human", int(df["needs_human"].sum()) if len(df) else 0)
    c3.metric("P0", int((df["приоритет"] == "p0").sum()) if len(df) else 0)
    c4.metric("Дубли/спам", int(df["дубль/спам"].sum()) if len(df) else 0)
    st.dataframe(df, hide_index=True, use_container_width=True, height=280)

    if len(df):
        sel = st.selectbox("Открыть обращение", df["#"].tolist(),
                           format_func=lambda k: f"{table[k]['ID']} · {table[k]['категория']} · {table[k]['приоритет']}")
        item = queue[sel]
        r, d, t = item["resp"], item["debug"], item["ticket"]
        tid = t.get("ticket_id") or f"row{sel}"
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Обращение")
            n_att = (t.get("images_count") or 0) + sum(m.get("attachments") or 0 for m in t.get("messages", []))
            st.caption(f"{t['kind']} · сервер {t.get('region')} · вложений {n_att} · "
                       f"пункт в форме: {t.get('rule_refs_raw') or '—'}")
            st.text(item["player"] or "(пусто)")
            if item["accused"]:
                st.caption("Комментарий обвиняемого (не слова заявителя):")
                st.text(item["accused"])
            st.subheader("Извлечённые сведения")
            st.json({**r["extracted"], "invalid_rule_refs": d["extraction"].get("invalid_rule_refs"),
                     "player_ids": d["extraction"].get("player_ids"), "дедупликация": d.get("duplicate_status")})
            st.subheader("Найденные правила (RAG)")
            for rule in d["retrieved"]:
                st.markdown(f"**{rule['rule_id']}** · {rule['section']} · score {rule['score']} · "
                            f"BM25 #{rule['bm25_rank']} / dense #{rule['dense_rank']} {rule['prior']}")
                st.caption(rule["text"][:300])
        with right:
            if t.get("demo_title"):
                st.info(f"Сценарий: {t['demo_title']}")
            st.subheader("Черновик ответа")
            st.caption(f"язык {r['draft_reply']['lang']} · цитаты {r['draft_reply']['cited']} · генератор {d['generator']}")
            st.markdown(f"<div style='padding:.6em .8em;border-left:3px solid #999'>"
                        f"{highlight_citations(r['draft_reply']['text'], r['draft_reply']['cited'])}</div>",
                        unsafe_allow_html=True)
            index = rules_by_id()
            for rid in r["draft_reply"]["cited"]:
                with st.expander(f"п. {rid} — текст правила (rules.json)"):
                    st.caption(index[rid].section)
                    st.write(index[rid].text)
            if r["needs_human"]:
                st.error("Нужен человек: черновик не отправляется без проверки администратора")
            else:
                st.success("Можно ответить автоматически")
            for reason in r["reasons"]:
                (st.error if reason.startswith(MANDATORY_REASONS) else st.warning)(reason, icon="⚠️")
            final_text = st.text_area("Текст (можно править)", r["draft_reply"]["text"], height=180, key=f"txt{tid}")
            final_cat = st.selectbox("Категория", category_codes(), index=category_codes().index(r["category"]), key=f"cat{tid}")
            st.metric("Confidence", r["confidence"])
            st.caption(f"категория {d['category_confidence']} ({d.get('category_source')}) · поиск {d['retrieval_confidence']} · "
                       f"язык {d['lang_confidence']} · приоритет: {d['priority_reason']}")
            rating = st.select_slider("Оценка черновика", options=[1, 2, 3, 4, 5], value=4, key=f"rate{tid}")
            comment = st.text_input("Комментарий", key=f"cm{tid}")
            b1, b2, b3 = st.columns(3)
            action = None
            if b1.button("Send", type="primary", disabled=r["needs_human"] and final_text == r["draft_reply"]["text"],
                         help="Для needs_human сначала проверьте/исправьте текст"):
                action = "send"
            if b2.button("Edit"):
                action = "edit"
            if b3.button("Escalate"):
                action = "escalate"
            if action:
                if action == "send" and final_text != r["draft_reply"]["text"]:
                    action = "edit"
                store.add(ticket_id=tid, action=action, predicted_category=r["category"], final_category=final_cat,
                          priority=r["priority"], lang=r["draft_reply"]["lang"], draft_text=r["draft_reply"]["text"],
                          final_text=final_text, cited=r["draft_reply"]["cited"], rating=rating, comment=comment)
                st.success(f"Сохранено локально: {action}. В игру ничего не отправлено (API игры нет).")

with tab_fb:
    s = store.stats()
    st.write(s)
    st.dataframe(pd.DataFrame(store.all()), use_container_width=True)
    st.caption("Автоматического переобучения нет: перед использованием правок как обучающего сигнала их нужно проверить.")

with tab_about:
    st.subheader("Как устроен сервис")
    st.markdown("Нормализация → язык → категория (TF-IDF + LogReg + правила) → факты → повторы → поиск правил "
                "(BM25 + эмбеддинги) → черновик по шаблону → проверки безопасности → решение об эскалации. "
                "Сервис готовит черновик и не принимает решений: не обещает разбан, возврат или наказание; "
                "платежи, апелляции, взлом и угрозы всегда уходят человеку.")
    st.caption("Все цифры ниже берутся из отчётов в reports/ без изменений.")
    for title, name, heading in [
        ("Классификация (модель A)", "quality_audit.md", "Окончательные результаты модели A"),
        ("Безопасность генерации (1000 обращений валидации)", "quality_audit.md", "3. Безопасность генерации"),
        ("Качество черновиков (ручная оценка)", "quality_audit.md", "2. Качество черновиков"),
        ("Стоимость и латентность", "evaluation_report.md", "Стоимость и латентность"),
    ]:
        body = report_section(name, heading)
        st.markdown(f"#### {title}")
        if body:
            st.markdown(body)
        else:
            st.info(f"Отчёт reports/{name} не найден — запустите scripts/evaluate.py")
    st.markdown("#### Оговорки")
    st.markdown("- Разметка для оценки выполнена разработчиком, а не экспертами Grand Mobile.\n"
                "- Реальные вызовы LLM API не выполнялись: в демо черновики строятся по шаблонам.\n"
                "- Данные по казахскому малочисленны, оценка по kk ненадёжна.")
