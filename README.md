# Grand Mobile Support Copilot

Сервис триажа обращений игроков Grand Mobile (Decentrathon 6.0, кейс 03). Он определяет язык, категорию и приоритет, отсеивает спам и повторы, извлекает факты, готовит черновик ответа на языке игрока со ссылками на пункты правил (RAG по `rules.json`) и честно сообщает, когда нужен человек.

Сервис помогает администратору, а не принимает решения сам. Он не обещает разбан, возврат средств или наказание другого игрока. Платежи, апелляции и подозрения на взлом всегда уходят человеку.

## Быстрый старт

Требуется Python 3.11+ (проверено на 3.12 в Docker и на 3.14 под Windows).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Положите архив кейса `grand-support-copilot-handout-2026-10-27.tgz` в корень проекта и распакуйте его в `data/raw/`:

```bash
make data          # = tar -xzf grand-support-copilot-handout-*.tgz -C data/raw
```

В результате должны появиться `data/raw/handout/{tickets_train.jsonl.gz, tickets_valid.jsonl, rules.json, taxonomy.json}`.

```bash
make train                                   # weak supervision + обучение и сравнение моделей (~8 мин на CPU)
make predict IN=tickets.jsonl OUT=pred.jsonl # пакетный прогноз
make run                                     # HTTP-сервис на :8000
make demo                                    # демо: проверка сценария + API + админка (см. DEMO.md)
make test                                    # автотесты (118, в т.ч. сквозные, отказы LLM и демо-сценарий)
make evaluate                                # отчёт reports/evaluation_report.md
python scripts/safety_audit.py               # аудит безопасности → reports/safety_audit.md
make dashboard                               # интерфейс администратора (Streamlit)
```

На Windows без `make` те же цели запускаются так: `.\tasks.ps1 predict -In tickets.jsonl -Out pred.jsonl`, `.\tasks.ps1 train`, `.\tasks.ps1 run` и т.д.

При первом запуске модель эмбеддингов `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~470 МБ) скачивается с Hugging Face. Без сети, или если в `.env` задано `EMBEDDINGS_ENABLED=false`, сервис работает в режиме «BM25 + TF-IDF»: эмбеддинги отключаются, и в `/health` это видно (`rag_dense: false`).

Docker (данные и модели монтируются томами, в образ не попадают):

```bash
docker build -t gm-copilot .
docker run -p 8000:8000 -v $PWD/data:/app/data -v $PWD/models:/app/models gm-copilot
```

## API

`POST /triage` — контракт из официального задания.

```bash
curl -s localhost:8000/triage -H 'content-type: application/json' -d '{"kind":"complaint","messages":[{"role":"player","text":"Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП","attachments":2}]}'
```

```json
{"lang":"ru","category":"complaint.insult","priority":"p2","is_spam_or_duplicate":false,
 "extracted":{"nickname":"Arman_Petrov","rule_refs":["2.1"],"has_evidence":true,"server":null},
 "draft_reply":{"lang":"ru","text":"Здравствуйте! Ваша жалоба на игрока Arman_Petrov принята. Описанная ситуация может относиться к п. 2.1 (оскорбления). Администратор проверит приложенные доказательства и примет решение в соответствии с правилами. Жалобы обычно рассматриваются в течение 48 часов (п. ЖБ 1.4.1).","cited":["2.1","ЖБ 1.4.1"]},
 "needs_human":true,"confidence":0.64,"reasons":["решение о наказании подтверждает администратор"]}
```

- Обязательны только `kind` и `messages`. Остальные поля выгрузки принимаются опционально и улучшают результат: `reported_nick`, `rule_refs_raw`, `images_count`, `created_at`, `reporter_hash`, `reported_hash`, `region`.
- `POST /triage?debug=true` добавляет поле `_debug` с найденными правилами, отдельными confidence, проверкой цитат и латентностью.
- `GET /health` возвращает готовность компонентов, `GET /version` — версии приложения и моделей.
- Некорректный JSON или неверная схема дают `422` с описанием ошибки. Ошибка внутри пайплайна даёт безопасный ответ с `needs_human: true`, сервис при этом не падает.
- Идентификаторы цитат: общие правила пишутся как в задании (`2.1`), пункты других разделов — с кодом раздела, потому что номера в `rules.json` повторяются между разделами: `ЖБ 1.2.7` (жалобы), `GZ 1.1` (зелёная зона), `ПГО`, `ОПГ`, `FW`, `ОПЛЗ`, `ПГВД`, `BW`, `FURA`. Коды взяты из того, как их пишут администраторы.

## Архитектура

```
INPUT → normalizer → language_detector → classifier → fact_extractor → deduplicator
      → rag (BM25 + FAISS) → response_generator (шаблоны / LLM) → safety gate → OUTPUT
```

| Модуль | Что делает |
|---|---|
| `app/services/normalizer.py` | NFC, мусор, пустые сообщения; cp1251-замены казахских букв (`ќ→қ` …) применяются только при признаках казахского текста, чтобы не портить украинские `є/ї` и узбекскую `ў`; хранит `original_text` и `normalized_text`; учитывает только сообщения игрока |
| `language_detector.py` | Письменность, специфичные буквы, частотные словари ru/kk/uz/en/es/pt и игровой сленг; `und` при нехватке признаков. Поле `lang` из выгрузки шумное и не используется как истина |
| `classifier.py` | Категория: TF-IDF (слова + символьные n-граммы) → линейная модель (по сравнению выбрана LogReg), отдельная модель на `complaint` и `report`, поверх — высокоточные правила для платежей, аккаунта и апелляций. Приоритет — детерминированная функция по критериям `taxonomy.json` |
| `category_rules.py` | Словари и правила; используются и для weak supervision, и при инференсе |
| `fact_extractor.py` | Ник (приоритет у `reported_nick`), заявленные пункты (несуществующие уходят в `reasons`), `has_evidence` = вложения или ссылки, сервер, ID игроков в репортах |
| `deduplicator.py` | Тот же `reporter_hash`, окно 60 мин, тот же `kind` и обвиняемый, косинус эмбеддингов (или символьные 3-граммы) → `duplicate` / `related` / `new_issue` |
| `rag.py` | 261 пункт `rules.json`, BM25 (стемминг + раскрытие сленга) + FAISS по мультиязычным эмбеддингам, Reciprocal Rank Fusion, приоры категории и заявленных пунктов; `validate_citations()` |
| `response_generator.py` | Шаблоны на 6 языках без вердиктов; цитируются только пункты из выдачи RAG, применимые к тексту; DM/DB/SK/TK объясняются по разделу «RP термины» без выдуманных номеров. LLM — для сложных случаев (длинные жалобы, вопросы) |
| `llm_provider.py` | `none` / `anthropic` (Claude Messages API, structured output) / `openai_compatible` (локальные модели: Ollama, vLLM); лимит входа, таймаут, подсчёт токенов и стоимости, отказ от вызова при превышении бюджета |
| `app/core/safety.py` | Централизованные условия эскалации, запрещённые обещания (6 языков), проверка цитат, языка и структуры |
| `pipeline.py` | Оркестратор; один и тот же код для API и `make predict` |
| `dashboard/app.py` | Очередь, карточка обращения (подсветка цитат и текст пунктов, причины эскалации), Send / Edit / Escalate, оценка 1–5; feedback в SQLite; демо-режим и вкладка «О системе» с метриками из `reports/`. Send ничего не отправляет в игру |

## LLM (опционально)

По умолчанию `LLM_PROVIDER=none`: черновики строятся по шаблонам, весь инференс локальный и бесплатный. Чтобы включить LLM, скопируйте `.env.example` в `.env` и задайте:

```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-5-5
LLM_PRICE_IN_PER_MTOK=0.10
LLM_PRICE_OUT_PER_MTOK=0.50
ANTHROPIC_API_KEY=...
```

`LLM_ENABLED=false` выключает LLM при любом провайдере. Если ключ или модель не заданы, сервис стартует в шаблонном режиме, а причина видна в `/health` (`llm_disabled_reason`). Таймаут, ошибки HTTP и битый ответ провайдера дают шаблонный черновик и `needs_human: true` (покрыто `tests/test_llm_failures.py`; реальные вызовы API в разработке не выполнялись).

Модель выбрана по ограничению кейса «не дороже $0.002 за обращение». Расчётная стоимость вызова около $0.00015, а LLM нужен примерно в 7.7% обращений. Ответ LLM проходит те же проверки (цитаты, обещания, язык); при любой ошибке используется шаблон. Для локальной модели: `LLM_PROVIDER=openai_compatible`, `LLM_BASE_URL=http://localhost:11434/v1`.

## Данные и разметка

В данных нет меток категорий и приоритетов. Порядок работы:

1. **Weak supervision** (`scripts/prepare_dataset.py`, отчёт `reports/weak_supervision.md`). Эвристики по тексту игрока и полям исхода из ответа админа (`verdict`, `violation`, формулировки об обжаловании, типовые ответы на репорты) размечают 83% train.
2. **Ручные выборки.** Разработчик вручную разметил 200 (control v1, использовалась при настройке) и 100 (test, на ней ничего не настраивалось) обращений valid по `taxonomy.json`. Это не экспертная разметка Grand Mobile.
3. **Сравнение моделей**: `reports/model_comparison.md` (TF-IDF+LogReg, TF-IDF+LinearSVC, эмбеддинги+LogReg).

Результаты: `reports/evaluation_report.md`, `reports/quality_audit.md` (итоговая проверка: найденные проблемы и приоритеты), `reports/safety_audit.md`, `reports/draft_quality.md`, `reports/dedup_evaluation.md`, `reports/data_analysis.md`.

## Конфиденциальность

- `data/`, `models/`, `*.jsonl`, `*.tgz`, `.env`, SQLite и `reports/evaluation_examples.md` (в нём есть фрагменты текстов) находятся в `.gitignore`. Отчёты в репозитории содержат только агрегаты.
- Во внешние сервисы данные уходят только при явно включённом LLM.
- Данные нужно удалить в течение 14 дней после объявления победителей: удалите `data/` и `models/`.

## Использованные модели и библиотеки

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — Apache-2.0.
- scikit-learn (BSD-3), faiss-cpu (MIT), rank-bm25 (Apache-2.0), FastAPI (MIT), Streamlit (Apache-2.0), PyTorch (BSD).
- Claude API (Anthropic) — опционально, по ключу пользователя.

Внешние датасеты не использовались. Архив `grand-loc-qa-handout` относится к другому кейсу и не использовался.
