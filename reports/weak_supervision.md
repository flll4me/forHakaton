# Weak supervision: покрытие labeling functions

Сгенерировано `scripts/prepare_dataset.py`. Только счётчики.

## train: 118337 записей, размечено 98313 (83.1%)

| kind | label | LF | n |
|---|---|---|---|
| complaint | None | abstain | 9494 |
| complaint | complaint.admin | LF1_verdict_appeal | 7701 |
| complaint | complaint.admin | LF2_text_appeal | 5117 |
| complaint | complaint.admin | LF1b_admin_reply_appeal | 2392 |
| complaint | complaint.cheat | LF5_text | 1787 |
| complaint | complaint.cheat | LF4_violation | 838 |
| complaint | complaint.dm | LF5_text | 10416 |
| complaint | complaint.dm | LF4_violation | 1787 |
| complaint | complaint.insult | LF5_text | 12505 |
| complaint | complaint.insult | LF4_violation | 9757 |
| complaint | complaint.invalid | LF6_empty | 917 |
| complaint | complaint.invalid | LF3_admin_112 | 65 |
| complaint | complaint.other | LF5_text | 1734 |
| complaint | complaint.other | LF4_violation | 890 |
| complaint | complaint.rmt_ads | LF5_text | 129 |
| complaint | complaint.rmt_ads | LF4_violation | 23 |
| complaint | complaint.scam | LF5_text | 2005 |
| complaint | complaint.scam | LF4_violation | 198 |
| report | None | abstain | 10530 |
| report | report.account | LF1_text | 34 |
| report | report.complaint | LF1_text | 10452 |
| report | report.complaint | LF2_admin_reply | 1847 |
| report | report.help | LF1_text | 6626 |
| report | report.help | LF2_admin_reply | 503 |
| report | report.offtopic | LF1_text | 14491 |
| report | report.offtopic | LF2_admin_reply | 2798 |
| report | report.payment | LF1_text | 101 |
| report | report.question | LF1_text | 3131 |
| report | report.question | LF2_admin_reply | 69 |

## valid: 1000 записей, размечено 844 (84.4%)

| kind | label | LF | n |
|---|---|---|---|
| complaint | None | abstain | 127 |
| complaint | complaint.admin | LF1_verdict_appeal | 89 |
| complaint | complaint.admin | LF2_text_appeal | 54 |
| complaint | complaint.admin | LF1b_admin_reply_appeal | 34 |
| complaint | complaint.cheat | LF5_text | 21 |
| complaint | complaint.cheat | LF4_violation | 9 |
| complaint | complaint.dm | LF5_text | 122 |
| complaint | complaint.dm | LF4_violation | 25 |
| complaint | complaint.insult | LF5_text | 148 |
| complaint | complaint.insult | LF4_violation | 117 |
| complaint | complaint.invalid | LF6_empty | 9 |
| complaint | complaint.invalid | LF3_admin_112 | 3 |
| complaint | complaint.other | LF5_text | 22 |
| complaint | complaint.other | LF4_violation | 13 |
| complaint | complaint.rmt_ads | LF5_text | 2 |
| complaint | complaint.scam | LF5_text | 17 |
| complaint | complaint.scam | LF4_violation | 5 |
| report | None | abstain | 29 |
| report | report.account | LF1_text | 1 |
| report | report.complaint | LF1_text | 36 |
| report | report.complaint | LF2_admin_reply | 4 |
| report | report.help | LF1_text | 28 |
| report | report.help | LF2_admin_reply | 1 |
| report | report.offtopic | LF1_text | 69 |
| report | report.offtopic | LF2_admin_reply | 5 |
| report | report.question | LF1_text | 9 |
| report | report.question | LF2_admin_reply | 1 |
