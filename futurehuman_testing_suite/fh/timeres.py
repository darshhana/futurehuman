"""Relative-date resolution test (Phase 4 extension).  Reference 'now' is fixed in the prompt."""
import json, re
from .schema import extract_json

NOW = "2026-11-15T09:00 (Sunday)"
SYSTEM = f"""Resolve the time expression in the text to an absolute local datetime.
Current datetime: {NOW}.
Return ONLY JSON: {{"datetime": "YYYY-MM-DDTHH:MM"}} or {{"datetime": "AMBIGUOUS"}} when the expression cannot be resolved
(for example Hindi 'kal'/'parso' without a verb tense, or 'next Monday' that could mean two different days)."""

CASES = [
 ("T01", "Remind me tomorrow at 10 AM.", "2026-11-16T10:00"),
 ("T02", "Meeting on Friday at 5 PM.", "2026-11-20T17:00"),
 ("T03", "Call him the day after tomorrow at noon.", "2026-11-17T12:00"),
 ("T04", "Let's meet Monday 9:30 AM.", "2026-11-16T09:30"),
 ("T05", "Dinner tonight at 8.", "2026-11-15T20:00"),
 ("T06", "Remind me in 30 minutes.", "2026-11-15T09:30"),
 ("T07", "Remind me in two hours.", "2026-11-15T11:00"),
 ("T08", "This Friday at 3 PM.", "2026-11-20T15:00"),
 ("T09", "Kal subah 10 baje Rahul ko call karna hai.", "2026-11-16T10:00"),
 ("T10", "Parso 4 baje meeting rakhni hai.", "2026-11-17T16:00"),
 ("T11", "Kal 10 baje.", "AMBIGUOUS"),
 ("T12", "Parso 4 baje.", "AMBIGUOUS"),
 ("T13", "Next Monday at 11.", "AMBIGUOUS"),
]


def run(gen):
    rows = []
    for cid, text, gold in CASES:
        r = gen([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], thinking=False, max_new_tokens=60)
        o = extract_json(r["text"]) or {}
        pred = str(o.get("datetime", "")).strip()
        rows.append(dict(id=cid, text=text, gold=gold, pred=pred, ok=(pred == gold), latency_s=r["latency_s"], raw=r["text"][:120]))
    return rows
