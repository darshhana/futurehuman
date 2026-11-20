import json, os, time
import pandas as pd
from .prompts import build_messages
from .schema import parse_prediction
from .metrics import intent_ok, entity_scores


def run_cases(cases, gen, prompt_id="A1_improved", fewshot=None, thinking=False, max_new_tokens=200,
              ref_date=None, progress=True, messages_fn=None):
    """Run the model on cases -> DataFrame (one row per case, raw output kept for error analysis)."""
    rows = []
    kw = {"ref_date": ref_date} if ref_date else {}
    skip = {c["id"] for c in (fewshot or [])}
    for i, c in enumerate(cases):
        if c["id"] in skip:
            continue
        msgs = messages_fn(c) if messages_fn else build_messages(c, prompt_id, fewshot, **kw)
        r = gen(msgs, thinking=thinking, max_new_tokens=max_new_tokens)
        p = parse_prediction(r["text"])
        d = p["data"] or {}
        valid_enum = d.get("actionability") in ("actionable", "non_actionable", "ambiguous")
        es = entity_scores(c.get("entities"), d.get("entities")) if c["label"] == "actionable" and isinstance(d.get("entities"), (dict, list)) else (
            dict(recall=0.0, precision=0.0) if c["label"] == "actionable" else None)
        rows.append(dict(
            id=c["id"], split=c.get("split"), group=c["group"], pair=c.get("pair"), family=c["family"], lang=c["lang"],
            gold_label=c["label"], gold_intents=json.dumps(c["intents"]), gold_rc=c["requires_confirmation"],
            pred_label=d.get("actionability") if valid_enum else "invalid", pred_intent=d.get("intent"),
            pred_conf=d.get("confidence") if isinstance(d.get("confidence"), (int, float)) and not isinstance(d.get("confidence"), bool) else None,
            pred_rc=d.get("requires_confirmation"), pred_entities=json.dumps(d.get("entities"), ensure_ascii=False),
            json_valid=p["json_valid"], schema_valid=p["schema_valid"], schema_errors=";".join(p["errors"]),
            intent_ok=intent_ok(d.get("intent"), c["intents"]), entity_recall=(es or {}).get("recall"),
            entity_precision=(es or {}).get("precision"), latency_s=r["latency_s"], n_out=r["n_out"], raw=r["text"]))
        if progress and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cases)}", end="\r")
    return pd.DataFrame(rows)


def save_run(df, run_id, config, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/{run_id}_predictions.csv", index=False, encoding="utf-8-sig")
    with open(f"{out_dir}/{run_id}_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
