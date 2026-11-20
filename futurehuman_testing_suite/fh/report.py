import glob, os
import pandas as pd
from .metrics import summarize, fmt


def _md(t):
    try:
        return t.to_markdown(index=False)
    except Exception:
        return t.to_string(index=False)


def load_runs(results_dir="results", split="test"):
    runs = {}
    for p in sorted(glob.glob(f"{results_dir}/*_predictions.csv")):
        rid = os.path.basename(p).replace("_predictions.csv", "")
        df = pd.read_csv(p)
        if split and "split" in df.columns:
            df = df[df["split"] == split]
        if len(df):
            runs[rid] = df
    return runs


def comparison_table(runs):
    rows = []
    for rid, df in runs.items():
        s = summarize(df)
        sf = s["safety"]
        rows.append({"run": rid, "n": s["n"], "macro_F1": round(s["macro_f1"], 3), "false_action": fmt(sf["false_action"]),
                     "missed_action": fmt(sf["missed_action"]), "unnecessary_ask": fmt(sf["unnecessary_ask"]),
                     "schema_valid": fmt(s["schema_valid"]), "ECE": round(s["ece"], 3), "distinct_conf_values": s["conf_unique"],
                     "pair_consistency": fmt(s["pair_consistency"]), "latency_p50_s": round(s["latency_p50"], 2)})
    return pd.DataFrame(rows)


def top_errors(df, k=10):
    bad = df[df["gold_label"] != df["pred_label"]].copy()
    bad["danger"] = (bad["pred_label"] == "actionable").astype(int)  # false actions first
    return bad.sort_values("danger", ascending=False).head(k)[["id", "lang", "family", "gold_label", "pred_label", "raw"]]


def markdown_report(runs, title="Futurehuman - model testing report"):
    t = comparison_table(runs)
    lines = [f"# {title}", "", "## Run comparison (locked TEST split)", "", _md(t), ""]
    for rid, df in runs.items():
        s = summarize(df)
        lines += [f"## {rid}", "", "Confusion matrix (rows = gold):", "", s["confusion"].to_string(), "", "Per class:", "", s["per_class"].round(3).to_string(), ""]
    return "\n".join(lines)


def print_summary(df, title=""):
    from .metrics import slice_report
    s = summarize(df)
    sf = s["safety"]
    print(f"===== {title}  (n={s['n']}) =====")
    print(f"macro-F1 {s['macro_f1']:.3f} | accuracy {fmt(sf['accuracy'])}")
    print(f"FALSE-ACTION   {fmt(sf['false_action'])}   (on non_actionable {fmt(sf['far_on_non_actionable'])}; on ambiguous {fmt(sf['far_on_ambiguous'])})")
    print(f"missed action  {fmt(sf['missed_action'])} | unnecessary ask {fmt(sf['unnecessary_ask'])} | over-ask {fmt(sf['over_ask'])}")
    print(f"json valid {fmt(s['json_valid'])} | schema valid {fmt(s['schema_valid'])} | invalid label {fmt(sf['invalid_rate'])}")
    print(f"intent acc (gold actionable) {fmt(s['intent_accuracy_on_actionable'])} | entity recall {s['entity_recall_mean']:.2f} precision {s['entity_precision_mean']:.2f}")
    print(f"ECE {s['ece']:.3f} | distinct confidence values: {s['conf_unique']} | pair consistency {fmt(s['pair_consistency'])}")
    print(f"latency p50 {s['latency_p50']:.2f}s p95 {s['latency_p95']:.2f}s")
    print("\nConfusion (rows=gold):\n", s["confusion"].to_string())
    print("\nPer class:\n", s["per_class"].round(3).to_string())
    print("\nBy language:\n", slice_report(df, "lang").round(2).to_string(index=False))
    print("\nBy family:\n", slice_report(df, "family").round(2).to_string(index=False))
    return s


import json as _json


def update_capabilities(tag, results_dir="results", **caps):
    """caps: name=dict(value=float, k=int, n=int) ; merged into results/capabilities_<tag>.json"""
    os.makedirs(results_dir, exist_ok=True)
    path = f"{results_dir}/capabilities_{tag}.json"
    cur = _json.load(open(path)) if os.path.exists(path) else {}
    cur.update(caps)
    _json.dump(cur, open(path, "w"), indent=2)
    return cur


def load_capabilities(tag, results_dir="results"):
    path = f"{results_dir}/capabilities_{tag}.json"
    return _json.load(open(path)) if os.path.exists(path) else {}


def caps_from_df(df):
    s = summarize(df)
    sf = s["safety"]
    r = lambda x: dict(value=x["rate"], k=x["k"], n=x["n"])
    return dict(actionability_macro_f1=dict(value=s["macro_f1"]), false_action_rate=r(sf["false_action"]),
                missed_action_rate=r(sf["missed_action"]), schema_valid_rate=r(s["schema_valid"]),
                intent_accuracy=r(s["intent_accuracy_on_actionable"]), ece=dict(value=s["ece"]),
                latency_p50_s=dict(value=s["latency_p50"]), latency_p95_s=dict(value=s["latency_p95"]))


# proposed starting points (the business owner must confirm these)
ACCEPTANCE = [
    ("Schema-valid output", "schema_valid_rate", ">=", 0.99),
    ("False-action rate (test)", "false_action_rate", "<=", 0.03),
    ("False-action rate, upper CI bound", "false_action_rate_hi", "<=", 0.05),
    ("False-execution rate (after gate)", "false_execution_rate", "<=", 0.01),
    ("Actionable recall (1 - missed)", "actionable_recall", ">=", 0.85),
    ("Calibration ECE", "ece", "<=", 0.10),
    ("Indian slices gap vs English (points)", "indian_gap_points", "<=", 5.0),
    ("Redaction recall (standard synthetic set)", "redaction_recall", ">=", 0.99),
    ("Redaction recall (adversarial set)", "redaction_recall_adversarial", ">=", 0.95),
    ("End-to-end p95 latency (s)", "latency_p95_s", "<=", 5.0),
]


def acceptance_table(caps):
    import pandas as pd
    flat = {k: v["value"] for k, v in caps.items() if isinstance(v, dict) and "value" in v}
    if "false_action_rate" in caps and "n" in caps["false_action_rate"]:
        from .metrics import wilson
        flat["false_action_rate_hi"] = wilson(caps["false_action_rate"]["k"], caps["false_action_rate"]["n"])[1]
    if "missed_action_rate" in flat:
        flat["actionable_recall"] = 1 - flat["missed_action_rate"]
    rows = []
    for name, key, op, thr in ACCEPTANCE:
        v = flat.get(key)
        if v is None:
            verdict = "NOT TESTED"
        else:
            verdict = "PASS" if ((v >= thr) if op == ">=" else (v <= thr)) else "FAIL"
        rows.append({"criterion": name, "target": f"{op} {thr}", "measured": None if v is None else round(v, 4), "verdict": verdict})
    return pd.DataFrame(rows)
