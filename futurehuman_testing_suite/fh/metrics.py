import math, re
import numpy as np
import pandas as pd

LABELS = ["actionable", "non_actionable", "ambiguous"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


def rate(k, n):
    lo, hi = wilson(k, n)
    return dict(k=int(k), n=int(n), rate=(k / n if n else float("nan")), lo=lo, hi=hi)


def fmt(r, pct=True):
    if r["n"] == 0:
        return "n/a"
    m = 100 if pct else 1
    return f"{r['rate']*m:.1f}{'%' if pct else ''} [{r['lo']*m:.1f}-{r['hi']*m:.1f}] ({r['k']}/{r['n']})"


def confusion(gold, pred):
    cols = LABELS + ["invalid"]
    m = pd.DataFrame(0, index=LABELS, columns=cols)
    for g, p in zip(gold, pred):
        m.loc[g, p if p in LABELS else "invalid"] += 1
    return m


def prf(gold, pred):
    rows = {}
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        rows[l] = dict(precision=P, recall=R, f1=F, support=tp + fn)
    df = pd.DataFrame(rows).T
    return df, float(df["f1"].mean())


def safety(gold, pred):
    """The headline numbers. FAR = non_actionable/ambiguous cases predicted actionable."""
    g = np.array(gold)
    p = np.array(pred)
    neg = g != "actionable"
    act = g == "actionable"
    return dict(
        false_action=rate(((p == "actionable") & neg).sum(), neg.sum()),
        far_on_non_actionable=rate(((p == "actionable") & (g == "non_actionable")).sum(), (g == "non_actionable").sum()),
        far_on_ambiguous=rate(((p == "actionable") & (g == "ambiguous")).sum(), (g == "ambiguous").sum()),
        missed_action=rate(((p == "non_actionable") & act).sum(), act.sum()),
        unnecessary_ask=rate(((p == "ambiguous") & act).sum(), act.sum()),
        over_ask=rate(((p == "ambiguous") & (g == "non_actionable")).sum(), (g == "non_actionable").sum()),
        accuracy=rate((g == p).sum(), len(g)),
        invalid_rate=rate((~np.isin(p, LABELS)).sum(), len(p)),
    )


# ---------------------------------------------------------------- entities (key-agnostic: values only)
def _norm(s):
    return re.sub(r"[^\w\u0900-\u097F: ]+", "", str(s).lower()).strip()


def _flat(v):
    if isinstance(v, dict):
        out = []
        for x in v.values():
            out += _flat(x)
        return out
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out += _flat(x)
        return out
    return [_norm(v)] if v not in (None, "") else []


def entity_scores(gold_entities, pred_entities):
    g = [x for x in _flat(gold_entities) if x]
    p = [x for x in _flat(pred_entities) if x]
    if not g:
        return None
    hit = lambda a, b: a in b or b in a
    rec = sum(any(hit(x, y) for y in p) for x in g) / len(g)
    prec = (sum(any(hit(x, y) for y in g) for x in p) / len(p)) if p else 0.0
    return dict(recall=rec, precision=prec)


def intent_ok(pred_intent, gold_intents):
    return pred_intent in gold_intents


# ---------------------------------------------------------------- calibration
def ece(conf, correct, bins=10):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def risk_coverage(df, thresholds=None):
    """For each confidence threshold: of cases the model calls 'actionable' with conf>=t (would EXECUTE),
    how many are wrong (gold != actionable)?  coverage = share of actionable gold cases still executed."""
    thresholds = thresholds if thresholds is not None else [0, .5, .6, .7, .8, .85, .9, .95, .99]
    out = []
    act_gold = (df["gold_label"] == "actionable").sum()
    for t in thresholds:
        ex = df[(df["pred_label"] == "actionable") & (df["pred_conf"] >= t)]
        wrong = (ex["gold_label"] != "actionable").sum()
        right = (ex["gold_label"] == "actionable").sum()
        out.append(dict(threshold=t, executed=len(ex), wrong_executions=int(wrong),
                        error_rate=(wrong / len(ex) if len(ex) else 0.0),
                        coverage=(right / act_gold if act_gold else 0.0)))
    return pd.DataFrame(out)


def pair_consistency(df):
    """Minimal pairs: share of pair-groups where EVERY variant is labelled correctly."""
    d = df[df["pair"].notna()]
    if d.empty:
        return rate(0, 0)
    ok = d.groupby("group").apply(lambda x: bool((x["gold_label"] == x["pred_label"]).all()), include_groups=False)
    return rate(int(ok.sum()), len(ok))


def summarize(df):
    gold, pred = df["gold_label"].tolist(), df["pred_label"].tolist()
    table, macro = prf(gold, pred)
    s = safety(gold, pred)
    act = df[(df["gold_label"] == "actionable")]
    intent_acc = rate(int(act["intent_ok"].sum()), len(act)) if len(act) else rate(0, 0)
    ent = df["entity_recall"].dropna()
    correct = (df["gold_label"] == df["pred_label"]).astype(float)
    return dict(n=len(df), macro_f1=macro, per_class=table, confusion=confusion(gold, pred), safety=s,
                json_valid=rate(int(df["json_valid"].sum()), len(df)), schema_valid=rate(int(df["schema_valid"].sum()), len(df)),
                intent_accuracy_on_actionable=intent_acc,
                entity_recall_mean=float(ent.mean()) if len(ent) else float("nan"),
                entity_precision_mean=float(df["entity_precision"].dropna().mean()) if df["entity_precision"].notna().any() else float("nan"),
                confirm_alignment=rate(int((df["pred_rc"] == df["gold_rc"]).sum()), len(df)),
                ece=ece(df["pred_conf"].fillna(0.5), correct), conf_unique=int(df["pred_conf"].nunique()),
                pair_consistency=pair_consistency(df),
                latency_p50=float(df["latency_s"].median()), latency_p95=float(df["latency_s"].quantile(.95)))


def slice_report(df, by):
    rows = []
    for k, d in df.groupby(by):
        s = safety(d["gold_label"].tolist(), d["pred_label"].tolist())
        rows.append({by: k, "n": len(d), "accuracy": s["accuracy"]["rate"], "false_action": s["false_action"]["rate"],
                     "missed_action": s["missed_action"]["rate"], "invalid": s["invalid_rate"]["rate"]})
    return pd.DataFrame(rows).sort_values("n", ascending=False)
