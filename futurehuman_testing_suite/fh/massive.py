import json, random, re
from .metrics import rate
from .schema import extract_json

SLOT_RE = re.compile(r"\[(\w+)\s*:\s*([^\]]+)\]")


def parse_slots(annot):
    out = {}
    for k, v in SLOT_RE.findall(annot or ""):
        out.setdefault(k, []).append(v.strip().lower())
    return out


def load_locale(locale, split="test", n=200, seed=0):
    from datasets import load_dataset
    try:
        ds = load_dataset("AmazonScience/massive", locale, split=split, trust_remote_code=True)
    except Exception:
        ds = load_dataset("AmazonScience/massive", locale, split=split)
    names = ds.features["intent"].names
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]
        items.append(dict(utt=r["utt"], intent=names[r["intent"]], slots=parse_slots(r["annot_utt"]), locale=locale))
    return items, names


def slot_inventory(locale="en-US", n=3000):
    from datasets import load_dataset
    try:
        tr = load_dataset("AmazonScience/massive", locale, split="train", trust_remote_code=True)
    except Exception:
        tr = load_dataset("AmazonScience/massive", locale, split="train")
    s = set()
    for r in tr.select(range(min(n, len(tr)))):
        s |= set(parse_slots(r["annot_utt"]))
    return sorted(s)


def system_prompt(intents, slots):
    return f"""You are a voice-assistant language-understanding module.
Given one user utterance (any language), return its intent and slots.
Allowed intents (choose exactly one): {', '.join(intents)}
Allowed slot types: {', '.join(slots)}
Return ONLY JSON: {{"intent": "<intent>", "slots": {{"<slot_type>": "<value copied from the utterance>"}}}}"""


def macro_f1(gold, pred):
    labels = sorted(set(gold))
    f = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        f.append(2 * P * R / (P + R) if P + R else 0)
    return sum(f) / len(f) if f else 0.0


def run_massive(items, gen, sysprompt, max_new_tokens=150):
    rows = []
    for it in items:
        r = gen([{"role": "system", "content": sysprompt}, {"role": "user", "content": it["utt"]}], thinking=False, max_new_tokens=max_new_tokens)
        o = extract_json(r["text"]) or {}
        ps = o.get("slots") if isinstance(o.get("slots"), dict) else {}
        pset = {(k, str(v).lower().strip()) for k, v in ps.items() for v in (v if isinstance(v, list) else [v])}
        gset = {(k, v) for k, vs in it["slots"].items() for v in vs}
        rows.append(dict(locale=it["locale"], utt=it["utt"], gold_intent=it["intent"], pred_intent=o.get("intent"), tp=len(pset & gset),
                         fp=len(pset - gset), fn=len(gset - pset), json_valid=bool(o), latency_s=r["latency_s"], raw=r["text"][:300]))
    return rows


def summarize(rows):
    n = len(rows)
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    gold = [r["gold_intent"] for r in rows]
    pred = [r["pred_intent"] for r in rows]
    return dict(n=n, intent_accuracy=rate(sum(g == p for g, p in zip(gold, pred)), n), intent_macro_f1=macro_f1(gold, pred),
                slot_f1=(2 * P * R / (P + R) if P + R else 0.0), slot_precision=P, slot_recall=R,
                json_valid=rate(sum(r["json_valid"] for r in rows), n))
