import csv, hashlib, json
from collections import Counter

LABELS = ["actionable", "non_actionable", "ambiguous"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def assign_splits(cases, dev_pct=35):
    """Split by GROUP (all variants of a scenario stay together). Groups containing a case that came from the
    notebook (src='notebook') are forced to dev, because prompts were already tuned on them."""
    forced = {c["group"] for c in cases if c.get("src") == "notebook"}
    for c in cases:
        h = int(hashlib.md5(c["group"].encode()).hexdigest(), 16) % 100
        c["split"] = "dev" if (c["group"] in forced or h < dev_pct) else "test"
    return cases


def load_actionability(path="data/actionability_v1.jsonl"):
    return assign_splits(load_jsonl(path))


def validate_dataset(cases):
    rep = {}
    rep["n"] = len(cases)
    rep["by_split"] = dict(Counter(c["split"] for c in cases))
    rep["by_label_split"] = {s: dict(Counter(c["label"] for c in cases if c["split"] == s)) for s in ("dev", "test")}
    rep["by_lang"] = dict(Counter(c["lang"] for c in cases))
    rep["by_family"] = dict(Counter(c["family"] for c in cases))
    leak = {}
    for c in cases:
        leak.setdefault(c["group"], set()).add(c["split"])
    rep["groups_in_both_splits"] = [g for g, s in leak.items() if len(s) > 1]
    texts = {}
    dup = []
    for c in cases:
        key = json.dumps(c["turns"], ensure_ascii=False)
        if key in texts and texts[key] != c["split"]:
            dup.append(c["id"])
        texts.setdefault(key, c["split"])
    rep["duplicate_text_across_splits"] = dup
    bad = [c["id"] for c in cases if c["label"] not in LABELS]
    rep["bad_labels"] = bad
    pairs = {}
    for c in cases:
        if c.get("pair"):
            pairs.setdefault(c["pair"], set()).add(c["label"])
    rep["pairs_with_single_label"] = [p for p, s in pairs.items() if len(s) < 2]
    return rep


def freeze_hash(cases, split="test"):
    blob = json.dumps([[c["id"], c["turns"], c["label"], c["intents"]] for c in cases if c["split"] == split],
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------- blind review + agreement
def conv_text(c):
    return "\n".join(f"{'USER' if s == 'user' else 'OTHER'}: {t}" for s, t in c["turns"])


def make_review_sheet(cases, path="data/review_sheet.csv", seed=7):
    """Blind sheet for the human labeler: NO gold labels in it, shuffled order."""
    import random
    order = list(cases)
    random.Random(seed).shuffle(order)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "conversation (label the LAST turn)", "your_label (actionable / non_actionable / ambiguous)", "comment"])
        for c in order:
            w.writerow([c["id"], conv_text(c), "", ""])
    return path


def cohens_kappa(a, b, labels=LABELS):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    pe = sum((pa[l] / n) * (pb[l] / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compare_review(cases, sheet_path):
    """Compare the human labels in the filled sheet with the drafted gold labels."""
    gold = {c["id"]: c for c in cases}
    rows = []
    with open(sheet_path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or row[0] not in gold:
                continue
            lab = (row[2] or "").strip().lower().replace("-", "_").replace(" ", "_")
            if lab in LABELS:
                rows.append((row[0], lab, row[3] if len(row) > 3 else ""))
    ids = [r[0] for r in rows]
    g = [gold[i]["label"] for i in ids]
    h = [r[1] for r in rows]
    dis = [dict(id=i, gold=gold[i]["label"], human=hl, comment=cm, conversation=conv_text(gold[i]), note=gold[i].get("note", ""))
           for (i, hl, cm) in rows if hl != gold[i]["label"]]
    return dict(n_labeled=len(rows), n_total=len(cases), agreement=sum(x == y for x, y in zip(g, h)) / max(len(rows), 1),
                kappa=cohens_kappa(g, h), disagreements=dis)


def apply_adjudication(cases, adjudicated):
    """adjudicated: dict id -> final label. Returns cases with updated labels (and intents reset for non-actionable)."""
    for c in cases:
        if c["id"] in adjudicated and adjudicated[c["id"]] != c["label"]:
            c["label_original"] = c["label"]
            c["label"] = adjudicated[c["id"]]
            if c["label"] == "non_actionable":
                c["intents"] = ["none"]
            elif c["label"] == "ambiguous":
                c["intents"] = ["unknown"]
    return cases
