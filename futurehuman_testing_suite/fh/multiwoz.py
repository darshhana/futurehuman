import json, random
from .metrics import rate
from .schema import extract_json


def extract_gold(frames):
    intents, slots, requested = {}, {}, {}
    for svc, st in zip(frames["service"], frames["state"]):
        if st["active_intent"] not in ("NONE", None, ""):
            intents[svc] = st["active_intent"]
        names, vals = st["slots_values"]["slots_values_name"], st["slots_values"]["slots_values_list"]
        for n, v in zip(names, vals):
            slots[n] = v
        if st["requested_slots"]:
            requested[svc] = st["requested_slots"]
    return dict(intents=intents, slot_values=slots, requested_slots=requested)


def build_inventory(split):
    slots, intents = set(), set()
    for d in split:
        for fr in d["turns"]["frames"]:
            for st in fr["state"]:
                if st["active_intent"] not in ("NONE", None, ""):
                    intents.add(st["active_intent"])
                slots |= set(st["slots_values"]["slots_values_name"])
    return sorted(slots), sorted(intents)


def system_prompt(slots, intents):
    return f"""You are a dialogue state tracker for a task-oriented assistant.
Given the conversation so far, output the CUMULATIVE dialogue state after the last USER message.

Allowed intents (per service): {', '.join(intents)}
Allowed slot names (use EXACTLY these, lowercase values): {', '.join(slots)}

Return ONLY JSON: {{"intents": {{"<service>": "<intent>"}}, "slot_values": {{"<slot_name>": "<value>"}}}}
Keep earlier slots unless the user changes them. Do not invent values. Do not add slots the user never mentioned."""


def make_samples(split, n_dialogues=100, max_samples=150, seed=0):
    rnd = random.Random(seed)
    idx = list(range(len(split)))
    rnd.shuffle(idx)
    samples = []
    for i in idx[:n_dialogues]:
        d = split[i]
        t = d["turns"]
        hist = []
        for k in range(len(t["turn_id"])):
            if t["speaker"][k] == 0:
                samples.append(dict(dialogue_id=d["dialogue_id"], turn=k, history=list(hist), utterance=t["utterance"][k],
                                    gold=extract_gold(t["frames"][k])))
            hist.append(("USER" if t["speaker"][k] == 0 else "ASSISTANT", t["utterance"][k]))
    rnd.shuffle(samples)
    return samples[:max_samples]


def to_messages(sample, sysprompt):
    conv = "\n".join(f"{r}: {u}" for r, u in sample["history"]) + (("\n" if sample["history"] else "") + f"USER: {sample['utterance']}")
    return [{"role": "system", "content": sysprompt}, {"role": "user", "content": conv}]


def _vals(v):
    if isinstance(v, (list, tuple)):
        return {str(x).lower().strip() for x in v}
    return {str(v).lower().strip()}


def score_state(gold_sv, pred_sv):
    pred_sv = pred_sv if isinstance(pred_sv, dict) else {}
    tp = sum(1 for k, v in pred_sv.items() if k in gold_sv and _vals(v) & _vals(gold_sv[k]))
    fp = len(pred_sv) - tp
    fn = len(gold_sv) - tp
    exact = (set(gold_sv) == set(pred_sv)) and all(_vals(pred_sv[k]) & _vals(gold_sv[k]) for k in gold_sv)
    return exact, tp, fp, fn


def run_multiwoz(samples, gen, sysprompt, max_new_tokens=300):
    rows = []
    for s in samples:
        r = gen(to_messages(s, sysprompt), thinking=False, max_new_tokens=max_new_tokens)
        o = extract_json(r["text"]) or {}
        exact, tp, fp, fn = score_state(s["gold"]["slot_values"], o.get("slot_values"))
        gi = s["gold"]["intents"]
        pi = o.get("intents") if isinstance(o.get("intents"), dict) else {}
        rows.append(dict(dialogue_id=s["dialogue_id"], turn=s["turn"], n_history=len(s["history"]), jga=exact, tp=tp, fp=fp, fn=fn,
                         intent_ok=(gi == pi), json_valid=bool(o), latency_s=r["latency_s"], raw=r["text"][:500]))
    return rows


def summarize(rows):
    n = len(rows)
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    ref = [r for r in rows if r["n_history"] >= 2]
    return dict(n=n, joint_goal_accuracy=rate(sum(r["jga"] for r in rows), n), slot_f1=(2 * P * R / (P + R) if P + R else 0.0),
                slot_precision=P, slot_recall=R, intent_exact=rate(sum(r["intent_ok"] for r in rows), n),
                json_valid=rate(sum(r["json_valid"] for r in rows), n),
                jga_with_history=rate(sum(r["jga"] for r in ref), len(ref)),
                jga_first_turn=rate(sum(r["jga"] for r in rows if r["n_history"] == 0), sum(r["n_history"] == 0 for r in rows)))
