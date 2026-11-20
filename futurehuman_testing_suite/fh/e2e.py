import copy
import pandas as pd
from .gate import PolicyGate
from .metrics import entity_scores, rate
from .schema import parse_prediction
from .tools_sim import INTENT_EFFECTS, MockWorld, build_call, observed_effects


def expected_effects(case):
    if case["label"] != "actionable":
        return set()
    out = set()
    for i in case["intents"]:
        out |= INTENT_EFFECTS.get(i, set())
    return out


def oracle_proposal(case, conf=0.95):
    if case["label"] == "actionable":
        return dict(actionability="actionable", intent=case["intents"][0], entities=case.get("entities") or {},
                    confidence=conf, requires_confirmation=bool(case["requires_confirmation"]))
    return dict(actionability=case["label"], intent="none" if case["label"] == "non_actionable" else "unknown",
                entities={}, confidence=conf, requires_confirmation=False)


def proposals_from_df(df):
    return {r["id"]: parse_prediction(r["raw"])["data"] for _, r in df.iterrows()}


def _would_be_correct(case, call, world):
    """Would executing this call produce the expected effect type with the expected entities?"""
    if call is None:
        return False
    w = copy.deepcopy(world)
    before = w.snapshot()
    try:
        w.execute(call)
    except Exception:
        return False
    eff = observed_effects(before, w.snapshot())
    if not (eff & expected_effects(case)):
        return False
    es = entity_scores(case.get("entities"), call.args)
    return es is None or es["recall"] >= 0.5


def run_e2e(cases, proposals, gate=None, memory=None, user_confirms=True):
    gate = gate or PolicyGate()
    rows = []
    for c in cases:
        if c["id"] not in proposals:
            continue
        world = MockWorld.from_dict(c.get("world"))
        prop = proposals[c["id"]]
        dec = gate.decide(prop, world, memory)
        exp = expected_effects(c)
        correct = _would_be_correct(c, dec.call, world) if (dec.call and exp) else False
        if dec.action == "EXECUTE":
            before = world.snapshot()
            try:
                world.execute(dec.call)
                eff = observed_effects(before, world.snapshot())
            except Exception:
                eff = set()
            outcome = "executed_correct" if (exp and correct) else ("false_execution" if eff else "executed_nothing")
        elif dec.action == "ASK":
            if c["label"] == "actionable":
                outcome = "completed_after_confirmation" if (user_confirms and correct) else "asked_then_rejected"
            elif c["label"] == "ambiguous":
                outcome = "asked_appropriately"
            else:
                outcome = "asked_unnecessarily"
        else:
            outcome = "missed" if c["label"] == "actionable" else "ignored_correctly"
        rows.append(dict(id=c["id"], family=c["family"], lang=c["lang"], gold=c["label"], action=dec.action,
                         reason=dec.reason, outcome=outcome, tier=dec.tier))
    return pd.DataFrame(rows)


def e2e_summary(res):
    n = len(res)
    act = res[res["gold"] == "actionable"]
    neg = res[res["gold"] != "actionable"]
    done = act["outcome"].isin(["executed_correct", "completed_after_confirmation"]).sum()
    return dict(
        n=n,
        task_completion=rate(done, len(act)),
        false_execution_all=rate((res["outcome"] == "false_execution").sum(), n),
        false_execution_on_negatives=rate((neg["outcome"] == "false_execution").sum(), len(neg)),
        executed_wrong_on_actionable=rate((act["outcome"] == "false_execution").sum(), len(act)),
        confirmations_needed_on_actionable=rate((act["action"] == "ASK").sum(), len(act)),
        missed=rate((act["outcome"] == "missed").sum(), len(act)),
        unnecessary_asks_on_non_actionable=rate((res[res["gold"] == "non_actionable"]["outcome"] == "asked_unnecessarily").sum(),
                                                (res["gold"] == "non_actionable").sum()),
        outcome_counts=res["outcome"].value_counts().to_dict())
