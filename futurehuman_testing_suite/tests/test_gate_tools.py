import copy
from fh.gate import PolicyGate
from fh.tools_sim import MockWorld, normalize_entities, build_call, observed_effects
from fh.memory import UserMemory
from fh.e2e import oracle_proposal, run_e2e, e2e_summary
from fh.dataset import load_actionability

def P(**kw):
    base = dict(actionability="actionable", intent="create_reminder", entities={"title": "call Rahul", "date": "tomorrow", "time": "10 AM"},
                confidence=0.9, requires_confirmation=False)
    base.update(kw); return base

def W(): return MockWorld.from_dict({"calendar": [dict(title="meeting with Rahul", person="Rahul", date="tomorrow", time="11 AM")]})

def test_entity_alias_normalisation():
    assert normalize_entities({"target": "Rahul", "day": "Friday", "meeting_time": "3 PM"}) == dict(person="Rahul", date="Friday", time="3 PM")
    assert normalize_entities({"date": "tomorrow at 10 AM"})["time"] == "10 AM"
    assert normalize_entities("junk") == {}

def test_gate_rule_order_and_decisions():
    g, w = PolicyGate(), W()
    assert PolicyGate(paused=True).decide(P(), w).action == "IGNORE"
    assert g.decide(None, w).reason == "invalid_proposal"
    assert g.decide(P(actionability="reschedule_meeting"), w).reason == "invalid_proposal"
    assert g.decide(P(actionability="non_actionable"), w).action == "IGNORE"
    assert g.decide(P(actionability="ambiguous"), w).action == "ASK"
    assert g.decide(P(intent="unknown"), w).reason == "intent_has_no_tool"
    assert g.decide(P(), w).action == "EXECUTE"
    assert PolicyGate(permissions={"calendar"}).decide(P(), w).reason == "no_permission"
    assert g.decide(P(entities={}), w).reason.startswith("missing")

def test_gate_confidence_boundaries():
    w = W(); g = PolicyGate()
    ent = dict(person="Priya", date="Monday", time="11 AM")
    base = P(intent="schedule_meeting", entities=ent)
    assert g.decide(dict(base, confidence=0.80), w).action == "EXECUTE"      # exactly at medium threshold
    assert g.decide(dict(base, confidence=0.79), w).reason == "low_confidence"
    assert g.decide(P(confidence=0.60), w).action == "EXECUTE"               # low tier
    assert g.decide(P(confidence=0.59), w).action == "ASK"
    assert g.decide(dict(base, confidence="high"), w).reason == "low_confidence"

def test_gate_state_checks_and_risk_tiers():
    g, w = PolicyGate(), W()
    cancel = P(intent="cancel_meeting", entities={"person": "Rahul"}, confidence=0.99)
    assert g.decide(cancel, w).reason == "high_risk_needs_confirmation"      # high tier never auto-executes
    assert g.decide(P(intent="cancel_meeting", entities={"person": "Zed"}), w).reason == "target_not_found_or_ambiguous"
    assert g.decide(P(intent="send_message", entities={"person": "Anil", "message": "late"}, confidence=1.0), w).action == "ASK"
    conflict = P(intent="schedule_meeting", entities={"person": "Z", "date": "tomorrow", "time": "11 AM"})
    assert g.decide(conflict, w).reason == "calendar_conflict"
    mod = P(intent="reschedule_meeting", entities={"person": "Rahul", "date": "Friday"}, requires_confirmation=True)
    assert g.decide(mod, w).reason == "model_requested_confirmation"
    assert PolicyGate(honor_model_confirmation=False).decide(dict(mod, requires_confirmation=True), w).action == "EXECUTE"

def test_gate_never_executes_without_required_slots_property():
    import itertools
    g, w = PolicyGate(), MockWorld()
    for intent in ["schedule_meeting", "create_reminder", "create_task", "send_message", "reschedule_meeting", "cancel_meeting"]:
        for keys in itertools.chain.from_iterable(itertools.combinations(["person", "date", "time", "title", "message"], r) for r in range(0, 3)):
            ent = {k: "x" for k in keys}
            d = g.decide(P(intent=intent, entities=ent, confidence=1.0), w)
            if d.action == "EXECUTE":
                from fh.tools_sim import missing_args
                assert not missing_args(d.call)

def test_execution_effects():
    w = W(); before = w.snapshot()
    w.execute(build_call(P(intent="reschedule_meeting", entities={"person": "Rahul", "date": "Friday"})))
    assert observed_effects(before, w.snapshot()) == {"calendar_modify"}
    assert w.calendar[0]["date"] == "Friday"

def test_memory_rules():
    m, g, w = UserMemory(), PolicyGate(), W()
    assert not m.apply_feedback(dict(channel="conversation", type="set_preference", key="confirm_before", value="reminder_add"))  # injection rejected
    assert g.decide(P(), w, m).action == "EXECUTE"
    m.apply_feedback(dict(channel="ui", type="rejected", tool="reminder_add"))
    assert g.decide(P(), w, m).action == "EXECUTE"                      # one rejection is not a pattern
    m.apply_feedback(dict(channel="ui", type="rejected", tool="reminder_add"))
    assert g.decide(P(), w, m).reason == "user_preference_confirm"
    m.apply_feedback(dict(channel="ui", type="set_preference", key="default_reminder_lead_min", value=10))
    m.apply_feedback(dict(channel="ui", type="set_preference", key="default_reminder_lead_min", value=30))
    assert m.profile["default_reminder_lead_min"] == 30                 # drift: newest wins
    mid = m.remember("prefers evening meetings", tags=["meeting"])
    assert m.retrieve("evening meeting")[0]["id"] == mid
    assert m.delete(mid) and not m.retrieve("evening meeting")          # deletion removes behaviour source

def test_memory_never_loosens_gate():
    m = UserMemory()
    for ev in [dict(channel="ui", type="accepted", tool="calendar_remove")] * 20:
        m.apply_feedback(ev)
    w = W()
    d = PolicyGate().decide(P(intent="cancel_meeting", entities={"person": "Rahul"}, confidence=1.0), w, m)
    assert d.action == "ASK"

def test_e2e_oracle_is_clean():
    cases = load_actionability("data/actionability_v1.jsonl")
    res = run_e2e(cases, {c["id"]: oracle_proposal(c) for c in cases})
    s = e2e_summary(res)
    assert s["false_execution_all"]["k"] == 0                            # gold proposals never cause a wrong state change
    assert s["task_completion"]["rate"] >= 0.9, res[res.outcome.isin(["missed", "asked_then_rejected", "false_execution"])]
