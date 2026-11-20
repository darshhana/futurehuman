import json
from fh.privacy import redact, evaluate_redaction, should_capture, PauseState
from fh.robustness import perturb_cases, speaker_swap_last, PERTURBATIONS
from fh.speech import wer, cer, entity_error_rate, vad_metrics, normalize
from fh.regression import check, compare_runs, GENERAL
from fh.multiwoz import extract_gold, score_state
from fh.massive import parse_slots, macro_f1
from fh.dataset import load_actionability

def test_redaction_set_quality():
    items = [json.loads(l) for l in open("data/privacy_test.jsonl", encoding="utf-8")]
    r = evaluate_redaction(items)
    print("recall", r["recall"], "fpr", r["false_positive_rate"], [x["text"] for x in r["leaks"]], [x["text"] for x in r["false_positives"]])
    assert r["recall"] >= 0.95
    assert r["false_positive_rate"] <= 0.1

def test_redaction_examples():
    assert "482913" not in redact("Your OTP is 482913")[0]
    assert redact("My PIN code is 411057")[0] == "My PIN code is 411057"      # postal PIN is not a secret
    assert "4111" not in redact("card 4111 1111 1111 1111")[0]
    assert redact("Order #48213 shipped")[0] == "Order #48213 shipped"

def test_capture_rules_and_pause():
    base = dict(permission_granted=True, app="Notes", domain="example.com", title="x")
    s = dict(excluded_apps=["Slack"], excluded_domains=["bank.com"], excluded_categories=["banking"])
    assert should_capture(base, s) == (True, "ok")
    assert should_capture(dict(base, permission_granted=False), s)[1] == "no_permission"
    assert should_capture(dict(base, title="Google - Incognito"), s)[1] == "private_window"
    assert should_capture(dict(base, app="1Password 8"), s)[1] == "password_manager"
    assert should_capture(dict(base, field_type="password"), s)[1] == "password_field"
    assert should_capture(dict(base, app="slack"), s)[1] == "excluded_app"
    assert should_capture(dict(base, domain="login.bank.com"), s)[1] == "excluded_domain"
    assert should_capture(dict(base, domain="www.hdfcbank.com"), s)[1].startswith("excluded_category")
    p = PauseState(); p.pause(15, now=1000)
    assert should_capture(base, s, p, now=1000 + 14 * 60)[1] == "paused"
    assert should_capture(base, s, p, now=1000 + 16 * 60)[0] is True
    p.pause(); assert p.is_paused(now=10 ** 9); p.resume(); assert not p.is_paused()

def test_perturbations_keep_labels_and_change_text():
    cases = load_actionability("data/actionability_v1.jsonl")[:20]
    for name in PERTURBATIONS:
        out = perturb_cases(cases, name)
        assert [c["label"] for c in out] == [c["label"] for c in cases]
        assert any(a["turns"] != b["turns"] for a, b in zip(out, cases))
    sw = speaker_swap_last(cases[0]); assert sw["turns"][-1][0] != cases[0]["turns"][-1][0]

def test_speech_metrics():
    assert wer("call rahul at ten am", "call rahul at ten am") == 0
    assert abs(wer("a b c d", "a x c d") - 0.25) < 1e-9
    assert cer("राहुल", "राहुल") == 0
    assert entity_error_rate(["Rahul", "10 AM"], "call rahul at ten a m") == 0
    assert entity_error_rate(["Rahul", "10 AM"], "call rohan at five") == 1
    v = vad_metrics([(1, 3)], [(1.5, 3.5)], 5)
    assert 0.7 < v["speech_recall"] < 0.8 and v["false_alarm_s"] > 0.4

def test_general_suite_checkers_and_compare():
    assert check("num", 391, "17 × 23 = 391") and not check("num", 391, "I think 400")
    assert check("words", 3, "Blue, simply blue"[:0] + "It is blue") and check("exact", "YES", "YES.")
    assert not check("nojson", None, '{"actionability": "none"}') and check("json_keys", ["name", "age"], '{"name":"A","age":3}')
    assert len(GENERAL) == 28
    before = {"actionability_macro_f1": dict(value=0.7), "false_action_rate": dict(value=0.05, k=5, n=100), "general_acc": dict(value=0.9)}
    after = {"actionability_macro_f1": dict(value=0.85), "false_action_rate": dict(value=0.04, k=4, n=100), "general_acc": dict(value=0.85)}
    table, verdict = compare_runs(before, after)
    assert verdict == "REJECT" and table.loc[table.capability == "general_acc", "verdict"].iloc[0] == "REGRESSION"
    after["general_acc"] = dict(value=0.89)
    assert compare_runs(before, after)[1] == "ACCEPT"
    after["false_action_rate"] = dict(value=0.20, k=20, n=100)
    assert compare_runs(before, after)[1] == "REJECT"

def test_multiwoz_and_massive_helpers():
    fr = {"service": ["train", "taxi"], "state": [
        {"active_intent": "find_train", "requested_slots": ["train-price"], "slots_values": {"slots_values_name": ["train-day"], "slots_values_list": [["monday"]]}},
        {"active_intent": "NONE", "requested_slots": [], "slots_values": {"slots_values_name": [], "slots_values_list": []}}]}
    g = extract_gold(fr)
    assert g["intents"] == {"train": "find_train"} and g["slot_values"] == {"train-day": ["monday"]} and g["requested_slots"] == {"train": ["train-price"]}
    assert score_state(g["slot_values"], {"train-day": "Monday"})[0] is True
    assert score_state(g["slot_values"], {"train-day": "Monday", "train-leaveat": "10:00"})[0] is False
    assert score_state(g["slot_values"], {"from": "x"})[0] is False
    assert parse_slots("wake me up at [time : five am] on [date : friday]") == {"time": ["five am"], "date": ["friday"]}
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0

def test_deletion_propagation_and_isolation():
    from fh.store import Store
    s = Store()
    s.add_capture("a", "secret plan alpha", 100); s.add_capture("a", "later note beta", 5000); s.add_capture("b", "secret plan alpha", 100)
    assert s.search("a", "alpha") == ["secret plan alpha"] and s.search("b", "beta") == []   # isolation
    s.delete_range("a", 0, 1000)
    assert s.search("a", "alpha") == [] and s.search("b", "alpha") == ["secret plan alpha"]   # only user a, only that range
    assert s.search("a", "beta") != []
    s.delete_account("a")
    assert s.search("a", "beta") == []
    leaky = Store(leaky={"vectors"})
    leaky.add_capture("a", "gamma", 1); leaky.delete_range("a", 0, 10)
    assert leaky.residue("gamma") == ["vectors"]                                          # the audit catches a forgetful store
    clean = Store(); clean.add_capture("a", "gamma", 1); clean.delete_range("a", 0, 10)
    assert clean.residue("gamma") == []
