import json
import pytest
from fh.schema import extract_json, validate, parse_prediction
from fh.dataset import load_actionability, validate_dataset, cohens_kappa, freeze_hash, make_review_sheet, compare_review
from fh.metrics import wilson, confusion, safety, ece, entity_scores, risk_coverage, pair_consistency
from fh.prompts import build_messages, select_fewshot, gold_json

GOOD = '{"actionability":"actionable","intent":"create_reminder","entities":{"person":"Rahul"},"confidence":0.8,"requires_confirmation":false}'

def test_extract_json_variants():
    assert extract_json("```json\n" + GOOD + "\n```")["intent"] == "create_reminder"
    assert extract_json("<think>blah {x}</think>\n" + GOOD)["actionability"] == "actionable"
    assert extract_json("no json here") is None
    assert extract_json('{"a": "}"}')["a"] == "}"

def test_validate_flags_enum_confusion():
    bad = json.loads(GOOD); bad["actionability"] = "reschedule_meeting"
    ok, errs = validate(bad)
    assert not ok and "bad_actionability_enum" in errs
    bad = json.loads(GOOD); bad["intent"] = ""
    assert "bad_intent_enum" in validate(bad)[1]
    assert validate(json.loads(GOOD))[0]
    assert not parse_prediction("garbage")["json_valid"]

def test_wilson_and_kappa():
    lo, hi = wilson(3, 100)
    assert 0.009 < lo < 0.02 and 0.08 < hi < 0.10
    a = ["actionable"] * 5 + ["non_actionable"] * 5
    assert cohens_kappa(a, a) == 1.0
    assert cohens_kappa(a, list(reversed(a))) < 0.2

def test_dataset_integrity():
    cases = load_actionability("data/actionability_v1.jsonl")
    rep = validate_dataset(cases)
    assert rep["groups_in_both_splits"] == []
    assert rep["bad_labels"] == []
    assert rep["pairs_with_single_label"] == []
    assert rep["duplicate_text_across_splits"] == []
    assert rep["by_split"]["dev"] > 20 and rep["by_split"]["test"] > 60
    for s in ("dev", "test"):
        assert set(rep["by_label_split"][s]) == {"actionable", "non_actionable", "ambiguous"}
    # notebook-origin cases (prompts tuned on them) are never in test
    assert all(c["split"] == "dev" for c in cases if c.get("src") == "notebook")
    assert freeze_hash(cases) == freeze_hash(cases)

def test_review_sheet_is_blind(tmp_path):
    cases = load_actionability("data/actionability_v1.jsonl")
    p = make_review_sheet(cases, str(tmp_path / "s.csv"))
    txt = open(p, encoding="utf-8-sig").read()
    assert "non_actionable" not in txt.split("\n", 1)[1]  # no gold labels leak into the sheet body

def test_review_comparison(tmp_path):
    import csv
    cases = load_actionability("data/actionability_v1.jsonl")[:6]
    p = tmp_path / "s.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "c", "your_label", "comment"])
        for i, c in enumerate(cases):
            w.writerow([c["id"], "x", c["label"] if i else "ambiguous", ""])
    r = compare_review(cases, str(p))
    assert r["n_labeled"] == 6 and len(r["disagreements"]) in (0, 1)

def test_safety_metrics():
    gold = ["actionable", "non_actionable", "ambiguous", "non_actionable", "actionable"]
    pred = ["actionable", "actionable", "ambiguous", "non_actionable", "ambiguous"]
    s = safety(gold, pred)
    assert s["false_action"]["k"] == 1 and s["false_action"]["n"] == 3
    assert s["unnecessary_ask"]["k"] == 1 and s["missed_action"]["k"] == 0
    c = confusion(gold, pred)
    assert c.loc["actionable", "ambiguous"] == 1

def test_entity_scores_key_agnostic():
    r = entity_scores({"person": "Rahul", "time": "10 AM"}, {"target": "rahul", "when": "tomorrow at 10 AM"})
    assert r["recall"] == 1.0 and r["precision"] == 1.0
    assert entity_scores({}, {"a": "b"}) is None

def test_ece_detects_overconfidence():
    assert ece([0.95] * 10, [1] * 5 + [0] * 5) > 0.4
    assert ece([0.5] * 10, [1] * 5 + [0] * 5) < 0.01

def test_prompt_builder_and_fewshot_from_dev_only():
    cases = load_actionability("data/actionability_v1.jsonl")
    dev = [c for c in cases if c["split"] == "dev"]
    fs = select_fewshot(dev)
    assert fs and all(c["split"] == "dev" for c in fs)
    msgs = build_messages(cases[0], "A1_improved", fs)
    assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user" and len(msgs) == 2 + 2 * len(fs)
    assert json.loads(gold_json(cases[0]))["actionability"] == cases[0]["label"]
