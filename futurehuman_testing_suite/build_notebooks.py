import nbformat as nbf, textwrap, os

SETUP = '''
# ---- Colab setup: upload futurehuman_testing_suite.zip when asked (once per session) ----
import os, sys, subprocess
SUITE = "/content/futurehuman_testing_suite"
if not os.path.isdir(SUITE):
    from google.colab import files
    up = files.upload()
    subprocess.run(["unzip", "-o", "-q", list(up)[0], "-d", "/content"], check=True)
os.chdir(SUITE); sys.path.insert(0, SUITE)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "accelerate", "bitsandbytes", "tabulate"])
print("ready in", os.getcwd())
'''

LOAD = '''
TAG = "base"          # change to "stage1" / "stage2" when you test a trained adapter (same notebook, same tests)
ADAPTER = None        # e.g. "/content/drive/MyDrive/adapters/stage1"  (None = untouched base model)
QUANT = "4bit"        # 4bit | 8bit | fp16
from fh import env
from fh.model import load_qwen, make_gen
env.set_seed(0)
model, tok = load_qwen("Qwen/Qwen3-4B", quant=QUANT, adapter=ADAPTER)
gen = make_gen(model, tok)
info = env.save_snapshot(f"results/env_{TAG}.json", model=model, model_id="Qwen/Qwen3-4B", quant=QUANT, seed=0,
                         extra=dict(adapter=ADAPTER))
info
'''


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.metadata["accelerator"] = "GPU"
    n.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    out = []
    for kind, src in cells:
        src = textwrap.dedent(src).strip("\n")
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n.cells = out
    nbf.write(n, path)


M, C = "md", "code"

# =====================================================================================================
nb1 = [
 (M, """
 # FH-01 · Environment, baseline and actionability (Phases 0, 1, 3, 4)
 **Runtime → Change runtime type → T4 GPU.** Run cells top to bottom.

 What this notebook does
 1. Phase 0 - smoke test (load, determinism, VRAM, versions recorded)
 2. Dataset checks, split by scenario group, **locked test hash**
 3. Blind-review agreement (you vs. the drafted labels) and adjudication
 4. Phase 1 - baseline conditions A0 (original prompt), A1 (improved prompt), B (few-shot), C (thinking, optional)
 5. Phase 3 - actionability: confusion matrix, **false-action rate**, slices, minimal pairs, calibration
 6. Phase 4 - structured output validity, entities, relative-time resolution

 **Rule:** tune prompts on **dev** only. The **test** split is run once, at the end of tuning.
 Colab is cloud - only synthetic/public data goes in here.
 """),
 (C, SETUP),
 (C, LOAD),
 (M, "## Phase 0 - smoke test"),
 (C, """
 import torch
 probes = [[{"role": "user", "content": p}] for p in ["Say hi in one word.", "What is 2+2?", "Name a colour.", "Capital of India?", "Say OK."]]
 a = [gen(m, max_new_tokens=20)["text"] for m in probes]
 b = [gen(m, max_new_tokens=20)["text"] for m in probes]
 print("deterministic (greedy, run twice):", a == b)
 r = gen(probes[0], max_new_tokens=50); print({k: r[k] for k in ("latency_s", "n_in", "n_out")})
 print("peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))
 """),
 (M, "## Dataset: integrity, splits, locked test hash"),
 (C, """
 import os, json
 from fh.dataset import *
 DATA = "data/actionability_v1_adjudicated.jsonl" if os.path.exists("data/actionability_v1_adjudicated.jsonl") else "data/actionability_v1.jsonl"
 cases = load_actionability(DATA)
 print("using", DATA)
 print(json.dumps(validate_dataset(cases), indent=1, ensure_ascii=False))
 dev = [c for c in cases if c["split"] == "dev"]; test = [c for c in cases if c["split"] == "test"]
 print(len(dev), "dev /", len(test), "test")
 h = freeze_hash(cases); open(f"results/TEST_HASH_{TAG}.txt", "w").write(h); print("TEST HASH", h)
 """),
 (M, """
 ## Blind review → agreement (do this BEFORE the locked test run)
 1. Open `data/review_sheet.csv` in Excel/Sheets. Read `GUIDELINE.md`. Fill **your_label** for every row **without looking at any labels**.
 2. Upload the filled file below. You get agreement + Cohen's kappa between you and the drafted labels, and a list of disagreements.
 3. For each disagreement decide the final label (the guideline may need fixing too) and put it in `FINAL`.
 Skip this cell block only if you have already adjudicated (the adjudicated file is then picked up automatically).
 """),
 (C, """
 from google.colab import files
 import pandas as pd
 up = files.upload()
 r = compare_review(cases, list(up)[0])
 print(r["n_labeled"], "of", r["n_total"], "labelled | agreement", round(r["agreement"], 3), "| kappa", round(r["kappa"], 3), "(rule of thumb: >= 0.7)")
 pd.set_option("display.max_colwidth", 200)
 pd.DataFrame(r["disagreements"])[["id", "gold", "human", "conversation", "note", "comment"]]
 """),
 (C, """
 FINAL = {}   # e.g. {"A049": "ambiguous", "A055": "non_actionable"}  - your decision for each disagreement
 cases = apply_adjudication(cases, FINAL)
 with open("data/actionability_v1_adjudicated.jsonl", "w", encoding="utf-8") as f:
     for c in cases: f.write(json.dumps(c, ensure_ascii=False) + "\\n")
 cases = load_actionability("data/actionability_v1_adjudicated.jsonl")
 dev = [c for c in cases if c["split"] == "dev"]; test = [c for c in cases if c["split"] == "test"]
 h = freeze_hash(cases); open(f"results/TEST_HASH_{TAG}.txt", "w").write(h); print("LOCKED TEST HASH", h, "| changed labels:", len(FINAL))
 """),
 (M, "## Phase 1 - baseline on DEV (iterate here)"),
 (C, """
 from fh.evaluate import run_cases, save_run
 from fh.report import print_summary
 from fh.prompts import select_fewshot
 dev_A0 = run_cases(dev, gen, "A0_original"); print_summary(dev_A0, "DEV  A0 original prompt")
 """),
 (C, """
 dev_A1 = run_cases(dev, gen, "A1_improved"); print_summary(dev_A1, "DEV  A1 improved prompt")
 """),
 (C, """
 fs = select_fewshot(dev, n=(3, 3, 2), seed=0)
 print("few-shot ids (dev only, excluded from scoring):", [c["id"] for c in fs])
 dev_B = run_cases(dev, gen, "A1_improved", fewshot=fs); print_summary(dev_B, "DEV  B few-shot")
 for k, d in {"A0": dev_A0, "A1": dev_A1, "B": dev_B}.items():
     save_run(d, f"{TAG}_{k}_dev", dict(prompt=k, split="dev"))
 """),
 (M, """
 ### Look at the errors before touching the prompt
 The auto-tags below map to the error taxonomy in the plan. If you edit `PROMPT_V1` in `fh/prompts.py`, **rename it** (e.g. `A2_...`) and note what changed - never overwrite silently.
 """),
 (C, """
 import pandas as pd
 pd.set_option("display.max_colwidth", 160)
 def auto_tag(r):
     if r.pred_label == "invalid": return "invalid_json_or_schema"
     if r.gold_label != "actionable" and r.pred_label == "actionable": return "OVER_ACTION (false action)"
     if r.gold_label == "actionable" and r.pred_label == "non_actionable": return "under_action (missed)"
     if r.gold_label == "actionable" and r.pred_label == "ambiguous": return "unnecessary_ask"
     if r.gold_label == "ambiguous" and r.pred_label == "non_actionable": return "ambiguity_ignored"
     if r.gold_label == "non_actionable" and r.pred_label == "ambiguous": return "over_ask"
     if r.gold_label == "actionable" and not r.intent_ok: return "wrong_intent"
     if r.gold_label == "actionable" and (r.entity_recall or 0) < 0.5: return "wrong_entity"
     return "ok"
 dev_B["tag"] = dev_B.apply(auto_tag, axis=1)
 print(dev_B["tag"].value_counts())
 dev_B[dev_B.tag != "ok"][["id", "lang", "gold_label", "pred_label", "pred_intent", "tag", "raw"]].sort_values("tag")
 """),
 (M, """
 ## Phase 1/3 - locked TEST run (after dev tuning is finished)
 Runs A0, A1 and B on the test split and saves raw predictions. **Do not change prompts after looking at these numbers.**
 """),
 (C, """
 RUNS = {}
 conds = {"A0_original": dict(prompt_id="A0_original"), "A1_improved": dict(prompt_id="A1_improved"),
          "B_fewshot": dict(prompt_id="A1_improved", fewshot=fs)}
 for rid, kw in conds.items():
     df = run_cases(test, gen, **kw); RUNS[rid] = df
     save_run(df, f"{TAG}_{rid}_test", dict(kw, fewshot=[c["id"] for c in kw.get("fewshot", [])], model="Qwen3-4B", quant=QUANT, adapter=ADAPTER, env=info))
     print_summary(df, f"TEST {rid}")
 """),
 (C, """
 # Condition C (optional): thinking mode. Qwen recommends SAMPLING here (greedy can loop) so results are noisier and ~10x slower.
 RUN_C = False
 if RUN_C:
     sub = test[:40]
     dfC = run_cases(sub, gen, "A1_improved", fewshot=fs, thinking=True, max_new_tokens=1024)
     RUNS["C_thinking_40"] = dfC; save_run(dfC, f"{TAG}_C_thinking_40_test", dict(thinking=True, n=40)); print_summary(dfC, "TEST C thinking (first 40)")
 """),
 (M, "## Phase 3 - analysis of the best condition"),
 (C, """
 BEST = "B_fewshot"          # choose from RUNS (decide by DEV results, not test)
 df = RUNS[BEST]
 from fh.metrics import risk_coverage, confusion
 import matplotlib.pyplot as plt
 print(df.assign(tag=df.apply(auto_tag, axis=1)).tag.value_counts())
 print("\\nMinimal pairs - groups with ANY wrong variant:")
 bp = df[df.pair.notna()].groupby("group").filter(lambda x: (x.gold_label != x.pred_label).any())
 bp[["pair", "id", "gold_label", "pred_label", "raw"]].sort_values(["pair", "id"])
 """),
 (C, """
 rc = risk_coverage(df); print(rc.to_string(index=False))
 fig, ax = plt.subplots(1, 2, figsize=(10, 4))
 ax[0].plot(rc.coverage, rc.error_rate, "o-"); ax[0].set_xlabel("coverage (share of actionable cases executed)"); ax[0].set_ylabel("error rate of executed"); ax[0].set_title("Risk-coverage")
 import numpy as np
 ok = (df.gold_label == df.pred_label).astype(float); c = df.pred_conf.fillna(.5)
 bins = np.linspace(0, 1, 11); idx = np.digitize(c, bins) - 1
 xs = [c[idx == i].mean() for i in range(10) if (idx == i).any()]; ys = [ok[idx == i].mean() for i in range(10) if (idx == i).any()]
 ax[1].plot([0, 1], [0, 1], "--", c="gray"); ax[1].plot(xs, ys, "o-"); ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Reliability (calibration)")
 plt.tight_layout(); plt.savefig(f"results/{TAG}_risk_calibration.png", dpi=150); plt.show()
 print("distinct confidence values:", df.pred_conf.nunique(), "-> if this is ~1-2 the confidence field cannot drive the gate")
 """),
 (M, "## Phase 4 - structured output, entities, time resolution"),
 (C, """
 errs = df.schema_errors.fillna("").replace("", pd.NA).dropna().str.split(";").explode().value_counts()
 print("schema error types:\\n", errs if len(errs) else "none")
 print("\\nintent accuracy on gold-actionable:", df[df.gold_label == "actionable"].intent_ok.mean().round(3))
 print("entity recall / precision (actionable):", df.entity_recall.mean().round(3), df.entity_precision.mean().round(3))
 print("\\nintents the model produced for gold-actionable, wrong ones:")
 df[(df.gold_label == "actionable") & (~df.intent_ok)][["id", "gold_intents", "pred_intent", "raw"]]
 """),
 (C, """
 from fh import timeres
 tr = pd.DataFrame(timeres.run(gen)); tr.to_csv(f"results/timeres_{TAG}.csv", index=False)
 print("exact-match", tr.ok.mean().round(3), "| 'kal/parso/next Monday' must be AMBIGUOUS:")
 tr[["id", "text", "gold", "pred", "ok"]]
 """),
 (C, """
 from fh.report import update_capabilities, caps_from_df
 caps = update_capabilities(TAG, **caps_from_df(df), timeres_exact=dict(value=float(tr.ok.mean())))
 print("saved results/capabilities_%s.json" % TAG); caps
 """),
 (M, "**Next:** FH-02 (MultiWOZ + MASSIVE), FH-03 (gate, simulated tools, memory, privacy). They read the predictions saved in `results/`."),
]
nb(nb1, "notebooks/FH_01_baseline_and_actionability.ipynb")

# =====================================================================================================
nb2 = [
 (M, """
 # FH-02 · Intent, slots and dialogue state (Phase 2)
 Two public datasets, scored **separately** (a gain on one must not hide a loss on the other):
 - **MultiWOZ 2.2** → joint goal accuracy (every slot right at a turn) and slot F1; also the subset of turns that rely on history.
 - **MASSIVE 1.1** → intent accuracy / macro-F1 and slot F1, English and Hindi.

 Caveats to say out loud: MultiWOZ domains (hotel, train, ...) are not calendar tasks - this tests *context handling*, not the product task.
 MASSIVE is scored on its own 60-intent labels (no mapping to the project ontology yet).
 """),
 (C, SETUP),
 (C, LOAD),
 (C, """
 QUICK = True                     # True = 60 samples each (~10 min total). False = full sizes (~45 min on T4)
 N_MW, N_MS = (60, 60) if QUICK else (150, 200)
 import json, pandas as pd
 from fh import multiwoz as mw_mod, massive as ms
 from fh.report import update_capabilities
 from fh.metrics import fmt
 """),
 (M, "## MultiWOZ 2.2"),
 (C, """
 from datasets import load_dataset
 mw = load_dataset("pfb30/multi_woz_v22", trust_remote_code=True)
 print(mw)
 print(json.dumps(mw["test"][0]["turns"]["frames"][0], indent=1)[:900])   # sanity: confirm the structure the code expects
 """),
 (C, """
 slots, intents = mw_mod.build_inventory(mw["train"].select(range(3000)))
 print(len(slots), "slot names;", len(intents), "intents"); print(slots[:12])
 SYS = mw_mod.system_prompt(slots, intents)
 samples = mw_mod.make_samples(mw["test"], n_dialogues=100, max_samples=N_MW, seed=0)
 rows = mw_mod.run_multiwoz(samples, gen, SYS)
 pd.DataFrame(rows).to_csv(f"results/{TAG}_multiwoz.csv", index=False)
 s = mw_mod.summarize(rows)
 print("JGA", fmt(s["joint_goal_accuracy"]), "| slot F1", round(s["slot_f1"], 3), "(P %.2f R %.2f)" % (s["slot_precision"], s["slot_recall"]))
 print("JGA first turn", fmt(s["jga_first_turn"]), "| JGA with history", fmt(s["jga_with_history"]), "| JSON valid", fmt(s["json_valid"]))
 """),
 (C, """
 bad = [r for r in rows if not r["jga"]][:6]
 for r in bad: print(r["dialogue_id"], "turn", r["turn"], "| tp/fp/fn", r["tp"], r["fp"], r["fn"], "\\n   ", r["raw"][:250])
 """),
 (M, "## MASSIVE 1.1 (English and Hindi)"),
 (C, """
 slots_en = ms.slot_inventory("en-US")
 res, allrows = {}, []
 for loc in ["en-US", "hi-IN"]:
     items, names = ms.load_locale(loc, "test", n=N_MS, seed=0)
     SYSM = ms.system_prompt(names, slots_en)
     r_ = ms.run_massive(items, gen, SYSM); allrows += r_; res[loc] = ms.summarize(r_)
     print(loc, "intent acc", fmt(res[loc]["intent_accuracy"]), "| macro-F1", round(res[loc]["intent_macro_f1"], 3), "| slot F1", round(res[loc]["slot_f1"], 3), "| json", fmt(res[loc]["json_valid"]))
 pd.DataFrame(allrows).to_csv(f"results/{TAG}_massive.csv", index=False)
 """),
 (C, """
 d = pd.DataFrame(allrows); d["ok"] = d.gold_intent == d.pred_intent
 print("most confused intents (gold -> pred):"); d[~d.ok].groupby(["locale", "gold_intent", "pred_intent"]).size().sort_values(ascending=False).head(12)
 """),
 (C, """
 r = lambda x: dict(value=x["rate"], k=x["k"], n=x["n"])
 update_capabilities(TAG, multiwoz_jga=r(s["joint_goal_accuracy"]), multiwoz_slot_f1=dict(value=s["slot_f1"]),
                     massive_intent_acc_en=r(res["en-US"]["intent_accuracy"]), massive_intent_acc_hi=r(res["hi-IN"]["intent_accuracy"]),
                     massive_slot_f1_en=dict(value=res["en-US"]["slot_f1"]), massive_slot_f1_hi=dict(value=res["hi-IN"]["slot_f1"]))
 """),
]
nb(nb2, "notebooks/FH_02_multiwoz_massive.ipynb")
print("part 1 notebooks written")
