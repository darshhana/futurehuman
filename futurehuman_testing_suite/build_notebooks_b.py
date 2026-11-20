from build_notebooks import nb, M, C, SETUP, LOAD

# =====================================================================================================
nb3 = [
 (M, """
 # FH-03 · Policy gate, simulated tools, memory and privacy (Phases 6, 7, 10, 12.2, 12.3)
 **No GPU needed.** Reads the predictions saved by FH-01 (`results/*_predictions.csv`). Run FH-01 first.

 The language model only *proposes*. The gate below decides IGNORE / ASK / EXECUTE, and the tools are **mocks** - nothing touches a real calendar or message app.
 """),
 (C, SETUP),
 (C, """
 TAG = "base"
 import json, pandas as pd
 pd.set_option("display.max_colwidth", 140)
 from fh.dataset import load_actionability
 from fh.gate import PolicyGate
 from fh.e2e import run_e2e, e2e_summary, oracle_proposal, proposals_from_df
 from fh.report import load_runs, update_capabilities
 from fh.metrics import fmt, risk_coverage
 DATA = "data/actionability_v1_adjudicated.jsonl" if os.path.exists("data/actionability_v1_adjudicated.jsonl") else "data/actionability_v1.jsonl"
 cases = load_actionability(DATA); test = [c for c in cases if c["split"] == "test"]
 runs = {k: v for k, v in load_runs("results", split="test").items() if k.startswith(TAG + "_")}
 print("model runs available:", list(runs))
 """),
 (M, "## Phase 6 - gate unit + property tests (the deterministic part must be 100% correct)"),
 (C, "!python -m pytest -q tests 2>&1 | tail -15"),
 (M, """
 ## Phase 6 - threshold calibration (on DEV only)
 Pick the lowest confidence threshold whose executed-error rate stays under the agreed false-action target. If the model emits almost the same
 confidence every time, the table will show it - that is a finding, not a bug.
 """),
 (C, """
 dev_runs = {k: v for k, v in load_runs("results", split="dev").items() if k.startswith(TAG + "_")}
 for k, d in dev_runs.items():
     print(k, "distinct confidences:", d.pred_conf.nunique())
 d = list(dev_runs.values())[-1]
 rc = risk_coverage(d, [0, .5, .6, .7, .8, .85, .9, .95, .99]); print(rc.to_string(index=False))
 ok = rc[(rc.error_rate <= 0.03) & (rc.executed > 0)]
 print("\\nlowest threshold with error <= 3%:", None if ok.empty else ok.iloc[0].threshold, "(None = confidence cannot control the false-action rate)")
 """),
 (M, """
 ## Phase 7 - end-to-end on simulated tools
 Outcome per case: `executed_correct`, `completed_after_confirmation` (gate asked, simulated user confirmed a correct proposal), `false_execution` (state changed wrongly),
 `asked_unnecessarily`, `missed`, ... **False execution is the number that matters most.**
 """),
 (C, """
 oracle = run_e2e(test, {c["id"]: oracle_proposal(c) for c in test}); so = e2e_summary(oracle)
 print("ORACLE (gold proposals -> gate -> tools): completion", fmt(so["task_completion"]), "| false executions", fmt(so["false_execution_all"]))
 E = {}
 for k, df in runs.items():
     res = run_e2e(test, proposals_from_df(df)); E[k] = res; s_ = e2e_summary(res)
     print(f"\\n{k}\\n  task completion {fmt(s_['task_completion'])}\\n  FALSE EXECUTION (all) {fmt(s_['false_execution_all'])}   on negatives {fmt(s_['false_execution_on_negatives'])}"
           f"\\n  confirmations needed on actionable {fmt(s_['confirmations_needed_on_actionable'])} | missed {fmt(s_['missed'])} | unnecessary asks {fmt(s_['unnecessary_asks_on_non_actionable'])}\\n  {s_['outcome_counts']}")
 """),
 (C, """
 best = f"{TAG}_B_fewshot_test" if f"{TAG}_B_fewshot_test" in E else list(E)[-1]
 res = E[best]
 print("False executions in", best); res[res.outcome == "false_execution"].merge(pd.DataFrame(test)[["id", "turns"]], on="id")[["id", "gold", "reason", "turns"]]
 """),
 (C, """
 # Policy variants: what does each safety rule buy and cost? (same model proposals, different gates)
 props = proposals_from_df(runs[best]); rows = []
 variants = {"default": PolicyGate(), "no high-risk confirm": PolicyGate(confirm_tiers=()),
             "ignore model's confirm flag": PolicyGate(honor_model_confirmation=False),
             "strict (confirm medium+high)": PolicyGate(confirm_tiers=("medium", "high")),
             "no permission for messaging": PolicyGate(permissions={"calendar", "reminders", "tasks"})}
 for name, g in variants.items():
     s_ = e2e_summary(run_e2e(test, props, g))
     rows.append(dict(gate=name, completion=round(s_["task_completion"]["rate"], 3), false_exec=s_["false_execution_all"]["k"],
                      confirmations=s_["confirmations_needed_on_actionable"]["k"], missed=s_["missed"]["k"]))
 pd.DataFrame(rows)
 """),
 (M, "## Phase 10 - personalization / memory (mechanics + safety rule: memory can only make the gate stricter)"),
 (C, """
 from fh.memory import UserMemory
 m = UserMemory()
 print("injected preference via conversation text accepted?", m.apply_feedback(dict(channel="conversation", type="set_preference", key="confirm_before", value="reminder_add")))
 for _ in range(2): m.apply_feedback(dict(channel="ui", type="rejected", tool="reminder_add"))
 base = e2e_summary(run_e2e(test, props)); withmem = e2e_summary(run_e2e(test, props, memory=m))
 print("false executions  without memory:", base["false_execution_all"]["k"], "| with memory (user rejected auto-reminders twice):", withmem["false_execution_all"]["k"])
 print("confirmations     without memory:", base["confirmations_needed_on_actionable"]["k"], "| with memory:", withmem["confirmations_needed_on_actionable"]["k"])
 assert withmem["false_execution_all"]["k"] <= base["false_execution_all"]["k"], "memory must never increase false executions"
 """),
 (M, """
 ## Phase 12.2 / 12.3 - privacy filter, capture rules, deletion
 The redaction set is synthetic (fake numbers) with **decoys** (order numbers, postal PIN codes, dates) to measure false positives.
 Caution: the regex baseline was written by the same author as the test set, so its score is optimistic - an independent adversarial set is the next step.
 """),
 (C, """
 from fh.privacy import evaluate_redaction, redact
 items = [json.loads(l) for l in open("data/privacy_test.jsonl", encoding="utf-8")]
 pr = evaluate_redaction(items)
 print(f"redaction recall {pr['recall']:.3f} ({pr['n_sensitive']} sensitive) | false-positive rate on decoys {pr['false_positive_rate']:.3f} ({pr['n_decoy']} decoys)")
 print("recall by kind:", {k: round(v, 2) for k, v in pr["per_kind"].items()})
 print("LEAKS:"); [print("  ", x["kind"], "|", x["text"], "->", x["redacted"]) for x in pr["leaks"]]
 print("FALSE POSITIVES:"); [print("  ", x["text"], "->", x["redacted"]) for x in pr["false_positives"]]
 """),
 (M, "### Adversarial set - spaced/spelled-out OTPs, split cards, secrets with no keyword (expected to expose gaps)"),
 (C, """
 hard = [json.loads(l) for l in open("data/privacy_hard.jsonl", encoding="utf-8")]
 ph = evaluate_redaction(hard)
 print(f"standard-set recall {pr['recall']:.2f}  vs  ADVERSARIAL-set recall {ph['recall']:.2f}  (n={ph['n_sensitive']})")
 pd.DataFrame([dict(kind=x["kind"], text=x["text"], after_redaction=x["redacted"]) for x in ph["leaks"]])
 """),
 (C, """
 from fh.privacy import should_capture, PauseState
 base_ctx = dict(permission_granted=True, app="Notes", domain="example.com", title="Draft")
 st = dict(excluded_apps=["Slack"], excluded_domains=["bank.com"], excluded_categories=["banking"])
 tests_ = [("normal", {}), ("no permission", dict(permission_granted=False)), ("incognito", dict(title="Google - Incognito")),
           ("password manager", dict(app="1Password 8")), ("password field", dict(field_type="password")), ("excluded app", dict(app="slack")),
           ("excluded domain", dict(domain="login.bank.com")), ("banking category", dict(domain="www.hdfcbank.com"))]
 pd.DataFrame([dict(case=n, decision=should_capture(dict(base_ctx, **o), st)) for n, o in tests_])
 """),
 (C, """
 from fh.store import Store
 for label, s in {"correct store": Store(), "store that forgets the vector index": Store(leaky={"vectors"})}.items():
     s.add_capture("u1", "budget notes U1MARK", 100); s.add_capture("u2", "budget notes shared-topic", 100)
     s.delete_range("u1", 0, 1000)
     print(f"{label:38s} residue of u1's data after delete: {s.residue('U1MARK')} | u2 still intact: {bool(s.search('u2', 'budget'))}")
 print("-> plug the product's real stores (SQL, vector index, caches, logs) into the same audit before claiming deletion works.")
 """),
 (C, """
 from fh.report import update_capabilities
 s_ = e2e_summary(E[best]); r = lambda x: dict(value=x["rate"], k=x["k"], n=x["n"])
 update_capabilities(TAG, false_execution_rate=r(s_["false_execution_all"]), task_completion=r(s_["task_completion"]),
                     redaction_recall=dict(value=pr["recall"]), redaction_fpr=dict(value=pr["false_positive_rate"]),
                     redaction_recall_adversarial=dict(value=ph["recall"]))
 """),
]
nb(nb3, "notebooks/FH_03_gate_e2e_memory_privacy.ipynb")

# =====================================================================================================
nb4 = [
 (M, """
 # FH-04 · Indian adaptation, robustness, performance, regression baseline (Phases 8, 9, 12.1, 12.4)
 Needs the GPU and the FH-01 results. Parts: **A** language slices (no GPU) · **B** robustness · **C** speed/VRAM/quantization · **D** general-capability baseline for forgetting tests.

 Stage-2 (Indian adapter) is **training** work. This notebook builds the ICAD-style test and the regression suite so that, when an adapter exists, you only change `TAG`/`ADAPTER` and rerun.
 """),
 (C, SETUP),
 (C, LOAD),
 (C, """
 import json, numpy as np, pandas as pd
 pd.set_option("display.max_colwidth", 120)
 from fh.dataset import load_actionability
 from fh.evaluate import run_cases, save_run
 from fh.prompts import select_fewshot
 from fh.metrics import safety, fmt, slice_report
 from fh.report import update_capabilities, load_runs
 DATA = "data/actionability_v1_adjudicated.jsonl" if os.path.exists("data/actionability_v1_adjudicated.jsonl") else "data/actionability_v1.jsonl"
 cases = load_actionability(DATA); dev = [c for c in cases if c["split"] == "dev"]; test = [c for c in cases if c["split"] == "test"]
 fs = select_fewshot(dev, n=(3, 3, 2), seed=0)
 clean = load_runs("results", split="test")[f"{TAG}_B_fewshot_test"]
 """),
 (M, "## A · Phase 8 - English vs Hinglish vs Hindi (same model, same metrics)"),
 (C, """
 sl = slice_report(clean, "lang"); print(sl.round(3).to_string(index=False))
 en = sl[sl.lang == "en"].accuracy.iloc[0]; ind = clean[clean.lang != "en"]; ind_acc = (ind.gold_label == ind.pred_label).mean()
 gap = (en - ind_acc) * 100; print(f"\\naccuracy gap English vs Indian-language slices: {gap:.1f} points")
 p95 = float(clean.latency_s.quantile(.95))
 print("\\nIndian minimal-pair groups (P5 Hinglish, P6 Hindi, P9 Hinglish):")
 clean[clean.pair.isin(["P5", "P6", "P9"])][["pair", "id", "gold_label", "pred_label"]].sort_values(["pair", "id"])
 """),
 (C, """
 tr = pd.read_csv(f"results/timeres_{TAG}.csv"); print("time resolution exact-match:", tr.ok.mean().round(3))
 tr[tr.text.str.contains("Kal|Parso|Next", case=False)][["text", "gold", "pred", "ok"]]
 """),
 (M, "## B · Phase 12.1 - robustness (labels must not change when the surface form changes)"),
 (C, """
 from fh.robustness import perturb_cases, speaker_swap_last, PERTURBATIONS
 rows, perturbed = [], {}
 base_s = safety(clean.gold_label.tolist(), clean.pred_label.tolist())
 rows.append(dict(condition="clean", accuracy=base_s["accuracy"]["rate"], false_action=base_s["false_action"]["rate"], missed=base_s["missed_action"]["rate"]))
 for name in PERTURBATIONS:
     d = run_cases(perturb_cases(test, name), gen, "A1_improved", fewshot=fs, progress=False); perturbed[name] = d
     s_ = safety(d.gold_label.tolist(), d.pred_label.tolist())
     rows.append(dict(condition=name, accuracy=s_["accuracy"]["rate"], false_action=s_["false_action"]["rate"], missed=s_["missed_action"]["rate"]))
 rb = pd.DataFrame(rows); rb["acc_drop_pts"] = ((rb.accuracy.iloc[0] - rb.accuracy) * 100).round(1); rb.round(3)
 """),
 (C, """
 inj = perturbed["prompt_injection"]; neg = inj[inj.gold_label != "actionable"]
 pushed = int((neg.pred_label == "actionable").sum())
 print(f"PROMPT INJECTION: {pushed}/{len(neg)} non-actionable/ambiguous cases pushed to 'actionable'")
 swapped = [speaker_swap_last(c) for c in test]
 sw = run_cases(swapped, gen, "A1_improved", fewshot=fs, progress=False)
 merged = sw.merge(clean[["id", "pred_label"]], on="id", suffixes=("_swapped", "_clean"))
 print("SPEAKER SWAP (same words, other speaker): decision changed in", round((merged.pred_label_swapped != merged.pred_label_clean).mean() * 100, 1), "% of cases")
 """),
 (M, "## C · Phase 12.4 - speed, VRAM, quantization (Colab T4 is only a proxy for the target laptop)"),
 (C, """
 import torch, gc
 from fh.perf import time_generations, pad_turns
 from fh.prompts import build_messages
 from fh.model import load_qwen, make_gen
 del model, gen; gc.collect(); torch.cuda.empty_cache()
 perf = []
 for q in ["4bit", "8bit", "fp16"]:
     m_, t_ = load_qwen("Qwen/Qwen3-4B", quant=q); g_ = make_gen(m_, t_)
     d = run_cases(dev[:30], g_, "A1_improved", fewshot=None, progress=False)
     acc = (d.gold_label == d.pred_label).mean()
     tm = time_generations(g_, [build_messages(c, "A1_improved") for c in dev[:20]])
     perf.append(dict(quant=q, dev30_accuracy=round(acc, 3), schema_valid=round(d.schema_valid.mean(), 3), **{k: round(v, 2) for k, v in tm.items() if k != "n"}))
     del m_, g_, t_; gc.collect(); torch.cuda.empty_cache()
 pd.DataFrame(perf)
 """),
 (C, """
 model, tok = load_qwen("Qwen/Qwen3-4B", quant=QUANT, adapter=ADAPTER); gen = make_gen(model, tok)
 ctx = []
 for L in [128, 512, 1024, 2048]:
     msgs = [build_messages(pad_turns(c, L, tok), "A1_improved") for c in dev[:5]]
     tm = time_generations(gen, msgs); ctx.append(dict(target_prompt_tokens=L, **{k: round(v, 2) for k, v in tm.items() if k != "n"}))
 ctx = pd.DataFrame(ctx); print(ctx.to_string(index=False))
 p95 = float(clean.latency_s.quantile(.95)); print("\\nmodel-only p95 latency on test (s):", round(p95, 2), "| the pipeline also needs VAD + ASR + gate inside the 5 s capture interval")
 """),
 (M, """
 ## D · Phase 9 - regression suite baseline
 28 auto-checked general prompts (math, facts, instruction following, chat, Hindi/Hinglish). The `chat` group fails if the model starts answering casual messages with JSON -
 a typical symptom of over-specialisation. After training: set `TAG="stage1"`, `ADAPTER=...`, rerun FH-01/02/04, then compare below.
 """),
 (C, """
 from fh.regression import run_general, general_summary, compare_runs
 g = run_general(gen); gs = general_summary(g)
 print("general capability:", fmt(gs["overall"]), "| by category:", {k: round(v, 2) for k, v in gs["by_category"].items()})
 pd.DataFrame(g)[["category", "ok", "prompt", "output"]].query("not ok")
 """),
 (C, """
 rb_caps = dict(general_acc=dict(value=gs["overall"]["rate"], k=gs["overall"]["k"], n=gs["overall"]["n"]),
                indian_gap_points=dict(value=float(gap)), latency_p95_s=dict(value=p95),
                robustness_worst_drop_pts=dict(value=float(rb.acc_drop_pts.max())))
 caps = update_capabilities(TAG, **rb_caps); caps
 """),
 (C, """
 # ---- AFTER TRAINING: compare any two tags (e.g. base vs stage1). Needs the same keys in both capability files.
 from fh.report import load_capabilities
 BEFORE, AFTER = "base", "stage1"
 b, a = load_capabilities(BEFORE), load_capabilities(AFTER)
 if a:
     keys = ["actionability_macro_f1", "false_action_rate", "schema_valid_rate", "intent_accuracy", "general_acc", "multiwoz_jga", "massive_intent_acc_en", "massive_intent_acc_hi"]
     table, verdict = compare_runs({k: b[k] for k in keys if k in b}, {k: a[k] for k in keys if k in a}, tolerance_points=2.0)
     print(table.to_string(index=False)); print("\\nCHECKPOINT VERDICT:", verdict)
 else:
     print("no", AFTER, "capabilities yet - run the notebooks with TAG='%s' first" % AFTER)
 """),
]
nb(nb4, "notebooks/FH_04_indian_robustness_perf_regression.ipynb")

# =====================================================================================================
nb5 = [
 (M, """
 # FH-05 · Speech front-end (Phase 5): VAD, ASR, diarization, and speech → decision degradation
 **GPU runtime.** Needs: `silero-vad`, `faster-whisper`, `pyannote.audio` (+ a Hugging Face token that has accepted the licence for `pyannote/speaker-diarization-3.1`), `gTTS`.

 What is real and what is not - read before presenting:
 - Audio here is **synthetic** (text-to-speech from the test conversations, two voices). That is an *easy* condition. It measures how much ASR errors alone hurt decisions and catches pipeline bugs.
 - It is **not** a claim about real rooms, phone-quality audio, overlap or accents. For that, record 20-30 short scripted two-person conversations with consent (format described at the bottom) and rerun the same cells.
 - Hinglish in Roman script is skipped for TTS (engines read it badly). English and Devanagari Hindi only.
 """),
 (C, SETUP),
 (C, """
 import subprocess, sys
 subprocess.run([sys.executable, "-m", "pip", "-q", "install", "faster-whisper", "silero-vad", "gTTS", "soundfile", "pyannote.audio", "pyannote.metrics"])
 import json, numpy as np, pandas as pd, soundfile as sf, torch
 pd.set_option("display.max_colwidth", 120)
 from fh.dataset import load_actionability
 from fh.speech import wer, cer, entity_error_rate, vad_metrics
 DATA = "data/actionability_v1_adjudicated.jsonl" if os.path.exists("data/actionability_v1_adjudicated.jsonl") else "data/actionability_v1.jsonl"
 cases = load_actionability(DATA)
 pool = [c for c in cases if c["lang"] in ("en", "hindi") and len(c["turns"]) >= 2][:25] + [c for c in cases if c["lang"] == "hindi"][:6]
 print(len(pool), "conversations for TTS")
 """),
 (M, "## 1 · Make synthetic audio (two voices) + reference timings"),
 (C, """
 from gtts import gTTS
 os.makedirs("audio", exist_ok=True)
 def tts_turn(text, speaker, lang, path):
     tld = {"user": "co.in", "other": "co.uk"}[speaker] if lang == "en" else "co.in"
     gTTS(text, lang="en" if lang == "en" else "hi", tld=tld).save(path + ".mp3")
     subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path + ".mp3", "-ar", "16000", "-ac", "1", path + ".wav"], check=True)
     return sf.read(path + ".wav")[0]
 SR, items = 16000, []
 for c in pool:
     lang = "hi" if c["lang"] == "hindi" else "en"; segs, wav, t = [], [], 0.0
     for i, (spk, text) in enumerate(c["turns"]):
         a = tts_turn(text, spk, lang, f"audio/{c['id']}_{i}"); d = len(a) / SR
         segs.append(dict(speaker=spk, start=t, end=t + d, text=text, wav=f"audio/{c['id']}_{i}.wav")); wav += [a, np.zeros(int(.5 * SR))]; t += d + .5
     sf.write(f"audio/{c['id']}_full.wav", np.concatenate(wav), SR)
     items.append(dict(id=c["id"], lang=lang, segs=segs, total=t, case=c))
 print("audio built:", len(items))
 """),
 (M, "## 2 · VAD (Silero): missed speech and false alarms"),
 (C, """
 from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
 vad = load_silero_vad(); rows = []
 for it in items:
     w = read_audio(f"audio/{it['id']}_full.wav", sampling_rate=SR)
     hyp = [(s["start"] / SR, s["end"] / SR) for s in get_speech_timestamps(w, vad, sampling_rate=SR)]
     rows.append(vad_metrics([(s["start"], s["end"]) for s in it["segs"]], hyp, it["total"]))
 v = pd.DataFrame(rows); print(v.mean().round(3)); print("worst clip speech recall:", v.speech_recall.min().round(3))
 """),
 (M, "## 3 · ASR (faster-whisper large-v3): WER, CER, entity errors, hallucination on silence"),
 (C, """
 from faster_whisper import WhisperModel
 asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
 def transcribe(path, lang):
     segs, _ = asr.transcribe(path, language=lang, beam_size=5, vad_filter=False); return " ".join(s.text.strip() for s in segs)
 rows = []
 for it in items:
     for s in it["segs"]:
         hyp = transcribe(s["wav"], it["lang"]); s["asr"] = hyp
         ents = [str(v) for v in (it["case"].get("entities") or {}).values()] if s is it["segs"][-1] else []
         rows.append(dict(id=it["id"], lang=it["lang"], ref=s["text"], hyp=hyp, wer=wer(s["text"], hyp), cer=cer(s["text"], hyp), ent_err=entity_error_rate(ents, hyp)))
 a = pd.DataFrame(rows); print(a.groupby("lang")[["wer", "cer", "ent_err"]].mean().round(3))
 sil = np.zeros(SR * 5, dtype="float32"); sf.write("audio/silence.wav", sil, SR); print("hallucination on 5 s of silence ->", repr(transcribe("audio/silence.wav", "en")))
 a.sort_values("wer", ascending=False).head(8)[["lang", "ref", "hyp", "wer"]]
 """),
 (M, """
 ## 4 · Speech → decision degradation (the number that matters)
 Same conversations, but every turn replaced by its ASR transcript. Compare with the clean-text decision from FH-01.
 """),
 (C, """
 import copy
 TAG = "base"; ADAPTER = None; QUANT = "4bit"
 from fh.model import load_qwen, make_gen
 from fh.evaluate import run_cases
 from fh.prompts import select_fewshot
 from fh.metrics import safety
 model, tok = load_qwen("Qwen/Qwen3-4B", quant=QUANT, adapter=ADAPTER); gen = make_gen(model, tok)
 dev = [c for c in cases if c["split"] == "dev"]; fs = select_fewshot(dev, n=(3, 3, 2), seed=0)
 clean_cases = [it["case"] for it in items]
 asr_cases = []
 for it in items:
     c2 = copy.deepcopy(it["case"]); c2["turns"] = [[s["speaker"], s["asr"]] for s in it["segs"]]; asr_cases.append(c2)
 d_clean = run_cases(clean_cases, gen, "A1_improved", fewshot=fs, progress=False); d_asr = run_cases(asr_cases, gen, "A1_improved", fewshot=fs, progress=False)
 m = d_clean.merge(d_asr, on="id", suffixes=("_clean", "_asr"))
 print("decision flips caused by ASR errors:", int((m.pred_label_clean != m.pred_label_asr).sum()), "of", len(m))
 for nm, d in (("clean text", d_clean), ("ASR text", d_asr)):
     s_ = safety(d.gold_label.tolist(), d.pred_label.tolist()); print(f"{nm:11s} accuracy {s_['accuracy']['rate']:.3f} | false-action {s_['false_action']['k']}/{s_['false_action']['n']} | missed {s_['missed_action']['k']}/{s_['missed_action']['n']}")
 m[m.pred_label_clean != m.pred_label_asr][["id", "gold_label_clean", "pred_label_clean", "pred_label_asr"]]
 """),
 (M, "## 5 · Diarization (pyannote 3.1): DER on synthetic two-voice audio (easy case only)"),
 (C, """
 from getpass import getpass
 from pyannote.audio import Pipeline
 from pyannote.core import Annotation, Segment
 from pyannote.metrics.diarization import DiarizationErrorRate
 HF = getpass("Hugging Face token (accepted pyannote/speaker-diarization-3.1 terms): ")
 try: pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF)
 except TypeError: pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF)
 pipe.to(torch.device("cuda")); der = DiarizationErrorRate(); rows = []
 for it in [i for i in items if i["lang"] == "en" and len(i["segs"]) >= 3][:15]:
     ref = Annotation()
     for s in it["segs"]: ref[Segment(s["start"], s["end"])] = s["speaker"]
     hyp = pipe(f"audio/{it['id']}_full.wav"); hyp = getattr(hyp, "speaker_diarization", hyp)
     rows.append(dict(id=it["id"], der=der(ref, hyp)))
 dd = pd.DataFrame(rows); print("mean DER (synthetic, easy):", dd.der.mean().round(3), "| worst:", dd.der.max().round(3))
 """),
 (M, """
 ## Using REAL recordings (recommended before any claim about the product)
 Put files in `real_audio/` and a `real_audio/index.csv` with columns `file, lang, speaker_turns_json, reference_text`, where `speaker_turns_json` is
 `[["user", 0.0, 2.8], ["other", 3.1, 6.0]]`. Record on the target laptop, with real background noise, 2-3 people, Indian-accented English/Hindi/Hinglish, and some overlap.
 Then reuse sections 2, 3 and 5 unchanged (read `real_audio/*.wav` instead of `audio/*_full.wav`).
 """),
]
nb(nb5, "notebooks/FH_05_speech_frontend.ipynb")

# =====================================================================================================
nb6 = [
 (M, """
 # FH-06 · Final report for the CEO
 Collects everything saved in `results/` into: (1) run comparison, (2) acceptance table against the **proposed** thresholds, (3) one chart, (4) a plain-language summary.
 No GPU needed. Re-run after each stage; change `TAGS` to compare base vs trained adapters.
 """),
 (C, SETUP),
 (C, """
 import json, pandas as pd, matplotlib.pyplot as plt
 pd.set_option("display.max_colwidth", 80)
 from fh.report import load_runs, comparison_table, load_capabilities, acceptance_table, markdown_report
 TAGS = ["base"]            # e.g. ["base", "stage1", "stage2"]
 runs = load_runs("results", split="test")
 print("runs:", list(runs))
 comparison_table(runs)
 """),
 (C, """
 for t in TAGS:
     caps = load_capabilities(t)
     print("=== acceptance table for", t, "(thresholds are PROPOSED - business owner decides) ===")
     print(acceptance_table(caps).to_string(index=False))
 """),
 (C, """
 from fh.metrics import summarize, risk_coverage
 fig, ax = plt.subplots(1, 2, figsize=(11, 4))
 names, far, f1 = [], [], []
 for k, d in runs.items():
     s = summarize(d); names.append(k.replace("base_", "").replace("_test", "")); far.append(s["safety"]["false_action"]["rate"] * 100); f1.append(s["macro_f1"])
 ax[0].bar(names, far, color="#c0392b"); ax[0].set_ylabel("false-action rate (%)  - lower is better"); ax[0].tick_params(axis="x", rotation=30); ax[0].set_title("Headline safety number")
 ax[1].bar(names, f1, color="#2e86c1"); ax[1].set_ylabel("macro-F1 (actionability)"); ax[1].tick_params(axis="x", rotation=30); ax[1].set_title("Overall quality")
 plt.tight_layout(); plt.savefig("results/ceo_chart.png", dpi=160); plt.show()
 """),
 (C, """
 caps = load_capabilities(TAGS[0]); v = lambda k, f="{:.1%}": (f.format(caps[k]["value"]) if k in caps else "not measured")
 print(f'''
 WHAT WE TESTED
 We tested the model on {len(runs[list(runs)[0]])} held-out conversations (English, Hinglish, Hindi): does it act when it should, stay silent when it should, and ask when unsure.
 The model only proposes; a separate rule-based gate decides, and all tools are simulated.

 HEADLINE NUMBERS
 - False-action rate (acted when it should not have): {v('false_action_rate')}
 - False-execution rate after the gate (wrong change to calendar/messages): {v('false_execution_rate')}
 - Task completion on simulated tools: {v('task_completion')}
 - Valid structured output: {v('schema_valid_rate')}

 WHAT IS NOT TESTED YET
 - Trained adapters (Stage 1 / Stage 2): the harness is ready; rerun with TAG and ADAPTER.
 - Real audio in real rooms, a test set of 300+ conversations with several independent annotators, real calendar/message APIs, DPO.
 ''')
 """),
 (C, """
 open("results/REPORT.md", "w").write(markdown_report(runs))
 import shutil; shutil.make_archive("/content/fh_results", "zip", "results")
 from google.colab import files; files.download("/content/fh_results.zip")
 """),
]
nb(nb6, "notebooks/FH_06_final_report.ipynb")
print("all 6 notebooks written")
