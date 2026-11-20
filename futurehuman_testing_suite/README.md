# Futurehuman — model testing suite (Qwen3-4B action intelligence)

Implements **everything in the testing plan except training** (LoRA/QLoRA, DPO fine-tuning are separate work).
Same harness is reused after training: set `TAG`/`ADAPTER` in the notebooks and rerun.

## How to use (Colab, T4)
1. Zip this folder as `futurehuman_testing_suite.zip` (already done if you got the zip) and upload it when a notebook asks.
2. Run notebooks in order. Each saves into `results/` (download from FH-06).

| Notebook | Phases | GPU | Time (T4) |
|---|---|---|---|
| FH_01 baseline + actionability | 0, 1, 3, 4 | yes | ~35 min |
| FH_02 MultiWOZ + MASSIVE | 2 | yes | ~10 min (QUICK) / ~45 min (full) |
| FH_03 gate, simulated tools, memory, privacy | 6, 7, 10, 12.2, 12.3 | no | ~2 min |
| FH_04 Indian slices, robustness, speed/VRAM, regression baseline | 8, 9, 12.1, 12.4 | yes | ~40 min |
| FH_05 speech (VAD, ASR, diarization, speech→decision) | 5 | yes | ~30 min |
| FH_06 final report + acceptance table + CEO chart | 13 | no | ~1 min |

## Workflow rules (these are what make the results believable)
- **Dev vs test.** Tune prompts on `dev` only. Run `test` once. Splits are by scenario *group*, so paraphrase variants never straddle the split. Cases from your first notebook are forced into dev (prompts were tuned on them).
- **Blind review first.** Fill `data/review_sheet.csv` before the locked test run (see `GUIDELINE.md`). FH_01 turns it into agreement/kappa and an adjudicated dataset + test hash.
- **Every run is saved** (raw outputs, config, versions) so errors can be inspected, not just counted.
- **Colab is cloud** → synthetic/public data only; never real conversations, screen text or OTPs.

## What is covered
Baseline conditions (A0 original prompt, A1 improved, B few-shot, C thinking) · intent/slot (MASSIVE) · dialogue state (MultiWOZ 2.2) ·
actionability with confusion matrix, **false-action rate**, slices (language, scenario family), **minimal pairs**, calibration (ECE, risk-coverage) ·
JSON/schema validity, entity and intent accuracy, relative-time resolution · gate rules (unit + property tests) and threshold calibration ·
end-to-end on simulated tools with outcome categories · memory mechanics (rejection→confirmation, drift, deletion, injection) ·
privacy redaction (standard + adversarial set), capture/exclusion/pause rules, deletion-propagation audit ·
English vs Hinglish vs Hindi · robustness (typos, disfluency, ASR-style, distractors, prompt injection, speaker swap) ·
speed/VRAM/quantization, context-length scaling · general-capability regression suite + before/after comparator with accept/reject ·
speech front-end (VAD, Whisper WER/CER/entity errors, pyannote DER, speech→decision degradation).

## Known limits (say these before someone else does)
1. **Test set is small.** v1 has 129 conversations (~83 in the locked test split). Confidence intervals are wide; the plan's target is 300–500 with several annotators.
2. **Labels were drafted, then checked by one reviewer.** Kappa here = drafted labels vs your blind labels. Better than nothing, not independent annotation.
3. **No code ran on a GPU while building this.** Library code is unit-tested (27 tests) and the notebooks were dry-run end to end with a mock model; real-model behaviour (and Colab/library version quirks, especially in FH_02 dataset loading and FH_05 speech libraries) will surface on first run.
4. **Single-proposal schema.** The model emits one intent per turn, so multi-assignment turns can only be partly satisfied; the e2e test treats any acceptable effect as a hit. A `actions: []` schema is a finding to raise.
5. **Entities are free-form**, mapped to typed tool arguments by an alias table. Poor mapping shows up as wrong/missing arguments - another reason to move to a typed schema.
6. **Privacy regex baseline** scores ~100% on the standard synthetic set but ~17% on the adversarial set (spaced/spelled-out OTPs, split cards, keyword-less secrets). The standard score alone would be misleading.
7. **Synthetic speech is an easy condition.** Real recordings (noise, overlap, accents) are needed before any product claim about audio.
8. MultiWOZ domains are not calendar tasks; MASSIVE is scored on its own labels, not the project ontology.
9. Acceptance thresholds in `fh/report.py` (`ACCEPTANCE`) are **proposals** for the business owner to confirm.

## Layout
`fh/` library · `data/` test sets (+ blind review sheet) · `tests/` 27 pytest tests (`python -m pytest -q tests`) · `notebooks/` FH_01…06 · `build_*.py` regenerate datasets/notebooks.
