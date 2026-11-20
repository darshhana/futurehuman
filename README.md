# FutureHuman — Conversational Action Intelligence

**Evaluation workspace for the FutureHuman conversational action-intelligence prototype using Qwen3-4B.**

This repository contains the evaluation code, datasets and review material, Colab notebooks, and saved outputs from an evaluation run. The work focuses on testing whether a conversational model can recognise when a user intends an action, identify the information needed for that action, and pass only appropriate proposals to a controlled execution layer.

> **Project scope:** This is an evaluation and prototype-testing repository. It is not a production assistant, and it does not contain model fine-tuning/training runs.

---

## 1. What has been done

A unified Google Colab notebook was prepared by bringing together the six evaluation modules (FH-01 to FH-06) from the testing suite. The workflow evaluates the base **Qwen3-4B** model and the prototype's supporting components.

The work completed in this repository includes:

- Organised the evaluation code into reusable Python modules under `fh/`.
- Prepared actionability, privacy, and review datasets, along with labelling guidelines.
- Combined the six evaluation stages into one notebook so the workflow can be run in sequence.
- Added a human-review step for the actionability dataset. The notebook compares the reviewer's labels with the draft labels, calculates agreement and Cohen's kappa, displays disagreements, and allows final adjudication.
- Added evaluation outputs and a saved notebook containing cell outputs from a run.
- Added a results archive so the generated outputs can be retained and shared as a bundle.

The repository includes both a clean notebook for future runs and an executed notebook that records a particular run.

## 2. Repository structure

```text
futurehuman/
│
├── README.md
├── .gitignore
│
├── FutureHuman_Unified_Evaluation.ipynb
├── FutureHuman_Evaluation_Run_<date>.ipynb
│
├── futurehuman_testing_suite.zip          # Keep the original suite ZIP for Colab upload
├── futurehuman_testing_suite/
│   ├── fh/                                # Evaluation and prototype Python modules
│   ├── data/                              # Actionability, privacy and review datasets
│   ├── notebooks/                         # Original FH-01 to FH-06 notebooks
│   ├── tests/                             # Automated tests for suite components
│   ├── results/                           # Results directory used by the suite
│   ├── GUIDELINE.md                       # Human-review labelling rules
│   ├── README.md                          # Testing-suite documentation
│   └── build_*.py                         # Dataset/notebook build scripts
│
├── review_sheet.csv                       # Completed human-review sheet
│
├── fh_results/                            # Extracted outputs from the evaluation run
│   ├── *_predictions.csv
│   ├── *_config.json
│   ├── *_massive.csv
│   ├── *_multiwoz.csv
│   ├── *_risk_calibration.png
│   ├── REPORT.md
│   ├── TEST_HASH_*.txt
│   └── other generated output files
│
└── futurehuman_evaluation_results.zip     # Compressed archive of fh_results
```

The executed notebook's filename may contain the date of the run. Replace `<date>` in the tree above with its actual filename. Keep the original testing-suite ZIP and the results ZIP separate: they serve different purposes.

## 3. What each important file means

| File or folder | What it contains / why it is here |
|---|---|
| `FutureHuman_Unified_Evaluation.ipynb` | Clean master notebook without saved cell outputs. Use this to start a fresh evaluation. |
| `FutureHuman_Evaluation_Run_<date>.ipynb` | Executed copy of the notebook with cell outputs saved. It is a record of that run, not a second implementation. |
| `futurehuman_testing_suite.zip` | Original input archive. Upload this to Colab when the notebook asks for the testing suite. |
| `futurehuman_testing_suite/` | Extracted suite. Contains the Python implementation, datasets, original notebooks, tests, and documentation. |
| `fh/` | Reusable modules for model loading, prompts, dataset handling, metrics, policy gating, simulated tools, privacy, memory, robustness, speech, and reporting. |
| `data/` | Actionability examples, privacy test cases, and the review sheet supplied with the suite. |
| `GUIDELINE.md` | Defines how to label the last turn of each conversation as `actionable`, `ambiguous`, or `non_actionable`. |
| `review_sheet.csv` | Human-review copy. The completed file is uploaded during the notebook's review step. |
| `tests/` | Automated tests for the suite's core logic and prototype components. |
| `fh_results/` | Extracted output files from the evaluation run. |
| `REPORT.md` | Generated summary report. Read this alongside the CSV and JSON files when interpreting the run. |
| `*_predictions.csv` | Per-example model predictions. Useful for inspecting individual errors, not just aggregate scores. |
| `*_config.json` | Records run settings/configuration associated with an evaluation condition. |
| `*_massive.csv` / `*_multiwoz.csv` | Outputs for the MASSIVE and MultiWOZ dataset evaluations. |
| `*_risk_calibration.png` | Visual output for confidence/risk calibration analysis. |
| `TEST_HASH_*.txt` | Hash used to identify/freeze the evaluated dataset version. |
| `futurehuman_evaluation_results.zip` | Packaged copy of the evaluation outputs. Keep it as an archive; extract it to inspect the files. |
| `.gitignore` | Lists local, generated, or sensitive files that Git should not track. |

## 4. Evaluation stages: FH-01 to FH-06

The unified notebook runs the available evaluation workflow in these stages.

### FH-01 — Baseline and actionability

Tests the model's conversational actionability decisions and structured proposals.

- Loads Qwen3-4B and records environment/model details, generation behaviour, latency, and GPU memory.
- Checks the actionability dataset, creates development and locked test splits, and records a test-set hash.
- Compares draft labels with the completed blind human review; calculates agreement and Cohen's kappa, displays disagreements, and supports adjudication.
- Compares prompt conditions: **A0** (original prompt), **A1** (improved prompt), **B** (few-shot), and optional **C** (thinking mode).
- Evaluates actionability classification, false-action behaviour, language/scenario slices, minimal pairs, and confidence/risk calibration.
- Checks structured output/schema validity, intent and entity handling, and relative-time interpretation.

**Important evaluation rule:** prompt iteration is done on the development split. The locked test split is intended for the final evaluation, not repeated prompt tuning.

### FH-02 — MultiWOZ 2.2 and MASSIVE 1.1

Evaluates related language-understanding capabilities on two public benchmarks, separately:

- **MultiWOZ 2.2:** dialogue-state tracking, including joint goal accuracy and slot-level measures.
- **MASSIVE 1.1:** intent and slot prediction. The notebook includes English (`en-US`) and Hindi (`hi-IN`) evaluation.

These benchmark results provide additional context; they are not a replacement for the project's own actionability evaluation.

### FH-03 — Policy gate, simulated tools, memory and privacy

Tests the deterministic layer that sits between a model proposal and an action.

- Checks policy-gate rules and threshold behaviour.
- Runs end-to-end scenarios against simulated tools, including correct execution, confirmation, asking for clarification, and false execution outcomes.
- Tests memory mechanics and safety behaviour, including whether conversational content can improperly change stored preferences.
- Evaluates privacy redaction on standard and adversarial synthetic test cases.
- Checks capture/exclusion/pause and deletion-related behaviour in the prototype store.

The tools are simulated. This stage does not connect to or execute real external services.

### FH-04 — Indian-language robustness, performance and regression

Checks behaviour beyond the basic English examples.

- Compares English, Hinglish, and Hindi actionability slices.
- Applies robustness variations such as typos, disfluencies, ASR-style text, distractors, prompt injection, and speaker changes.
- Measures generation speed, memory/VRAM use, quantization behaviour, and context-length effects.
- Runs a general-capability regression set to check whether action-oriented prompting harms ordinary responses.
- Includes a before/after comparison path for evaluating a future fine-tuned checkpoint.

### FH-05 — Speech front end

Evaluates the speech pipeline using generated test audio.

- **VAD:** voice activity detection and speech-region detection.
- **ASR:** Whisper transcription, including word error rate (WER), character error rate (CER), entity errors, and silence-hallucination checks.
- **Diarization:** speaker-turn evaluation using pyannote and diarization error rate (DER).
- **Speech-to-decision:** compares actionability decisions from clean text with decisions made using ASR transcripts.

The included generated audio is an easy, synthetic test condition. It does not establish performance on real recordings, noisy environments, overlapping speech, phone-quality audio, or all accents. The diarization step requires a Hugging Face token and acceptance of the relevant gated-model terms.

### FH-06 — Final report

Collects saved evaluation runs and creates:

- A comparison of available runs.
- An acceptance table against proposed thresholds.
- A chart for the report.
- A plain-language summary in `REPORT.md`.

The thresholds are proposals in the testing code, not confirmed product acceptance criteria. They should be reviewed and approved by the project/business owner.

## 5. Understanding the results

The `fh_results/` folder contains the generated evidence from the run. The main files answer different questions:

| Result type | What to look for |
|---|---|
| Prediction CSVs | Which examples were classified correctly or incorrectly? Where did false actions occur? |
| Configuration JSONs | Which prompt/model/runtime settings produced the predictions? |
| MASSIVE and MultiWOZ CSVs | How did the model perform on intent/slot and dialogue-state tasks? |
| Risk/calibration chart and metrics | How does confidence relate to error risk, and what happens when the system handles fewer uncertain cases? |
| Privacy outputs | How well did the redaction checks work on the included standard and adversarial cases? |
| Speech outputs | What were the VAD, transcription, diarization, and speech-to-decision results on the generated audio? |
| `REPORT.md` | What summary and acceptance-table results were generated for this run? |
| Test hash | Which frozen actionability dataset version was evaluated? |

**Use the actual values in `fh_results/REPORT.md` and the corresponding CSV/JSON files when reporting numerical results.** This README intentionally does not state accuracy, F1, false-action rate, WER, DER, or task-completion numbers because those values must be taken from the saved run outputs, not assumed from the test design.

### Known limitations of the current evaluation setup

- The actionability dataset is a small v1 set: 129 conversations, with approximately 83 in the locked test split. Results therefore have uncertainty; a larger dataset and multiple independent annotators are needed for stronger claims.
- The draft actionability labels were checked through a single blind reviewer. Agreement with one reviewer is useful quality control, but it is not multi-annotator validation.
- The action proposal schema currently represents one proposal per turn, so turns containing multiple assignments may not be fully represented.
- Entity values are free-form and mapped to typed tool arguments through an alias table; this can lead to missing or incorrect arguments.
- The privacy regex baseline has a known gap on adversarial patterns such as spaced/spelled-out OTPs, split card numbers, and secrets without keywords. Standard-set performance alone should not be treated as sufficient.
- Synthetic speech is an easy condition. Real recordings are needed before making product claims about audio.
- MultiWOZ domains are not calendar-task data, and MASSIVE is evaluated using its own labels rather than the FutureHuman action ontology.
- Acceptance thresholds are proposed and require business-owner confirmation.

## 6. How to run the evaluation in Google Colab

### Before starting

Keep these files available locally:

- `FutureHuman_Unified_Evaluation.ipynb` — clean master notebook.
- `futurehuman_testing_suite.zip` — original testing-suite ZIP. **This is the ZIP the setup cell asks for.**
- `review_sheet.csv` — completed human-review file.

Do not upload `futurehuman_evaluation_results.zip` when the notebook asks for the testing suite. That ZIP contains outputs, not the code and data needed to initialise the suite.

### Run steps

1. Open `FutureHuman_Unified_Evaluation.ipynb` in Google Colab.
2. Select a GPU runtime (the notebook was designed with a Colab T4 workflow in mind).
3. Run the notebook from the top, in order. Do not skip around between sections.
4. In the setup cell, upload `futurehuman_testing_suite.zip` when prompted. The notebook extracts it and installs the required packages.
5. Keep the experiment settings as configured for the base-model run unless you intentionally evaluate another checkpoint.
6. At the human-review step, upload the completed `review_sheet.csv`.
7. Inspect the disagreement table. Add final adjudication decisions in the `FINAL` dictionary in the next cell. If there are no changes to make, leave it empty.
8. Continue through FH-01 to FH-06. Some stages require a GPU; FH-03 and FH-06 are marked as not requiring one.
9. At the end, download the generated results ZIP. Extract it locally into `fh_results/` to inspect the report and individual outputs.
10. Keep the executed notebook as a dated run record. For a new evaluation, start again from the clean master notebook.

### Evaluating a future fine-tuned model

The notebook has `TAG` and `ADAPTER` settings for evaluating another checkpoint with the same test workflow. The testing suite does **not** train the adapter: LoRA/QLoRA and DPO training are separate work and are outside the current repository's implementation.

## 7. Test suite

The testing-suite documentation reports **27 automated tests** across the core, gate/tools, privacy, robustness, and speech-related components. These are software tests for the suite; they are not the same as model benchmark scores or proof that every Colab stage has run successfully on a real model.

To run them in an environment with the suite dependencies installed:

```bash
cd futurehuman_testing_suite
python -m pytest -q tests
```

## 8. Data, privacy and publishing

- Use only public or synthetic data in Colab. Do not upload real user conversations, private messages, personal recordings, OTPs, or other sensitive information.
- Before making the repository public, inspect `review_sheet.csv`, all files in `fh_results/`, the executed notebook outputs, and both ZIP archives for sensitive or restricted material.
- Do not commit API keys, Hugging Face tokens, passwords, credentials, model weights, or local model caches.
- Check the licences and sharing conditions for all external datasets and gated speech models.
- The presence of privacy tests does not mean the prototype is production-secure.

---

**In short:** this repository provides a repeatable evaluation workflow for the base Qwen3-4B conversational action-intelligence prototype, the supporting test code and review process, and the saved evidence from a particular evaluation run. It is intended to make model behaviour, action-safety checks, limitations, and future improvements inspectable.
