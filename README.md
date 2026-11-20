FutureHuman — Conversational Action Intelligence
This repository contains the FutureHuman evaluation workspace: the testing suite, unified evaluation notebook, human-review file, and outputs from an evaluation run using Qwen3-4B.
Repository contents

FutureHuman/
│
├── README.md
├── .gitignore
│
├── FutureHuman_Unified_Evaluation.ipynb
├── FutureHuman_Evaluation_Run_2026-11-19.ipynb
│
├── futurehuman_testing_suite/
│   ├── fh/
│   ├── data/
│   ├── notebooks/
│   ├── tests/
│   ├── results/
│   ├── GUIDELINE.md
│   ├── README.md
│   └── build scripts
│
├── review_sheet.csv
│
├── fh_results/
│   └── Evaluation output files (CSV, JSON, PNG, Markdown, and TXT)
│
└── futurehuman_evaluation_results.zip
The executed notebook's filename may include its actual run date. Keep the clean notebook and executed notebook separate.
Main files and folders
Item	Purpose
FutureHuman_Unified_Evaluation.ipynb	Clean master notebook, without saved cell outputs. Use this as the starting point for a fresh run.
FutureHuman_Evaluation_Run_2026-11-19.ipynb	Notebook saved with cell outputs from the completed run. It preserves the run's visible results.
futurehuman_testing_suite/	Extracted testing-suite source files, datasets, helper modules, notebooks, tests, and guidelines.
review_sheet.csv	Human-review sheet used for reviewing actionability labels.
fh_results/	Extracted evaluation results from the run, including prediction CSVs, configuration JSONs, charts, report, and test hash files.
futurehuman_evaluation_results.zip	ZIP archive of the evaluation results. Keep this as the packaged copy of the results.


What the evaluation covers
The unified notebook brings together the available evaluation stages:
- FH-01 — Baseline and actionability: baseline prompts, actionability decisions, review/adjudication workflow, and related analysis.
- FH-02 — MultiWOZ and MASSIVE: dialogue-state, intent, and slot evaluation using the included datasets.
- FH-03 — Gate, end-to-end, memory and privacy: simulated action execution and checks around gating, memory, and privacy.
- FH-04 — Indian robustness, performance and regression: language/robustness checks, performance measurements, and regression evaluation.
- FH-05 — Speech front end: speech-to-text and speaker-diarization evaluation on the notebook's generated test audio.
- FH-06 — Final report: consolidates available evaluation outputs into a report.
Running the notebook
1. Open FutureHuman_Unified_Evaluation.ipynb in Google Colab.
2. Select a GPU runtime when the notebook requires it.
3. When prompted, upload futurehuman_testing_suite.zip (the original testing-suite archive, if kept alongside this repository).
4. Upload review_sheet.csv when the review workflow requests it.
5. Run the notebook cells in order and review the generated outputs.
The executed notebook is retained separately as a record of a particular run. For a new evaluation, start from the clean master notebook rather than editing the saved-run copy.
Scope and limitations
- This repository contains an evaluation workflow and its outputs; it is not, by itself, a production action-execution system.
- The notebook evaluates the base Qwen3-4B workflow and prototype components. It does not perform LoRA/QLoRA or DPO training.
- Speech results from generated test audio should not be treated as proof of performance on real-world recordings, noisy environments, overlapping speakers, or all accents.
- Privacy checks and simulated tools are evaluation components, not a production security certification.
- Review the generated reports and test conditions before making performance or readiness claims.
