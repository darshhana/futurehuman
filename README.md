# FutureHuman — Evaluation Suite

This repository contains the testing suite and a unified Google Colab notebook for evaluating FutureHuman's conversational action-intelligence workflow.

The current setup uses **Qwen3-4B** as the base model and tests how well it understands conversations, identifies possible actions, and handles them safely.

## What is included

- `futurehuman_testing_suite/` — Python modules, test datasets, and supporting files used by the evaluation notebook.
- `FutureHuman_Unified_Evaluation_with_Review.csv.ipynb` — the main notebook that brings the available evaluation steps together.
- `GUIDELINE.md` — instructions for manually reviewing and labeling conversations.
- `review_sheet.csv` — sheet used for independent review of actionability labels.

## What the notebook does

The notebook is divided into six parts:

| Part | What is tested | Why it is needed |
|---|---|---|
| **FH-01: Baseline and actionability** | Loads Qwen3-4B, checks basic model behaviour, evaluates whether conversations are actionable, ambiguous, or non-actionable, and compares prompt conditions. | To establish how the base model performs before further development. |
| **FH-02: MultiWOZ and MASSIVE** | Tests intent recognition, slot extraction, and dialogue-state understanding using public datasets. | To check whether the model understands the user's intent and relevant details across conversations. |
| **FH-03: Gate, tools, memory and privacy** | Tests the policy gate, simulated tool execution, confirmation requirements, memory behaviour, and privacy-related checks. | To check whether proposed actions are handled safely instead of being executed blindly. |
| **FH-04: Indian-language robustness and performance** | Compares English, Hindi and Hinglish performance; tests variations in messages, prompt injection, speaker changes, speed, memory usage, and general capabilities. | To identify weaknesses under different language and input conditions, and measure performance. |
| **FH-05: Speech front-end** | Tests voice activity detection, speech-to-text, speaker diarization, and the effect of speech errors on action decisions. | To understand how errors in the audio pipeline can affect the final decision. |
| **FH-06: Final report** | Collects saved results, compares runs, checks proposed acceptance thresholds, and generates a summary. | To make evaluation results easier to review and compare. |

## How the evaluation works

The notebook follows this general flow:

1. Load the testing suite and Qwen3-4B.
2. Run basic checks on the model.
3. Validate and review the actionability dataset.
4. Compare different prompt conditions on the development data.
5. Evaluate the selected conditions on the locked test data.
6. Run the remaining dataset, safety, language, performance, and speech tests.
7. Collect the results into a final report.

The actionability dataset uses three labels:

- `actionable` — the conversation contains a clear action.
- `ambiguous` — an action may be intended, but clarification is needed.
- `non_actionable` — there is no action for the assistant to take.

## Setup and execution

The notebook is designed to run in Google Colab with a GPU runtime (T4 recommended).

1. Open `FutureHuman_Unified_Evaluation_with_Review.csv.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Upload `futurehuman_testing_suite.zip` when prompted.
4. Run the notebook cells in order.
5. Upload the completed `review_sheet.csv` when the human-review section asks for it.
6. Review disagreements and enter final decisions where required.
7. Continue through the evaluation sections.
8. Check the generated files in `futurehuman_testing_suite/results/`.

Some sections require additional model downloads or access permissions. In particular, speech diarization requires a Hugging Face token and acceptance of the relevant pyannote model terms.

## Important notes

- This repository currently contains an **evaluation workflow**, not a completed model-training pipeline.
- LoRA/QLoRA and DPO training are not performed by this notebook. It can be used later to evaluate fine-tuned adapters.
- The speech tests currently use synthetic audio. They should not be treated as proof of performance on real-world recordings.
- Privacy tests include synthetic examples. They do not establish production-level privacy or deletion guarantees.
- Acceptance thresholds in the final report are proposed evaluation targets, not verified product requirements.
- Do not upload real user conversations, private recordings, API keys, or access tokens to the repository or public Colab sessions.
