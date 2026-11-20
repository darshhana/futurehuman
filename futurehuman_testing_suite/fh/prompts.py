import json
from . import REF_DATE
from .schema import INTENTS

# A0: the original prompt from the notebook (kept to measure how much prompt work helped)
PROMPT_ORIGINAL = """You are FutureHuman's conversational action intelligence engine.

Analyse the conversation context and the user's latest message.

Classify actionability as exactly one of:
- actionable
- non_actionable
- ambiguous

Rules:
1. Do not treat casual mentions as instructions.
2. Consider the conversation context.
3. Recognise explicit requests, commitments and cancellations.
4. If the meaning is unclear, classify as ambiguous.
5. Never invent missing information.

Return only valid JSON in this format:

{
  "actionability": "actionable",
  "intent": "create_reminder",
  "entities": {},
  "confidence": 0.95,
  "requires_confirmation": false
}"""

# A1: improved zero-shot prompt (mirrors GUIDELINE.md so the model and the labelers share one definition)
PROMPT_V1 = """You are FutureHuman, a conversational action-intelligence engine.
You read a conversation between USER (the person you assist) and OTHER (another person or an assistant).
Classify ONLY THE LAST TURN of the conversation, using all earlier turns as context.
Conversation may be English, Hindi (Devanagari), Hinglish (Roman Hindi) or mixed.

actionability:
- actionable: the USER gives a clear command/request, commits to something themselves, accepts a concrete
  proposal made earlier, or cancels/changes an existing item, AND the needed details (what + when/who)
  are present in the conversation.
- non_actionable: nothing for the assistant to do: facts, opinions, past events, hypotheticals/conditionals,
  declines, statements by OTHER (their proposals or promises), general questions, or vague wishes with no who/when.
  An open proposal is NOT an assignment until the USER accepts it.
- ambiguous: an action is plausible but the commitment is hedged (maybe/might/we could) or a needed detail
  is missing or cannot be resolved (which meeting? what time? who is 'him'?). The assistant should ASK.

intent (choose exactly one): none, unknown, create_reminder, cancel_reminder, schedule_meeting,
reschedule_meeting, cancel_meeting, confirm_meeting, decline_meeting, send_message, create_task
- non_actionable -> intent none. ambiguous -> intent unknown.
- accepting a proposed meeting -> schedule_meeting. Changing an existing meeting -> reschedule_meeting.

entities: only values that appear in the conversation; use keys person, date, time, title, message.
Never invent information. Hindi words like 'kal'/'parso' can mean past or future: if tense does not
settle it, treat the date as unresolved (ambiguous).

confidence: 0.0-1.0, how sure you are (use lower values when unsure; do not default to 0.95).
requires_confirmation: true when the action is implicit, destructive, or sends something to another person.
You never execute anything; you only propose.

Return ONLY a JSON object:
{"actionability": "...", "intent": "...", "entities": {}, "confidence": 0.0, "requires_confirmation": false}"""

PROMPTS = {"A0_original": PROMPT_ORIGINAL, "A1_improved": PROMPT_V1}


def format_conversation(turns):
    return "\n".join(f"{'USER' if s == 'user' else 'OTHER'}: {t}" for s, t in turns)


def user_message(case, ref_date=REF_DATE):
    return (f"Today is {ref_date}.\nConversation:\n{format_conversation(case['turns'])}\n\n"
            f"Classify the LAST turn.")


def gold_json(case):
    intent = case["intents"][0] if case["label"] == "actionable" else ("none" if case["label"] == "non_actionable" else "unknown")
    conf = {"actionable": 0.9, "non_actionable": 0.9, "ambiguous": 0.6}[case["label"]]
    return json.dumps({"actionability": case["label"], "intent": intent, "entities": case.get("entities") or {},
                       "confidence": conf, "requires_confirmation": bool(case["requires_confirmation"])},
                      ensure_ascii=False)


def select_fewshot(dev_cases, n=(3, 3, 2), seed=0):
    """Pick few-shot examples from DEV only: n=(actionable, non_actionable, ambiguous), varied families, mixed languages."""
    import random
    rnd = random.Random(seed)
    chosen, used_fam = [], set()
    for label, k in zip(("actionable", "non_actionable", "ambiguous"), n):
        pool = [c for c in dev_cases if c["label"] == label]
        rnd.shuffle(pool)
        pool.sort(key=lambda c: (c["family"] in used_fam, c["lang"] == "en"))  # prefer new families, non-English first
        for c in pool[:k]:
            chosen.append(c)
            used_fam.add(c["family"])
    rnd.shuffle(chosen)
    return chosen


def build_messages(case, prompt_id="A1_improved", fewshot=None, ref_date=REF_DATE):
    msgs = [{"role": "system", "content": PROMPTS[prompt_id]}]
    for fs in fewshot or []:
        msgs.append({"role": "user", "content": user_message(fs, ref_date)})
        msgs.append({"role": "assistant", "content": gold_json(fs)})
    msgs.append({"role": "user", "content": user_message(case, ref_date)})
    return msgs
