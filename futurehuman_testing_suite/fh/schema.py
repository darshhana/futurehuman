import json, re

ACTIONABILITY = ("actionable", "non_actionable", "ambiguous")
INTENTS = ("none", "unknown", "create_reminder", "cancel_reminder", "schedule_meeting",
           "reschedule_meeting", "cancel_meeting", "confirm_meeting", "decline_meeting",
           "send_message", "create_task")
REQUIRED = ("actionability", "intent", "entities", "confidence", "requires_confirmation")


def extract_json(text):
    """First balanced {...} object after an optional </think> block. Returns dict or None."""
    if text is None:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def validate(obj):
    """Return (schema_valid, errors)."""
    errs = []
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    for k in REQUIRED:
        if k not in obj:
            errs.append(f"missing:{k}")
    if errs:
        return False, errs
    if obj["actionability"] not in ACTIONABILITY:
        errs.append("bad_actionability_enum")
    if obj["intent"] not in INTENTS:
        errs.append("bad_intent_enum")
    if not isinstance(obj["entities"], dict):
        errs.append("entities_not_dict")
    c = obj["confidence"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not (0 <= c <= 1):
        errs.append("bad_confidence")
    if not isinstance(obj["requires_confirmation"], bool):
        errs.append("bad_requires_confirmation")
    return (not errs), errs


def parse_prediction(text):
    obj = extract_json(text)
    if obj is None:
        return dict(json_valid=False, schema_valid=False, errors=["no_json"], data=None)
    ok, errs = validate(obj)
    return dict(json_valid=True, schema_valid=ok, errors=errs, data=obj)
