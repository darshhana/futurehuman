import json, re
from .metrics import rate

GENERAL = [
 # (category, prompt, check_type, check_arg)
 ("math", "What is 17 * 23? Give only the number.", "num", 391),
 ("math", "A train travels 60 km/h for 2.5 hours. How many km? Give only the number.", "num", 150),
 ("math", "What is 15% of 240? Give only the number.", "num", 36),
 ("math", "If x + 7 = 19, what is x? Give only the number.", "num", 12),
 ("math", "How many minutes are in 3.5 hours? Give only the number.", "num", 210),
 ("math", "What is the sum of the first 10 positive integers? Give only the number.", "num", 55),
 ("math", "A shirt costs 800 rupees after a 20% discount. What was the original price in rupees? Give only the number.", "num", 1000),
 ("math", "What is 144 divided by 12? Give only the number.", "num", 12),
 ("facts", "What is the capital of Japan? One word.", "contains", ["tokyo"]),
 ("facts", "Which planet is known as the Red Planet? One word.", "contains", ["mars"]),
 ("facts", "Who wrote the play Romeo and Juliet? Short answer.", "contains", ["shakespeare"]),
 ("facts", "What is the chemical formula of water? Short answer.", "contains", ["h2o"]),
 ("facts", "What is the largest ocean on Earth? One word.", "contains", ["pacific"]),
 ("facts", "In which country is the Taj Mahal? One word.", "contains", ["india"]),
 ("instruction", "Reply in all lowercase letters: say hello to my friend.", "lower", None),
 ("instruction", "Answer with exactly three words: what colour is the sky on a clear day?", "words", 3),
 ("instruction", "Return a JSON object with keys name and age for a person called Asha aged 30. JSON only.", "json_keys", ["name", "age"]),
 ("instruction", "List two fruits separated by a comma, nothing else.", "regex", r"^\s*[A-Za-z ]+\s*,\s*[A-Za-z ]+\s*\.?\s*$"),
 ("instruction", "Reply with the single word YES in capital letters.", "exact", "YES"),
 ("instruction", "Count from 1 to 5, separated by single spaces, nothing else.", "exact", "1 2 3 4 5"),
 ("chat", "Hi! How are you?", "nojson", None),
 ("chat", "Tell me a one-line joke.", "nojson", None),
 ("chat", "Thanks, that's all for now.", "nojson", None),
 ("chat", "What can you help me with?", "nojson", None),
 ("hindi", "भारत की राजधानी क्या है? एक शब्द में उत्तर दें।", "contains_any", ["दिल्ली", "delhi"]),
 ("hindi", "Mujhe ek line mein batao: paani kitne degree Celsius par ubalta hai?", "contains", ["100"]),
 ("hindi", "5 aur 7 ka sum kya hota hai? Sirf number batao.", "num", 12),
 ("hindi", "'Good morning' ko Hindi mein kaise kahenge? Sirf Devanagari mein jawab do.", "devanagari", None),
]


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def check(ctype, arg, text):
    t = strip_think(text)
    if ctype == "num":
        nums = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
        return bool(nums) and any(abs(float(n) - arg) < 1e-6 for n in nums[-3:] + nums[:1])
    if ctype == "contains":
        return all(k in t.lower() for k in arg)
    if ctype == "contains_any":
        return any(k in t.lower() for k in arg)
    if ctype == "lower":
        return len(t) > 0 and t == t.lower() and any(ch.isalpha() for ch in t)
    if ctype == "words":
        return len(re.findall(r"\w+", t)) == arg
    if ctype == "json_keys":
        m = re.search(r"\{.*\}", t, re.S)
        try:
            o = json.loads(m.group()) if m else None
            return isinstance(o, dict) and all(k in o for k in arg)
        except Exception:
            return False
    if ctype == "regex":
        return re.match(arg, t, re.S) is not None
    if ctype == "exact":
        return t.strip().strip(".") == arg
    if ctype == "nojson":
        return len(t) > 3 and "{" not in t and '"actionability"' not in t
    if ctype == "devanagari":
        return re.search(r"[\u0900-\u097F]", t) is not None
    return False


def run_general(gen, max_new_tokens=160):
    rows = []
    for cat, prompt, ct, arg in GENERAL:
        r = gen([{"role": "system", "content": "You are a helpful assistant. Be concise."},
                 {"role": "user", "content": prompt}], thinking=False, max_new_tokens=max_new_tokens)
        rows.append(dict(category=cat, prompt=prompt, ok=check(ct, arg, r["text"]), output=r["text"][:300]))
    return rows


def general_summary(rows):
    ok = sum(r["ok"] for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["category"], [0, 0])
        by[r["category"]][0] += r["ok"]
        by[r["category"]][1] += 1
    return dict(overall=rate(ok, len(rows)), by_category={k: v[0] / v[1] for k, v in by.items()})


def compare_runs(before, after, tolerance_points=2.0):
    """before/after: dict capability -> {'value': float(0-1), 'k':int,'n':int (optional, for rate caps)}.
    Rule: any capability dropping more than tolerance_points => REJECT.  For 'false_action' (lower is better) the
    after-value must not exceed the upper CI bound of before."""
    import pandas as pd
    from .metrics import wilson
    rows, reject = [], False
    for cap, b in before.items():
        a = after.get(cap)
        if a is None:
            continue
        lower_better = cap.startswith("false_") or cap.startswith("invalid")
        delta = (a["value"] - b["value"]) * 100
        if lower_better:
            hi = wilson(b["k"], b["n"])[1] if b.get("n") else b["value"]
            bad = a["value"] > hi
        else:
            bad = delta < -tolerance_points
        reject |= bad
        rows.append(dict(capability=cap, before=round(b["value"] * 100, 1), after=round(a["value"] * 100, 1),
                         delta_points=round(delta, 1), verdict="REGRESSION" if bad else "ok"))
    return pd.DataFrame(rows), ("REJECT" if reject else "ACCEPT")
