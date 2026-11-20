import re, unicodedata

NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}


def normalize(text):
    t = unicodedata.normalize("NFC", text.lower())
    t = re.sub(r"[^\w\s\u0900-\u097F]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    return _edit(r, h) / max(len(r), 1)


def cer(ref, hyp):
    r, h = normalize(ref).replace(" ", ""), normalize(hyp).replace(" ", "")
    return _edit(list(r), list(h)) / max(len(r), 1)


def _ent(s):
    s = normalize(s)
    s = " ".join(NUMW.get(w, w) for w in s.split())
    return re.sub(r"\s+", "", s.replace("a m", "am").replace("p m", "pm"))


def entity_error_rate(ref_entities, hyp_text):
    """Share of reference entities (names, times, dates, numbers) missing from the transcript."""
    if not ref_entities:
        return None
    h = _ent(hyp_text)
    miss = sum(1 for e in ref_entities if _ent(e) not in h)
    return miss / len(ref_entities)


def _frames(segs, total, step):
    n = int(total / step) + 1
    f = [False] * n
    for s, e in segs:
        for i in range(int(s / step), min(int(e / step) + 1, n)):
            f[i] = True
    return f


def vad_metrics(ref_segments, hyp_segments, total_dur, step=0.01):
    """Frame-level: speech recall (missed speech) and false-alarm rate (hyp speech over non-speech)."""
    r, h = _frames(ref_segments, total_dur, step), _frames(hyp_segments, total_dur, step)
    sp = sum(r)
    ns = len(r) - sp
    hit = sum(a and b for a, b in zip(r, h))
    fa = sum((not a) and b for a, b in zip(r, h))
    return dict(speech_recall=hit / sp if sp else float("nan"), false_alarm_rate=fa / ns if ns else float("nan"),
                missed_speech_s=(sp - hit) * step, false_alarm_s=fa * step)
