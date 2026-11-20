"""Privacy layer under test: redaction (runs BEFORE storage) + capture rules + pause.
Regex baseline - the test set deliberately includes hard cases so weaknesses show up."""
import re, time

def luhn_ok(digits):
    d = [int(x) for x in digits][::-1]
    s = sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])
    return len(d) >= 13 and s % 10 == 0

_P = []
def _add(kind, pat, group=0, flags=0):
    _P.append((kind, re.compile(pat, flags), group))

_add("otp", r"\b(?:otp|one[- ]time (?:password|code)|verification code|auth(?:entication)? code|security code|login code)\b[^\n]{0,40}?\b(\d{4,8})\b", 1, re.I)
_add("otp", r"\b(\d{4,8})\b(?=[^\n]{0,20}\b(?:is|hai|है)\b[^\n]{0,15}\b(?:otp|verification code|code)\b)", 1, re.I)
_add("pin", r"\b(?:atm |card |upi )?pin\b(?!\s*code)[^0-9\n]{0,20}(\d{4,6})\b", 1, re.I)
_add("cvv", r"\bcvv2?\b[^0-9\n]{0,10}(\d{3,4})\b", 1, re.I)
_add("aadhaar", r"\b\d{4}\s\d{4}\s\d{4}\b")
_add("aadhaar", r"\b(?:aadhaar|aadhar|आधार)\b[^0-9\n]{0,20}(\d{12})\b", 1, re.I)
_add("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b")
_add("upi", r"\b[\w.\-]{2,}@(?:oksbi|okaxis|okhdfcbank|okicici|ybl|paytm|ibl|axl|upi|sbi|hdfcbank|icici|apl|fbl)\b", 0, re.I)
_add("bank_account", r"\b(?:a/c|acc(?:ount)?(?:\s*(?:no|number|num))?\.?)[^0-9\n]{0,15}(\d{9,18})\b", 1, re.I)
_add("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_add("api_key", r"\bsk-[A-Za-z0-9_\-]{16,}\b")
_add("api_key", r"\bAKIA[0-9A-Z]{16}\b")
_add("api_key", r"\bghp_[A-Za-z0-9]{30,}\b")
_add("api_key", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")
_add("api_key", r"\bAIza[0-9A-Za-z_\-]{30,}\b")
_add("jwt", r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")
_add("token", r"\bBearer\s+([A-Za-z0-9\-._~+/]{16,}=*)", 1, re.I)
_add("secret", r"\b(?:api[_ -]?key|secret|token|passwd|pwd)\b\s*[:=]\s*(\S{6,})", 1, re.I)
_add("password", r"\b(?:password|passcode|pwd|पासवर्ड)\b\s*(?:is|hai|है|:|=)\s*(\S+)", 1, re.I)
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def redact(text):
    """Return (redacted_text, findings[kind])."""
    found = []
    spans = []
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if luhn_ok(digits) or re.search(r"card", text[max(0, m.start() - 25):m.start()], re.I):
            spans.append((m.start(), m.end(), "card"))
    for kind, pat, g in _P:
        for m in pat.finditer(text):
            s, e = (m.start(g), m.end(g)) if g else (m.start(), m.end())
            spans.append((s, e, kind))
    spans.sort()
    merged = []
    for s, e, k in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]), merged[-1][2])
        else:
            merged.append((s, e, k))
    out, last = [], 0
    for s, e, k in merged:
        out.append(text[last:s])
        out.append(f"[REDACTED:{k}]")
        found.append(k)
        last = e
    out.append(text[last:])
    return "".join(out), found


def evaluate_redaction(items):
    """items: {text, sensitive:[substrings that must NOT survive], kind, decoy:bool}.
    Leak = any sensitive substring still present.  False positive = a decoy changed."""
    leaks, fps, sens_n, decoy_n, per_kind = [], [], 0, 0, {}
    for it in items:
        red, _ = redact(it["text"])
        if it.get("decoy"):
            decoy_n += 1
            if red != it["text"]:
                fps.append(dict(text=it["text"], redacted=red))
        else:
            sens_n += 1
            leaked = [s for s in it["sensitive"] if s in red]
            k = per_kind.setdefault(it["kind"], [0, 0])
            k[1] += 1
            if leaked:
                leaks.append(dict(text=it["text"], redacted=red, leaked=leaked, kind=it["kind"]))
            else:
                k[0] += 1
    return dict(n_sensitive=sens_n, n_decoy=decoy_n, leaks=leaks, false_positives=fps,
                recall=(sens_n - len(leaks)) / sens_n if sens_n else float("nan"),
                false_positive_rate=len(fps) / decoy_n if decoy_n else float("nan"),
                per_kind={k: v[0] / v[1] for k, v in per_kind.items()})


# ---------------------------------------------------------------- capture rules + pause
PASSWORD_MANAGERS = ("1password", "bitwarden", "lastpass", "keepass", "dashlane", "enpass", "keeper")
CATEGORY_MAP = {"banking": ("hdfcbank.com", "icicibank.com", "sbi.co.in", "axisbank.com", "paytm.com", "phonepe.com"),
                "health": ("practo.com", "1mg.com"), "social": ("facebook.com", "instagram.com", "x.com")}
PRIVATE_TITLE = re.compile(r"incognito|private browsing|inprivate|private window", re.I)


class PauseState:
    def __init__(self):
        self.until = 0.0
        self.indefinite = False

    def pause(self, minutes=None, now=None):
        now = time.time() if now is None else now
        if minutes is None:
            self.indefinite = True
        else:
            self.until = now + minutes * 60
            self.indefinite = False

    def resume(self):
        self.until, self.indefinite = 0.0, False

    def is_paused(self, now=None):
        now = time.time() if now is None else now
        return self.indefinite or now < self.until


def should_capture(ctx, settings, pause=None, now=None):
    """ctx: app, title, domain, is_private, field_type, permission_granted.  Returns (bool, reason)."""
    if not ctx.get("permission_granted"):
        return False, "no_permission"
    if pause is not None and pause.is_paused(now):
        return False, "paused"
    app = (ctx.get("app") or "").lower()
    domain = (ctx.get("domain") or "").lower()
    if ctx.get("is_private") or PRIVATE_TITLE.search(ctx.get("title") or ""):
        return False, "private_window"
    if any(pm in app for pm in PASSWORD_MANAGERS):
        return False, "password_manager"
    if ctx.get("field_type") == "password":
        return False, "password_field"
    if app in {a.lower() for a in settings.get("excluded_apps", ())}:
        return False, "excluded_app"
    for d in settings.get("excluded_domains", ()):
        if domain == d or domain.endswith("." + d):
            return False, "excluded_domain"
    for cat in settings.get("excluded_categories", ()):
        if any(domain == d or domain.endswith("." + d) for d in CATEGORY_MAP.get(cat, ())):
            return False, f"excluded_category:{cat}"
    return True, "ok"
