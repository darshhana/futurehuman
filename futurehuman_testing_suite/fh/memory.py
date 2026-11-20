"""Explicit user memory (profile + episodic).  Design rule under test: memory may make the gate STRICTER
or enrich arguments, but can never lower a confirmation requirement below the default policy."""
import itertools, re, time


class UserMemory:
    TRUSTED_CHANNELS = {"ui"}  # only explicit UI feedback can change preferences; conversation text cannot

    def __init__(self):
        self.profile = dict(confirm_before=set(), nicknames={}, default_reminder_lead_min=None, preferred_hours=None)
        self.episodes = {}
        self._ids = itertools.count(1)
        self._rejects = {}

    # ---- feedback -> structured preferences
    def apply_feedback(self, ev):
        if ev.get("channel") not in self.TRUSTED_CHANNELS:
            return False  # e.g. text inside a conversation trying to write a preference
        kind, tool = ev.get("type"), ev.get("tool")
        if kind == "rejected" and tool:
            self._rejects[tool] = self._rejects.get(tool, 0) + 1
            if self._rejects[tool] >= 2:
                self.profile["confirm_before"].add(tool)
        elif kind == "set_preference":
            k, v = ev["key"], ev["value"]
            if k == "confirm_before":
                self.profile["confirm_before"].add(v)
            elif k in ("default_reminder_lead_min", "preferred_hours"):
                self.profile[k] = v  # newest value wins (preference drift)
            elif k == "nickname":
                self.profile["nicknames"][v[0].lower()] = v[1]
        elif kind == "corrected" and ev.get("field") == "reminder_lead_min":
            self.profile["default_reminder_lead_min"] = ev["new"]
        else:
            return False
        self.remember(f"{kind}: {ev}", tags=[tool or "", ev.get("field", "")])
        return True

    def remember(self, text, tags=()):
        mid = next(self._ids)
        self.episodes[mid] = dict(id=mid, text=text, tags=list(tags), ts=time.time())
        return mid

    def delete(self, mid):
        return self.episodes.pop(mid, None) is not None

    def retrieve(self, query, k=3):
        q = set(re.findall(r"\w+", query.lower()))
        scored = []
        for m in self.episodes.values():
            words = set(re.findall(r"\w+", (m["text"] + " " + " ".join(m["tags"])).lower()))
            s = len(q & words)
            if s:
                scored.append((s, m["ts"], m))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [m for _, _, m in scored[:k]]

    # ---- used by the gate (can only add strictness)
    def requires_confirmation(self, tool):
        return tool in self.profile["confirm_before"]

    def resolve_name(self, name):
        return self.profile["nicknames"].get((name or "").lower(), name)
