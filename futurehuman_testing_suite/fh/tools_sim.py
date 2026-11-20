"""Simulated tools (no real APIs).  Typed tool calls built from loose model entities."""
import copy, re
from dataclasses import dataclass, field

ALIASES = dict(
    person=["person", "name", "participant", "participants", "with", "target", "contact", "attendee", "attendees",
            "recipient", "to", "meeting_participants", "who", "meeting_with"],
    date=["date", "day", "when", "meeting_day", "meeting_date", "due", "due_date", "deadline"],
    time=["time", "at", "start_time", "meeting_time", "hour", "reminder_time"],
    title=["title", "meeting", "event", "task", "topic", "subject", "project", "action", "assignment",
           "meeting_topic", "reminder", "what", "description"],
    message=["message", "text", "content", "body", "msg"],
)
INTENT_TOOL = {"schedule_meeting": "calendar_add", "confirm_meeting": "calendar_add", "reschedule_meeting": "calendar_modify",
               "cancel_meeting": "calendar_remove", "create_reminder": "reminder_add", "cancel_reminder": "reminder_remove",
               "send_message": "message_send", "create_task": "task_add"}
TOOL_CATEGORY = {"calendar_add": "calendar", "calendar_modify": "calendar", "calendar_remove": "calendar",
                 "reminder_add": "reminders", "reminder_modify": "reminders", "reminder_remove": "reminders",
                 "task_add": "tasks", "message_send": "messaging"}
# every tuple = at least ONE of these args must be present
REQUIRED = {"calendar_add": [("date",), ("time",)], "calendar_modify": [("date", "time")], "calendar_remove": [],
            "reminder_add": [("title", "person"), ("date", "time")], "reminder_modify": [("date", "time")],
            "reminder_remove": [("title", "person")], "task_add": [("title", "person")],
            "message_send": [("person",), ("message",)]}
INTENT_EFFECTS = {"schedule_meeting": {"calendar_add"}, "confirm_meeting": {"calendar_add"}, "reschedule_meeting": {"calendar_modify", "reminder_modify"},
                  "cancel_meeting": {"calendar_remove"}, "create_reminder": {"reminder_add", "reminder_modify"},
                  "cancel_reminder": {"reminder_remove"}, "send_message": {"message_send"}, "create_task": {"task_add"}}


def _s(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(_s(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(_s(x) for x in v.values())
    return "" if v is None else str(v).strip()


def normalize_entities(ents):
    out = {}
    if not isinstance(ents, dict):
        return out
    for k, v in ents.items():
        kk = str(k).lower().strip()
        for canon, names in ALIASES.items():
            if kk in names or kk == canon:
                if _s(v) and canon not in out:
                    out[canon] = _s(v)
                break
    if "time" not in out and "date" in out:
        m = re.search(r"\bat\s+(.+)$", out["date"], re.I)
        if m:
            out["time"] = m.group(1)
            out["date"] = out["date"][:m.start()].strip()
    return out


@dataclass
class ToolCall:
    tool: str
    args: dict
    def text(self):
        return " ".join(str(v) for v in self.args.values()).lower()


def build_call(proposal):
    """proposal dict (validated model JSON) -> ToolCall or None if the intent maps to no tool."""
    tool = INTENT_TOOL.get(proposal.get("intent"))
    if tool is None:
        return None
    return ToolCall(tool, normalize_entities(proposal.get("entities")))


def missing_args(call):
    miss = []
    for alts in REQUIRED[call.tool]:
        if not any(call.args.get(a) for a in alts):
            miss.append("/".join(alts))
    return miss


@dataclass
class MockWorld:
    calendar: list = field(default_factory=list)
    reminders: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):
        d = copy.deepcopy(d or {})
        return cls(calendar=d.get("calendar", []), reminders=d.get("reminders", []), tasks=d.get("tasks", []), messages=d.get("messages", []))

    def snapshot(self):
        return copy.deepcopy((self.calendar, self.reminders, self.tasks, self.messages))

    # ---- lookup used by the gate (external-state check) and by execution
    def _items(self, tool):
        return self.reminders if tool.startswith("reminder") else self.calendar

    def find(self, call):
        items = self._items(call.tool)
        keys = ("person", "title") if call.tool.endswith("modify") else ("person", "title", "date", "time")
        hits = []
        for it in items:
            ok = True
            for k in keys:
                v = call.args.get(k)
                if v and v.lower() not in str(it.get(k, "")).lower() and str(it.get(k, "")).lower() not in v.lower():
                    # title fuzzy: allow overlap of any word
                    if k == "title" and set(v.lower().split()) & set(str(it.get("title", "")).lower().split()):
                        continue
                    ok = False
                    break
            if ok:
                hits.append(it)
        return hits

    def conflicts(self, call):
        if call.tool != "calendar_add" or not call.args.get("time"):
            return []
        return [e for e in self.calendar if str(e.get("date", "")).lower() == call.args.get("date", "").lower()
                and str(e.get("time", "")).lower() == call.args.get("time", "").lower()]

    def execute(self, call):
        a = call.args
        if call.tool == "calendar_add":
            self.calendar.append(dict(title=a.get("title", "meeting"), person=a.get("person", ""), date=a.get("date", ""), time=a.get("time", "")))
        elif call.tool in ("calendar_modify", "reminder_modify"):
            hits = self.find(call)
            if len(hits) != 1:
                raise LookupError("not_found_or_ambiguous")
            for k in ("date", "time"):
                if a.get(k):
                    hits[0][k] = a[k]
        elif call.tool == "calendar_remove":
            hits = self.find(call)
            if len(hits) != 1:
                raise LookupError("not_found_or_ambiguous")
            self.calendar.remove(hits[0])
        elif call.tool == "reminder_add":
            self.reminders.append(dict(title=a.get("title", ""), person=a.get("person", ""), date=a.get("date", ""), time=a.get("time", "")))
        elif call.tool == "reminder_remove":
            hits = self.find(call)
            if len(hits) != 1:
                raise LookupError("not_found_or_ambiguous")
            self.reminders.remove(hits[0])
        elif call.tool == "task_add":
            self.tasks.append(dict(title=a.get("title", a.get("person", "")), due=a.get("date", "")))
        elif call.tool == "message_send":
            self.messages.append(dict(to=a.get("person", ""), text=a.get("message", "")))
        else:
            raise ValueError("unknown_tool")


def observed_effects(before, after):
    """before/after: MockWorld.snapshot() tuples -> set of effect names."""
    names = ("calendar", "reminder", "task", "message")
    eff = set()
    for name, b, a in zip(names, before, after):
        if len(a) > len(b):
            eff.add(f"{name}_add" if name != "message" else "message_send")
        elif len(a) < len(b):
            eff.add(f"{name}_remove")
        elif a != b:
            eff.add(f"{name}_modify")
    return eff
