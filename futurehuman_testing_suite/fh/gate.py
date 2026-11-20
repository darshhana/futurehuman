from dataclasses import dataclass
from .tools_sim import INTENT_TOOL, TOOL_CATEGORY, build_call, missing_args

RISK = {"calendar_add": "medium", "calendar_modify": "medium", "calendar_remove": "high", "reminder_add": "low",
        "reminder_modify": "low", "reminder_remove": "low", "task_add": "low", "message_send": "high"}
DEFAULT_THRESHOLDS = {"low": 0.6, "medium": 0.8, "high": 0.9}
ALL_PERMISSIONS = {"calendar", "reminders", "tasks", "messaging"}


@dataclass
class Decision:
    action: str          # IGNORE | ASK | EXECUTE
    reason: str
    call: object = None
    tier: str = None


class PolicyGate:
    """Deterministic gate. The LLM only PROPOSES; this decides. Rule order is fixed and unit-tested."""

    def __init__(self, thresholds=None, permissions=None, paused=False, confirm_tiers=("high",), honor_model_confirmation=True):
        self.th = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
        self.permissions = set(ALL_PERMISSIONS if permissions is None else permissions)
        self.paused = paused
        self.confirm_tiers = set(confirm_tiers)
        self.honor_model_confirmation = honor_model_confirmation

    def decide(self, proposal, world, memory=None):
        if self.paused:
            return Decision("IGNORE", "paused")
        if not isinstance(proposal, dict) or proposal.get("actionability") not in ("actionable", "non_actionable", "ambiguous"):
            return Decision("IGNORE", "invalid_proposal")
        if proposal["actionability"] == "non_actionable":
            return Decision("IGNORE", "non_actionable")
        if proposal["actionability"] == "ambiguous":
            return Decision("ASK", "ambiguous")
        call = build_call(proposal)
        if call is None:
            return Decision("ASK", "intent_has_no_tool")
        tier = RISK[call.tool]
        if TOOL_CATEGORY[call.tool] not in self.permissions:
            return Decision("IGNORE", "no_permission", call, tier)
        miss = missing_args(call)
        if miss:
            return Decision("ASK", "missing:" + ",".join(miss), call, tier)
        if call.tool in ("calendar_modify", "calendar_remove", "reminder_modify", "reminder_remove"):
            if len(world.find(call)) != 1:
                return Decision("ASK", "target_not_found_or_ambiguous", call, tier)
        if world.conflicts(call):
            return Decision("ASK", "calendar_conflict", call, tier)
        conf = proposal.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or conf < self.th[tier]:
            return Decision("ASK", "low_confidence", call, tier)
        if tier in self.confirm_tiers:
            return Decision("ASK", f"{tier}_risk_needs_confirmation", call, tier)
        if self.honor_model_confirmation and proposal.get("requires_confirmation") is True and tier != "low":
            return Decision("ASK", "model_requested_confirmation", call, tier)
        if memory is not None and memory.requires_confirmation(call.tool):
            return Decision("ASK", "user_preference_confirm", call, tier)
        return Decision("EXECUTE", "all_checks_passed", call, tier)
