"""Builds data/actionability_v1.jsonl  (run:  python build_dataset.py)

Each example:
  id, group (split unit), pair (minimal-pair id or null), variant, family, lang,
  turns [[speaker, text], ...]  (LAST turn is the one being classified),
  label, intents (list of acceptable intents), entities {person,date,time,title,message},
  requires_confirmation, world (initial mock state for E2E), src, note
Splits are assigned by `group` (see fh/dataset.py) so variants never leak across splits.
"""
import json, os

SM = ["schedule_meeting", "confirm_meeting"]
RM = ["reschedule_meeting"]
CM = ["cancel_meeting"]
CR = ["create_reminder"]
CT = ["create_task"]
MSG = ["send_message"]
NONE = ["none"]
UNK = ["unknown"]

EX = []
_gc = [0]


def T(s):
    out = []
    for part in s.split(" || "):
        sp, tx = part.split(": ", 1)
        out.append(["user" if sp.strip() == "U" else "other", tx.strip()])
    return out


def ex(turns, label, intents, ents=None, fam="", lang="en", pair=None, var=None,
       src=None, note="", rc=None, world=None, group=None):
    if group is None:
        _gc[0] += 1
        group = f"g{_gc[0]:03d}"
    if rc is None:
        if label == "ambiguous":
            rc = True
        elif label == "non_actionable":
            rc = False
        else:
            rc = (fam in ("implicit_multiturn", "agreement_confirmation", "correction_cancel_reschedule",
                          "multi_assignment", "safety_high_risk")
                  or intents[0] in ("send_message", "cancel_meeting", "cancel_reminder"))
    EX.append(dict(group=group, pair=pair, variant=var, family=fam, lang=lang,
                   turns=T(turns), label=label, intents=intents, entities=ents or {},
                   requires_confirmation=rc, world=world or {}, src=src, note=note))
    return group


def cal(*events):
    return {"calendar": list(events)}


def evt(title, person, date, time):
    return dict(title=title, person=person, date=date, time=time)


# ------------------------------------------------------------------ ENGLISH: explicit commands
ex("U: Remind me to call Rahul tomorrow at 10 AM.", "actionable", CR,
   dict(person="Rahul", date="tomorrow", time="10 AM"), "explicit_command", pair="P7", var="explicit", src="notebook")
ex("U: Set a reminder to submit the invoice on Friday at 5 PM.", "actionable", CR,
   dict(title="submit the invoice", date="Friday", time="5 PM"), "explicit_command")
ex("U: Schedule a meeting with Priya on Monday at 11 AM.", "actionable", SM,
   dict(person="Priya", date="Monday", time="11 AM"), "explicit_command")
ex("U: Book a call with the design team on Wednesday at 3 PM.", "actionable", SM,
   dict(person="design team", date="Wednesday", time="3 PM"), "explicit_command")
ex("U: Send a message to Anil saying I will be ten minutes late.", "actionable", MSG,
   dict(person="Anil", message="I will be ten minutes late"), "explicit_command")
ex("U: Add a task to renew my passport by the end of this month.", "actionable", CT,
   dict(title="renew my passport"), "explicit_command")
ex("U: Create a task to review the pull request today.", "actionable", CT,
   dict(title="review the pull request", date="today"), "explicit_command")
ex("U: Remind me in half an hour to take the laundry out.", "actionable", CR,
   dict(title="take the laundry out", time="half an hour"), "explicit_command")
ex("U: Please tell Rahul the meeting is moved to 5.", "actionable", MSG,
   dict(person="Rahul", message="the meeting is moved to 5"), "explicit_command")

# ------------------------------------------------------------------ cancel / reschedule / change of mind
ex("U: Move my Thursday meeting with Neha to Friday at 2 PM.", "actionable", RM,
   dict(person="Neha", date="Friday", time="2 PM"), "correction_cancel_reschedule",
   world=cal(evt("meeting with Neha", "Neha", "Thursday", "10 AM")))
ex("U: Schedule a meeting with Rahul on Friday. || O: Sure, Friday meeting noted. || U: Actually, make it Saturday instead.",
   "actionable", RM, dict(person="Rahul", date="Saturday"), "correction_cancel_reschedule", src="notebook",
   world=cal(evt("meeting with Rahul", "Rahul", "Friday", "10 AM")))
ex("U: Schedule my meeting with Rahul tomorrow at 11 AM. || O: Your meeting has been scheduled. || U: Actually, cancel it.",
   "actionable", CM, dict(person="Rahul", date="tomorrow", time="11 AM"), "correction_cancel_reschedule", src="notebook",
   world=cal(evt("meeting with Rahul", "Rahul", "tomorrow", "11 AM")))
ex("U: Remind me to water the plants at 7 PM. || O: Reminder set. || U: Never mind, delete that reminder.",
   "actionable", ["cancel_reminder"], dict(title="water the plants"), "change_of_mind",
   world={"reminders": [dict(title="water the plants", person="", date="today", time="7 PM")]})
ex("U: Remind me to call Rahul at 10. || O: Done. || U: Make that 11 instead.",
   "actionable", ["create_reminder", "reschedule_meeting"], dict(person="Rahul", time="11"), "change_of_mind",
   world={"reminders": [dict(title="call Rahul", person="Rahul", date="today", time="10")]})
ex("U: I cannot attend the meeting tomorrow. || O: Thursday? || U: Thursday works.",
   "actionable", RM, dict(date="Thursday"), "implicit_multiturn",
   world=cal(evt("meeting", "", "tomorrow", "10 AM")))
ex("U: I have the review at 3 on Thursday. || O: A client call was just added at 3 on Thursday. || U: Move the review to 4 PM then.",
   "actionable", RM, dict(title="review", time="4 PM"), "calendar_conflict",
   world=cal(evt("review", "", "Thursday", "3 PM")))
ex("U: Cancel the Friday review and move the Monday sync to Tuesday at 11 AM.", "actionable",
   ["cancel_meeting", "reschedule_meeting"], dict(date="Tuesday", time="11 AM"), "multi_assignment",
   world=cal(evt("review", "", "Friday", "2 PM"), evt("sync", "", "Monday", "10 AM")))
ex("U: Remind me to buy milk tonight and schedule a call with Kiran for Monday at 10 AM.", "actionable",
   ["create_reminder", "schedule_meeting"], dict(person="Kiran", date="Monday", time="10 AM"), "multi_assignment")

# ------------------------------------------------------------------ P1: the 'Thursday works' family
P1c = "O: Can we schedule the project review for Thursday at 4 PM?"
g = ex(P1c + " || U: Thursday works for me.", "actionable", SM,
       dict(title="project review", date="Thursday", time="4 PM"), "agreement_confirmation", pair="P1", var="accept", src="notebook")
ex(P1c + " || U: Thursday doesn't work for me.", "non_actionable", NONE, None, "negative", pair="P1", var="decline", group=g)
ex(P1c + " || U: If Thursday suits the others, we'll see.", "non_actionable", NONE, None, "negative", pair="P1", var="hypothetical", group=g)
ex(P1c + " || U: Maybe Thursday, let me check my calendar.", "ambiguous", UNK, None, "ambiguous", pair="P1", var="hedge", group=g)
ex(P1c, "non_actionable", NONE, None, "negative", pair="P1", var="proposal_only", group=g,
   note="Open proposal by another person is not an assignment until the user accepts.")

ex("U: We need to review the project. || O: Should we schedule it for Friday at 3 PM? || U: Yes, that works for me.",
   "actionable", SM, dict(title="project review", date="Friday", time="3 PM"), "agreement_confirmation", src="notebook")
ex("O: Lunch tomorrow at 1? || U: Sounds good, see you then.", "actionable", SM,
   dict(title="lunch", date="tomorrow", time="1"), "agreement_confirmation")
ex("O: I'll be free after 6 on Tuesday. Dinner then? || U: Perfect, let's do Tuesday at 6:30.", "actionable", SM,
   dict(title="dinner", date="Tuesday", time="6:30"), "agreement_confirmation")
ex("U: My dentist appointment is at 5 on Friday. || O: Want me to put a reminder? || U: Yes please, an hour before.",
   "actionable", CR, dict(title="dentist appointment", date="Friday", time="an hour before"), "agreement_confirmation")
ex("O: Don't forget the team standup tomorrow at 9:30. || U: Thanks, remind me at 9 sharp.", "actionable", CR,
   dict(title="team standup", date="tomorrow", time="9"), "agreement_confirmation")
ex("O: Will you submit the assignment by Friday? || U: Yes, I'll submit it by Friday.", "actionable", CT,
   dict(title="submit the assignment", date="Friday"), "implicit_multiturn")

# P3: who commits
g = ex("O: Please send me the report by Wednesday noon. || U: Will do.", "actionable", CT,
       dict(title="send the report", date="Wednesday", time="noon"), "implicit_multiturn", pair="P3", var="user_commits")
ex("O: I'll send you the report by Wednesday noon.", "non_actionable", NONE, None, "negative", pair="P3", var="other_commits", group=g)

# urgent
ex("U: Urgent, message the vendor right now that the shipment is delayed.", "actionable", MSG,
   dict(person="vendor", message="the shipment is delayed"), "urgent_action")
ex("O: The server is down! Please inform the client immediately. || U: On it, messaging the client now.", "actionable", MSG,
   dict(person="client"), "urgent_action")

# ------------------------------------------------------------------ P2: single-turn
g = ex("U: I'll meet Rahul on Thursday at 5 PM.", "actionable", SM,
       dict(person="Rahul", date="Thursday", time="5 PM"), "explicit_command", pair="P2", var="self_commit")
ex("U: Put a meeting with Rahul on Thursday at 5 PM in my calendar.", "actionable", SM,
   dict(person="Rahul", date="Thursday", time="5 PM"), "explicit_command", pair="P2", var="command", group=g)
ex("U: Maybe I should meet Rahul on Thursday.", "ambiguous", UNK, dict(person="Rahul", date="Thursday"), "ambiguous", pair="P2", var="maybe", group=g)
ex("U: We should meet sometime next week.", "non_actionable", NONE, None, "no_action_mention", pair="P2", var="vague", group=g,
   note="Vague social statement: no who/when. Manual says must not auto-create an event.")

# P4: cancel family
g = ex("U: Cancel my meeting with Rahul tomorrow at 11 AM.", "actionable", CM,
       dict(person="Rahul", date="tomorrow", time="11 AM"), "correction_cancel_reschedule", pair="P4", var="command",
       world=cal(evt("meeting with Rahul", "Rahul", "tomorrow", "11 AM")))
ex("U: Should I cancel my meeting with Rahul tomorrow at 11 AM?", "ambiguous", UNK, None, "ambiguous", pair="P4", var="question", group=g,
   world=cal(evt("meeting with Rahul", "Rahul", "tomorrow", "11 AM")))
ex("U: My meeting with Rahul tomorrow at 11 AM is the one I dread most.", "non_actionable", NONE, None, "no_action_mention", pair="P4", var="mention", group=g,
   world=cal(evt("meeting with Rahul", "Rahul", "tomorrow", "11 AM")))

# P7: reminder family
g = [e["group"] for e in EX if e["pair"] == "P7"][0]
ex("U: I need to call Rahul tomorrow at 10 AM, I keep forgetting.", "ambiguous", UNK, dict(person="Rahul", date="tomorrow", time="10 AM"), "ambiguous", pair="P7", var="implied", group=g)
ex("U: I called Rahul yesterday at 10 AM.", "non_actionable", NONE, None, "no_action_mention", pair="P7", var="past", group=g)

# P8: long context with distractors
P8c = ("U: Did you watch the match last night? || O: Yes, what a game. || U: The weather was bad though. || "
       "O: True. By the way, can we review the budget on Wednesday at 2 PM?")
g = ex(P8c + " || U: Wednesday at 2 works.", "actionable", SM, dict(title="budget review", date="Wednesday", time="2 PM"),
       "long_context", pair="P8", var="accept")
ex(P8c + " || U: Wednesday is usually busy for me.", "non_actionable", NONE, None, "long_context", pair="P8", var="decline", group=g)
ex(P8c + " || U: I'll check my calendar and let you know.", "ambiguous", UNK, None, "long_context", pair="P8", var="hedge", group=g)

# ------------------------------------------------------------------ ENGLISH non-actionable
ex("U: I usually wake up at 7 AM.", "non_actionable", NONE, None, "no_action_mention", src="notebook")
ex("U: Thursday might be a good day for meetings.", "non_actionable", NONE, None, "no_action_mention", src="notebook")
ex("U: I love Thursdays. || O: Why do you prefer Thursdays? || U: Because my schedule is usually free.",
   "non_actionable", NONE, None, "no_action_mention", src="notebook")
ex("U: I had a meeting with Rahul yesterday and it went well.", "non_actionable", NONE, None, "no_action_mention")
ex("U: My meeting tomorrow is at 11 and I am a bit nervous.", "non_actionable", NONE, None, "no_action_mention")
ex("U: Rahul always cancels meetings at the last minute.", "non_actionable", NONE, None, "no_action_mention")
ex("U: The weather is great today.", "non_actionable", NONE, None, "negative")
ex("U: Did you see the match yesterday?", "non_actionable", NONE, None, "negative")
ex("U: I'm thinking about what to cook tonight.", "non_actionable", NONE, None, "negative")
ex("O: The client meeting is at 11 tomorrow. || U: Yes, I know.", "non_actionable", NONE, None, "negative")
ex("U: I don't need a reminder for that, I'll remember it myself.", "non_actionable", NONE, None, "negative")
ex("U: I have a client meeting tomorrow at 11 AM. || O: Would you like me to add it to your calendar? || U: No, don't add it to my calendar. I'll manage it myself.",
   "non_actionable", NONE, None, "negative", src="notebook")
ex("U: I have a client meeting tomorrow at 11 AM. || O: Would you like me to add it to your calendar? || U: No, forget it. It has been cancelled.",
   "non_actionable", NONE, None, "negative", src="notebook",
   note="Notebook gold was actionable/cancel_meeting. Re-labelled: the event was never added, so there is nothing to cancel. DISCUSS in review.")
ex("O: What time is the call? || U: It's at 3 PM, I think.", "non_actionable", NONE, None, "negative")
ex("U: Friday is my favorite day of the week.", "non_actionable", NONE, None, "negative")
ex("U: I scheduled a meeting with Priya last week and she never replied.", "non_actionable", NONE, None, "no_action_mention")
ex("U: Priya said she would call me tomorrow.", "non_actionable", NONE, None, "no_action_mention")
ex("U: What is a good time for team meetings in general?", "non_actionable", NONE, None, "qa_request")
ex("U: Tell me a joke.", "non_actionable", NONE, None, "qa_request", note="Request, but no assignment for tools.")
ex("U: What is the capital of France?", "non_actionable", NONE, None, "qa_request")
ex("U: Explain how GPS works.", "non_actionable", NONE, None, "qa_request")
ex("U: I wonder if I should join a gym.", "non_actionable", NONE, None, "negative")
ex("U: Meetings are exhausting.", "non_actionable", NONE, None, "negative")
ex("U: The deadline is Friday, which is stressful.", "non_actionable", NONE, None, "no_action_mention")
ex("U: I should really start going to bed early.", "non_actionable", NONE, None, "negative")

# ------------------------------------------------------------------ ENGLISH ambiguous
ex("U: Maybe I should meet the client tomorrow.", "ambiguous", UNK, dict(person="client", date="tomorrow"), "ambiguous", src="notebook")
ex("U: We should catch up sometime next week. || O: Would you like to schedule something? || U: Let's see, maybe.",
   "ambiguous", UNK, None, "ambiguous", src="notebook")
ex("U: I have a call with Neha at 2 and a review with Rahul at 4. || O: Noted. || U: Remind me about the meeting before it starts.",
   "ambiguous", UNK, None, "ambiguous", note="Two candidate meetings.")
ex("U: Schedule a meeting with Rahul.", "ambiguous", UNK, dict(person="Rahul"), "ambiguous", note="Missing when.")
ex("U: Remind me to call him tomorrow.", "ambiguous", UNK, dict(date="tomorrow"), "ambiguous", note="'him' unresolved.")
ex("O: How about Thursday or Friday? || U: Either is fine.", "ambiguous", UNK, None, "ambiguous")
ex("U: Cancel the meeting.", "ambiguous", UNK, None, "ambiguous", note="No referent in the conversation.",
   world=cal(evt("meeting", "", "tomorrow", "10 AM")))
ex("U: Schedule a call with Kiran on Monday. || O: Done. || U: Move it to the afternoon.", "ambiguous", UNK, None, "ambiguous",
   note="No exact time.", world=cal(evt("call with Kiran", "Kiran", "Monday", "10 AM")))
ex("U: Message Anil about tomorrow.", "ambiguous", UNK, dict(person="Anil"), "ambiguous", note="No content.")
ex("O: Can you send me the file today? || U: I'll try.", "ambiguous", UNK, None, "ambiguous", note="Hedged commitment.")
ex("O: When can we review the project? || U: Maybe we could do Thursday at 4.", "ambiguous", UNK, None, "ambiguous")
ex("U: Put something on my calendar for Friday.", "ambiguous", UNK, dict(date="Friday"), "ambiguous", note="What and when?")
ex("O: Shall we do 3 or 4 PM on Thursday? || U: 4 PM... maybe.", "ambiguous", UNK, None, "ambiguous")
ex("U: I might have to reschedule the Friday review.", "ambiguous", UNK, None, "ambiguous",
   world=cal(evt("review", "", "Friday", "2 PM")))
ex("U: Cancel everything tomorrow.", "ambiguous", UNK, None, "safety_high_risk",
   note="Broad, destructive; must be confirmed.", world=cal(evt("call", "", "tomorrow", "10 AM"), evt("review", "", "tomorrow", "3 PM")))

# ------------------------------------------------------------------ HINGLISH (Roman script)
H = "hinglish"
ex("U: Kal subah 10 baje Rahul ko call karne ka reminder laga do.", "actionable", CR, dict(person="Rahul", date="kal", time="10"), "explicit_command", H)
ex("U: Rahul ke saath Friday 3 baje meeting schedule kar do.", "actionable", SM, dict(person="Rahul", date="Friday", time="3"), "explicit_command", H,
   pair="P9", var="command")
ex("U: Priya ko message bhej do ki main 10 minute late aaunga.", "actionable", MSG, dict(person="Priya", message="10 minute late"), "explicit_command", H)
ex("U: Kal 11 baje wali Rahul ki meeting cancel kar do.", "actionable", CM, dict(person="Rahul", time="11"), "correction_cancel_reschedule", H,
   world=cal(evt("meeting with Rahul", "Rahul", "kal", "11")))
ex("U: Friday ki meeting Saturday ko shift kar do, 2 baje.", "actionable", RM, dict(date="Saturday", time="2"), "correction_cancel_reschedule", H,
   world=cal(evt("meeting", "", "Friday", "11")))
ex("U: Mujhe yaad dilana ki Monday ko invoice bhejna hai.", "actionable", CR, dict(title="invoice", date="Monday"), "explicit_command", H)
ex("U: Ek task bana do: client ka proposal aaj shaam tak review karna hai.", "actionable", CT, dict(title="client ka proposal review", date="aaj shaam"), "explicit_command", H)
ex("U: Meri Thursday wali meeting Neha ke saath hai na, usko Friday 5 baje kar do.", "actionable", RM, dict(person="Neha", date="Friday", time="5"), "correction_cancel_reschedule", H,
   world=cal(evt("meeting with Neha", "Neha", "Thursday", "11")))
ex("O: Dinner Tuesday ko 7 baje? || U: Done, pakka Tuesday 7 baje.", "actionable", SM, dict(title="dinner", date="Tuesday", time="7"), "agreement_confirmation", H)
ex("O: Report Wednesday tak bhej dena. || U: Theek hai, bhej dunga.", "actionable", CT, dict(title="report", date="Wednesday"), "implicit_multiturn", H)
ex("U: Monday meeting rakho Anil ke saath 10 baje. || O: Ho gaya. || U: Nahi yaar, Tuesday kar do.", "actionable", RM, dict(person="Anil", date="Tuesday"), "change_of_mind", H,
   world=cal(evt("meeting with Anil", "Anil", "Monday", "10")))
P5c = "O: Kya hum Thursday ko 4 baje project review rakh sakte hain?"
g = ex(P5c + " || U: Haan, Thursday theek hai.", "actionable", SM, dict(title="project review", date="Thursday", time="4"), "agreement_confirmation", H,
       pair="P5", var="accept")
ex(P5c + " || U: Nahi, Thursday mere liye kaam nahi karega.", "non_actionable", NONE, None, "negative", H, pair="P5", var="decline", group=g)
ex(P5c + " || U: Shayad, main dekhta hoon.", "ambiguous", UNK, None, "ambiguous", H, pair="P5", var="hedge", group=g)
ex(P5c, "non_actionable", NONE, None, "negative", H, pair="P5", var="proposal_only", group=g)

g = [e["group"] for e in EX if e["pair"] == "P9"][0]
ex("U: Rahul ke saath ek meeting rakh do.", "ambiguous", UNK, dict(person="Rahul"), "ambiguous", H, pair="P9", var="missing_time", group=g)
ex("U: Rahul ke saath meetings hamesha lambi chalti hain.", "non_actionable", NONE, None, "no_action_mention", H, pair="P9", var="mention", group=g)

ex("U: Main roz subah 7 baje uthta hoon.", "non_actionable", NONE, None, "no_action_mention", H)
ex("U: Thursday meetings ke liye achha din ho sakta hai.", "non_actionable", NONE, None, "no_action_mention", H)
ex("U: Kal Rahul ke saath meeting bahut achhi rahi.", "non_actionable", NONE, None, "no_action_mention", H, note="'kal' = yesterday here (past tense 'rahi').")
ex("U: Rahul hamesha last minute pe meeting cancel kar deta hai.", "non_actionable", NONE, None, "no_action_mention", H)
ex("U: Mujhe Thursday pasand hai kyunki us din schedule free rehta hai.", "non_actionable", NONE, None, "no_action_mention", H)
ex("U: Agar Rahul free hua toh Friday ko dekhenge.", "non_actionable", NONE, None, "negative", H)
ex("U: Client meeting hai kya kal 11 baje? || O: Haan, add kar du calendar mein? || U: Nahi, calendar mein add mat karo, main khud dekh lunga.", "non_actionable", NONE, None, "negative", H)
ex("O: Main tumhe slides Wednesday tak bhej dunga.", "non_actionable", NONE, None, "negative", H)
ex("U: Aaj mausam bahut accha hai.", "non_actionable", NONE, None, "negative", H)
ex("U: Client meeting kal 11 baje hai, thoda tension hai.", "non_actionable", NONE, None, "no_action_mention", H)
ex("U: Kabhi milte hain next week.", "non_actionable", NONE, None, "no_action_mention", H, note="Vague 'sometime' with no who/when.")
ex("U: Shayad mujhe kal client se milna chahiye.", "ambiguous", UNK, dict(person="client", date="kal"), "ambiguous", H)
ex("U: Meeting cancel kar do.", "ambiguous", UNK, None, "ambiguous", H, note="No referent.", world=cal(evt("meeting", "", "kal", "11")))
ex("O: Thursday ya Friday? || U: Dono chalega.", "ambiguous", UNK, None, "ambiguous", H)
ex("O: Kya hum next week milein? || U: Dekhte hain, shayad.", "ambiguous", UNK, None, "ambiguous", H)
ex("U: Usse kal call karne ka reminder laga do.", "ambiguous", UNK, None, "ambiguous", H, note="'usse' unresolved.")
ex("U: Subah meeting rakh do Anil ke saath.", "ambiguous", UNK, dict(person="Anil"), "ambiguous", H, note="Vague time (no exact time).")

# ------------------------------------------------------------------ HINDI (Devanagari)
D = "hindi"
ex("U: कल सुबह 9 बजे मुझे डॉक्टर को कॉल करने की याद दिलाना।", "actionable", CR, dict(title="डॉक्टर को कॉल", date="कल", time="9"), "explicit_command", D)
ex("U: शुक्रवार को 3 बजे राहुल के साथ मीटिंग रख दो।", "actionable", SM, dict(person="राहुल", date="शुक्रवार", time="3"), "explicit_command", D)
ex("U: राहुल की कल वाली मीटिंग रद्द कर दो।", "actionable", CM, dict(person="राहुल"), "correction_cancel_reschedule", D,
   world=cal(evt("राहुल के साथ मीटिंग", "राहुल", "कल", "11")))
P6c = "O: क्या हम गुरुवार को 4 बजे मिल सकते हैं?"
g = ex(P6c + " || U: हाँ, गुरुवार ठीक है।", "actionable", SM, dict(date="गुरुवार", time="4"), "agreement_confirmation", D, pair="P6", var="accept")
ex(P6c + " || U: नहीं, गुरुवार मेरे लिए ठीक नहीं है।", "non_actionable", NONE, None, "negative", D, pair="P6", var="decline", group=g)
ex(P6c + " || U: शायद, मैं देखता हूँ।", "ambiguous", UNK, None, "ambiguous", D, pair="P6", var="hedge", group=g)
ex("U: मैं रोज़ सुबह 7 बजे उठता हूँ।", "non_actionable", NONE, None, "no_action_mention", D)
ex("U: गुरुवार मीटिंग के लिए अच्छा दिन हो सकता है।", "non_actionable", NONE, None, "no_action_mention", D)
ex("U: शायद मुझे कल क्लाइंट से मिलना चाहिए।", "ambiguous", UNK, None, "ambiguous", D)
ex("U: मीटिंग रद्द कर दो।", "ambiguous", UNK, None, "ambiguous", D, note="No referent.", world=cal(evt("मीटिंग", "", "कल", "11")))


def main():
    os.makedirs("data", exist_ok=True)
    for i, e in enumerate(EX, 1):
        e["id"] = f"A{i:03d}"
    with open("data/actionability_v1.jsonl", "w", encoding="utf-8") as f:
        for e in EX:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote", len(EX), "examples")


if __name__ == "__main__":
    main()
