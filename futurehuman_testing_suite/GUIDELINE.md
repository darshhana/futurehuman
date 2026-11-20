# Labeling guideline (v1) — read before filling `data/review_sheet.csv`

You label **the LAST turn** of each conversation, using the earlier turns as context.
**USER** = the person the assistant works for. **OTHER** = another person (or an assistant).

## The three labels

**actionable** — the USER clearly (a) gives a command/request, (b) commits to something themselves ("I'll send it by Friday"),
(c) **accepts a concrete proposal** made earlier ("Thursday works"), or (d) cancels/changes an item that exists —
AND the needed details (what + when / who) are in the conversation or recoverable from it.

**ambiguous** — an action is plausible but the assistant should **ask first**, because
- the commitment is hedged ("maybe", "might", "we could", "I'll try", "let's see"), or
- a needed detail is missing or unresolvable (no time, "him" with no referent, "the meeting" when there are two, "afternoon" with no exact time), or
- the request is broad and destructive ("cancel everything tomorrow").

**non_actionable** — nothing for the assistant to do:
- facts, feelings, opinions, past events ("I met Rahul yesterday")
- hypotheticals / conditionals ("if Thursday suits the others, we'll see")
- the USER **declines** or counter-proposes ("Thursday doesn't work") — negotiation is still open
- statements by OTHER: their proposals, their promises ("I'll send you the slides")
- vague social wishes with no who/when ("we should meet sometime")
- general questions / chit-chat ("what's the capital of France", "tell me a joke")
- the user says *no* to an action ("don't add it to my calendar")

## Tie-breakers
1. An **open proposal is not an assignment** until the USER accepts it.
2. **Nothing to cancel → not actionable**: cancelling something that was never created is non_actionable.
3. Hedge words + a specific who/when → **ambiguous**. Hedge words with no specifics → **non_actionable**.
4. When it is truly 50/50 between actionable and ambiguous, choose **ambiguous** (asking is cheap; a wrong action is not).
5. Hindi/Hinglish **"kal"/"parso"** can mean yesterday/tomorrow and day-before-yesterday/day-after-tomorrow. If the tense doesn't settle it, treat the date as unresolved (→ ambiguous when the action depends on it).

## How to fill the sheet
- Write exactly one of: `actionable`, `non_actionable`, `ambiguous`.
- Do **not** look at any other file with labels while you do this (that is the point of a blind review).
- Use the `comment` column when you were unsure — those rows are where the guideline needs fixing.
