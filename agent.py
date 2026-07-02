"""
Agent core logic.

Per-turn flow:
  1. extract_context()  →  slots + intent in ONE LLM call (uses FULL history)
  2. Python-level overrides (turn cap, prior-shortlist detection)
  3. Route to handler
  4. Handler retrieves catalog data and generates grounded reply (truncated history for speed)
  5. Validate all URLs before returning
"""

import json
import os
import re

from groq import Groq
from prompts import (
    COMBINED_EXTRACTION_PROMPT,
    CLARIFY_PROMPT,
    RECOMMEND_PROMPT,
    REFINE_PROMPT,
    COMPARE_PROMPT,
    REFUSE_PROMPT,
    EOC_CHECK_PROMPT,
)
from retrieval import CatalogRetriever
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

VALID_INTENTS = {"clarify", "recommend", "refine", "compare", "refuse", "confirm"}


def _llm(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    resp = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _conv_text(messages: list[dict], max_turns: int = None) -> str:
    """
    Format conversation history as text.
    max_turns=None → full history (use for extraction so no slot is lost).
    max_turns=N    → last N user+assistant pairs (use for reply generation to save tokens).
    """
    if max_turns is not None:
        messages = messages[-(max_turns * 2):]
    lines = []
    for m in messages:
        label = "User" if m["role"] == "user" else "Agent"
        lines.append(f"{label}: {m['content']}")
    return "\n".join(lines)


def _slots_summary(slots: dict) -> str:
    pairs = []
    skip = {"has_enough_context", "intent", "compare_names"}
    for k, v in slots.items():
        if k in skip or not v:
            continue
        pairs.append(f"{k}={json.dumps(v)}")
    return ", ".join(pairs) if pairs else "none yet"


def _strip_json(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()


def _parse_slots(raw: str) -> dict:
    defaults = {
        "role": None,
        "seniority": None,
        "purpose": None,
        "skills": [],
        "test_types_wanted": [],
        "test_types_excluded": [],
        "duration_max_minutes": None,
        "remote": None,
        "has_enough_context": False,
        "compare_names": [],
        "intent": "clarify",
    }
    try:
        parsed = json.loads(_strip_json(raw))
        if parsed.get("intent") not in VALID_INTENTS:
            parsed["intent"] = "clarify"
        return {**defaults, **parsed}
    except (json.JSONDecodeError, TypeError):
        return defaults


def _build_query(slots: dict) -> str:
    parts = []
    role = slots.get("role")
    seniority = slots.get("seniority")
    purpose = slots.get("purpose")
    skills = slots.get("skills") or []

    if role:
        parts.append(str(role))

    if seniority in ["director", "executive"] or "leadership" in str(role).lower():
        parts.append("leadership personality benchmark executive assessment")

    if skills:
        parts.extend(str(s) for s in skills)
    if purpose:
        parts.append(str(purpose))

    return " ".join(parts).strip() or "assessment"


def _has_prior_shortlist(messages: list[dict]) -> bool:
    """
    Check whether a previous agent turn already returned a recommendation shortlist.
    Tightened to only match URLs or explicit shortlist language — NOT generic 'assessment' mentions.
    """
    for m in messages[:-1]:
        if m["role"] == "assistant" and any(
            kw in m["content"].lower()
            for kw in ["shl.com", "shortlist", "here are", "full shortlist", "battery"]
        ):
            return True
    return False


def _count_assistant_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


def _check_eoc(last_user_msg: str) -> bool:
    signals = [
        "perfect", "that's what we need", "great", "thanks", "thank you",
        "looks good", "that works", "exactly", "done", "all set", "got it",
        "confirmed", "locking", "lock it in", "that covers it",
    ]
    lower = last_user_msg.lower()
    if any(s in lower for s in signals):
        return True
    # Only call LLM for short ambiguous messages
    if len(last_user_msg.split()) <= 10:
        try:
            raw = _llm(EOC_CHECK_PROMPT.format(last_message=last_user_msg), max_tokens=5)
            return "yes" in raw.lower()
        except Exception:
            pass
    return False


# ------------------------------------------------------------------ #
#  Intent handlers                                                     #
# ------------------------------------------------------------------ #

def _handle_refuse(messages: list[dict]) -> dict:
    reply = _llm(REFUSE_PROMPT.format(conversation=_conv_text(messages, max_turns=3)), max_tokens=120)
    return {"reply": reply, "recommendations": [], "end_of_conversation": False}


def _handle_clarify(messages: list[dict], slots: dict, retriever: CatalogRetriever) -> dict:
    # Only fetch catalog hints after at least 1 exchange (not on first message)
    if _count_assistant_turns(messages) >= 1:
        query = _build_query(slots)
        matches = retriever.search(query, top_k=2)
        catalog_hints = "\n".join(retriever.format_for_context(m) for m in matches) if matches else "None"
    else:
        catalog_hints = "None"

    reply = _llm(
        CLARIFY_PROMPT.format(
            conversation=_conv_text(messages, max_turns=4),
            slots_summary=_slots_summary(slots),
            preliminary_matches=catalog_hints,
        ),
        max_tokens=200,
    )
    return {"reply": reply, "recommendations": [], "end_of_conversation": False}


def _handle_compare(messages: list[dict], slots: dict, retriever: CatalogRetriever) -> dict:
    names = list(slots.get("compare_names") or [])

    # Fallback: scan last user message for catalog item names
    if not names:
        last_msg = messages[-1]["content"].lower()
        for item in retriever.catalog:
            name_lower = item["name"].lower()
            if len(name_lower) > 4 and name_lower in last_msg:
                names.append(item["name"])

    catalog_data_parts = []
    found_items = []
    for name in names[:3]:
        item = retriever.get_by_name(name)
        if item:
            catalog_data_parts.append(retriever.format_for_context(item))
            found_items.append(item)

    if not catalog_data_parts:
        # Named items not found — fall back to recommend
        return _handle_recommend(messages, slots, retriever, intent="recommend")

    reply = _llm(
        COMPARE_PROMPT.format(
            conversation=_conv_text(messages, max_turns=4),
            catalog_data="\n".join(catalog_data_parts),
        ),
        max_tokens=400,
    )

    # KEY FIX: If a prior shortlist exists, carry it forward alongside the compare reply.
    # This matches C5 T2 behaviour (compare answer shown + prior recs maintained).
    # If no prior shortlist, return [] (C6 T2 behaviour).
    if _has_prior_shortlist(messages):
        prior = _handle_recommend(messages, slots, retriever, intent="refine")
        return {"reply": reply, "recommendations": prior["recommendations"], "end_of_conversation": False}

    return {"reply": reply, "recommendations": [], "end_of_conversation": False}


def _handle_recommend(
    messages: list[dict],
    slots: dict,
    retriever: CatalogRetriever,
    intent: str = "recommend",
) -> dict:
    query = _build_query(slots)
    filters = {
        "test_types_wanted": slots.get("test_types_wanted") or [],
        "test_types_excluded": slots.get("test_types_excluded") or [],
        "remote": slots.get("remote"),
        "seniority": slots.get("seniority"),
    }

    matches = retriever.search(query, top_k=10, filters=filters)

    # Pin OPQ32r for executive/director/leadership roles if not already in results
    seniority = slots.get("seniority") or ""
    role_str = str(slots.get("role") or "").lower()
    if seniority in ["executive", "director"] or "leadership" in role_str:
        opq = retriever.get_by_name("Occupational Personality Questionnaire OPQ32r")
        if opq and opq not in matches:
            matches.insert(0, opq)
            matches = matches[:10]

    catalog_matches_text = "\n".join(retriever.format_for_context(m) for m in matches)

    prompt_template = REFINE_PROMPT if intent == "refine" else RECOMMEND_PROMPT
    reply = _llm(
        prompt_template.format(
            conversation=_conv_text(messages, max_turns=4),
            slots_summary=_slots_summary(slots),
            catalog_matches=catalog_matches_text,
        ),
        max_tokens=350,
    )

    recs = [
        r for r in (retriever.to_recommendation(m) for m in matches)
        if retriever.validate_url(r["url"])
    ]

    # EOC check only makes sense for non-refine intents (refine = user just changed something)
    last_user_msg = messages[-1]["content"]
    eoc = _check_eoc(last_user_msg) if intent != "refine" else False

    return {"reply": reply, "recommendations": recs, "end_of_conversation": eoc}


# ------------------------------------------------------------------ #
#  Main entry point                                                    #
# ------------------------------------------------------------------ #

def process_chat(messages: list[dict], retriever: CatalogRetriever) -> dict:
    turn_count = len(messages)  # total individual messages (user + assistant)

    # BUG FIX 1: Use FULL history for extraction — never truncate here.
    # Truncation in _conv_text was causing early slot values (e.g. JD skills from turn 1 in C9)
    # to be invisible to the extractor on later turns.
    raw = _llm(
        COMBINED_EXTRACTION_PROMPT.format(conversation=_conv_text(messages)),
        max_tokens=500,
    )
    slots = _parse_slots(raw)
    intent = slots["intent"]

    # Refuse is absolute — no overrides
    if intent == "refuse":
        return _handle_refuse(messages)

    # ---------------------------------------------------------------- #
    #  Python-level overrides (more reliable than LLM classification)   #
    # ---------------------------------------------------------------- #

    assistant_turns = _count_assistant_turns(messages)

    # BUG FIX 2: Force recommend after 2 complete exchanges if we have any role/seniority.
    # This fixes C1 T3: leadership + executive + selection was STILL returning clarify.
    # The LLM kept being overly cautious; Python override is deterministic.
    if (
        intent == "clarify"
        and assistant_turns >= 2
        and (slots.get("role") or slots.get("seniority"))
    ):
        intent = "recommend"

    # BUG FIX 3: Turn cap — spec says 8 turns max (user + assistant = 8 messages).
    # Force recommend at turn_count=7 (the 4th user message) so we always land before the cap.
    # Previous threshold of 9 was too late.
    if turn_count >= 7 and intent == "clarify":
        intent = "recommend"

    # Hard EOC force at message 13 (consistent with original logic for long traces)
    if turn_count >= 13:
        result = _handle_recommend(messages, slots, retriever, intent="refine")
        result["end_of_conversation"] = True
        return result

    # BUG FIX 4: Detect prior shortlist BEFORE routing recommend → refine.
    # Previous code did this only inside recommend handler, missing the confirm path.
    has_shortlist = _has_prior_shortlist(messages)
    if intent == "recommend" and has_shortlist:
        intent = "refine"

    # Route
    if intent == "compare":
        return _handle_compare(messages, slots, retriever)

    if intent == "clarify":
        return _handle_clarify(messages, slots, retriever)

    # BUG FIX 5: confirm should use "refine" not "recommend" so slots are respected
    # and the same shortlist is returned, not a fresh retrieval that might differ.
    if intent == "confirm":
        result = _handle_recommend(messages, slots, retriever, intent="refine")
        result["end_of_conversation"] = True
        return result

    return _handle_recommend(messages, slots, retriever, intent=intent)