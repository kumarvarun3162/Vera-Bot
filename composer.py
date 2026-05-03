"""
composer.py — Groq-powered message composition for Vera-Bot.
Uses llama-3.3-70b-versatile (fast + capable) via Groq API.
Temperature = 0 for determinism as required by the challenge.
"""

import os
import json
import logging
import re
from typing import Optional

from groq import Groq

logger = logging.getLogger("vera-bot.composer")

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — THE BRAIN OF VERA
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages for Indian merchants to help them grow their Google Business Profile and run marketing.

## STRICT RULES (violating any = point loss)

1. **SPECIFICITY**: Always anchor on a concrete, verifiable fact — a number, date, research stat, peer stat, or source. "Haircut @ ₹99" beats "great discount". "2,100-patient trial shows 38% better outcome" beats "research shows".

2. **CATEGORY VOICE** (match exactly):
   - dentists/pharmacies: peer-clinical, technical vocab OK, FORBIDDEN: "cure", "guaranteed", "100% safe"
   - salons/gyms: aspirational but grounded, service+price format
   - restaurants: local + food-specific, conversational

3. **MERCHANT FIT**: Use the merchant's actual name, their actual numbers (views/calls/CTR), their actual active offers with real prices. Never generic.

4. **TRIGGER RELEVANCE**: The message must answer "why NOW?" — reference the specific event that triggered it.

5. **SINGLE CTA LAST**: One call-to-action at the very end only.
   - Action triggers → binary: "Reply YES" / "Reply STOP"
   - Information triggers → open-ended question
   - Pure FYI → no CTA

6. **LANGUAGE**: Match the merchant's language preference. Hindi-English code-mix is preferred for Indian merchants (e.g., "Aapka profile 62% complete hai — description missing hai. Chalega update?")

7. **NO PREAMBLES**: Never start with "I hope you're doing well" or "I'm reaching out today". Start with substance.

8. **NO FABRICATION**: Only use data present in the given context. No fake citations, no invented competitor names, no made-up statistics.

9. **NO URLS**: Never include http/https links — they get rejected.

10. **CONCISE**: 2-4 sentences for information triggers. Up to 6 for action triggers.

## COMPULSION LEVERS (use 1-2 per message for engagement):
- **Specificity**: "Your CTR is 2.1% vs peer median 3.0% — gap of 0.9%"
- **Loss aversion**: "78 patients haven't returned in 6 months — they're slipping away"
- **Social proof**: "3 dentists in your locality ran this campaign this month"
- **Effort externalization**: "I've already drafted it — just say GO"
- **Curiosity**: "Want to see exactly who's lapsing?"
- **Single binary ask**: "Reply YES / STOP"

## OUTPUT FORMAT
Return a JSON object — ONLY the JSON, no markdown fences, no extra text:
{
  "body": "<the WhatsApp message text>",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "none",
  "template_name": "<vera_[trigger_kind]_v1>",
  "template_params": ["<merchant_name>", "<key_stat_or_hook>", "<cta_text>"],
  "rationale": "<1-2 sentences: why this message, which compulsion levers used>"
}"""

# ──────────────────────────────────────────────────────────────────────────────
# TRIGGER-KIND ROUTING (specialized guidance per trigger type)
# ──────────────────────────────────────────────────────────────────────────────
TRIGGER_GUIDANCE = {
    "research_digest": (
        "Lead with the specific research finding stat + source. Explain why it matters for THIS merchant's "
        "patient/customer cohort using their actual signals. Offer to draft shareable patient content. "
        "CTA: open-ended (want me to pull the abstract + draft a post?)."
    ),
    "recall_due": (
        "Send on behalf of the merchant to their customer. Use customer's name, name the specific service, "
        "state time since last visit, offer 2 specific time slots matching customer's preference, include the "
        "real price from merchant's active offer catalog. CTA: numbered slot choice (Reply 1 for X, 2 for Y)."
    ),
    "perf_spike": (
        "Lead with the exact spike number (+X% views/calls). Frame as great news. Build momentum — offer to "
        "draft a post to capture the surge or run a timed offer. CTA: binary YES/STOP."
    ),
    "perf_dip": (
        "Lead with the specific dip stat. Frame as early warning (not alarm). Offer one concrete action to recover "
        "(refresh offer, add a post, update GBP hours). Use loss-aversion but stay constructive. CTA: binary YES/STOP."
    ),
    "milestone_reached": (
        "One sentence of genuine celebration with the specific milestone number. Immediately pivot to a next-step "
        "action that builds on the momentum. Keep it short and energetic. CTA: binary YES/STOP."
    ),
    "dormant_with_vera": (
        "Re-engage a merchant who hasn't replied in 14+ days. Use curiosity + a new piece of information (peer stat "
        "or trend) to earn attention. Don't re-introduce yourself. Short hook, low-friction ask. CTA: open-ended."
    ),
    "festival_upcoming": (
        "Name the festival + exact days away. Offer a category-appropriate campaign — use service+price format "
        "not generic % off. Mention effort externalization (I've drafted it). CTA: binary YES/STOP."
    ),
    "competitor_opened": (
        "Frame defensively — a new competitor is nearby. Use loss-aversion (they're targeting your locality). "
        "Offer a differentiating action (highlight unique service, add a post, refresh offer). No competitor name "
        "unless it's in the context. CTA: binary YES/STOP."
    ),
    "review_theme_emerged": (
        "Name the specific review theme (e.g., 'wait time mentioned 3 times this month'). Frame as actionable intelligence "
        "— offer to draft a response post or add a GBP note. CTA: open-ended."
    ),
    "customer_lapsed_soft": (
        "Send on behalf of merchant to lapsed customer. Name the customer, time since last visit, "
        "last service received. Offer a specific slot + price incentive. Hi-en mix if language pref says hi."
    ),
    "customer_lapsed_hard": (
        "Win-back message on behalf of merchant to long-lapsed customer. Acknowledge the gap warmly, "
        "offer a strong specific incentive (price + service), easy reply CTA."
    ),
    "appointment_tomorrow": (
        "Send appointment reminder on behalf of merchant to customer. Name customer, service, exact time. "
        "Include clinic address or landmark. Friendly, short. No CTA needed (or 'Reply 1 to confirm')."
    ),
    "gbp_unverified": (
        "Alert merchant their Google Business Profile is unverified — hurting visibility. Give the specific impact "
        "(e.g., 'unverified profiles show 40% fewer in Maps results'). Offer to guide verification. CTA: binary YES."
    ),
    "regulation_change": (
        "Alert merchant to a specific regulatory change affecting their category. Name the authority + effective date. "
        "Explain the practical impact on their business. Offer to help them comply. CTA: open-ended."
    ),
    "renewal_due": (
        "Alert merchant their subscription is expiring. Give exact days_remaining. Frame as loss-aversion "
        "(what they'll lose: visibility, leads). CTA: binary YES to renew."
    ),
    "category_seasonal": (
        "Connect a seasonal trend to this merchant's specific situation (their locality, their offers). "
        "Offer a seasonal campaign in service+price format. CTA: binary YES/STOP."
    ),
    "ipl_match_today": (
        "Tie the IPL match (team + timing) to a food/beverage offer if it's a restaurant. Suggest a "
        "timed combo deal. Effort externalization — I've drafted the post. CTA: binary YES/STOP."
    ),
    "curious_ask_due": (
        "Ask a single, genuinely interesting question that makes the merchant think about their business. "
        "Frame it as knowledge exchange — Vera shares a peer insight first, then asks. No hard CTA. "
        "Example: '3 dentists in your area added clear-aligner consultations last month — is that something you offer?'"
    ),
    "cde_opportunity": (
        "Mention a continuing education / professional development opportunity in their field. "
        "Be specific about the topic, date, and organizer. Offer to register or share details. CTA: open-ended."
    ),
    "supply_alert": (
        "Alert pharmacy/health merchant to a supply disruption for a key product. Be specific about "
        "which product and the timeline. Suggest action (stock up, note to customers). CTA: open-ended."
    ),
    "chronic_refill_due": (
        "Pharmacy trigger — customer is due for chronic medication refill. On behalf of merchant, "
        "remind customer, offer easy reorder path. Sensitive tone, factual, no clinical overclaims."
    ),
    "winback_eligible": (
        "Merchant has a lapsed subscription or inactive account — offer re-engagement with a specific "
        "value prop (what's improved since they left). CTA: binary YES."
    ),
    "active_planning_intent": (
        "Merchant explicitly said they want to plan something. Pick up exactly where they left off — "
        "reference their last message. Draft a specific plan or next step immediately. "
        "NO qualifying questions. They said yes — execute. CTA: confirm/cancel."
    ),
    "trial_followup": (
        "Follow up on a trial period or campaign that recently ran. Share the specific results "
        "(views, clicks, leads generated). Ask if they want to continue or upgrade. CTA: binary YES."
    ),
    "wedding_package_followup": (
        "Follow up on a wedding package inquiry or campaign. Name the specific package discussed. "
        "Create urgency (wedding season slots filling). CTA: binary YES/STOP."
    ),
    "seasonal_perf_dip": (
        "Acknowledge the seasonal dip is normal but act now to minimize it. Give a peer comparison "
        "(X% of similar merchants ran a seasonal offer last month). Offer a specific seasonal campaign. CTA: binary YES."
    ),
}


def _get_trigger_guidance(kind: str) -> str:
    return TRIGGER_GUIDANCE.get(kind, (
        "Compose a message relevant to this trigger kind. Lead with the most specific data point "
        "from the context. One compulsion lever. Single CTA at the end."
    ))


def _extract_json(text: str) -> dict:
    """Safely extract JSON from LLM response."""
    # Strip markdown fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    
    # Find first { to last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw text: {text[:300]}")
        return {}


def _build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> str:
    """Build the user prompt for initial message composition."""

    # Extract the most relevant pieces — don't dump the whole dict to save tokens
    merchant_name = merchant.get("identity", {}).get("name", "Merchant")
    owner_name = merchant.get("identity", {}).get("owner_first_name", merchant_name)
    city = merchant.get("identity", {}).get("city", "")
    locality = merchant.get("identity", {}).get("locality", "")
    languages = merchant.get("identity", {}).get("languages", ["en"])
    lang_pref = "Hindi-English code-mix" if "hi" in languages else "English"

    perf = merchant.get("performance", {})
    ctr = perf.get("ctr", 0)
    views = perf.get("views", 0)
    calls = perf.get("calls", 0)
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0)

    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})

    # Resolve digest item if trigger references one
    digest_ref = ""
    if "top_item_id" in trg_payload:
        item_id = trg_payload["top_item_id"]
        for d in category.get("digest", []):
            if d.get("id") == item_id:
                digest_ref = f"Digest item: {d.get('title')} — Source: {d.get('source', '')}. Trial N: {d.get('trial_n', '')}. Patient segment: {d.get('patient_segment', '')}. Summary: {d.get('summary', '')}"
                break

    # Customer context for customer-facing triggers
    customer_section = ""
    if customer:
        cust_name = customer.get("identity", {}).get("name", "Customer")
        cust_lang = customer.get("identity", {}).get("language_pref", lang_pref)
        rel = customer.get("relationship", {})
        state = customer.get("state", "")
        prefs = customer.get("preferences", {})
        services = rel.get("services_received", [])
        last_visit = rel.get("last_visit", "")
        customer_section = f"""
## CUSTOMER CONTEXT (send on behalf of merchant to this customer)
- Name: {cust_name}
- Language preference: {cust_lang}
- Last visit: {last_visit}
- Total visits: {rel.get("visits_total", 0)}
- Services received: {', '.join(services)}
- State: {state}
- Preferred slot: {prefs.get("preferred_slots", "any")}
- Consent scope: {customer.get("consent", {}).get("scope", [])}
"""

    # Seasonal + trend context
    season_notes = "; ".join([f"{s['month_range']}: {s['note']}" for s in category.get("seasonal_beats", [])[:2]])
    trend_notes = "; ".join([f"{t['query']}: +{int(t.get('delta_yoy',0)*100)}% YoY" for t in category.get("trend_signals", [])[:2]])

    prompt = f"""## TASK
Compose a WhatsApp message for this exact trigger. Follow all rules in your system prompt.

## TRIGGER
- Kind: {trg_kind}
- Source: {trigger.get("source", "")}
- Urgency: {trigger.get("urgency", 2)}/5
- Suppression key: {trigger.get("suppression_key", "")}
- Payload: {json.dumps(trg_payload)}
{f"- {digest_ref}" if digest_ref else ""}

## TRIGGER GUIDANCE (for kind="{trg_kind}")
{_get_trigger_guidance(trg_kind)}

## MERCHANT CONTEXT
- Name: {merchant_name} ({owner_name})
- City/Locality: {city}, {locality}
- Category: {category.get("slug", "")} ({category.get("display_name", "")})
- Language preference: {lang_pref}
- Subscription: {merchant.get("subscription", {}).get("status", "")} plan={merchant.get("subscription", {}).get("plan", "")} days_remaining={merchant.get("subscription", {}).get("days_remaining", "")}
- Performance (30d): views={views}, calls={calls}, CTR={ctr:.3f} | Peer median CTR={peer_ctr:.3f}
- 7-day delta: views {perf.get("delta_7d", {}).get("views_pct", 0)*100:+.0f}%, calls {perf.get("delta_7d", {}).get("calls_pct", 0)*100:+.0f}%
- Active offers: {', '.join(active_offers) if active_offers else "none"}
- Signals: {', '.join(signals)}
- Customer aggregate: total_ytd={cust_agg.get("total_unique_ytd", 0)}, lapsed_180d={cust_agg.get("lapsed_180d_plus", 0)}, retention_6mo={cust_agg.get("retention_6mo_pct", 0):.0%}
- Recent conversation history (last 2 turns): {json.dumps(merchant.get("conversation_history", [])[-2:])}
- Review themes: {json.dumps(merchant.get("review_themes", [])[:2])}

## CATEGORY CONTEXT
- Voice: {json.dumps(category.get("voice", {}))}
- Top offers in catalog: {', '.join([o.get("title","") for o in category.get("offer_catalog", [])[:3]])}
- Peer stats: {json.dumps(category.get("peer_stats", {}))}
- Seasonal beats: {season_notes}
- Trend signals: {trend_notes}
{customer_section}

Compose the message now. Return only the JSON object."""

    return prompt


def _build_reply_prompt(
    category: Optional[dict],
    merchant: Optional[dict],
    trigger: Optional[dict],
    customer: Optional[dict],
    conversation_turns: list,
    latest_message: str,
    is_intent_transition: bool,
) -> str:
    """Build the user prompt for replying to a merchant/customer message."""

    merchant_name = merchant.get("identity", {}).get("name", "Merchant") if merchant else "Merchant"
    languages = merchant.get("identity", {}).get("languages", ["en"]) if merchant else ["en"]
    lang_pref = "Hindi-English code-mix" if "hi" in languages else "English"
    
    # Format conversation history
    history_str = ""
    for t in conversation_turns[-6:]:  # last 6 turns
        role = t.get("role", "?")
        body = t.get("body", "")
        history_str += f"[{role.upper()}]: {body}\n"

    trg_kind = trigger.get("kind", "") if trigger else ""
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"] if merchant else []

    intent_note = ""
    if is_intent_transition:
        intent_note = """
⚠️ INTENT TRANSITION DETECTED: The merchant just said "let's do it" / "go ahead" / confirmed intent.
DO NOT ask another qualifying question. Switch immediately to action mode.
Draft the specific next step, name concrete deliverable, give a CONFIRM/CANCEL CTA."""

    prompt = f"""## TASK
Generate the bot's next reply in this ongoing conversation.

## CONVERSATION HISTORY
{history_str.strip()}

## MERCHANT'S LATEST MESSAGE
"{latest_message}"
{intent_note}

## MERCHANT CONTEXT
- Name: {merchant_name}
- Language: {lang_pref}
- Category: {category.get("slug", "") if category else ""}
- Active offers: {', '.join(active_offers) if active_offers else "none"}
- Trigger kind: {trg_kind}

## YOUR REPLY OPTIONS
Return one of three action types:
1. {{"action": "send", "body": "...", "cta": "...", "rationale": "..."}} — send a reply
2. {{"action": "wait", "wait_seconds": <int>, "rationale": "..."}} — back off
3. {{"action": "end", "rationale": "..."}} — gracefully close conversation

Use "end" if: merchant says stop/not interested/bye, or repeated auto-replies.
Use "wait" if: merchant needs time or asked to be contacted later.
Use "send" for everything else — continue the conversation usefully.

Return only the JSON object."""

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    existing_conversations: Optional[dict] = None,
) -> dict:
    """
    Compose an outbound message from Vera to a merchant (or from merchant to customer).
    Returns: {body, cta, template_name, template_params, rationale}
    """
    prompt = _build_compose_prompt(category, merchant, trigger, customer)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # deterministic as required
            max_tokens=600,
        )
        raw = response.choices[0].message.content
        result = _extract_json(raw)

        if not result.get("body"):
            logger.warning("Empty body from LLM — using fallback")
            return _fallback_compose(merchant, trigger)

        return result

    except Exception as e:
        logger.error(f"Groq API error in compose_message: {e}")
        return _fallback_compose(merchant, trigger)


def compose_reply(
    category: Optional[dict],
    merchant: Optional[dict],
    trigger: Optional[dict],
    customer: Optional[dict],
    conversation_turns: list,
    latest_message: str,
    is_intent_transition: bool = False,
) -> dict:
    """
    Compose a reply to a merchant/customer message.
    Returns: {action, body?, cta?, wait_seconds?, rationale}
    """
    prompt = _build_reply_prompt(
        category, merchant, trigger, customer,
        conversation_turns, latest_message, is_intent_transition
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content
        result = _extract_json(raw)

        if not result.get("action"):
            result["action"] = "send"
        if result["action"] == "send" and not result.get("body"):
            result["body"] = "Noted! Thoda sa time do — main draft kar raha hoon."
            result["cta"] = "none"

        return result

    except Exception as e:
        logger.error(f"Groq API error in compose_reply: {e}")
        return {
            "action": "send",
            "body": "Samajh gaya! Main abhi kaam shuru karta hoon — ek minute.",
            "cta": "none",
            "rationale": "Fallback reply due to API error.",
        }


def _fallback_compose(merchant: dict, trigger: dict) -> dict:
    """Deterministic fallback if LLM fails."""
    name = merchant.get("identity", {}).get("owner_first_name", "")
    kind = trigger.get("kind", "update")
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str = active_offers[0] if active_offers else "aapki services"

    body = f"{name}, aapke account mein ek important update hai regarding {kind.replace('_', ' ')}. {offer_str} ke baare mein baat karein? Reply YES."

    return {
        "body": body,
        "cta": "binary_yes_no",
        "template_name": f"vera_{kind}_v1",
        "template_params": [name, kind, "YES"],
        "rationale": f"Fallback message for trigger kind={kind}.",
    }
