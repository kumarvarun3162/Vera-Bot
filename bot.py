"""
bot.py — Vera-Bot: magicpin AI Challenge
FastAPI server implementing all 5 required endpoints.

Run:  uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from composer import compose_message, compose_reply

load_dotenv()

# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vera-bot")

app = FastAPI(title="Vera-Bot", version="1.0.0", description="magicpin AI Challenge Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()

# ──────────────────────────────────────────────
# IN-MEMORY STATE
# ──────────────────────────────────────────────
# (scope, context_id) → {version: int, payload: dict}
contexts: dict[tuple, dict] = {}

# conversation_id → {merchant_id, customer_id, trigger_id, turns[], ended, auto_reply_streak}
conversations: dict[str, dict] = {}

# suppression_key → True  (prevents duplicate sends)
sent_suppression_keys: set[str] = set()

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# ──────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_ctx(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None

def merchant_and_category(merchant_id: str):
    merchant = get_ctx("merchant", merchant_id)
    if not merchant:
        return None, None
    category = get_ctx("category", merchant.get("category_slug", ""))
    return merchant, category

def count_contexts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts

# ── Auto-reply detection ──────────────────────
AUTO_REPLY_PATTERNS = [
    "thank you for contacting", "thanks for contacting",
    "our team will respond", "will get back to you",
    "i am currently unavailable", "i'm currently unavailable",
    "this is an automated", "automated response", "auto-reply",
    "out of office", "i am away", "i'm away",
    "aapki jaankari ke liye bahut-bahut shukriya",
    "main ek automated assistant hoon",
    "main aapki yeh sabhi baatein aur sujhaav hamari team",
]
def is_auto_reply(msg: str) -> bool:
    m = msg.lower()
    return any(p in m for p in AUTO_REPLY_PATTERNS)

# ── Opt-out detection ─────────────────────────
OPT_OUT_PATTERNS = [
    "stop messaging", "stop sending", "don't message", "dont message",
    "not interested", "unsubscribe", "do not contact", "remove me",
    "leave me alone", "band karo", "mat bhejo", "zaroorat nahi",
    "why are you bothering", "useless", "spam", "block",
]
def is_opt_out(msg: str) -> bool:
    m = msg.lower()
    return any(p in m for p in OPT_OUT_PATTERNS)

# ── Intent-transition detection ───────────────
INTENT_PATTERNS = [
    "let's do it", "lets do it", "ok let's", "ok lets",
    "go ahead", "yes proceed", "yes, proceed", "proceed",
    "haan chalte", "chalo shuru", "kar do", "karo",
    "confirm", "i'm in", "im in", "sign me up",
    "what's next", "whats next", "next steps",
    "ok do it", "please do it", "yes do it",
]
def is_intent_transition(msg: str) -> bool:
    m = msg.lower().strip()
    return any(p in m for p in INTENT_PATTERNS)

# ── Consecutive auto-reply count ──────────────
def auto_reply_streak(conv: dict) -> int:
    count = 0
    for t in reversed(conv.get("turns", [])):
        if t.get("role") == "merchant" and t.get("is_auto_reply"):
            count += 1
        else:
            break
    return count

# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": count_contexts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Vera-Pro"),
        "team_members": [os.environ.get("TEAM_MEMBER", "Submission")],
        "model": "llama-3.3-70b-versatile via Groq",
        "approach": (
            "4-context composer with trigger-kind routing. "
            "Auto-reply detection (3-strike rule), intent-transition handler, "
            "graceful opt-out exits. Hindi-English code-mix, specificity-first composition."
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", ""),
        "version": "1.0.0",
        "submitted_at": utcnow(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    # Validate scope
    if body.scope not in VALID_SCOPES:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(VALID_SCOPES)}",
        }

    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    # Idempotent: same version = already have it
    if cur and cur["version"] == body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    # Stale: we have a newer version
    if cur and cur["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    # Accept: new or version bump
    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"[CONTEXT] stored {body.scope}/{body.context_id} v{body.version}")

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": utcnow(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        # Load trigger
        trigger = get_ctx("trigger", trg_id)
        if not trigger:
            logger.warning(f"[TICK] trigger {trg_id} not in store — skipping")
            continue

        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue

        # Suppression check
        sup_key = trigger.get("suppression_key", "")
        if sup_key and sup_key in sent_suppression_keys:
            logger.info(f"[TICK] suppressed {trg_id} (key={sup_key})")
            continue

        # Load merchant + category
        merchant, category = merchant_and_category(merchant_id)
        if not merchant or not category:
            logger.warning(f"[TICK] missing merchant/category for {merchant_id}")
            continue

        # Load customer (optional)
        customer_id = trigger.get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None

        # Don't re-open an ended conversation
        conv_id = f"conv_{merchant_id}_{trg_id}"
        existing = conversations.get(conv_id)
        if existing and existing.get("ended"):
            continue

        # Compose the message
        logger.info(f"[TICK] composing for merchant={merchant_id} trigger={trg_id}")
        try:
            result = compose_message(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
            )
        except Exception as e:
            logger.error(f"[TICK] compose failed: {e}")
            continue

        if not result or not result.get("body"):
            continue

        # Mark suppressed
        if sup_key:
            sent_suppression_keys.add(sup_key)

        # Init conversation state
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "turns": [{"role": "vera", "body": result["body"], "ts": body.now}],
            "ended": False,
        }

        send_as = "merchant_on_behalf" if customer_id else "vera"
        m_name = merchant.get("identity", {}).get("name", "")

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": result.get("template_name", f"vera_{trigger.get('kind','generic')}_v1"),
            "template_params": result.get("template_params", [m_name, result["body"][:60], "YES"]),
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": sup_key,
            "rationale": result.get("rationale", ""),
        })

        # Hard cap: 10 actions per tick (avoid overload)
        if len(actions) >= 10:
            break

    logger.info(f"[TICK] now={body.now} triggers_checked={len(body.available_triggers)} actions={len(actions)}")
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id

    # Ensure conversation exists
    if conv_id not in conversations:
        conversations[conv_id] = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "trigger_id": None,
            "turns": [],
            "ended": False,
        }
    conv = conversations[conv_id]

    # If already ended, don't respond
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already closed."}

    # Detect message type
    auto = is_auto_reply(body.message)
    opt_out = is_opt_out(body.message)
    intent = is_intent_transition(body.message)

    # Record incoming turn
    conv["turns"].append({
        "role": body.from_role,
        "body": body.message,
        "ts": body.received_at,
        "turn_number": body.turn_number,
        "is_auto_reply": auto,
    })

    # ── HARD OPT-OUT ─────────────────────────
    if opt_out:
        conv["ended"] = True
        logger.info(f"[REPLY] {conv_id} opted out — ending")
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Conversation closed; suppressing future outreach.",
        }

    # ── AUTO-REPLY HANDLING ───────────────────
    if auto:
        streak = auto_reply_streak(conv)
        logger.info(f"[REPLY] {conv_id} auto-reply streak={streak}")

        if streak == 1:
            # First auto-reply: one gentle probe for the owner
            reply_body = "Looks like an auto-reply 😊 Jab owner/manager dekhen, please reply 'YES' to continue where we left off."
            conv["turns"].append({"role": "vera", "body": reply_body, "ts": body.received_at})
            return {
                "action": "send",
                "body": reply_body,
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected. One nudge to reach the owner before backing off.",
            }
        elif streak == 2:
            # Second: back off 4 hours
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Auto-reply for the 2nd time. Owner not at phone. Backing off 4 hours.",
            }
        else:
            # Third+: end gracefully
            conv["ended"] = True
            return {
                "action": "end",
                "rationale": "Auto-reply 3+ consecutive times. No real engagement. Closing gracefully.",
            }

    # ── NORMAL REPLY (compose with LLM) ──────
    merchant_id = conv.get("merchant_id") or body.merchant_id
    customer_id = conv.get("customer_id") or body.customer_id
    merchant, category = merchant_and_category(merchant_id) if merchant_id else (None, None)
    customer = get_ctx("customer", customer_id) if customer_id else None
    trigger_id = conv.get("trigger_id")
    trigger = get_ctx("trigger", trigger_id) if trigger_id else None

    logger.info(f"[REPLY] {conv_id} turn={body.turn_number} intent_transition={intent}")

    try:
        result = compose_reply(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            conversation_turns=conv["turns"],
            latest_message=body.message,
            is_intent_transition=intent,
        )
    except Exception as e:
        logger.error(f"[REPLY] compose_reply failed: {e}")
        result = {
            "action": "send",
            "body": "Noted! Ek second — main abhi draft karta hoon.",
            "cta": "none",
            "rationale": "Fallback due to API error.",
        }

    action = result.get("action", "send")
    if action == "end":
        conv["ended"] = True

    if action == "send" and result.get("body"):
        conv["turns"].append({"role": "vera", "body": result["body"], "ts": body.received_at})

    response = {"action": action, "rationale": result.get("rationale", "")}
    if action == "send":
        response["body"] = result.get("body", "")
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 3600)

    return response


# ── Optional teardown ─────────────────────────
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    logger.info("[TEARDOWN] All state wiped")
    return {"status": "wiped"}


# ── Root health check ─────────────────────────
@app.get("/")
async def root():
    return {"bot": "Vera-Pro", "status": "running", "endpoints": [
        "GET /v1/healthz", "GET /v1/metadata",
        "POST /v1/context", "POST /v1/tick", "POST /v1/reply"
    ]}
