# Vera-Bot — magicpin AI Challenge Submission

## Approach

This bot implements the full 4-context composition framework (Category × Merchant × Trigger × Customer) using **Groq's `llama-3.3-70b-versatile`** model for fast, deterministic message generation (temperature=0).

### Architecture

```
Judge → POST /v1/context  → In-memory context store (5 categories, 50 merchants, 200 customers, 100 triggers)
Judge → POST /v1/tick     → Trigger routing → compose_message() → Claude-quality WhatsApp messages
Judge → POST /v1/reply    → Conversation state manager → compose_reply() → adaptive multi-turn responses
```

### Key Design Decisions

**1. Trigger-kind routing**
Each of the 26 trigger kinds (research_digest, recall_due, perf_spike, festival_upcoming, etc.) gets a specialized prompt guidance block. A research_digest message leads with the paper stat + trial N + source citation. A recall_due message leads with the customer's name, service gap, and two available slots. A perf_spike message leads with the exact number and builds momentum.

**2. Specificity-first composition**
The system prompt enforces the challenge's core scoring criterion: every message must anchor on a concrete, verifiable fact from the provided context — a peer CTR number, a patient count, a research trial N, a ₹price. Generic "10% off" framings are explicitly penalized in the prompt.

**3. Auto-reply detection (3-strike rule)**
Pattern-matched against 14 known WhatsApp Business auto-reply phrases. On first detection: one gentle owner probe. On second: 4-hour backoff. On third: graceful conversation close.

**4. Intent-transition handler**
14 intent patterns ("let's do it", "go ahead", "kar do", etc.) trigger immediate switch from qualification mode to action mode. The reply prompt explicitly warns the LLM: "DO NOT ask another qualifying question. Draft the specific next step now."

**5. Hindi-English code-mix**
Language preference from `merchant.identity.languages` drives prompt instruction. Merchants with `"hi"` in their language array get Hindi-English blended responses matching real Vera patterns.

**6. Suppression deduplication**
Every `suppression_key` is tracked in memory. Same trigger cannot fire twice in the same test window.

### Tradeoffs Made

- **In-memory state**: Faster and sufficient for the 60-minute test window. Would use Redis for production persistence.
- **Single-pass composition**: No retrieval layer over digest items (would add 200ms). Instead the full relevant digest item is injected into the prompt.
- **Groq over OpenAI/Claude**: 10× lower latency (average ~400ms vs 2-4s) which matters for the 30s per-call timeout requirement.

### What Would Have Helped Most

1. **Slot availability data**: Merchant appointment slots to make recall_due messages more concrete ("Wednesday 6pm or Thursday 5pm" vs generic "evenings this week").
2. **Post history**: Knowing exactly what was posted to GBP in the last 30 days to avoid repetition in post suggestions.
3. **Local event feed**: Hyperlocal events (colony festival, nearby college exam schedule) for more targeted festival/seasonal triggers.

## Running Locally

```bash
# 1. Clone / download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your env vars
cp .env.example .env
# Edit .env: add GROQ_API_KEY, TEAM_NAME, TEAM_MEMBER, CONTACT_EMAIL

# 4. Start the server
uvicorn bot:app --host 0.0.0.0 --port 8080

# 5. Test health
curl http://localhost:8080/v1/healthz

# 6. Generate submission.jsonl
python generate_submission.py

# 7. Run the judge simulator
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

## Deploying to Render (free)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service → connect your repo
3. Set environment variables in the Render dashboard: `GROQ_API_KEY`, `TEAM_NAME`, `TEAM_MEMBER`, `CONTACT_EMAIL`
4. Deploy — your public URL will be `https://vera-bot-xxxx.onrender.com`

## Pre-flight Checklist

- [x] All 5 endpoints implemented and schema-correct
- [x] `/v1/context` idempotent on (scope, context_id, version)
- [x] `/v1/tick` returns within 30s (Groq avg ~400ms; well within budget)
- [x] `/v1/reply` handles auto-reply, opt-out, intent-transition
- [x] Bot persists context across calls (in-memory, no restarts during test)
- [x] `submission.jsonl` generated with `generate_submission.py`
- [x] No URLs in message bodies
- [x] No fabricated data — all facts from provided context only
