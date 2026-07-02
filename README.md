# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Assessments via dialogue.
FastAPI + Groq (Llama 3.3 70B) + BM25 + sentence-transformers.

---

## Project structure

```
shl-recommender/
├── main.py          # FastAPI app, endpoints
├── agent.py         # Intent routing, LLM calls, reply generation
├── retrieval.py     # Hybrid BM25 + dense search, RRF fusion
├── schemas.py       # Pydantic request/response models
├── prompts.py       # All LLM prompts
├── catalog.json     # SHL catalog — YOU MUST ADD THIS (see below)
├── requirements.txt
├── render.yaml      # Render deploy config
└── .env.example
```

---

## Setup (local)



### 2. Get a Groq API key

Free at https://console.groq.com — no credit card needed.

### 3. Install and run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


# Load env and start
uvicorn main:app --reload --port 8000
```

### 4. Test

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer"}
    ]
  }'
```


## API reference

### GET /health
```json
{"status": "ok"}
```

### POST /chat

Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Got it. Here are assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is `[]` (empty array) when the agent is still clarifying.  
`end_of_conversation` is `true` only when the agent considers the task complete.

---

## Design decisions

### Retrieval: Hybrid BM25 + dense (RRF)
- **BM25** catches exact name matches ("OPQ32r", "Java 8 (New)") that pure semantic search misses.
- **Dense** (all-MiniLM-L6-v2) catches semantic matches ("communication skills" → OPQ).
- **RRF** (Reciprocal Rank Fusion) combines both without score normalization.

### One LLM call per turn for slot extraction + intent
Combining these into one JSON output halves latency and stays well within the 30s timeout.

### Turn cap guard
If `len(messages) >= 6`, the agent is forced into `recommend` mode even if it would normally clarify. This ensures we never hit the 8-turn evaluator cap without a shortlist.

### URL validation
Every recommendation URL is checked against the in-memory catalog set before returning. Hallucinated URLs are silently dropped. This satisfies the "catalog only" hard eval.

### Stateless design
No DB, no session store. Full conversation history is replayed per call. Works naturally with the evaluator's stateless harness.
