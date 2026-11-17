# Chat Safety Agent: pre-filter + Chat Safety

A fast, **non-LLM pre-filter** that decides, for every chat turn, whether the expensive
Chat Safety model needs to run. It's integrated with the
[chat-safety](https://github.com/BahulamAI/chat-safety) library and shown in a small chat app.

```
user message
   │
[1] Pre-filter  (classifier.py, no LLM, ~20 ms)
   ├── safe ───────────────────────────────► answer normally (no safety call)
   └── risky / drifting
          │
[2] chat_safety.RiskAssessor   (message + session scope, 10 risk families)
          │
[3] Policy (agent.py): answer · answer with care · supportive reply · refuse
          │
[4] reply   (+ the reply is fed back into the pre-filter: assistant-led drift)
```

The `chat-safety` library is used **unchanged**. This project only adds files around it.

## How the pre-filter works

For each of the 10 risk families (same IDs as `chat_safety.Category`) it combines:

| Signal | What it catches | Example |
|---|---|---|
| Keyword rules (regex) | Clear risky terms, deterministic and explainable | "explosion", "hack into", "kill myself" |
| Topic words (weak) | Context like "chemicals", "mix", "password". **Never escalates on its own.** | "combine two chemicals" stays safe |
| Semantic similarity | Rephrased requests. Local `all-MiniLM-L6-v2`: the message must be clearly closer to risky examples than to normal ones. | "cause the most injuries at a concert" |
| Pretext / detail phrases | Intent hiding and the step from curious to how-to. These only count when a risk is already present. | "for educational purposes", "exact amounts" |
| **Session drift** | Risk carried across turns (`max(now, 0.8 × previous)` plus a boost when the topic repeats). Slow escalation is caught. | "What exact amounts would make it bigger?" after a chemicals question |

The highest family score at or above **0.35** is sent to Chat Safety. The threshold is low on purpose:
a missed hazardous conversation costs far more than one extra safety call. Self-harm topic
words are weighted higher for the same reason.

## Results (offline, `python eval_prefilter.py`)

36 turns in 18 labelled conversations (benign, direct risk, rephrased risk, slow drift):

| Metric | Value |
|---|---|
| Recall (risky turns sent to Chat Safety) | **12 / 12** |
| False positives (safe turns flagged) | **0 / 24** |
| Turns that skipped Chat Safety | **67%** |
| Pre-filter latency | **~20–30 ms** per turn on CPU |

Caveat: the set is small and rules were tuned while building it. Treat these as a sanity check,
not a calibrated accuracy figure. Real use needs a larger held-out set.

## Chat Safety evaluator (pluggable)

`RiskAssessor` accepts any evaluator with `async evaluate(request, questions)`. Pick one in `.env`:

| `SAFETY_PROVIDER` | Uses |
|---|---|
| `jev` | The library's `JevRiskEvaluator` (production; needs `SAFETY_API_KEY`) |
| `llm` | Any OpenAI-compatible model, Groq free tier by default (demo) |
| `offline` | Replays pre-filter scores; for trying the UI only |

If the safety check is **unavailable**, the agent does not treat that as safe. It answers
carefully (fail-safe), as the library README recommends.

## Run

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                               # add LLM_API_KEY (free key: console.groq.com)
streamlit run app.py
```

Each reply shows a badge (🟢 safe · 🟡 cleared · 🟠 careful · 💙 support · 🔴 refused) and a
**Why?** panel with the pre-filter reasons, Chat Safety scores and timings.

Tests (offline, no keys): `python -m unittest discover -s tests -v`

## Files

| File | Purpose |
|---|---|
| `classifier.py` | The pre-filter |
| `agent.py` | Pre-filter → RiskAssessor → policy → reply |
| `evaluators.py` | Jev / LLM / offline evaluators for `RiskAssessor` |
| `risk-registry.json` | 10 risk families with message and session questions (library schema `chat-safety.risks/1`) |
| `app.py` | Streamlit chat for the live demo |
| `eval_prefilter.py`, `data/eval_set.json` | Routing evaluation |
| `tests/` | Offline unit tests |

## Limits and next steps
- Keyword lists are English-only, and a determined user can phrase around them. The semantic signal and drift reduce this but don't remove it. Next step: train a small classifier (logistic regression on embeddings) on logged, labelled traffic.
- The policy thresholds (0.4 / 0.7) are illustrative and need calibration on real data.
- No output check yet: the library supports `Stage.OUTPUT_RELEASE` to assess a draft reply before showing it.
- Session state is in memory. Production would persist it per session revision, as the library describes.
