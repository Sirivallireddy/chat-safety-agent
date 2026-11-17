"""Small chat agent: pre-filter -> (only if needed) Chat Safety -> policy -> reply.

    user message
        |
    [1] PreFilter (classifier.py, no LLM)          ~10 ms
        |-- safe ------------------------------> [4] answer normally
        |-- escalate
              |
    [2] chat_safety.RiskAssessor (message + session scope)
              |
    [3] Policy (this file): answer / answer_with_care / support / refuse
              |
    [4] reply
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from chat_safety import (
    AssessmentInput, AssessmentScope, AssessmentStatus, RecentTurn, RiskAssessor, Stage,
    load_risk_registry,
)

from classifier import PreFilter, SessionState
from evaluators import make_safety_evaluator

HERE = Path(__file__).parent
log = logging.getLogger("chat-safety-agent")

# ------------------------------------------------- application policy (ours, not the library's)
REFUSE_AT = 0.7    # family probability at/above this -> refuse
CAUTION_AT = 0.4   # between CAUTION_AT and REFUSE_AT -> answer, but carefully
MAX_TURNS = 8      # library default context bound
MAX_TURN_CHARS = 3000
MAX_INPUT_CHARS = 4000

CHAT_SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers short and clear."

CAUTION_NUDGE = (
    "Safety note from the application: this conversation touches the risk area '{family}'. "
    "Answer only at a general, educational level. Do not give quantities, step-by-step "
    "procedures, sources to obtain materials, or ways to avoid detection. "
    "If the request needs those details, politely decline that part."
)
SUPPORT_NUDGE = (
    "Safety note from the application: the user may be in distress. Respond with warmth and "
    "empathy, do not give any method or means information, encourage them to talk to someone "
    "they trust, and gently mention they can reach a helpline."
)
HELPLINE = ("\n\nIf you're going through a hard time, you don't have to face it alone. In India you "
            "can call Tele-MANAS on **14416** (free, 24x7). If you're in immediate danger, call **112**.")
REFUSALS = {
    "violence_weapons": "I can't help with making or using weapons, explosives or anything meant to hurt people. "
                        "I'm happy to explain the science safely or talk about something else.",
    "unauthorized_intrusion": "I can't help with getting into accounts or systems without permission. "
                              "I can help you secure your own accounts instead.",
    "fraud_theft": "I can't help with scams, forgery or stealing. If you think you've been targeted by fraud, "
                   "I can help you figure out how to report it.",
    "harassment_hate": "I can't help with threatening, harassing or targeting people.",
    "privacy_exfiltration": "I can't help with finding, tracking or exposing someone's private information.",
    "prompt_injection": "I can't ignore my guidelines, but I'm happy to help with your actual question.",
    "medical_advice": "I can't give personal dosing or treatment changes. Please check with a doctor or pharmacist.",
    "child_exploitation": "I can't help with that.",
    "nonconsensual_intimate_content": "I can't help with creating or sharing intimate images of anyone without consent.",
}


@dataclass
class Trace:
    """Everything the UI shows under a reply, so the live demo is explainable."""
    route: str = ""
    prefilter_risk: float = 0.0
    prefilter_family: str | None = None
    prefilter_reasons: list[str] = field(default_factory=list)
    prefilter_scores: dict[str, float] = field(default_factory=dict)
    drift: bool = False
    prefilter_ms: float = 0.0
    safety_called: bool = False
    safety_status: str | None = None
    safety_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    safety_ms: float = 0.0
    action: str = "answer"
    action_family: str | None = None
    total_ms: float = 0.0


@dataclass
class Turn:
    role: str
    text: str
    trace: Trace | None = None


class ChatAgent:
    def __init__(self, prefilter: PreFilter | None = None):
        self.prefilter = prefilter or PreFilter()
        self.registry = load_risk_registry(HERE / "risk-registry.json")
        evaluator, self.safety_label = make_safety_evaluator(self.prefilter)
        self.assessor = RiskAssessor(self.registry, evaluator)
        self._llm, self.llm_label = self._make_llm()
        self.reset()

    def reset(self) -> None:
        self.history: list[Turn] = []
        self.state = SessionState()
        self.session_ref = f"chat-{uuid.uuid4().hex[:12]}"
        self.revision = 0

    # ------------------------------------------------------------- main entry
    def handle(self, text: str) -> Turn:
        return asyncio.run(self.ahandle(text))

    async def ahandle(self, text: str) -> Turn:
        start = time.perf_counter()
        text = text.strip()[:MAX_INPUT_CHARS]
        trace = Trace()

        # [1] pre-filter
        pre = self.prefilter.assess(text, self.state)
        trace.route, trace.prefilter_risk = pre.route, pre.risk
        trace.prefilter_family, trace.prefilter_reasons = pre.top_family, pre.reasons
        trace.prefilter_scores, trace.drift, trace.prefilter_ms = pre.family_scores, pre.drift, pre.latency_ms
        log.info("USER: %s", text[:100])
        log.info("  [1] PRE-FILTER  -> %s | risk %.2f %s%s | %s ms | %s", pre.route.upper(), pre.risk,
                 pre.top_family or "", " (drift)" if pre.drift else "", pre.latency_ms, "; ".join(pre.reasons))

        nudge = None
        if pre.escalate:
            # [2] Chat Safety, only for flagged turns
            trace.safety_called = True
            t0 = time.perf_counter()
            result = await self._assess(text)
            trace.safety_ms = round((time.perf_counter() - t0) * 1000, 1)
            trace.safety_status = result.status.value
            # [3] policy
            if result.status is AssessmentStatus.COMPLETE:
                for s in result.scores:
                    trace.safety_scores.setdefault(s.family_id, {})[s.scope.value] = round(s.probability, 2)
                trace.action, trace.action_family = self._policy(trace.safety_scores)
            else:
                # Unavailable is NOT evidence of safety: answer carefully (fail-safe).
                trace.action, trace.action_family = "answer_with_care", pre.top_family
            top = sorted(((f, max(v.values())) for f, v in trace.safety_scores.items()), key=lambda x: -x[1])[:3]
            log.info("  [2] CHAT SAFETY -> %s | %s ms | top: %s", trace.safety_status, trace.safety_ms,
                     ", ".join(f"{f} {p:.2f}" for f, p in top) or "-")
            log.info("  [3] POLICY      -> %s %s", trace.action.upper(), trace.action_family or "")
            if trace.action == "answer_with_care":
                nudge = CAUTION_NUDGE.format(family=trace.action_family)
            elif trace.action == "support":
                nudge = SUPPORT_NUDGE

        # [4] reply
        self.history.append(Turn("user", text))
        if trace.action == "refuse":
            reply = REFUSALS.get(trace.action_family, "I can't help with that.")
        else:
            try:
                reply = await asyncio.to_thread(self._chat, nudge)
            except Exception as exc:  # bad key, rate limit, network: show it, don't crash the chat
                reply = f"⚠️ Chat model error: {type(exc).__name__}. Check LLM_API_KEY / LLM_MODEL in .env and restart."
            if trace.action == "support":
                reply += HELPLINE
        self.prefilter.observe_assistant(reply, self.state)
        trace.total_ms = round((time.perf_counter() - start) * 1000, 1)
        if not trace.safety_called:
            log.info("  [2] CHAT SAFETY -> skipped (saved an LLM call)")
        log.info("  [4] REPLY       -> %s | total %s ms\n", trace.action, trace.total_ms)
        turn = Turn("assistant", reply, trace)
        self.history.append(turn)
        self.revision += 1
        return turn

    # ------------------------------------------------------------- helpers
    async def _assess(self, text: str):
        recent = tuple(RecentTurn(t.role, t.text[:MAX_TURN_CHARS]) for t in self.history[-MAX_TURNS:])
        # The library needs history for a session check, so the first turn is message-only.
        session = bool(recent)
        request = AssessmentInput(
            request_id=f"{self.session_ref}-{self.revision}",
            registry_version=self.registry.version,
            stage=Stage.INPUT,
            content=text,
            assessment_scopes=((AssessmentScope.MESSAGE, AssessmentScope.SESSION) if session
                               else (AssessmentScope.MESSAGE,)),
            session_ref=self.session_ref if session else None,
            session_revision=self.revision if session else None,
            recent_turns=recent,
        )
        return await self.assessor.assess_families(request)

    @staticmethod
    def _policy(scores: dict[str, dict[str, float]]) -> tuple[str, str | None]:
        # Use the worse of message and session scope for each family.
        worst = {fam: max(by_scope.values()) for fam, by_scope in scores.items()}
        fam = max(worst, key=worst.get)
        p = worst[fam]
        if worst.get("self_harm", 0) >= CAUTION_AT:
            return "support", "self_harm"
        if p >= REFUSE_AT:
            return "refuse", fam
        if p >= CAUTION_AT:
            return "answer_with_care", fam
        return "answer", None  # Chat Safety says it was a false alarm

    def _make_llm(self):
        key = os.getenv("LLM_API_KEY")
        if not key:
            return None, "Offline (no LLM_API_KEY)"
        from openai import OpenAI
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                        timeout=30)
        return (client, model), model

    def _chat(self, nudge: str | None) -> str:
        if self._llm is None:
            note = " (answering with care)" if nudge else ""
            return f"[offline mode{note}] Add LLM_API_KEY to .env to get real answers."
        client, model = self._llm
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        messages += [{"role": t.role, "content": t.text} for t in self.history[-MAX_TURNS:]]
        if nudge:  # placed last so the model sees it right before answering
            messages.append({"role": "system", "content": nudge})
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3, max_tokens=500)
        return resp.choices[0].message.content.strip()
