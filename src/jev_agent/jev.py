from __future__ import annotations
import os
import httpx
from .core import Action, Decision, Observation

class JevChoicePolicy:
    """Choice + independent escalate question; errors fail closed to LLM."""
    def __init__(self, threshold=0.85, return_threshold=0.65, max_candidates=20,
                 url=None, model=None, api_key=None, client=None):
        self.threshold = threshold
        self.return_threshold = return_threshold
        self.max_candidates = max_candidates
        self.url = url or os.getenv("JEV_API_URL", "https://jev-ai.org/api/v1/systemone/")
        self.model = model or os.getenv("JEV_MODEL", "jev-1.13")
        self.api_key = api_key or os.getenv("JEV_API_KEY")
        self.client = client

    async def decide(self, goal: str, obs: Observation, history: list[dict]) -> Decision:
        candidates = obs.actions
        if not candidates or len(candidates) > self.max_candidates:
            return Decision(return_to_llm=True, source="jev", confidence=0)
        if not self.api_key:
            raise RuntimeError("JEV_API_KEY not set; use --mock-jev for offline smoke test")
        criteria = {f"a{i}": a.label() for i, a in enumerate(candidates)}
        state = {"goal": goal, "screen": obs.text[:12000], "recent_history": history[-6:]}
        payload = {"model": self.model, "state": state, "questions": {
            "next_action": {"type": "choice", "instructions": "Choose the safest next GUI action that progresses the goal using only available actions.", "criteria": criteria},
            "escalate": {"type": "noul", "instructions": "Is this GUI decision ambiguous, requiring multi-step planning, an irreversible action, or likely to fail without a stronger reasoning model?"}
        }}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            if self.client is None:
                async with httpx.AsyncClient(timeout=30) as client:
                    res = await client.post(self.url, json=payload, headers=headers)
            else:
                res = await self.client.post(self.url, json=payload, headers=headers)
            res.raise_for_status()
            answers = res.json()["answers"]
            choice = answers["next_action"]
            probs = choice["probabilities"]
            escalate = float(answers["escalate"]["noul"])
            selected = choice["choice"]
            if selected not in criteria:
                return Decision(return_to_llm=True, source="jev")
            score = float(probs.get(selected, 0.0))
            idx = int(selected[1:])
            if score < self.threshold or escalate >= self.return_threshold:
                return Decision(return_to_llm=True, source="jev", confidence=score, probabilities=probs)
            return Decision(action=candidates[idx], confidence=score, probabilities=probs, source="jev")
        except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError):
            return Decision(return_to_llm=True, source="jev_error")

class MockJevPolicy:
    """Deterministic offline testing only; does NOT measure Jev quality."""
    async def decide(self, goal, obs, history):
        if not obs.actions:
            return Decision(return_to_llm=True, source="mock_jev")
        return Decision(action=obs.actions[0], confidence=0.99, source="mock_jev")