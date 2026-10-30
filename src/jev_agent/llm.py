from __future__ import annotations
import os, json
import httpx
from .core import Decision, Observation

SYSTEM = """You are a GUI agent. Output ONLY JSON: {"decision":"action","index":0} or {"decision":"delegate"}. Indices refer to current candidate actions. If action list lacks a suitable action, output {"decision":"action","index":0} only as a last resort. Delegate only if another lightweight decision model can safely perform straightforward navigation; do not delegate complex reasoning or risky actions."""

class OpenAICompatiblePolicy:
    """INFERENCE/SANITY CHECK ONLY: Chat completions cannot be used as verl PPO token trajectories."""
    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
        self.url = (base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/") + "/chat/completions"
        self.key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
    async def decide(self, goal, obs: Observation, history, allow_delegate):
        text = json.dumps({"goal": goal, "screen": obs.text[:12000],
                           "actions": [a.label() for a in obs.actions],
                           "recent_history": history[-6:], "can_delegate": allow_delegate}, ensure_ascii=False)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.url, headers={"Authorization":f"Bearer {self.key}"}, json={
                "model": self.model, "messages":[{"role":"system","content":SYSTEM}, {"role":"user","content":text}],
                "temperature": 0.3, "max_tokens": 128})
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        try:
            decision = json.loads(content[content.index('{'):content.rindex('}')+1])
        except (ValueError, TypeError):
            decision = {"decision":"action","index":0}
        if decision.get("decision") == "delegate" and allow_delegate:
            return Decision(delegate=True, source="llm")
        idx = decision.get("index", 0)
        if not isinstance(idx,int) or idx<0 or idx>=len(obs.actions):
            return Decision(source="llm_invalid")
        return Decision(action=obs.actions[idx], source="llm")

class MockLLMPolicy:
    async def decide(self, goal, obs, history, allow_delegate):
        if allow_delegate and not history:
            return Decision(delegate=True, source="mock_llm")
        return Decision(action=obs.actions[0] if obs.actions else None, source="mock_llm")