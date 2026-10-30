from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from .core import Environment, LLMPolicy, JevPolicy

@dataclass
class RolloutResult:
    success: bool
    steps: int
    llm_calls: int
    jev_calls: int
    reward: float
    elapsed_seconds: float
    history: list[dict]

async def rollout(env: Environment, llm: LLMPolicy, jev: JevPolicy, cfg: dict, task: str | None = None) -> RolloutResult:
    goal, obs = env.reset(task)
    mode = "jev" if cfg.get("controller") == "jev" else "llm"
    steps = llm_calls = jev_calls = jev_streak = 0
    history: list[dict] = []
    max_steps = int(cfg.get("max_env_steps", 25))
    max_jev = int(cfg.get("max_jev_steps_per_handoff", 4))
    started = time.monotonic()
    try:
        while steps < max_steps:
            if env.success():
                break
            if mode == "jev":
                if jev_streak >= max_jev:
                    mode, jev_streak = "llm", 0
                    continue
                decision = await jev.decide(goal, obs, history)
                jev_calls += 1
                if decision.return_to_llm or decision.action is None:
                    mode, jev_streak = "llm", 0
                    if cfg.get("controller") == "jev":
                        break
                    continue
                jev_streak += 1
            else:
                decision = await llm.decide(goal, obs, history, allow_delegate=cfg.get("controller") == "hybrid")
                llm_calls += 1
                if decision.delegate and cfg.get("controller") == "hybrid":
                    mode, jev_streak = "jev", 0
                    continue
                if decision.action is None:
                    break
                jev_streak = 0
            if decision.action not in obs.actions:
                break
            new_obs, done = env.step(decision.action)
            history.append({"source": decision.source, "action": decision.action.label(), "confidence": decision.confidence})
            steps += 1
            obs = new_obs
            if done:
                break
        success = bool(env.success())
        costs = cfg.get("cost", {})
        r_cfg = cfg.get("reward", {})
        cost = (llm_calls*costs.get("llm_call",1) + jev_calls*costs.get("jev_call",.05) + steps*costs.get("environment_action",.01))
        reward = (r_cfg.get("success",1) if success else r_cfg.get("failure",0)) - r_cfg.get("cost_weight",.02)*cost
        return RolloutResult(success, steps, llm_calls, jev_calls, reward, time.monotonic()-started, history)
    finally:
        env.close()