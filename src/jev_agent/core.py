from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol

@dataclass(frozen=True)
class Action:
    kind: str
    target: str = ""
    value: str = ""
    def label(self) -> str:
        return f"{self.kind}({self.target}{', ' + self.value if self.value else ''})"

@dataclass
class Observation:
    text: str
    actions: list[Action]

@dataclass
class Decision:
    action: Action | None = None
    delegate: bool = False
    return_to_llm: bool = False
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    source: str = ""

class Environment(Protocol):
    def reset(self, task: str | None = None) -> tuple[str, Observation]: ...
    def step(self, action: Action) -> tuple[Observation, bool]: ...
    def success(self) -> bool: ...
    def close(self) -> None: ...

class LLMPolicy(Protocol):
    async def decide(self, goal: str, obs: Observation, history: list[dict], allow_delegate: bool) -> Decision: ...

class JevPolicy(Protocol):
    async def decide(self, goal: str, obs: Observation, history: list[dict]) -> Decision: ...