import asyncio
from jev_agent.mock_env import MockEnvironment
from jev_agent.llm import MockLLMPolicy
from jev_agent.jev import MockJevPolicy
from jev_agent.runner import rollout

def test_delegation_mock():
    cfg={"controller":"hybrid", "max_env_steps":5,"max_jev_steps_per_handoff":4}
    r=asyncio.run(rollout(MockEnvironment(), MockLLMPolicy(),MockJevPolicy(),cfg))
    assert r.success and r.steps==3 and r.llm_calls==1 and r.jev_calls==3

def test_llm_only():
    cfg={"controller":"llm", "max_env_steps":5}
    r=asyncio.run(rollout(MockEnvironment(), MockLLMPolicy(),MockJevPolicy(),cfg))
    assert r.success and r.jev_calls==0

def test_max_handoff():
    cfg={"controller":"hybrid", "max_env_steps":5,"max_jev_steps_per_handoff":1}
    r=asyncio.run(rollout(MockEnvironment(),MockLLMPolicy(),MockJevPolicy(),cfg))
    assert r.success and r.llm_calls>1