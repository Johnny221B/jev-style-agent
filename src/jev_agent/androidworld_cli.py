"""Run one official AndroidWorld task with real Jev/Qwen or mock policies."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from .androidworld_env import AndroidWorldConfig, AndroidWorldEnvironment
from .runner import rollout
from .jev import JevChoicePolicy, MockJevPolicy
from .llm import OpenAICompatiblePolicy, MockLLMPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='ContactsAddContact')
    p.add_argument('--controller', choices=['llm', 'jev', 'hybrid'], default='hybrid')
    p.add_argument('--mock-models', action='store_true', help='Only tests environment interactions; not model quality')
    p.add_argument('--adb-path', default=os.getenv('ADB_PATH', 'adb'))
    p.add_argument('--console-port', type=int, default=5554)
    p.add_argument('--grpc-port', type=int, default=8554)
    p.add_argument('--emulator-setup', action='store_true')
    p.add_argument('--max-steps', type=int, default=20)
    p.add_argument('--jev-max-candidates', type=int, default=20)
    p.add_argument('--out', default='runs/androidworld_single.json')
    args = p.parse_args()
    env = AndroidWorldEnvironment(AndroidWorldConfig(
        adb_path=args.adb_path, console_port=args.console_port,
        grpc_port=args.grpc_port, emulator_setup=args.emulator_setup,
        max_candidates=args.jev_max_candidates,
    ))
    llm = MockLLMPolicy() if args.mock_models else OpenAICompatiblePolicy()
    jev = MockJevPolicy() if args.mock_models else JevChoicePolicy(max_candidates=args.jev_max_candidates)
    cfg = {'controller': args.controller, 'max_env_steps': args.max_steps,
           'max_jev_steps_per_handoff': 4, 'cost': {'llm_call': 1, 'jev_call': .05,
           'environment_action': .01}, 'reward': {'success': 1, 'failure': 0, 'cost_weight': .01}}
    result = asyncio.run(rollout(env, llm, jev, cfg, task=args.task))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    print(json.dumps({'success': result.success, 'steps': result.steps,
                      'llm_calls': result.llm_calls, 'jev_calls': result.jev_calls,
                      'file': str(dest)}, indent=2))

if __name__ == '__main__':
    main()