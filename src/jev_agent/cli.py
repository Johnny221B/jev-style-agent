import argparse, asyncio, json
from pathlib import Path
import yaml
from dataclasses import asdict
from .runner import rollout
from .mock_env import MockEnvironment
from .llm import MockLLMPolicy, OpenAICompatiblePolicy
from .jev import MockJevPolicy, JevChoicePolicy

async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/experiment.yaml')
    p.add_argument('--controller', choices=['llm','jev','hybrid'])
    p.add_argument('--mock-llm', action='store_true')
    p.add_argument('--mock-jev', action='store_true')
    p.add_argument('--output', default='outputs/smoke.json')
    p.add_argument('--task')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if args.controller: cfg['controller']=args.controller
    if cfg['environment']!='mock':
        raise SystemExit('AndroidWorld is not enabled in the generic smoke-test CLI. See docs/ANDROIDWORLD.md and server integration checklist.')
    env=MockEnvironment()
    llm=MockLLMPolicy() if args.mock_llm else OpenAICompatiblePolicy()
    jev=MockJevPolicy() if args.mock_jev else JevChoicePolicy(cfg['confidence_threshold'],cfg['return_threshold'],cfg['max_candidates'])
    result=await rollout(env,llm,jev,cfg,args.task)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(asdict(result),indent=2))
    print(json.dumps(asdict(result),indent=2))

if __name__=='__main__': asyncio.run(main())