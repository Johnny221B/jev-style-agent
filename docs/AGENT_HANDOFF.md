# Server Coding Agent handoff (work incrementally)

You are extending a tested **mock-only research starter** into a real AndroidWorld + verl GRPO pipeline. Preserve and extend existing code; do not rewrite it wholesale. DO NOT report AndroidWorld/GRPO success until the real emulator and verl token-accurate training tests pass.

## Phase A: environment first
1. Inspect README, `docs/ANDROIDWORLD.md`, upstream AndroidWorld checkout, installed APKs, `adb devices`, `/dev/kvm`, and job scheduler constraints. Record upstream commits and `python -V`, `pip freeze` into `outputs/environment.txt`.
2. Run official AndroidWorld `minimal_task_runner.py` once on an allowed task. If simulator cannot run on GPU node, test on CPU/login node or request appropriate VM/KVM resources; do not fake task success.
3. Implement `src/jev_agent/androidworld_env.py` satisfying the `Environment` Protocol. Use installed native action definitions, not guessed API calls. Keep `env.close()` and task cleanup reliable.
4. Add a real episode smoke test behind `RUN_ANDROIDWORLD_INTEGRATION=1`, and leave offline unit tests independent of Android SDK.

## Phase B: real inference
5. Verify real accessibility-tree candidate mapping, no invalid action indices, input text handling, and progression/recovery on 10 sampled tasks. Save JSONL rollout traces.
6. Start local vLLM Qwen3-4B; run full-LLM baseline and Jev-assisted runner on same task set. Store actual token usage, emulator wall time and Jev call usage. Calibrate threshold on validation only.
7. Add safeguards against repeated LLM/Jev bounce: after a Jev return, an LLM direct action or changed subgoal must occur before another delegation. Track repeated screen hashes, nonprogress, and delegated step caps.

## Phase C: verl RL (separate PR)
8. Pin a specific verl commit; inspect its `AgentLoopBase`, `AgentLoopOutput`, and LLM server signatures. Implement a token-exact custom agent loop, ensuring Jev and environment text have `response_mask=0`. Avoid reconstructing generated tokens from chat-completion strings.
9. Create train/val/test template & seed manifests. Use environment success reward plus logged cost components; no LLM judge. First prove nonzero GRPO gradient and stable update on a 2-task synthetic rollout before launching expensive jobs.
10. Run 4B LoRA-GRPO on 4×H100, then 8B. Keep full-LLM, fixed-route and random-route baselines with equal aggregate cost limits.

## Reporting
For each phase summarize commands run, real test output, changed files, remaining blocker, and next smallest test. Do not silently reinterpret mock tests as real training.

## Phase A implementation already included (v0.2)
- `jev_agent.androidworld_env.AndroidWorldEnvironment`: official task lifecycle + accessible UI click/back/scroll candidate actions, task reward.
- `jev_agent.androidworld_cli`: single real environment run with mock or configured API policies.
- `scripts/check_android_host.sh`: cluster environment preflight.
- `tests/test_androidworld_adapter.py`: offline shape/guard tests.

**First server-side task:** verify official upstream `json_action.JSONAction` action names, actual `UIElement` clickable property, and action navigation/scroll parameters using the pinned installed AndroidWorld version; execute a single emulator task; add an integration test. Next add input-text actions supplied by LLM (do not ask Jev to generate arbitrary strings), stronger step logging, avoid call ping-pong, and only then integrate verl multi-turn token IDs. No claim of real emulator validation yet.