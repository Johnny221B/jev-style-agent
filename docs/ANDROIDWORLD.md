# AndroidWorld integration plan

Upstream: https://github.com/google-research/android_world

## Server setup
- Check virtualization/KVM permission on GPU nodes; CPU emulators may be slow or blocked by cluster policy. Test `ls -l /dev/kvm`, `adb devices`, disk quota, and Android SDK licenses.
- Clone upstream AndroidWorld at a recorded commit; follow upstream emulator/APK setup documentation. Run `python minimal_task_runner.py --help` and then one official example before integrating RL.
- Each concurrent rollout requires isolated emulator/device state and unique console port. **Never** share one emulator between concurrent trajectories.
- Use `env_launcher.load_and_setup_env(console_port=..., emulator_setup=False, adb_path=...)` after one-time setup; `env.reset(go_home=True)`; `registry.TaskRegistry().get_registry(TaskRegistry.ANDROID_WORLD_FAMILY)`; instantiate tasks from `generate_random_params()`, call `initialize_task(env)`, use `is_successful(env)` for terminal success; close environment.

## Agent implementation checklist (server-side)
1. Read current installed `android_world.agents.t3a` / `agent_utils` / `env.interface` to extract a text observation from `env.get_state()` and map UI indices to action targets.
2. Enumerate clickable UI elements from accessibility / UI hierarchy; validate indices and limits. Do not silently invent actions.
3. Implement `Action(kind,target,value)` to native `AndroidAction` conversion for TAP, SCROLL, BACK, INPUT_TEXT. This depends on the installed AndroidWorld commit: **do not assume enum names**.
4. Only include INPUT_TEXT if its value has already been supplied by task spec or LLM; Jev should choose an existing candidate, not generate arbitrary strings.
5. After every step, re-read UI state, check success and no-progress. Add failure handling and max-step termination.
6. Split task templates by name and seeds (train/validation/test); store a manifest and never tune thresholds on test.
7. Record per-step `controller`, chosen action, Jev `probabilities`, `return_probability`, timing, screen hash, and task success.

## Why no fake adapter?
The project intentionally does not claim a working AndroidWorld adapter without emulator/API tests. This avoids sending you code that looks complete but invokes nonexistent methods on the installed version.

Official reference: https://github.com/google-research/android_world/blob/main/minimal_task_runner.py