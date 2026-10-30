"""AndroidWorld text-accessibility adapter. Requires upstream android_world at runtime.

Uses the official task lifecycle and JSONAction execution contract from T3A.
The env does NOT launch Android emulator or call any commercial LLM.
"""
from __future__ import annotations
import random
import shutil
import time
from dataclasses import dataclass
from .core import Action, Observation


def extract_candidates(ui_elements, screen_size, limit=80):
    """Keep official UI indices stable; never renumber or invent targets."""
    from android_world.agents import m3a_utils
    rows, actions = [], []
    for idx, element in enumerate(ui_elements):
        if not m3a_utils.validate_ui_element(element, screen_size):
            continue
        desc = str(element)
        rows.append(f'UI element {idx}: {desc}')
        # This is a conservative first version: clickable components only.
        if getattr(element, 'is_clickable', False) or getattr(element, 'is_checkable', False):
            actions.append(Action('click', str(idx)))
        if len(actions) >= limit:
            break
    actions.append(Action('back'))
    actions.append(Action('scroll', 'down'))
    actions.append(Action('scroll', 'up'))
    return '\n'.join(rows) or '(No accessible elements)', actions


@dataclass
class AndroidWorldConfig:
    adb_path: str = 'adb'
    console_port: int = 5554
    grpc_port: int = 8554
    emulator_setup: bool = False
    pause_seconds: float = 0.5
    max_candidates: int = 80
    seed: int = 0


class AndroidWorldEnvironment:
    def __init__(self, config: AndroidWorldConfig | None = None):
        self.config = config or AndroidWorldConfig()
        self.env = None
        self.task = None
        self.last_state = None
        self._rng = random.Random(self.config.seed)

    def _connect(self):
        if self.env is not None:
            return
        from android_world.env import env_launcher
        adb = shutil.which(self.config.adb_path) or self.config.adb_path
        self.env = env_launcher.load_and_setup_env(
            console_port=self.config.console_port,
            emulator_setup=self.config.emulator_setup,
            adb_path=adb, grpc_port=self.config.grpc_port,
        )
        self.config.emulator_setup = False

    def reset(self, task: str | None = None) -> tuple[str, Observation]:
        from android_world import registry
        self._connect()
        self.env.reset(go_home=True)
        tasks = registry.TaskRegistry()
        available = tasks.get_registry(tasks.ANDROID_WORLD_FAMILY)
        if task is not None and task not in available:
            raise ValueError(f'Unknown AndroidWorld task {task!r}; examples: {list(available)[:12]}')
        name = task or self._rng.choice(sorted(available))
        task_cls = available[name]
        # The upstream registry exposes a classmethod to generate task parameters.
        params = task_cls.generate_random_params()
        self.task = task_cls(params)
        self.task.initialize_task(self.env)
        return str(self.task.goal), self.observe()

    def observe(self) -> Observation:
        if self.env is None:
            raise RuntimeError('Call reset() first')
        self.last_state = self.env.get_state(wait_to_stabilize=True)
        text, actions = extract_candidates(self.last_state.ui_elements,
                                           self.env.logical_screen_size,
                                           self.config.max_candidates)
        return Observation(text=text, actions=actions)

    def step(self, action: Action) -> tuple[Observation, bool]:
        from android_world.env import json_action
        if self.last_state is None:
            raise RuntimeError('No current observation')
        _, valid = extract_candidates(self.last_state.ui_elements,
                                      self.env.logical_screen_size,
                                      self.config.max_candidates)
        if action not in valid:
            raise ValueError(f'Stale or unsupported action: {action}')
        if action.kind == 'click':
            native = json_action.JSONAction(action_type='click', index=int(action.target))
        elif action.kind == 'back':
            native = json_action.JSONAction(action_type='navigate_back')
        elif action.kind == 'scroll':
            native = json_action.JSONAction(action_type='scroll', direction=action.target)
        else:
            raise ValueError(f'Unsupported action kind {action.kind}')
        self.env.execute_action(native)
        if self.config.pause_seconds:
            time.sleep(self.config.pause_seconds)
        obs = self.observe()
        return obs, self.success()

    def success(self) -> bool:
        return bool(self.task is not None and self.task.is_successful(self.env) == 1)

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
                self.task = None
                self.last_state = None