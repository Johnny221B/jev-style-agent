from .core import Action, Observation

class MockEnvironment:
    """Three-step fake UI for testing architecture; not AndroidWorld."""
    labels = ("Settings", "Display", "Dark mode")
    def __init__(self): self.pos = 0
    def reset(self, task=None):
        self.pos = 0
        return (task or "Enable dark mode", self._obs())
    def _obs(self):
        return Observation(text=f"Page {self.pos}: {self.labels[min(self.pos,2)]}", actions=[Action("click", self.labels[min(self.pos,2)])] if self.pos<3 else [])
    def step(self, action):
        if self.pos<3 and action == self._obs().actions[0]: self.pos+=1
        return self._obs(), self.pos>=3
    def success(self): return self.pos>=3
    def close(self): pass