from types import SimpleNamespace
from unittest.mock import patch
from jev_agent.androidworld_env import AndroidWorldConfig, AndroidWorldEnvironment
from jev_agent.core import Action


def test_default_config():
    cfg = AndroidWorldConfig()
    assert cfg.console_port == 5554
    assert cfg.grpc_port == 8554


def test_action_identity():
    a = Action('click', '12')
    assert a.label() == 'click(12)'


def test_environment_requires_reset():
    env = AndroidWorldEnvironment()
    try:
        env.observe()
    except RuntimeError as err:
        assert 'reset' in str(err)
    else:
        raise AssertionError('observe must require reset')