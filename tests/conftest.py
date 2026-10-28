"""Puts the ``gr00t`` test stand-ins (``tests/fake_gr00t``) first on ``sys.path``; the ``fixed_yaml`` fixture."""

import sys
from pathlib import Path

import pytest

FAKE_GR00T = Path(__file__).resolve().parent / "fake_gr00t"
sys.path.insert(0, str(FAKE_GR00T))
for name in [m for m in sys.modules if m == "gr00t" or m.startswith("gr00t.")]:
    del sys.modules[name]


@pytest.fixture
def fixed_yaml(tmp_path) -> Path:
    """``tests.helpers.TEST_CONFIG`` as an eval config file."""
    from tests.helpers import eval_variant

    return eval_variant(tmp_path / "fixed")
