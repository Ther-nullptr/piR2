"""Fixtures load model dependencies only after explicit GPU test selection."""

import pytest


@pytest.fixture
def policy_and_inputs():
    from tests.helpers.groot_streaming import make_policy_and_inputs

    return make_policy_and_inputs()
