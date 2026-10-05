"""Test the shared runtime patch lifecycle without importing model libraries."""

from types import SimpleNamespace

import pytest

from coexecution.patching import ScopedReplacements


class TwoReplacements(ScopedReplacements):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def _install(self):
        self.replace(self.target, "vision", object())
        self.replace(self.target, "text", object())
        return self


def test_partial_entry_failure_restores_previous_replacement():
    original = object()
    target = SimpleNamespace(vision=original)
    adapter = TwoReplacements(target)
    with pytest.raises(AttributeError), adapter:
        pass
    assert target.vision is original
    assert adapter.restore == []


def test_body_failure_restores_both_replacements():
    vision, text = object(), object()
    target = SimpleNamespace(vision=vision, text=text)
    adapter = TwoReplacements(target)
    with pytest.raises(RuntimeError, match="body failed"), adapter:
        assert target.vision is not vision
        assert target.text is not text
        raise RuntimeError("body failed")
    assert target.vision is vision
    assert target.text is text
    assert adapter.restore == []
