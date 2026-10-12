"""Image ownership and scoped rebinding, without neural computation."""

import hashlib
import importlib
import importlib.util
import sys
import threading
from functools import wraps
from types import FunctionType, ModuleType, SimpleNamespace

import numpy as np
import pytest

MODULE = "coexecution.optimization.execution.images"


@pytest.fixture
def image_interface(monkeypatch):
    assert importlib.util.find_spec(MODULE) is not None, "Image preparation missing"
    module = importlib.import_module(MODULE)
    pil = SimpleNamespace(fromarray=lambda value: value.copy())
    namespace = {"np": np, "Image": pil}
    exec(  # noqa: S102 -- isolated pinned-interface fixture, never external input
        """
class Processor:
    def _get_vlm_inputs(self):
        pass
    def _apply_vlm_processing(self, images, language):
        values = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]
        conversation = [{"role": "user", "content": [
            *[{"type": "image", "image": value} for value in values],
            {"type": "text", "text": language}]}]
        return {"vlm_content": {"images": values, "conversation": conversation,
            "text": self.processor.apply_chat_template(conversation,
                tokenize=False, add_generation_prompt=False)}}
""",
        namespace,
    )

    class Consumer:
        pause = None

        def __init__(self):
            self.image_processor = Pixels()

        def __call__(self, features):
            return features

        def apply_chat_template(self, conversation, **options):
            if self.pause:
                self.pause()
            assert options == {"tokenize": False, "add_generation_prompt": False}
            return conversation[0]["content"][-1]["text"]

    class BasePixels:
        def preprocess(self, images):
            return self._preprocess(images)

    class Pixels(BasePixels):
        def __call__(self, images):
            return self.preprocess(images)

        def preprocess(self, images):
            return super().preprocess(images)

        def _preprocess(self, images):
            return images

    class Collator:
        def __init__(self):
            self.processor = Consumer()

        def __call__(self, features):
            return self.processor(features)

    cls = namespace["Processor"]
    processor = cls()
    processor.processor, processor.collator = Consumer(), Collator()
    policy = SimpleNamespace(
        model=SimpleNamespace(), processor=processor, collate_fn=processor.collator
    )
    monkeypatch.setattr(
        module, "_interfaces", lambda: (cls, Collator, Consumer, pil, np, (Pixels,))
    )
    definitions = {}
    for item in (cls, Collator, Consumer, Pixels, BasePixels):
        for name, function in vars(item).items():
            if hasattr(function, "__code__"):
                definitions.setdefault(item.__module__, {})[
                    item.__qualname__ + "." + name
                ] = function.__code__
    monkeypatch.setattr(module, "_source_definitions", definitions.__getitem__)
    return SimpleNamespace(module=module, policy=policy, processor=processor)


def test_arrays_are_owned_contiguous_rgb_and_preserve_conversation(image_interface):
    item = image_interface
    source = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
    original = item.processor._apply_vlm_processing
    with item.module.ImageArrayPreparation(item.policy) as adapter:
        result = item.processor._apply_vlm_processing(source, "move")["vlm_content"]
        for index, value in enumerate(result["images"]):
            np.testing.assert_array_equal(value, source[index].transpose(1, 2, 0))
            assert value.flags.c_contiguous and value.flags.owndata
            assert result["conversation"][0]["content"][index]["image"] is value
        saved = result["images"][0].copy()
        source.fill(0)
        np.testing.assert_array_equal(result["images"][0], saved)
        assert result["text"] == "move"
        assert adapter.evidence()["images"] == 2
    assert item.processor._apply_vlm_processing == original
    assert "_apply_vlm_processing" not in vars(item.processor)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((3, 4, 5), np.uint8),
        np.zeros((1, 4, 5, 3), np.uint8),
        np.zeros((1, 3, 4, 5), np.float32),
        [],
    ],
)
def test_unsupported_input_fails_clearly(image_interface, source):
    with (
        image_interface.module.ImageArrayPreparation(image_interface.policy),
        pytest.raises(ValueError, match="uint8.*TCHW"),
    ):
        image_interface.processor._apply_vlm_processing(source, "move")


def test_scope_reentry_stale_bindings_double_ownership_and_active_close(
    image_interface,
):
    item = image_interface
    scope = item.module.ImageArrayPreparation(item.policy)
    source = np.zeros((1, 3, 4, 5), np.uint8)
    with scope:
        old = item.processor._apply_vlm_processing
        with pytest.raises(RuntimeError, match="already active"):
            scope.__enter__()
        with pytest.raises(ValueError, match="implementation"):
            item.module.ImageArrayPreparation(item.policy).__enter__()
        scope.lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="active call"):
                scope.close()
        finally:
            scope.lock.release()
    scope.close()
    with scope:
        with pytest.raises(RuntimeError, match="closed"):
            old(source, "move")
        assert item.processor._apply_vlm_processing(source, "move")


def test_consumer_replacement_is_rejected(image_interface):
    item = image_interface
    with item.module.ImageArrayPreparation(item.policy):
        item.processor.collator.processor = object()
        with pytest.raises(RuntimeError, match="replaced"):
            item.processor._apply_vlm_processing(
                np.zeros((1, 3, 4, 5), np.uint8), "move"
            )


def test_active_call_blocks_overlapping_submission_and_close(image_interface):
    item = image_interface
    entered, release = threading.Event(), threading.Event()
    failures = []

    def pause():
        entered.set()
        assert release.wait(5)

    item.processor.processor.pause = pause
    source = np.zeros((1, 3, 4, 5), np.uint8)
    with item.module.ImageArrayPreparation(item.policy) as scope:

        def call():
            try:
                item.processor._apply_vlm_processing(source, "move")
            except BaseException as error:  # noqa: BLE001 -- report worker failures to main test
                failures.append(error)

        worker = threading.Thread(target=call)
        worker.start()
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="serial submissions"):
                item.processor._apply_vlm_processing(source, "move")
            with pytest.raises(RuntimeError, match="active call"):
                scope.close()
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive() and not failures


@pytest.mark.parametrize("target", ["collator", "template", "qwen"])
def test_unknown_consumer_implementation_is_rejected_on_entry(image_interface, target):
    item = image_interface
    if target == "collator":
        item.processor.collator.__call__ = lambda *args: None
    elif target == "template":
        item.processor.processor.apply_chat_template = lambda *args: None
    else:
        item.processor.collator.processor.__call__ = lambda *args: None
    original = item.processor._apply_vlm_processing
    with pytest.raises(ValueError, match="consumer implementation"):
        item.module.ImageArrayPreparation(item.policy).__enter__()
    assert item.processor._apply_vlm_processing == original


def test_close_restores_owned_instance_method_and_never_overwrites_later_scope(
    image_interface,
):
    item = image_interface
    original = item.processor._apply_vlm_processing
    item.processor._apply_vlm_processing = original
    first = item.module.ImageArrayPreparation(item.policy)
    with first:
        pass
    assert vars(item.processor)["_apply_vlm_processing"] is original
    with item.module.ImageArrayPreparation(item.policy):
        replacement = item.processor._apply_vlm_processing
        first.close()
        assert item.processor._apply_vlm_processing is replacement


@pytest.mark.parametrize("target", ["processor", "collator", "qwen", "template"])
@pytest.mark.parametrize("mutation", ["class", "code"])
def test_entry_rejects_changed_class_implementation(
    image_interface, monkeypatch, target, mutation
):
    item = image_interface
    owner, name = {
        "processor": (item.processor, "_apply_vlm_processing"),
        "collator": (item.processor.collator, "__call__"),
        "qwen": (item.processor.processor, "__call__"),
        "template": (item.processor.processor, "apply_chat_template"),
    }[target]
    original = getattr(type(owner), name)

    def foreign(self, *args, **kwargs):
        return {"foreign": True}

    if mutation == "class":
        monkeypatch.setattr(
            type(owner), name, FunctionType(foreign.__code__, original.__globals__)
        )
    else:
        monkeypatch.setattr(original, "__code__", foreign.__code__)
    with pytest.raises(ValueError, match="implementation"):
        item.module.ImageArrayPreparation(item.policy).__enter__()


@pytest.mark.parametrize("when", ["before", "active"])
def test_pixel_processor_replacement_is_rejected(image_interface, when):
    item = image_interface
    if when == "before":
        item.processor.collator.processor.image_processor = object()
        with pytest.raises(ValueError, match="pixel processor"):
            item.module.ImageArrayPreparation(item.policy).__enter__()
    else:
        with item.module.ImageArrayPreparation(item.policy):
            item.processor.collator.processor.image_processor = object()
            with pytest.raises(RuntimeError, match="replaced"):
                item.processor._apply_vlm_processing(
                    np.zeros((1, 3, 4, 5), np.uint8), "move"
                )


@pytest.mark.parametrize("method", ["__call__", "preprocess", "_preprocess"])
@pytest.mark.parametrize("when", ["before", "active"])
def test_pixel_processor_method_mutation_is_rejected(
    image_interface, monkeypatch, method, when
):
    item = image_interface
    pixels = item.processor.collator.processor.image_processor
    original = getattr(type(pixels), method)

    def foreign(self, *args, **kwargs):
        return []

    foreign_code = foreign.__code__.replace(co_freevars=original.__code__.co_freevars)

    if when == "before":
        monkeypatch.setattr(original, "__code__", foreign_code)
        with pytest.raises(ValueError, match="implementation"):
            item.module.ImageArrayPreparation(item.policy).__enter__()
    else:
        with item.module.ImageArrayPreparation(item.policy):
            monkeypatch.setattr(original, "__code__", foreign_code)
            with pytest.raises(RuntimeError, match="replaced"):
                item.processor._apply_vlm_processing(
                    np.zeros((1, 3, 4, 5), np.uint8), "move"
                )


def test_cold_source_admission_compiles_reference_and_rejects_source_or_code_edits(
    monkeypatch, tmp_path
):
    module = importlib.import_module(MODULE)
    source = b"class Pinned:\n    def run(self, value):\n        return value\n"
    path = tmp_path / "pinned_fixture.py"
    path.write_bytes(source)
    fixture = ModuleType("pinned_fixture")
    fixture.__file__ = str(path)
    exec(compile(source, str(path), "exec"), vars(fixture))  # noqa: S102 -- static test fixture
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    monkeypatch.setitem(
        module._SOURCE_HASHES, fixture.__name__, hashlib.sha256(source).hexdigest()
    )
    original = fixture.Pinned.run
    assert module._admit_method(fixture.Pinned(), "run", {})

    def foreign(self, value):
        return None

    fixture.Pinned.run = FunctionType(foreign.__code__, original.__globals__)
    with pytest.raises(ValueError, match="implementation"):
        module._admit_method(fixture.Pinned(), "run", {})
    fixture.Pinned.run = original
    path.write_bytes(source + b"# unpinned revision\n")
    with pytest.raises(ValueError, match="source revision"):
        module._admit_method(fixture.Pinned(), "run", {})


def test_source_admission_never_runs_in_per_image_path(image_interface, monkeypatch):
    item = image_interface
    with item.module.ImageArrayPreparation(item.policy):
        monkeypatch.setattr(
            item.module,
            "_source_definitions",
            lambda *args: pytest.fail("Hot path accessed source"),
        )
        item.processor._apply_vlm_processing(np.zeros((1, 3, 4, 5), np.uint8), "move")


@pytest.mark.parametrize("when", ["before", "active"])
def test_decorator_invoked_callable_cell_cannot_change(
    image_interface, monkeypatch, when
):
    item = image_interface
    cls = type(item.processor.processor)
    original = cls.apply_chat_template

    def decorate(func):
        spare = func

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            assert callable(spare)
            return func(*args, **kwargs)

        return wrapped_func

    wrapper = decorate(original)
    monkeypatch.setattr(cls, "apply_chat_template", wrapper)
    references = item.module._source_definitions

    def source(module):
        if module == "transformers.utils.deprecation":
            return {"deprecate_kwarg.wrapper.wrapped_func": wrapper.__code__}
        return references(module)

    monkeypatch.setattr(item.module, "_source_definitions", source)
    cell = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))["func"]

    def foreign(*args, **kwargs):
        pytest.fail("Changed decorator callable was invoked")

    if when == "before":
        cell.cell_contents = foreign
        with pytest.raises(ValueError, match="implementation wrapper"):
            item.module.ImageArrayPreparation(item.policy).__enter__()
    else:
        with item.module.ImageArrayPreparation(item.policy):
            cell.cell_contents = foreign
            with pytest.raises(RuntimeError, match="replaced"):
                item.processor._apply_vlm_processing(
                    np.zeros((1, 3, 4, 5), np.uint8), "move"
                )


@pytest.mark.parametrize("height", [1, 3])
def test_ambiguous_hwc_channel_dimensions_fail_closed(image_interface, height):
    item = image_interface
    with (
        item.module.ImageArrayPreparation(item.policy),
        pytest.raises(ValueError, match="ambiguous.*channel"),
    ):
        item.processor._apply_vlm_processing(
            np.zeros((1, 3, height, 24), np.uint8), "move"
        )


@pytest.mark.parametrize("when", ["before", "active"])
@pytest.mark.parametrize("mutation", ["replacement", "code"])
def test_pixel_super_implementation_mutation_is_rejected(
    image_interface, monkeypatch, when, mutation
):
    item = image_interface
    pixels = item.processor.collator.processor.image_processor
    base = type(pixels).__mro__[1]

    def foreign(*args, **kwargs):
        pytest.fail("Changed superclass pixel method was invoked")

    def mutate():
        if mutation == "replacement":
            monkeypatch.setattr(base, "preprocess", foreign)
        else:
            monkeypatch.setattr(base.preprocess, "__code__", foreign.__code__)

    if when == "before":
        mutate()
        with pytest.raises(ValueError, match="implementation"):
            item.module.ImageArrayPreparation(item.policy).__enter__()
    else:
        with item.module.ImageArrayPreparation(item.policy):
            mutate()
            with pytest.raises(RuntimeError, match="replaced"):
                item.processor._apply_vlm_processing(
                    np.zeros((1, 3, 4, 5), np.uint8), "move"
                )
