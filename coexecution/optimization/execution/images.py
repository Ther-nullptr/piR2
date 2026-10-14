"""Instance-local removal of the pinned processor's PIL round trip.

The original conversation/template code runs with fresh owned uint8 RGB arrays.
Importing this module loads neither image/model dependencies nor a compiler.
"""

import hashlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, MethodType, SimpleNamespace

from coexecution.optimization.execution.patching import (
    clone_forward,
    effective_callable_stamp,
)

# Pinned GR00T fork and Transformers 4.57.3; checked only on scope entry.
_SOURCE_HASHES = {
    "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7": "f5df900d4a1e4c2eb4d12d871cec0946bc06affb29111ba80645d30125193098",
    "transformers.processing_utils": "6c29de5c3fc674cf21ba6377a39cbafb2ba3e4bc2d1202d4b944f8cf840e6f33",
    "transformers.utils.deprecation": "aec6dcedb6c73e6bde3d29a4a5ffe75d0ece217e884c548aea72711efbb46d8e",
    "transformers.models.qwen3_vl.processing_qwen3_vl": "efd8d64aaf608aad1ffb3e6d503d6a99e5227d007df95c1d9fa905d998cda4a9",
    "transformers.image_processing_utils": "740d1fecbdf24983538443b24d9215b50c494b8abed460346bcd8f3c3d69d249",
    "transformers.image_processing_utils_fast": "038a4aa9e57716be46a18822f5f2f977c8a5b9d0e9cc3db63f032cddafd7d1ac",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl": "7820a0fcca107e75605e08d9b774285ca2b0316f857bc0225c779794705ecf4f",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast": "09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c",
}


def _source_definitions(module_name):
    module = sys.modules.get(module_name)
    path = Path(module.__file__) if module is not None else None
    if path is None or module_name not in _SOURCE_HASHES:
        raise ValueError("Unsupported image consumer implementation source")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != _SOURCE_HASHES[module_name]:
        raise ValueError("Unsupported image consumer implementation source revision")
    result = {}

    def visit(code, parents=()):
        for value in code.co_consts:
            if isinstance(value, CodeType):
                key = (*parents, value.co_name)
                result[".".join(key)] = value
                visit(value, key)

    visit(compile(source, str(path), "exec", dont_inherit=True))
    return result


def _code_equal(actual, expected):
    if not isinstance(actual, CodeType):
        return False
    return actual.replace(co_filename=expected.co_filename) == expected


def _admit_method(owner, name, definitions):
    classes = [cls for cls in type(owner).__mro__ if name in vars(cls)]
    bound = getattr(owner, name)
    function = getattr(bound, "__func__", None)
    if function is not getattr(classes[0], name):
        raise ValueError("Unsupported image consumer implementation")
    references = []
    for cls in classes:
        function = vars(cls)[name]
        references.append((cls, name, function))
        references.extend(_admit_definition(cls, name, function, definitions))
    return references


def _admit_definition(cls, name, function, definitions):
    """Validate each definition in the call chain, including super() targets."""
    expected_name = cls.__qualname__ + "." + name
    if cls.__module__ not in definitions:
        definitions[cls.__module__] = _source_definitions(cls.__module__)
    expected = definitions[cls.__module__].get(expected_name)
    wrappers = []
    while getattr(function, "__wrapped__", None) is not None:
        wrapped = function.__wrapped__
        if "transformers.utils.deprecation" not in definitions:
            definitions["transformers.utils.deprecation"] = _source_definitions(
                "transformers.utils.deprecation"
            )
        wrapper = definitions["transformers.utils.deprecation"].get(
            "deprecate_kwarg.wrapper.wrapped_func"
        )
        cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        if (
            wrapper is None
            or not _code_equal(function.__code__, wrapper)
            or "func" not in cells
            or cells["func"].cell_contents is not wrapped
        ):
            raise ValueError("Unsupported image consumer implementation wrapper")
        wrappers.extend(
            (
                (function, "__code__", function.__code__),
                (function, "__wrapped__", wrapped),
            )
        )
        wrappers.extend(
            (cell, "cell_contents", cell.cell_contents)
            for cell in cells.values()
            if callable(cell.cell_contents)
        )
        function = wrapped
    if expected is None or not _code_equal(function.__code__, expected):
        raise ValueError("Unsupported image consumer implementation")
    wrappers.append((function, "__code__", function.__code__))
    return wrappers


def _interfaces():
    import numpy as np
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        Gr00tN1d7DataCollator,
        Gr00tN1d7Processor,
    )
    from PIL import Image
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen2VLImageProcessorFast,
        Qwen3VLProcessor,
    )

    return (
        Gr00tN1d7Processor,
        Gr00tN1d7DataCollator,
        Qwen3VLProcessor,
        Image,
        np,
        (Qwen2VLImageProcessor, Qwen2VLImageProcessorFast),
    )


class ImageArrayPreparation:
    """One serial processor; close before replacing processors or consumers."""

    def __init__(self, policy):
        self.policy = policy
        self.lock = threading.Lock()
        self.closed = True
        self._close_owner = None
        self._generation = 0
        self._references, self._consumers = [], []
        self._binding = None
        self._images = self._calls = 0

    def __enter__(self):
        if not self.closed:
            raise RuntimeError("Image preparation scope is already active")
        processor_type, collator_type, qwen_type, pil, np, pixel_types = _interfaces()
        definitions = {}
        self._validate_references()
        processor = self.policy.processor
        if type(processor) is not processor_type:
            raise ValueError("Image preparation requires the pinned GR00T processor")
        original = processor._apply_vlm_processing
        function = getattr(original, "__func__", None)
        if (
            function is not processor_type._apply_vlm_processing
            or function.__globals__.get("Image") is not pil
            or function.__globals__.get("np") is not np
            or getattr(processor._get_vlm_inputs, "__func__", None)
            is not processor_type._get_vlm_inputs
        ):
            raise ValueError("Unsupported image preparation implementation")
        refs = [
            (self.policy, "processor", processor),
            (self.policy, "model", self.policy.model),
        ]
        for name in ("_apply_vlm_processing", "_get_vlm_inputs"):
            refs.extend(_admit_method(processor, name, definitions))
        consumers = [
            (
                processor,
                "_get_vlm_inputs",
                effective_callable_stamp(processor._get_vlm_inputs),
            )
        ]
        collators = [(processor, "collator", processor.collator)]
        if hasattr(self.policy, "collate_fn"):
            if self.policy.collate_fn is not processor.collator:
                raise ValueError("Unsupported policy image collator consumer")
            refs.append((self.policy, "collate_fn", processor.collator))
        if hasattr(self.policy.model, "collator"):
            collators.append(
                (self.policy.model, "collator", self.policy.model.collator)
            )
        processors = [processor.processor]
        refs.append((processor, "processor", processor.processor))
        for owner, name, collator in collators:
            if type(collator) is not collator_type:
                raise ValueError("Image preparation requires the pinned GR00T collator")
            refs.extend(_admit_method(collator, "__call__", definitions))
            if (
                getattr(collator.__call__, "__func__", None)
                is not collator_type.__call__
            ):
                raise ValueError("Unsupported image collator consumer implementation")
            refs.extend(
                ((owner, name, collator), (collator, "processor", collator.processor))
            )
            consumers.append(
                (collator, "__call__", effective_callable_stamp(collator.__call__))
            )
            processors.append(collator.processor)
        for consumer in processors:
            if type(consumer) is not qwen_type:
                raise ValueError(
                    "Image preparation requires Qwen3VLProcessor consumers"
                )
            for name in ("__call__", "apply_chat_template"):
                refs.extend(_admit_method(consumer, name, definitions))
                if getattr(getattr(consumer, name), "__func__", None) is not getattr(
                    qwen_type, name
                ):
                    raise ValueError("Unsupported Qwen image consumer implementation")
                consumers.append(
                    (consumer, name, effective_callable_stamp(getattr(consumer, name)))
                )
            pixels = consumer.image_processor
            if type(pixels) not in pixel_types:
                raise ValueError(
                    "Image preparation requires a stock Qwen2VL pixel processor"
                )
            refs.append((consumer, "image_processor", pixels))
            for name in (
                "__call__",
                "preprocess",
                "_preprocess",
                "rescale",
                "normalize",
                "_preprocess_image_like_inputs",
                "_prepare_image_like_inputs",
                "_prepare_images_structure",
                "_process_image",
                "convert_to_rgb",
                "resize",
                "rescale_and_normalize",
                "_further_process_kwargs",
                "_validate_preprocess_kwargs",
            ):
                if callable(getattr(pixels, name, None)):
                    refs.extend(_admit_method(pixels, name, definitions))
                    consumers.append(
                        (pixels, name, effective_callable_stamp(getattr(pixels, name)))
                    )
        self._references, self._consumers = refs, consumers
        self._generation += 1
        generation = self._generation

        def owned_rgb(value):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.uint8
                or value.ndim != 3
                or value.shape[-1] != 3
            ):
                raise ValueError("Image preparation requires uint8 HWC RGB arrays")
            self._images += 1
            return np.array(value, copy=True, order="C")

        cloned = MethodType(
            clone_forward(function, {"Image": SimpleNamespace(fromarray=owned_rgb)}),
            processor,
        )

        def prepare(_processor, images, language):
            if self.closed or generation != self._generation:
                raise RuntimeError("Image preparation scope is closed")
            if not self.lock.acquire(blocking=False):
                raise RuntimeError("Image preparation requires serial submissions")
            try:
                if self.closed:
                    raise RuntimeError("Image preparation scope is closed")
                self._validate_references()
                if processor._apply_vlm_processing is not self._binding[1]:
                    raise RuntimeError("Image preparation implementation was replaced")
                if (
                    type(images) is not np.ndarray
                    or images.dtype != np.uint8
                    or images.ndim != 4
                    or images.shape[1] != 3
                    or any(size < 1 for size in images.shape)
                ):
                    raise ValueError(
                        "Image preparation requires nonempty uint8 RGB TCHW arrays"
                    )
                if not isinstance(language, str):
                    raise TypeError("Image preparation requires string language")
                if images.shape[2] in (1, 3):
                    raise ValueError(
                        "Image preparation rejects ambiguous HWC channel dimensions (height 1 or 3)"
                    )
                result = cloned(images, language)
                self._calls += 1
                return result
            finally:
                self.lock.release()

        bound = MethodType(prepare, processor)
        self._binding = (
            processor,
            bound,
            "_apply_vlm_processing" in vars(processor),
            original,
        )
        processor._apply_vlm_processing = bound
        self.closed = False
        return self

    def _validate_references(self):
        if any(
            getattr(owner, name, None) is not expected
            for owner, name, expected in self._references
        ):
            raise RuntimeError("Image processor, model or collator was replaced")
        if any(
            effective_callable_stamp(getattr(owner, name, None)) != expected
            for owner, name, expected in self._consumers
        ):
            raise RuntimeError("Image preparation consumer was replaced")

    def evidence(self):
        return {
            "active": not self.closed,
            "calls": self._calls,
            "images": self._images,
            "representation": "owned contiguous uint8 HWC RGB",
            "scope": "original GR00T conversation/template and Qwen collation",
        }

    @contextmanager
    def close_guard(self):
        owner = threading.get_ident()
        if self.closed or self._close_owner == owner:
            yield
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close image preparation during an active call")
        self._close_owner = owner
        try:
            yield
        finally:
            self._close_owner = None
            self.lock.release()

    def close(self):
        with self.close_guard():
            self.closed = True
            if self._binding is not None:
                processor, bound, had, original = self._binding
                if getattr(processor, "_apply_vlm_processing", None) is bound:
                    if had:
                        processor._apply_vlm_processing = original
                    else:
                        delattr(processor, "_apply_vlm_processing")
                self._binding = None

    def __exit__(self, *exception):
        self.close()
