"""Cold discovery and batched host validation of frozen inference objects."""

_MISSING = object()
_SCALARS = (bool, int, float, str)


def _identity(value):
    """Cold/reference fingerprint; the hot checker does not call this helper."""
    return (
        value
        if value is _MISSING or value is None or isinstance(value, _SCALARS)
        else id(value)
    )


def tensor_version(tensor):
    """Inference tensors expose a cheap flag, but deliberately have no version."""
    flag = getattr(tensor, "is_inference", None)
    if callable(flag) and flag():
        return None
    try:
        return tensor._version
    except RuntimeError:
        # Compatibility for metadata-only interfaces without is_inference().
        return None


def _tensor_state(tensor):
    return (
        tensor_version(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class ModelTopologyGuard:
    """Cache complete policy modules cold; check registry/training fields hot."""

    def __init__(self, model):
        self.modules = tuple(model.modules())
        self.records = []
        for module in self.modules:
            state = vars(module)
            registry = state.get("_modules", {})
            self.records.append(
                (module, state, type(module), registry, tuple(registry.items()))
            )

    def training_changed(self):
        return any(
            state.get("training", False) is not False
            for _, state, _, _, _ in self.records
        )

    def unchanged(self):
        for module, state, cls, registry, entries in self.records:
            if vars(module) is not state or type(module) is not cls:
                return False
            if state.get("_modules", registry) is not registry or len(registry) != len(
                entries
            ):
                return False
            for name, expected in entries:
                if registry.get(name, _MISSING) is not expected:
                    return False
        return True


class HostGuardPlan:
    """Discover once; batch owner/header/container checks without hot recursion."""

    def __init__(self, model, *, methods, fields, global_bindings):
        self.checks, self.seen, self.attributes, self.owners = [], set(), set(), set()
        self.owner_checks, self.attribute_checks, self.method_checks = [], [], []
        self.tensor_checks, self.function_checks, self.cell_checks = [], [], []
        self.mapping_checks, self.sequence_checks = [], []
        self.methods, self.fields, self.global_bindings = (
            methods,
            fields,
            global_bindings,
        )
        modules = tuple(model.named_modules())
        owners = dict(modules)
        for _, module in modules:
            self.attribute(module, "training")
            for name in (
                "_modules",
                "_parameters",
                "_buffers",
                "_forward_hooks",
                "_forward_pre_hooks",
            ):
                if name in vars(module):
                    self.attribute(module, name)
            config = getattr(module, "config", None)
            if config is not None:
                self.attribute(module, "config")
                for name in ("_attn_implementation", "use_cache"):
                    if hasattr(config, name):
                        self.attribute(config, name)
            for name in methods:
                if callable(getattr(module, name, None)):
                    self.attribute(module, name, method=True)
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            owner_name, _, field = name.rpartition(".")
            owner = owners.get(owner_name)
            if owner is not None:
                # Registered replacement is already checked through the module's
                # parameter/buffer registry. Standalone metadata interfaces need
                # their explicit attribute getter instead.
                state = vars(owner)
                if field not in state.get("_parameters", {}) and field not in state.get(
                    "_buffers", {}
                ):
                    self.attribute(owner, field)
            self.value(tensor)

    def _append(self, group, record):
        group.append(record)
        self.checks.append(record)

    def _attribute_check(self, getter, expected):
        self._append(
            self.attribute_checks,
            (getter, expected, expected is None or isinstance(expected, _SCALARS)),
        )

    def attribute(self, owner, name, *, method=False):
        if id(owner) not in self.owners:
            self.owners.add(id(owner))
            self._append(self.owner_checks, (owner, vars(owner), type(owner)))
        identity = id(owner), name
        if identity in self.attributes:
            return
        self.attributes.add(identity)
        values = vars(owner)
        if name in values:
            getter = lambda values=values, name=name: values.get(name, _MISSING)
        elif name in values.get("_parameters", {}):
            self.attribute(owner, "_parameters")
            registry = values["_parameters"]
            getter = lambda registry=registry, name=name: registry.get(name, _MISSING)
        elif name in values.get("_buffers", {}):
            self.attribute(owner, "_buffers")
            registry = values["_buffers"]
            getter = lambda registry=registry, name=name: registry.get(name, _MISSING)
        else:
            getter = lambda owner=owner, name=name: getattr(owner, name, _MISSING)
        value = getter()
        if method and hasattr(value, "__func__"):
            self._append(self.method_checks, (getter, value.__func__, value.__self__))
        else:
            self._attribute_check(getter, value)
        self.value(value)

    def value(self, value):
        if value is None or value is _MISSING or isinstance(value, _SCALARS):
            return
        identity = (
            (id(value.__func__), id(value.__self__))
            if hasattr(value, "__func__")
            else id(value)
        )
        if identity in self.seen:
            return
        self.seen.add(identity)
        if all(hasattr(value, name) for name in ("shape", "dtype", "device", "stride")):
            self._append(self.tensor_checks, (value, _tensor_state(value)))
        elif isinstance(value, dict):
            entries = tuple(
                (key, item, item is None or isinstance(item, _SCALARS))
                for key, item in value.items()
            )
            self._append(self.mapping_checks, (value, entries))
            for item in value.values():
                self.value(item)
        elif isinstance(value, (tuple, list)):
            # Tuples cannot change their elements; the containing field/header
            # checks replacement. Only nested mutable objects need validation.
            if isinstance(value, list):
                entries = tuple(
                    (item, item is None or isinstance(item, _SCALARS)) for item in value
                )
                self._append(self.sequence_checks, (value, entries))
            for item in value:
                self.value(item)
        else:
            function = getattr(value, "__func__", value)
            if hasattr(function, "__code__"):
                if id(function) not in self.owners:
                    self.owners.add(id(function))
                    self._append(
                        self.function_checks,
                        (
                            function,
                            function.__code__,
                            function.__defaults__,
                            function.__kwdefaults__,
                            function.__closure__,
                        ),
                    )
                    for item in (function.__defaults__, function.__kwdefaults__):
                        self.value(item)
                    for cell in function.__closure__ or ():
                        item = cell.cell_contents
                        self._append(
                            self.cell_checks,
                            (cell, item, item is None or isinstance(item, _SCALARS)),
                        )
                        self.value(item)
                    for name in self.global_bindings:
                        if (
                            name in function.__code__.co_names
                            and name in function.__globals__
                        ):
                            namespace = function.__globals__
                            marker = id(namespace), name
                            if marker not in self.attributes:
                                self.attributes.add(marker)
                                self._attribute_check(
                                    lambda namespace=namespace, name=name: (
                                        namespace.get(name, _MISSING)
                                    ),
                                    namespace[name],
                                )
                            self.value(namespace[name])
                owner = getattr(value, "__self__", None)
                if owner is not None:
                    self.value(owner)
            elif hasattr(value, "_global_mapping"):
                self._attribute_check(lambda value=value: value["sdpa"], value["sdpa"])
                self.value(value["sdpa"])
            else:
                for name in self.fields:
                    if hasattr(value, name):
                        self.attribute(value, name)

    def unchanged(self):
        """No discovery, recursive stamps, fingerprint callbacks or GPU reads."""
        try:
            for owner, state, cls in self.owner_checks:
                if vars(owner) is not state or type(owner) is not cls:
                    return False
            for function, code, defaults, kwdefaults, closure in self.function_checks:
                if (
                    function.__code__ is not code
                    or function.__defaults__ is not defaults
                    or function.__kwdefaults__ is not kwdefaults
                    or function.__closure__ is not closure
                ):
                    return False
            for mapping, entries in self.mapping_checks:
                if len(mapping) != len(entries):
                    return False
                for key, expected, scalar in entries:
                    actual = mapping.get(key, _MISSING)
                    if actual is not expected and (
                        not scalar
                        or type(actual) is not type(expected)
                        or actual != expected
                    ):
                        return False
            for values, entries in self.sequence_checks:
                if len(values) != len(entries):
                    return False
                for actual, (expected, scalar) in zip(values, entries):
                    if actual is not expected and (
                        not scalar
                        or type(actual) is not type(expected)
                        or actual != expected
                    ):
                        return False
            for cell, expected, scalar in self.cell_checks:
                actual = cell.cell_contents
                if actual is not expected and (
                    not scalar
                    or type(actual) is not type(expected)
                    or actual != expected
                ):
                    return False
            for getter, expected, scalar in self.attribute_checks:
                actual = getter()
                if actual is not expected and (
                    not scalar
                    or type(actual) is not type(expected)
                    or actual != expected
                ):
                    return False
            for getter, function, owner in self.method_checks:
                actual = getter()
                if (
                    getattr(actual, "__func__", None) is not function
                    or getattr(actual, "__self__", None) is not owner
                ):
                    return False
            for tensor, expected in self.tensor_checks:
                if _tensor_state(tensor) != expected:
                    return False
            return True
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return False
