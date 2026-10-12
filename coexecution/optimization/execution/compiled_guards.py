"""Opt-in native predicates compiled from the exact Python guard source.

Compilation happens once at scope startup. All discovery records stay owned by
the Python guard; only unbound predicates are rebound to that guard instance.
"""

import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from types import MethodType

SOURCE = Path(__file__).with_name("guards.py")
CYTHON_VERSION = "3.1.2"
_loaded = {}
_lock = threading.Lock()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _build_settings():
    from importlib.metadata import PackageNotFoundError, version

    if platform.python_implementation() != "CPython":
        raise RuntimeError("Compiled host guards require CPython")
    try:
        cython, setuptools = version("Cython"), version("setuptools")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Compiled host guards require Cython==3.1.2 and setuptools"
        ) from error
    if cython != CYTHON_VERSION:
        raise RuntimeError(
            f"Compiled host guards require Cython=={CYTHON_VERSION}; found {cython}"
        )
    configuration = {
        key: sysconfig.get_config_var(key)
        for key in (
            "CC",
            "CFLAGS",
            "CCSHARED",
            "LDSHARED",
            "LDFLAGS",
            "SOABI",
            "EXT_SUFFIX",
            "Py_DEBUG",
        )
    }
    environment = {
        key: os.environ.get(key)
        for key in (
            "CC",
            "CXX",
            "CFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "LDSHARED",
            "AR",
            "ARFLAGS",
            "_PYTHON_SYSCONFIGDATA_NAME",
        )
    }
    compiler = shlex.split(environment["CC"] or configuration["CC"] or "cc")[0]
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError(f"Compiled host guards compiler is unavailable: {compiler}")
    return {
        "cython": cython,
        "setuptools": setuptools,
        "python": sys.version,
        "abi": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
        "configuration": configuration,
        "environment": environment,
        "compiler": {
            "path": executable,
            "sha256": _sha256(Path(executable).read_bytes()),
        },
        "directives": {"language_level": 3, "binding": True},
        "extra_compile_args": ["-O3"],
    }


def _build(directory, name, source_bytes, settings):
    import subprocess

    source = directory / (name + ".py")
    source.write_bytes(source_bytes)
    script = """
import json, sys
from setuptools import Extension, setup
from Cython.Build import cythonize
name, source, settings = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
extensions = cythonize([Extension(name, [source], extra_compile_args=settings['extra_compile_args'])], compiler_directives=settings['directives'], quiet=True)
setup(name=name, ext_modules=extensions, script_args=['build_ext', '--inplace', '--build-temp', 'build'])
"""
    log = directory / "build.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", script, name, str(source), json.dumps(settings)],
            cwd=directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    binary = directory / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    if result.returncode or not binary.is_file():
        raise RuntimeError(
            f"Compiled host guard build failed: {log.read_text()[-6000:]}"
        )
    return binary


def _import_extension(name, binary):
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the compiled host guard extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompiledGuardBackend:
    def __init__(self, module, evidence):
        self.module, self._evidence = module, evidence

    def bind(self, guard, kind):
        predicates = (
            ("unchanged", "training_changed")
            if kind == "ModelTopologyGuard"
            else ("unchanged",)
        )
        cls = getattr(self.module, kind)
        for name in predicates:
            setattr(guard, name, MethodType(getattr(cls, name), guard))

    def evidence(self):
        return deepcopy(self._evidence)


def _cached(directory, expected):
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            return None
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        binary = directory / manifest["binary"]
        if (
            binary.parent != directory
            or _sha256(binary.read_bytes()) != manifest["binary_sha256"]
        ):
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_compiled_guards(*, cache_dir=None):
    """Load once per source/build identity, with process lock and atomic publish.

    A requested backend fails explicitly when dependencies/build/load fail.
    The default cache is outside the repository; its path may be overridden by
    PIR2_HOST_GUARD_CACHE. No cache or dependency is touched on module import.
    """
    import fcntl

    with _lock:
        source = SOURCE.read_bytes()
        settings = _build_settings()
        identity = {"source_sha256": _sha256(source), "build_settings": settings}
        key = _sha256(json.dumps(identity, sort_keys=True).encode())
        root = (
            Path(
                cache_dir
                or os.environ.get(
                    "PIR2_HOST_GUARD_CACHE",
                    Path.home() / ".cache" / "pir2" / "host-guards",
                )
            )
            .expanduser()
            .resolve()
        )
        memory_key = str(root), key
        if memory_key in _loaded:
            return _loaded[memory_key]
        root.mkdir(parents=True, exist_ok=True)
        directory = root / key
        name = "_pir2_host_guards_" + key
        expected = {**identity, "cache_key": key, "backend": "cython"}
        with (root / (key + ".lock")).open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            manifest = _cached(directory, expected)
            if manifest is None:
                with tempfile.TemporaryDirectory(
                    prefix=key + ".", dir=root
                ) as temporary:
                    staging = Path(temporary) / "extension"
                    staging.mkdir()
                    binary = _build(staging, name, source, settings)
                    manifest = {
                        **expected,
                        "binary": binary.name,
                        "binary_sha256": _sha256(binary.read_bytes()),
                    }
                    (staging / "manifest.json").write_text(
                        json.dumps(manifest, indent=2) + "\n"
                    )
                    # The interprocess lock owns this key. Readers cannot see a
                    # partial build, and a corrupt prior result cannot be reused.
                    if directory.exists():
                        shutil.rmtree(directory)
                    staging.rename(directory)
            module = _import_extension(name, directory / manifest["binary"])
        backend = CompiledGuardBackend(module, manifest)
        _loaded[memory_key] = backend
        return backend
