"""Backend validation uses fake runtime metadata, never tensors or GPU execution."""

import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from coexecution.optimization import capabilities


def runtime(monkeypatch, precision, capability, *, loaded=()):
    spec = capabilities.BACKENDS.get(precision)
    namespace = SimpleNamespace()
    ops = SimpleNamespace(loaded_libraries=set(map(str, loaded)))
    if spec:
        setattr(ops, spec["backend_id"], namespace)
    fake = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: capability,
            get_device_name=lambda device: "test-device",
        ),
        ops=ops,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(
        capabilities,
        "_source_identity",
        lambda module: {
            "expected_revision": "pinned",
            "revision": "pinned",
            "dirty": False,
        },
    )
    return fake, namespace, SimpleNamespace(type="cuda")


def loader(monkeypatch, precision, function):
    spec = capabilities.BACKENDS[precision]
    module = SimpleNamespace(**{spec["loader"]: function})
    monkeypatch.setattr(capabilities.importlib, "import_module", lambda name: module)


def operators(namespace, precision, *, exclude=()):
    for name in capabilities.BACKENDS[precision]["operators"]:
        if name not in exclude:
            setattr(
                namespace, name, lambda *args: pytest.fail("No operator may execute")
            )


def config(precision):
    return SimpleNamespace(precision=precision, shape_tactics=None)


def test_capabilities_import_does_not_import_model_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import coexecution.optimization.capabilities; "
                "assert not any(x.startswith(('torch', 'triton', 'robotics_kernels')) for x in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "precision,arch",
    [
        ("fp8", (8, 9)),
        ("fp4", (10, 0)),
    ],
)
def test_wrong_hardware_rejected_before_loading(monkeypatch, precision, arch):
    _, _, device = runtime(monkeypatch, precision, arch)
    monkeypatch.setattr(
        capabilities.importlib,
        "import_module",
        lambda name: pytest.fail("Unsupported hardware cannot load backend"),
    )
    with pytest.raises(ValueError, match="no supported backend"):
        capabilities.preflight_backend(config(precision), device)


@pytest.mark.parametrize(
    "precision,missing",
    [
        ("fp8", "linear_forward_packed_tactic"),
        ("fp4", "pack_linear_weight_gemm_bf16"),
        ("w8a8", "gemm_biasless"),
    ],
)
def test_exact_selected_api_is_required(monkeypatch, precision, missing):
    _, namespace, device = runtime(
        monkeypatch, precision, (8, 9) if precision == "w8a8" else (11, 0)
    )
    loader(
        monkeypatch,
        precision,
        lambda *args: operators(namespace, precision, exclude=(missing,)),
    )
    with pytest.raises(RuntimeError, match=missing):
        capabilities.preflight_backend(config(precision), device)


def test_fp4_swap_ab_requires_new_api_only_when_enabled(monkeypatch):
    _, namespace, device = runtime(monkeypatch, "fp4", (11, 0))
    loader(monkeypatch, "fp4", lambda *args: operators(namespace, "fp4"))
    cfg = config("fp4")
    report = capabilities.preflight_backend(cfg, device)
    name = "linear_forward_packed_gemm_bf16_swap_ab"
    assert name not in report["required_operators"]
    cfg.fp4_swap_ab = True
    with pytest.raises(RuntimeError, match=name):
        capabilities.preflight_backend(cfg, device)
    setattr(namespace, name, lambda *args: pytest.fail("No GPU execution in CI"))
    assert name in capabilities.preflight_backend(cfg, device)["required_operators"]


@pytest.mark.parametrize("precision", ["w8a8", "w4a4"])
def test_integer_loader_rejection_propagates_without_precision_fallback(
    monkeypatch, precision
):
    _, _, device = runtime(monkeypatch, precision, (9, 0))

    def fail():
        raise RuntimeError("backend does not support this build or device")

    loader(monkeypatch, precision, fail)
    with pytest.raises(RuntimeError, match="does not support"):
        capabilities.preflight_backend(config(precision), device)


@pytest.mark.parametrize(
    "architecture,validated",
    [((8, 9), True), ((8, 0), False), ((9, 0), False), ((12, 0), False)],
)
def test_integer_loadability_does_not_claim_hardware_validation(
    monkeypatch, architecture, validated
):
    _, namespace, device = runtime(monkeypatch, "w8a8", architecture)
    loader(monkeypatch, "w8a8", lambda: operators(namespace, "w8a8"))
    report = capabilities.preflight_backend(config("w8a8"), device)
    assert report["backend_id"] == "pir2_integer"
    assert report["validated_architectures"] == ["sm_89"]
    assert report["runtime_arch_validated"] is validated


def test_observed_library_hash_uses_loaded_path_not_environment(monkeypatch, tmp_path):
    actual = tmp_path / "pir2_cutlass_fp8_ext.so"
    actual.write_bytes(b"loaded backend")
    requested = tmp_path / "requested-but-not-loaded.so"
    requested.write_bytes(b"unrelated binary")
    monkeypatch.setenv("ROBOTICS_CUTLASS_FP8_SO", str(requested))
    fake, namespace, device = runtime(monkeypatch, "fp8", (11, 0))

    def load(*args):
        operators(namespace, "fp8")
        fake.ops.loaded_libraries.add(str(actual))

    loader(monkeypatch, "fp8", load)
    report = capabilities.preflight_backend(config("fp8"), device)
    assert report["binary"]["status"] == "observed_backend_load"
    assert (
        report["binary"]["selected"]["sha256"]
        == hashlib.sha256(actual.read_bytes()).hexdigest()
    )
    assert report["backend_id"] == "pir2_cutlass_fp8"
    assert report["device_arch"] == "sm_110"
    assert report["runtime"]["torch"] == "test-torch"


def test_preloaded_operator_origin_remains_unknown(monkeypatch, tmp_path):
    actual = tmp_path / "pir2_integer_ext.so"
    actual.write_bytes(b"already loaded")
    _, namespace, device = runtime(monkeypatch, "w8a8", (8, 9), loaded=(actual,))
    operators(namespace, "w8a8")
    loader(monkeypatch, "w8a8", lambda: None)
    report = capabilities.preflight_backend(config("w8a8"), device)
    assert report["binary"]["status"] == "unknown"
    assert report["binary"]["selected"] is None
    assert (
        report["binary"]["loaded_candidates"][0]["sha256"]
        == hashlib.sha256(actual.read_bytes()).hexdigest()
    )


def test_partially_preloaded_namespace_has_no_unique_binary_identity(
    monkeypatch, tmp_path
):
    actual = tmp_path / "pir2_integer_ext.so"
    actual.write_bytes(b"additional operators")
    fake, namespace, device = runtime(monkeypatch, "w8a8", (8, 9))
    namespace.gemm = lambda: None

    def load():
        namespace.gemm_biasless = lambda: None
        fake.ops.loaded_libraries.add(str(actual))

    loader(monkeypatch, "w8a8", load)
    report = capabilities.preflight_backend(config("w8a8"), device)
    assert report["binary"]["status"] == "unknown"
    assert report["binary"]["selected"] is None


def test_source_identity_does_not_mistake_parent_repository_for_backend(tmp_path):
    module = tmp_path / "site-packages/robotics_kernels/integer.py"
    module.parent.mkdir(parents=True)
    module.write_text("# packaged module\n")
    result = capabilities._source_identity(SimpleNamespace(__file__=str(module)))
    assert result["revision"] == "unknown"
    assert result["dirty"] == "unknown"
    assert result["module_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert result["expected_revision"].startswith("sha256:")
    assert result["upstream_revision"] == "239b4a3ef2268048571c9f508ad5700f98398c2f"


def test_external_git_tracking_cannot_claim_owned_kernel_identity(tmp_path):
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Contract test",
                "-c",
                "user.email=test@example.invalid",
                *args,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )

    git("init", "-q")
    (tmp_path / "README.md").write_text("test repository\n")
    git("add", "README.md")
    git("commit", "-qm", "test fixture")
    module = tmp_path / "src/robotics_kernels/ampere_ada/integer.py"
    module.parent.mkdir(parents=True)
    module.write_text("# untracked runtime source\n")
    result = capabilities._source_identity(SimpleNamespace(__file__=str(module)))
    assert result["module_tracked"] is False
    assert result["revision"] == "unknown"
    assert result["dirty"] == "unknown"

    git("add", "src/robotics_kernels/ampere_ada/integer.py")
    git("commit", "-qm", "track runtime source")
    result = capabilities._source_identity(SimpleNamespace(__file__=str(module)))
    assert result["module_tracked"] is False
    assert result["revision"] == "unknown"
    assert result["dirty"] == "unknown"

    (module.parent / "custom_kernel.cu").write_text("// untracked backend source\n")
    result = capabilities._source_identity(SimpleNamespace(__file__=str(module)))
    assert result["dirty"] == "unknown"


def test_modified_source_cannot_claim_a_clean_profile_revision(monkeypatch):
    _, namespace, device = runtime(monkeypatch, "w8a8", (8, 9))
    loader(monkeypatch, "w8a8", lambda: operators(namespace, "w8a8"))
    monkeypatch.setattr(
        capabilities,
        "_source_identity",
        lambda module: {
            "expected_revision": "pinned",
            "revision": "pinned",
            "dirty": True,
        },
    )
    report = capabilities.preflight_backend(config("w8a8"), device)
    assert report["source"]["revision"] == "pinned"
    assert report["backend_revision"] == "unknown"


@pytest.mark.parametrize("architecture", [(8, 9), (9, 0), (12, 0)])
def test_bf16_does_not_load_quantization_backend(monkeypatch, architecture):
    _, _, device = runtime(monkeypatch, "bf16", architecture)
    monkeypatch.setattr(
        capabilities.importlib,
        "import_module",
        lambda name: pytest.fail("BF16 must not load a low-bit backend"),
    )
    report = capabilities.preflight_backend(config("bf16"), device)
    assert report["backend_id"] == "torch"
    assert report["quant_format"] == "bf16"
