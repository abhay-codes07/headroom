import sys

from headroom.onnx_runtime import (
    ONNX_ALLOW_SPINNING_ENV,
    ONNX_CPU_ARENA_ENV,
    cpu_arena_enabled,
    create_cpu_session_options,
    onnx_thread_spinning_enabled,
)


class _FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True
        self.config_entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config_entries[key] = value


class _FakeOrt:
    SessionOptions = _FakeSessionOptions


class _FakeSessionOptionsWithoutToggles:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None

    def add_session_config_entry(self, key: str, value: str) -> None:
        # No config storage on this stand-in; ORT here just accepts the call.
        return None


class _FakeOrtWithoutToggles:
    SessionOptions = _FakeSessionOptionsWithoutToggles


def test_create_cpu_session_options_disables_retention_features(monkeypatch):
    """Non-Windows keeps the legacy low-RSS behavior: arena + mem pattern off."""
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    options = create_cpu_session_options(
        _FakeOrt,
        intra_op_num_threads=1,
        inter_op_num_threads=2,
    )

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_darwin_unchanged(monkeypatch):
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    options = create_cpu_session_options(_FakeOrt)

    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_keeps_arena_on_windows(monkeypatch):
    """Disabling the arena on Windows degrades inference by orders of
    magnitude (onnxruntime#11627) — ORT defaults must stay untouched there."""
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    options = create_cpu_session_options(_FakeOrt, intra_op_num_threads=3)

    assert options.enable_cpu_mem_arena is True
    assert options.enable_mem_pattern is True
    assert options.intra_op_num_threads == 3


def test_arena_env_override_forces_on(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "1")

    assert cpu_arena_enabled() is True
    options = create_cpu_session_options(_FakeOrt)
    assert options.enable_cpu_mem_arena is True


def test_arena_env_override_forces_off(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "0")

    assert cpu_arena_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.enable_cpu_mem_arena is False


def test_arena_env_invalid_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "bananas")

    monkeypatch.setattr(sys, "platform", "win32")
    assert cpu_arena_enabled() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert cpu_arena_enabled() is False


def test_create_cpu_session_options_handles_older_session_options(monkeypatch):
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    options = create_cpu_session_options(_FakeOrtWithoutToggles)

    assert options.intra_op_num_threads is None
    assert options.inter_op_num_threads is None


def test_thread_spinning_disabled_by_default(monkeypatch):
    # #2495: ORT thread pools spin-wait on all cores between inferences, so a
    # long-lived proxy pegs every core while idle. Disable spinning by default.
    monkeypatch.delenv(ONNX_ALLOW_SPINNING_ENV, raising=False)
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)

    assert onnx_thread_spinning_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.config_entries.get("session.intra_op.allow_spinning") == "0"
    assert options.config_entries.get("session.inter_op.allow_spinning") == "0"


def test_thread_spinning_env_can_reenable(monkeypatch):
    monkeypatch.setenv(ONNX_ALLOW_SPINNING_ENV, "1")
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)

    assert onnx_thread_spinning_enabled() is True
    options = create_cpu_session_options(_FakeOrt)
    assert "session.intra_op.allow_spinning" not in options.config_entries
    assert "session.inter_op.allow_spinning" not in options.config_entries


def test_thread_spinning_env_explicit_off(monkeypatch):
    monkeypatch.setenv(ONNX_ALLOW_SPINNING_ENV, "0")

    assert onnx_thread_spinning_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.config_entries.get("session.intra_op.allow_spinning") == "0"


def test_spinning_disable_is_best_effort_on_older_ort(monkeypatch):
    # An ORT build that rejects the config key must not break session creation.
    monkeypatch.delenv(ONNX_ALLOW_SPINNING_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    class _RejectingSessionOptions(_FakeSessionOptions):
        def add_session_config_entry(self, key: str, value: str) -> None:
            raise RuntimeError(f"unknown config key: {key}")

    class _RejectingOrt:
        SessionOptions = _RejectingSessionOptions

    # Must not raise.
    options = create_cpu_session_options(_RejectingOrt)
    assert options.enable_cpu_mem_arena is False
