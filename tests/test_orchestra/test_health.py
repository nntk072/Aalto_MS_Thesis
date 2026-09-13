"""Tests for orchestra health registry."""

from __future__ import annotations

from orchestra.failures import next_midnight_pacific
from orchestra.health import HealthRegistry, ProbeFacts, RoutingStatus, derive_status
from orchestra.models import Model


def _model(**kwargs: object) -> Model:
    defaults: dict[str, object] = {
        "name": "default",
        "provider": "opencode",
        "cli": "opencode",
        "roles": ["triage"],
        "priority": 1,
        "quota_daily": 0,
    }
    defaults.update(kwargs)
    return Model(
        name=defaults["name"],  # type: ignore[arg-type]
        provider=defaults["provider"],  # type: ignore[arg-type]
        cli=defaults["cli"],  # type: ignore[arg-type]
        roles=defaults["roles"],  # type: ignore[arg-type]
        priority=defaults["priority"],  # type: ignore[arg-type]
        quota_daily=defaults["quota_daily"],  # type: ignore[arg-type]
        probe_forbidden=defaults.get("probe_forbidden", False),  # type: ignore[arg-type]
        registry_id=defaults.get("registry_id", "test-model"),  # type: ignore[arg-type]
    )


def test_derive_status_exhausted(tmp_path) -> None:
    facts = ProbeFacts(
        registry_id="x",
        binary_ok=True,
        auth_ok=True,
        exhausted_until=next_midnight_pacific(),
    )
    assert derive_status(facts) == RoutingStatus.EXHAUSTED


def test_probe_tier0_sets_binary_ok(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _c: "/usr/bin/opencode")
    reg = HealthRegistry(tmp_path)
    model = _model()
    reg.probe_tier0([model])
    facts = reg.get(model)
    assert facts.binary_ok is True
    assert facts.probe_tier == 0


def test_no_tier2_probe_function() -> None:
    """Health module must not define a generation probe path."""
    import orchestra.health as health_mod

    assert not hasattr(health_mod, "probe_tier2")


def test_vibe_auth_reads_dotenv(tmp_path, monkeypatch) -> None:
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / ".env").write_text('MISTRAL_API_KEY="test-key"\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    from orchestra.health import vibe_mistral_auth_ok

    assert vibe_mistral_auth_ok() is True


def test_vibe_model_in_config_active_model(tmp_path, monkeypatch) -> None:
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / "config.toml").write_text('active_model = "mistral-medium-3.5"\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    from orchestra.health import _vibe_model_in_config

    model = _model(
        name="mistral-medium-3.5",
        provider="mistral",
        cli="vibe",
        registry_id="vibe-mistral-medium-3.5",
    )
    assert _vibe_model_in_config(model) is True


def test_probe_tier0_vibe_mistral_available(tmp_path, monkeypatch) -> None:
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / ".env").write_text("MISTRAL_API_KEY=abc\n")
    (vibe_dir / "config.toml").write_text('active_model = "mistral-medium-3.5"\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda c: "/usr/bin/vibe" if c == "vibe" else None)
    reg = HealthRegistry(tmp_path / "state")
    model = _model(
        name="mistral-medium-3.5",
        provider="mistral",
        cli="vibe",
        env_var="MISTRAL_API_KEY",
        registry_id="vibe-mistral-medium-3.5",
    )
    reg.probe_tier0([model])
    facts = reg.get(model)
    assert facts.binary_ok is True
    assert facts.auth_ok is True
    assert facts.in_catalog is True
    assert derive_status(facts) == RoutingStatus.AVAILABLE
