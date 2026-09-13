"""Tests for quota-aware model routing."""

from __future__ import annotations

from orchestra.models import Model
from orchestra.router import ModelRouter


def _model(**kwargs: object) -> Model:
    defaults: dict[str, object] = {
        "name": "m",
        "provider": "p",
        "cli": "opencode",
        "roles": ["triage", "planner"],
        "priority": 2,
        "quota_daily": 1000,
        "env_var": "ORCHESTRA_TEST_KEY",
    }
    defaults.update(kwargs)
    return Model(
        name=defaults["name"],  # type: ignore[arg-type]
        provider=defaults["provider"],  # type: ignore[arg-type]
        cli=defaults["cli"],  # type: ignore[arg-type]
        roles=defaults["roles"],  # type: ignore[arg-type]
        priority=defaults["priority"],  # type: ignore[arg-type]
        quota_daily=defaults["quota_daily"],  # type: ignore[arg-type]
        escalation_only=defaults.get("escalation_only", False),  # type: ignore[arg-type]
        env_var=defaults["env_var"],  # type: ignore[arg-type]
    )


def test_select_skips_escalation_for_trivial(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    monkeypatch.setenv("GEMINI_TEST_KEY", "1")
    gemini = _model(
        name="gemini-3.1-pro-preview",
        provider="gemini",
        cli="gemini",
        roles=["triage"],
        priority=1,
        escalation_only=True,
        env_var="GEMINI_TEST_KEY",
    )
    local = _model(name="default", provider="opencode", priority=2)
    router = ModelRouter(models=[gemini, local], state_dir=tmp_path)
    selected = router.select("triage", count=1, task_complexity="trivial", escalate=False)
    assert selected == [local]


def test_select_uses_gemini_when_escalating(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    monkeypatch.setenv("GEMINI_TEST_KEY", "1")
    gemini = _model(
        name="g",
        provider="gemini",
        roles=["triage"],
        priority=1,
        escalation_only=True,
        env_var="GEMINI_TEST_KEY",
    )
    local = _model(name="default", provider="opencode", priority=2)
    router = ModelRouter(models=[gemini, local], state_dir=tmp_path)
    selected = router.select("triage", count=1, task_complexity="trivial", escalate=True)
    assert selected == [gemini]


def test_quota_record_and_exceeded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    model = _model(quota_daily=10)
    router = ModelRouter(models=[model], state_dir=tmp_path)
    assert not router.quota.quota_exceeded(model)
    router.quota.record_usage(model.display_name, 10, model=model)
    assert router.quota.quota_exceeded(model)
    stats = router.stats()
    assert stats[model.display_name]["tokens_used"] == 10
    assert stats[model.display_name]["calls"] == 1


def test_select_synthesizer_uses_opencode_not_gemini(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    monkeypatch.setenv("GEMINI_TEST_KEY", "1")
    gemini = _model(
        name="g",
        provider="gemini",
        roles=["synthesizer"],
        priority=1,
        escalation_only=True,
        env_var="GEMINI_TEST_KEY",
    )
    local = _model(
        name="default",
        provider="opencode",
        roles=["synthesizer"],
        priority=2,
    )
    router = ModelRouter(models=[gemini, local], state_dir=tmp_path)
    selected = router.select("synthesizer", count=1, task_complexity="medium", escalate=False)
    assert selected == [local]


def test_fallback_chain_excludes_primary_and_respects_escalation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    monkeypatch.setenv("GEMINI_TEST_KEY", "1")
    primary = _model(name="a", provider="opencode", roles=["triage"], priority=1)
    backup = _model(
        name="b",
        provider="gemini",
        roles=["triage"],
        priority=2,
        env_var="GEMINI_TEST_KEY",
    )
    escal = _model(
        name="c",
        provider="ollama",
        roles=["triage"],
        priority=3,
        escalation_only=True,
    )
    router = ModelRouter(models=[primary, backup, escal], state_dir=tmp_path)

    assert router.fallback_chain(primary, "triage", task_complexity="trivial") == [backup]
    assert router.fallback_chain(primary, "triage", task_complexity="trivial", escalate=True) == [
        backup,
        escal,
    ]
    assert primary not in router.fallback_chain(primary, "triage")


def test_fallback_chain_allows_same_name_different_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    opencode_default = _model(name="default", provider="opencode", roles=["triage"], priority=1)
    kilo_default = _model(name="default", provider="kilo", roles=["triage"], priority=2)
    router = ModelRouter(models=[opencode_default, kilo_default], state_dir=tmp_path)
    assert router.fallback_chain(opencode_default, "triage") == [kilo_default]
    assert router.fallback_chain(kilo_default, "triage") == [opencode_default]


def test_t0_returns_empty_when_only_gemini(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_TEST_KEY", "1")
    gemini = _model(
        name="gemini-3.1-pro-preview",
        provider="gemini",
        cli="gemini",
        roles=["triage"],
        priority=1,
        escalation_only=True,
        min_tier="T3",
        env_var="GEMINI_TEST_KEY",
        registry_id="gemini-3.1-pro-preview",
    )
    router = ModelRouter(models=[gemini], state_dir=tmp_path)
    assert router.select("triage", count=1, task_tier="T0", escalate=False) == []


def test_quota_requests_counter(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    model = _model(rpd=5, registry_id="rpd-model")
    router = ModelRouter(models=[model], state_dir=tmp_path)
    router.quota.record_usage(model.display_name, 1, model=model)
    usage = router.quota.get_usage(model.display_name, model)
    assert usage["requests"] == 1
