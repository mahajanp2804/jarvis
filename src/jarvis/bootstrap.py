"""Composition root: wires configuration, platform adapters, tools, policy and AI.

Both the CLI and (from Phase 3) the background service build their object graph
here, so they share exactly the same policy and audit behaviour.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from jarvis.agent.orchestrator import Orchestrator
from jarvis.agent.planner import Planner, PlannerSettings
from jarvis.ai.base import LLMBackend
from jarvis.ai.llama_cpp_backend import LlamaCppBackend
from jarvis.ai.model_manager import ModelManager
from jarvis.ai.subprocess_backend import SubprocessBackend
from jarvis.api.dispatcher import Dispatcher
from jarvis.config import AIConfig, JarvisConfig
from jarvis.platforms.base import PlatformAdapter
from jarvis.runtime.audit import AuditLog
from jarvis.runtime.permissions import Confirmer, Decision, PolicyEngine
from jarvis.tools.applications import AppCatalog, ApplicationTools
from jarvis.tools.base import ToolRegistry, ToolSpec


@dataclass
class Runtime:
    """Everything needed to serve requests."""

    config: JarvisConfig
    platform: PlatformAdapter
    registry: ToolRegistry
    dispatcher: Dispatcher
    orchestrator: Orchestrator
    catalog: AppCatalog
    audit: AuditLog
    models: ModelManager | None
    planner: Planner | None

    def close(self) -> None:
        """Unload the model (if loaded) and close the audit log."""
        if self.models is not None:
            self.models.shutdown()
        self.audit.close()


def build_backend(ai: AIConfig) -> LLMBackend:
    """Create the configured model backend (nothing is loaded yet)."""
    if ai.isolation == "subprocess":
        return SubprocessBackend(
            {
                "model_path": str(ai.resolved_model_path()),
                "context_length": ai.context_length,
                "gpu_layers": ai.gpu_layers,
                "threads": ai.threads,
                "chat_format": ai.chat_format,
                "metal_residency_sets": ai.metal_residency_sets,
            }
        )
    return LlamaCppBackend(
        ai.resolved_model_path(),
        context_length=ai.context_length,
        gpu_layers=ai.gpu_layers,
        threads=ai.threads,
        chat_format=ai.chat_format,
        metal_residency_sets=ai.metal_residency_sets,
    )


def build_runtime(
    config: JarvisConfig,
    platform: PlatformAdapter,
    confirmer: Confirmer,
    *,
    cancel_event: threading.Event | None = None,
    poll_interval: float = 0.1,
    backend: LLMBackend | None = None,
) -> Runtime:
    """Construct the runtime object graph for ``config`` on ``platform``.

    ``backend`` overrides the configured model backend (used by tests).
    """
    app_cfg = config.applications
    aliases = {**platform.apps.default_aliases(), **app_cfg.aliases}
    catalog = AppCatalog(
        adapter=platform.apps,
        search_paths=app_cfg.search_paths or platform.apps.default_search_paths(),
        aliases=aliases,
        cache_seconds=app_cfg.discovery_cache_seconds,
    )
    policy = PolicyEngine(config.policy)
    app_tools = ApplicationTools(
        platform.apps, catalog, policy, app_cfg, poll_interval=poll_interval
    )
    registry = ToolRegistry(app_tools.specs())
    audit = AuditLog(config.general.resolved_log_dir() / "audit.jsonl")
    dispatcher = Dispatcher(
        registry,
        policy,
        confirmer,
        audit,
        dry_run=config.general.dry_run,
        cancel_event=cancel_event,
    )

    models: ModelManager | None = None
    planner: Planner | None = None
    ai = config.ai
    if ai.enabled:
        models = ModelManager(backend or build_backend(ai), ai.idle_unload_seconds)

        def enabled_tools() -> list[ToolSpec]:
            # The model only sees tools the policy would allow at all.
            return [s for s in registry if s.name in config.policy.enabled_tools]

        def permitted_app_names() -> list[str]:
            return [
                a.name
                for a in catalog.all()
                if policy.check_app(a.name, a.bundle_id).decision is not Decision.DENY
            ]

        planner = Planner(
            models,
            enabled_tools,
            PlannerSettings(
                max_steps=ai.max_steps,
                max_tokens=ai.max_plan_tokens,
                temperature=ai.temperature,
            ),
            app_names=permitted_app_names if ai.include_app_names else None,
        )

    return Runtime(
        config=config,
        platform=platform,
        registry=registry,
        dispatcher=dispatcher,
        orchestrator=Orchestrator(dispatcher, planner=planner),
        catalog=catalog,
        audit=audit,
        models=models,
        planner=planner,
    )
