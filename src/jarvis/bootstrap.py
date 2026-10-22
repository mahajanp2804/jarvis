"""Composition root: wires configuration, platform adapters, tools, policy and AI.

Both the CLI and (from Phase 3) the background service build their object graph
here, so they share exactly the same policy and audit behaviour.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.agent.orchestrator import Orchestrator
from jarvis.agent.planner import Planner, PlannerSettings
from jarvis.ai.base import LLMBackend
from jarvis.ai.llama_cpp_backend import LlamaCppBackend
from jarvis.ai.model_manager import ModelManager
from jarvis.ai.prompt_cache import PromptCache
from jarvis.ai.subprocess_backend import SubprocessBackend
from jarvis.api.dispatcher import Dispatcher, RetryPolicy
from jarvis.config import AIConfig, JarvisConfig
from jarvis.errors import JarvisError
from jarvis.memory.store import MemoryStore
from jarvis.paths import config_dir
from jarvis.platforms.base import PlatformAdapter
from jarvis.runtime.audit import AuditLog
from jarvis.runtime.permissions import Confirmer, Decision, PolicyEngine
from jarvis.tools.applications import AppCatalog, ApplicationTools
from jarvis.tools.base import ToolRegistry, ToolSpec
from jarvis.tools.browser import BrowserSession, BrowserTools, playwright_available
from jarvis.tools.files import FileTools
from jarvis.tools.memory import MemoryTools
from jarvis.tools.paths import PathPolicy
from jarvis.tools.projects import ProjectTools
from jarvis.tools.screen import ScreenTools
from jarvis.tools.terminal import ProcessManager, TerminalTools

log = logging.getLogger(__name__)


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
    processes: ProcessManager
    browser: BrowserSession | None = None
    memory: MemoryStore | None = None

    def close(self) -> None:
        """Stop background processes and the browser, unload the model, close files."""
        self.processes.shutdown()
        if self.browser is not None:
            self.browser.close()
        if self.models is not None:
            self.models.shutdown()
        if self.memory is not None:
            self.memory.close()
        self.audit.close()


def build_memory(config: JarvisConfig) -> MemoryStore | None:
    """The memory store (the file is opened on first use), or None if disabled."""
    memory = config.memory
    if not memory.enabled:
        return None
    return MemoryStore(
        memory.resolved_path(), history_days=memory.history_days, max_history=memory.max_history
    )


def build_path_policy(config: JarvisConfig) -> PathPolicy:
    files = config.files
    return PathPolicy(
        files.read_roots, files.write_roots, files.blocked_paths, allow_hidden=files.allow_hidden
    )


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
                "prompt_cache_dir": (
                    str(ai.resolved_prompt_cache_dir()) if ai.prompt_cache else None
                ),
                "prompt_cache_max_mb": ai.prompt_cache_max_mb,
            }
        )
    return LlamaCppBackend(
        ai.resolved_model_path(),
        context_length=ai.context_length,
        gpu_layers=ai.gpu_layers,
        threads=ai.threads,
        chat_format=ai.chat_format,
        metal_residency_sets=ai.metal_residency_sets,
        prompt_cache=(
            PromptCache(ai.resolved_prompt_cache_dir(), ai.prompt_cache_max_mb << 20)
            if ai.prompt_cache
            else None
        ),
    )


def _preferences_context(memory: MemoryStore | None) -> Callable[[], str | None] | None:
    if memory is None:
        return None

    def remembered_preferences() -> str | None:
        try:
            return memory.planner_context()
        except JarvisError as exc:  # a broken database must not stop planning
            log.warning("Remembered preferences unavailable: %s", exc.message)
            return None

    return remembered_preferences


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
    paths = build_path_policy(config)
    file_tools = FileTools(
        platform,
        paths,
        catalog,
        policy,
        config.files,
        undo_dir=config.general.resolved_log_dir() / "undo",
    )
    project_tools = ProjectTools(file_tools, paths, config.files)
    processes = ProcessManager(config.general.resolved_log_dir() / "processes")
    terminal_tools = TerminalTools(platform, paths, config.terminal, processes)
    specs = [
        *app_tools.specs(),
        *file_tools.specs(),
        *project_tools.specs(),
        *terminal_tools.specs(),
        *ScreenTools(platform, paths).specs(),  # opt-in via policy.enabled_tools
    ]
    browser: BrowserSession | None = None
    if playwright_available():  # optional extra: no browser tools without it
        browser = BrowserSession(config.browser, config_dir() / "browser-profile")
        specs += BrowserTools(browser, config.browser).specs()
    memory = build_memory(config)
    if memory is not None:
        specs += MemoryTools(memory).specs()
    registry = ToolRegistry(specs)
    audit = AuditLog(config.general.resolved_log_dir() / "audit.jsonl")
    dispatcher = Dispatcher(
        registry,
        policy,
        confirmer,
        audit,
        dry_run=config.general.dry_run,
        cancel_event=cancel_event,
        retry=RetryPolicy(
            max_retries=config.agent.max_tool_retries,
            backoff_seconds=config.agent.retry_backoff_seconds,
        ),
    )

    models: ModelManager | None = None
    planner: Planner | None = None
    ai = config.ai
    if ai.enabled:
        models = ModelManager(backend or build_backend(ai), ai.idle_unload_seconds)

        def enabled_tools() -> list[ToolSpec]:
            # The model only sees tools the policy would allow at all.
            return [s for s in registry if policy.is_enabled(s.name)]

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
                context_tokens=ai.context_length,
            ),
            app_names=permitted_app_names if ai.include_app_names else None,
            context=_preferences_context(memory) if config.memory.share_preferences else None,
        )

    return Runtime(
        config=config,
        platform=platform,
        registry=registry,
        dispatcher=dispatcher,
        orchestrator=Orchestrator(
            dispatcher,
            planner=planner,
            history_ttl_seconds=config.agent.history_ttl_seconds,
            max_rounds=config.agent.max_rounds,
            max_observation_chars=config.agent.max_observation_chars,
            memory=memory if config.memory.record_history else None,
            workflows=memory,
        ),
        catalog=catalog,
        audit=audit,
        models=models,
        planner=planner,
        processes=processes,
        browser=browser,
        memory=memory,
    )
