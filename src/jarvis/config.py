"""Configuration management.

Precedence (lowest to highest): built-in defaults < TOML file < ``JARVIS_*``
environment variables < command-line flags (applied by :mod:`jarvis.main`).

Unknown keys are rejected so a typo in ``config.toml`` fails loudly instead of
silently falling back to a default, which matters for security settings.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from jarvis.ai.registry import DEFAULT_MODEL
from jarvis.api.schemas import RiskLevel
from jarvis.errors import ConfigError
from jarvis.paths import (
    default_config_file,
    default_log_dir,
    default_memory_path,
    default_models_dir,
    default_prompt_cache_dir,
)

CONFIG_ENV_VAR = "JARVIS_CONFIG"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Tool groups enabled by default ("*" wildcards). Screen capture is opt-in:
# add "screen.*" to use it.
DEFAULT_TOOLS = ["apps.*", "files.*", "projects.*", "terminal.*", "browser.*", "memory.*"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class GeneralConfig(_Section):
    """Process-wide behaviour."""

    dry_run: bool = False
    log_level: LogLevel = "INFO"
    log_dir: Path | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("log_dir")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    def resolved_log_dir(self) -> Path:
        """The configured log directory, or the platform default."""
        return self.log_dir or default_log_dir()


class ApplicationsConfig(_Section):
    """Application discovery and launch behaviour."""

    # Empty means "use the platform adapter's standard install locations".
    search_paths: list[Path] = Field(default_factory=list)
    # Extra spoken names, e.g. {"browser" = "Safari"}. Merged over platform defaults.
    aliases: dict[str, str] = Field(default_factory=dict)
    launch_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    quit_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    # How long the list of installed apps is reused before rescanning disk.
    discovery_cache_seconds: float = Field(default=60.0, ge=0, le=3600)

    @field_validator("search_paths")
    @classmethod
    def _expand_paths(cls, value: list[Path]) -> list[Path]:
        return [p.expanduser() for p in value]

    @field_validator("aliases")
    @classmethod
    def _normalise_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, target in value.items():
            if not key.strip() or not target.strip():
                raise ValueError("alias names and targets must be non-empty")
            cleaned[" ".join(key.lower().split())] = target.strip()
        return cleaned


class PolicyConfig(_Section):
    """Permission policy. See README "Security model"."""

    # Allowlist of tools. Any tool not listed here is denied.
    enabled_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    # Actions at or above this risk level need explicit confirmation.
    confirm_at_or_above: RiskLevel = RiskLevel.MEDIUM
    # If non-empty, only these apps (names or bundle IDs) may be opened or quit.
    app_allowlist: list[str] = Field(default_factory=list)
    # These apps (names or bundle IDs) may never be opened or quit.
    app_blocklist: list[str] = Field(default_factory=list)

    @field_validator("confirm_at_or_above", mode="before")
    @classmethod
    def _parse_risk(cls, value: object) -> RiskLevel:
        return RiskLevel.parse(value)

    @field_validator("confirm_at_or_above")
    @classmethod
    def _cap_threshold(cls, value: RiskLevel) -> RiskLevel:
        if value > RiskLevel.HIGH:
            raise ValueError(
                "high and critical actions always require confirmation; "
                "set confirm_at_or_above to 'high' or lower"
            )
        return value

    @field_serializer("confirm_at_or_above")
    def _serialise_risk(self, value: RiskLevel) -> str:
        return value.name.lower()


class AIConfig(_Section):
    """Local language model used for requests the rule-based router can't handle."""

    enabled: bool = True
    backend: Literal["llama_cpp"] = "llama_cpp"
    # "subprocess": run the model in a worker process, so unloading frees all of
    # its memory, a running request can be aborted at shutdown, and a native
    # crash can't take Jarvis down. "in_process": no worker (slightly faster load).
    isolation: Literal["subprocess", "in_process"] = "subprocess"
    # GGUF file. Unset = the default model in the platform models directory.
    model_path: Path | None = None
    context_length: int = Field(default=4096, ge=512, le=131072)
    # Layers offloaded to the GPU (Metal on Apple Silicon); -1 = all, 0 = CPU only.
    gpu_layers: int = Field(default=-1, ge=-1)
    # CPU threads for inference; unset = llama.cpp default.
    threads: int | None = Field(default=None, ge=1, le=64)
    # Chat template; unset = the template embedded in the GGUF file.
    chat_format: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_plan_tokens: int = Field(default=384, ge=64, le=4096)
    max_steps: int = Field(default=5, ge=1, le=10)
    # Keep the model loaded this long after the last request, then free its
    # memory. 0 = unload immediately after every request. Measured on an M4:
    # staying loaded costs ~0.7 GB (plus reclaimable file cache) but ~0% CPU;
    # reloading adds ~3 s to the next request. See README "Performance".
    idle_unload_seconds: float = Field(default=300.0, ge=0, le=86400)
    # llama.cpp's Metal "residency set" keep-alive wakes the CPU ~250 times/s
    # while a model is loaded. Measured: no latency benefit for Jarvis's short
    # requests, so it is off unless enabled here (or GGML_METAL_NO_RESIDENCY is set).
    metal_residency_sets: bool = False
    # Give the model the names of installed apps so it can map "my browser" to an
    # app. Names never leave the machine.
    include_app_names: bool = True
    # Save llama.cpp's state for the system prompt to disk, so the first request
    # after a model load doesn't re-read it. Measured on an M4: 7.7 s -> 2.4 s
    # for that request (5.3 s of prompt reading skipped); ~250 MB of disk per
    # saved prompt, no extra memory. See README "Prompt cache on disk".
    prompt_cache: bool = True
    # Disk cap; least recently used prompts are deleted beyond it.
    prompt_cache_max_mb: int = Field(default=512, ge=1, le=65536)
    # Unset = ~/Library/Application Support/Jarvis/prompt-cache on macOS.
    prompt_cache_dir: Path | None = None

    @field_validator("model_path", "prompt_cache_dir")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    def resolved_model_path(self) -> Path:
        """The configured model file, or the default model in the models directory."""
        return self.model_path or default_models_dir() / DEFAULT_MODEL.filename

    def resolved_prompt_cache_dir(self) -> Path:
        return self.prompt_cache_dir or default_prompt_cache_dir()


DEFAULT_PROJECT_ROOTS = (
    "~/Desktop",
    "~/Documents",
    "~/Developer",
    "~/Projects",
    "~/code",
    "~/src",
    "~/repos",
    "~/dev",
    "~/GitHub",
    "~/workspace",
)


def _paths(*items: str) -> list[Path]:
    return [Path(item) for item in items]


class FilesConfig(_Section):
    """Where the file, project and terminal tools may read and write.

    Paths may start with "~". Symlinks are resolved before checking, so a link
    can't escape these folders.
    """

    read_roots: list[Path] = Field(default_factory=lambda: _paths("~"))
    write_roots: list[Path] = Field(
        default_factory=lambda: _paths("~/Desktop", "~/Documents", "~/Downloads")
    )
    # Searched by projects.find, in this order.
    project_roots: list[Path] = Field(default_factory=lambda: _paths(*DEFAULT_PROJECT_ROOTS))
    # Never read or changed, even inside a root. ~/Library also holds Jarvis's
    # own config and audit log, so tools can't alter them.
    blocked_paths: list[Path] = Field(default_factory=lambda: _paths("~/Library", "~/.Trash"))
    # Hidden files and folders (.ssh, .env, .git, ...) are off-limits unless true.
    allow_hidden: bool = False
    max_read_bytes: int = Field(default=1_000_000, ge=1_000, le=50_000_000)
    search_seconds: float = Field(default=4.0, gt=0, le=60)


DEFAULT_SEARCH_PATH = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)


class TerminalConfig(_Section):
    """Development commands (see jarvis/tools/commands.py for the allowlist)."""

    # PATH for commands; Jarvis's own environment is never passed on.
    search_path: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_PATH))
    # Extra programs you trust; they always ask for confirmation (medium risk).
    extra_programs: list[str] = Field(default_factory=list)
    # A command that is still running after this long is stopped.
    timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    # How long to watch a background server's output for a "ready"/URL message.
    startup_seconds: float = Field(default=10.0, gt=0, le=120)

    @field_validator("search_path")
    @classmethod
    def _absolute(cls, value: list[str]) -> list[str]:
        if any(not item.startswith("/") for item in value):
            raise ValueError("search_path entries must be absolute folders")
        return value


class BrowserConfig(_Section):
    """Browser automation (Playwright with an installed Chrome, Edge or Brave)."""

    # Show the browser window (recommended, so you see what Jarvis does).
    headless: bool = False
    # Browser executable; unset = Chrome, then Edge, Brave, Chromium, Playwright's own.
    executable: str | None = None
    # Bing is the default because it serves results to automated browsers; DuckDuckGo,
    # Brave Search and Startpage answered with bot checks when tested (Oct 2026).
    search_url: str = "https://www.bing.com/search?q={query}"
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    # Close the browser after this many idle seconds (0 = keep it open).
    idle_close_seconds: float = Field(default=600.0, ge=0, le=86400)

    @field_validator("search_url")
    @classmethod
    def _search_url(cls, value: str) -> str:
        if not value.startswith("https://") or "{query}" not in value:
            raise ValueError("search_url must be an https:// URL containing {query}")
        return value


class AgentConfig(_Section):
    """How requests are planned, executed, verified and retried."""

    # Plan -> execute -> observe rounds per request (each round is one model call).
    max_rounds: int = Field(default=6, ge=1, le=20)
    # Extra attempts for idempotent tools after a transient failure.
    max_tool_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0, le=10)
    # Longest piece of tool output shown to the model per step.
    max_observation_chars: int = Field(default=1500, ge=200, le=20000)
    # Forget the conversation after this much inactivity.
    history_ttl_seconds: float = Field(default=600.0, ge=0, le=86400)


class ServiceConfig(_Section):
    """Background service (Launch Agent) behaviour."""

    # Global shortcut that opens the request dialog. Must include cmd or ctrl.
    hotkey: str = "ctrl+option+cmd+j"
    # launchd label; also the Launch Agent file name.
    label: str = Field(
        default="local.jarvis.assistant", pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]{2,100}$"
    )
    # Show a dialog even when an action's result is visible (e.g. an app opened).
    show_success: bool = False
    # Longest `jarvis ask` waits for the service to finish a request.
    request_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)

    @field_validator("hotkey")
    @classmethod
    def _valid_hotkey(cls, value: str) -> str:
        return _check_hotkey(value)


class MemoryConfig(_Section):
    """Persistent memory (SQLite): remembered preferences and task history.

    Approvals are never stored, and secrets and outside content are left out.
    Inspect, export or erase it with `jarvis memory show|export|clear`.
    """

    enabled: bool = True
    # Unset = ~/Library/Application Support/Jarvis/memory.sqlite3 on macOS.
    path: Path | None = None
    # Keep a history of requests, their outcome and the tools used.
    record_history: bool = True
    history_days: int = Field(default=30, ge=1, le=3650)
    max_history: int = Field(default=1000, ge=10, le=100_000)
    # Give remembered preferences ("my editor is VS Code") to the model.
    share_preferences: bool = True

    @field_validator("path")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    def resolved_path(self) -> Path:
        return self.path or default_memory_path()


def _check_hotkey(value: str) -> str:
    from jarvis.runtime.hotkeys import Hotkey, HotkeyError

    try:
        return str(Hotkey.parse(value))
    except HotkeyError as exc:
        raise ValueError(exc.message + (f" {exc.hint}" if exc.hint else "")) from None


class VoiceConfig(_Section):
    """Spoken replies and push-to-talk voice input (both on-device, both off by default)."""

    # Say a short summary of each reply (shortcut dialog, `jarvis shell`, `jarvis run`).
    speak_replies: bool = False
    # Installed system voice, e.g. "Samantha"; empty = the system voice.
    voice: str = Field(default="", max_length=100)
    # Words per minute; 0 = the system default.
    rate: int = Field(default=0, ge=0, le=500)
    # Longest summary spoken per reply.
    max_spoken_chars: int = Field(default=300, ge=40, le=2000)

    # Push-to-talk input: press the shortcut, speak, pause (or press it again).
    # Needs `pip install -e ".[voice]"` and `jarvis voice download` (60 MB).
    push_to_talk: bool = False
    push_to_talk_hotkey: str = "ctrl+option+cmd+k"
    # whisper.cpp model; unset = the default model in the models directory.
    stt_model_path: Path | None = None
    # Spoken language (two-letter code); "en" models only understand English.
    language: str = Field(default="en", pattern=r"^(?:[a-z]{2,3}|auto)$")
    max_record_seconds: float = Field(default=15.0, ge=2, le=60)
    # Stop recording after this much quiet following speech (0 = only when told).
    silence_seconds: float = Field(default=1.5, ge=0, le=10)
    # Transcribe on the GPU (Metal): 0.08 s instead of 0.2 s per command on an M4,
    # but the first use after macOS resets its shader cache takes ~7 s.
    stt_use_gpu: bool = False

    @field_validator("rate")
    @classmethod
    def _sensible_rate(cls, value: int) -> int:
        if 0 < value < 80:
            raise ValueError("rate must be 0 (system default) or 80-500 words per minute")
        return value

    @field_validator("push_to_talk_hotkey")
    @classmethod
    def _valid_hotkey(cls, value: str) -> str:
        return _check_hotkey(value)

    @field_validator("stt_model_path")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    def resolved_stt_model_path(self) -> Path:
        from jarvis.voice.models import DEFAULT_STT_MODEL

        return self.stt_model_path or default_models_dir() / DEFAULT_STT_MODEL.filename


class JarvisConfig(_Section):
    """Root configuration object."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    applications: ApplicationsConfig = Field(default_factory=ApplicationsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    terminal: TerminalConfig = Field(default_factory=TerminalConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    @model_validator(mode="after")
    def _distinct_hotkeys(self) -> JarvisConfig:
        if self.voice.push_to_talk and self.voice.push_to_talk_hotkey == self.service.hotkey:
            raise ValueError("voice.push_to_talk_hotkey must differ from service.hotkey")
        return self


EXAMPLE_CONFIG_TOML = """\
# Jarvis configuration. Unknown keys are rejected.
# Environment variables override these values: JARVIS_DRY_RUN, JARVIS_LOG_LEVEL,
# JARVIS_LOG_DIR, JARVIS_AI_ENABLED, JARVIS_MODEL_PATH.

[general]
# Validate and describe actions without executing them.
dry_run = false
# DEBUG, INFO, WARNING or ERROR.
log_level = "INFO"
# Uncomment to override the default (~/Library/Logs/Jarvis on macOS).
# log_dir = "~/Library/Logs/Jarvis"

[applications]
# Empty list = standard install locations for this OS.
search_paths = []
launch_timeout_seconds = 10.0
quit_timeout_seconds = 10.0
discovery_cache_seconds = 60.0

[applications.aliases]
# Extra names for apps, merged over built-in aliases. Example:
# "browser" = "Safari"

[policy]
# Tools Jarvis may use ("*" wildcards allowed). Anything not listed is denied.
# Screen capture/OCR is opt-in: add "screen.*".
enabled_tools = ["apps.*", "files.*", "projects.*", "terminal.*", "browser.*", "memory.*"]
# Actions at or above this risk need confirmation: read_only, low, medium, high.
# High and critical actions always need confirmation.
confirm_at_or_above = "medium"
# If non-empty, ONLY these apps (names or bundle IDs) may be opened or quit.
app_allowlist = []
# These apps may never be opened or quit by Jarvis.
app_blocklist = []

[ai]
# Use the local model for requests the built-in rules don't understand.
enabled = true
backend = "llama_cpp"
# "subprocess" (model in a worker process; fully freed on unload) or "in_process".
isolation = "subprocess"
# Uncomment to use a different GGUF file. Default:
# ~/Library/Application Support/Jarvis/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
# model_path = "~/models/my-model.gguf"
context_length = 4096
# Layers on the GPU (Metal): -1 = all, 0 = CPU only.
gpu_layers = -1
temperature = 0.0
max_plan_tokens = 384
max_steps = 5
# Free the model's memory after this many idle seconds (0 = after every request).
idle_unload_seconds = 300.0
# llama.cpp Metal residency keep-alive: ~250 CPU wakeups/s while loaded; off by default.
metal_residency_sets = false
include_app_names = true
# Save the system prompt's llama.cpp state to disk (~250 MB) so the first request
# after a model load skips re-reading it (~5 s faster on an M4).
prompt_cache = true
prompt_cache_max_mb = 512
# Uncomment to override the default (~/Library/Application Support/Jarvis/prompt-cache).
# prompt_cache_dir = "~/Library/Application Support/Jarvis/prompt-cache"

[files]
# Folders Jarvis may read, and the subset it may change ("~" = your home folder).
read_roots = ["~"]
write_roots = ["~/Desktop", "~/Documents", "~/Downloads"]
# Searched by projects.find, in this order.
project_roots = [
    "~/Desktop", "~/Documents", "~/Developer", "~/Projects", "~/code",
    "~/src", "~/repos", "~/dev", "~/GitHub", "~/workspace",
]
# Never read or changed (~/Library holds app data and Jarvis's own config and logs).
blocked_paths = ["~/Library", "~/.Trash"]
# Hidden files and folders (.ssh, .env, .git, ...) are off-limits unless true.
allow_hidden = false
max_read_bytes = 1000000
search_seconds = 4.0

[terminal]
# PATH for commands (Jarvis's own environment is never passed on).
search_path = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
# Extra programs you trust; they always ask for confirmation.
extra_programs = []
timeout_seconds = 120.0
startup_seconds = 10.0

[browser]
# Show the browser window so you can see (and take over) what Jarvis does.
headless = false
# Unset = Google Chrome, then Edge, Brave, Chromium, or Playwright's own Chromium.
# executable = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
# Must contain {query}. Bing answers automated browsers; several others show bot checks.
search_url = "https://www.bing.com/search?q={query}"
timeout_seconds = 20.0
idle_close_seconds = 600.0

[agent]
# Plan -> execute -> observe rounds per request (each is one model call).
max_rounds = 6
# Extra attempts for idempotent tools after a transient failure.
max_tool_retries = 2
retry_backoff_seconds = 0.5
# Longest piece of tool output shown to the model per step.
max_observation_chars = 1500
# Forget the conversation after this many idle seconds.
history_ttl_seconds = 600.0

[service]
# Global shortcut that opens Jarvis (must include cmd or ctrl): ⌃⌥⌘J by default.
hotkey = "ctrl+option+cmd+j"
label = "local.jarvis.assistant"
# Also show a dialog when an action visibly succeeded (e.g. an app opened).
show_success = false
request_timeout_seconds = 300.0

[memory]
# Remember preferences you state ("remember that my editor is VS Code") and keep a
# task history. Approvals are never stored; secrets and outside content are left out.
# Inspect, export or erase it: jarvis memory show | export | clear
enabled = true
# Uncomment to override the default (~/Library/Application Support/Jarvis/memory.sqlite3).
# path = "~/Library/Application Support/Jarvis/memory.sqlite3"
record_history = true
history_days = 30
max_history = 1000
# Give remembered preferences to the model so "open my editor" works.
share_preferences = true

[voice]
# Say a short summary of each reply (on-device; macOS `say`). Off by default.
# Speech stops when you press the shortcut again. `jarvis voice voices` lists voices.
speak_replies = false
# Empty = the system voice (System Settings > Accessibility > Spoken Content).
voice = ""
# Words per minute; 0 = system default.
rate = 0
max_spoken_chars = 300
# Push-to-talk: press the shortcut (⌃⌥⌘K), speak, then pause or press it again.
# You see (and can edit) the transcript before anything runs. Nothing listens
# in the background. Needs: pip install -e ".[voice]" and `jarvis voice download`.
push_to_talk = false
push_to_talk_hotkey = "ctrl+option+cmd+k"
# Uncomment to use another whisper.cpp model. Default:
# ~/Library/Application Support/Jarvis/models/ggml-base.en-q5_1.bin
# stt_model_path = "~/models/ggml-small.en-q5_1.bin"
language = "en"
max_record_seconds = 15.0
# Stop after this much quiet following speech (0 = only when you press again).
silence_seconds = 1.5
# Metal: 0.08 s instead of 0.2 s per command, but ~7 s after a shader-cache reset.
stt_use_gpu = false
"""


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got '{raw}'.")


def resolve_config_path(
    explicit: Path | None = None, env: Mapping[str, str] | None = None
) -> tuple[Path, bool]:
    """Return ``(path, user_specified)`` for the configuration file to load."""
    env = os.environ if env is None else env
    if explicit is not None:
        return explicit.expanduser(), True
    if env.get(CONFIG_ENV_VAR):
        return Path(env[CONFIG_ENV_VAR]).expanduser(), True
    return default_config_file(), False


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror}") from exc


def _apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> None:
    general = data.setdefault("general", {})
    if not isinstance(general, dict):
        raise ConfigError("[general] must be a table.")
    if "JARVIS_DRY_RUN" in env:
        general["dry_run"] = _parse_bool("JARVIS_DRY_RUN", env["JARVIS_DRY_RUN"])
    if env.get("JARVIS_LOG_LEVEL"):
        general["log_level"] = env["JARVIS_LOG_LEVEL"]
    if env.get("JARVIS_LOG_DIR"):
        general["log_dir"] = env["JARVIS_LOG_DIR"]

    ai = data.setdefault("ai", {})
    if not isinstance(ai, dict):
        raise ConfigError("[ai] must be a table.")
    if "JARVIS_AI_ENABLED" in env:
        ai["enabled"] = _parse_bool("JARVIS_AI_ENABLED", env["JARVIS_AI_ENABLED"])
    if env.get("JARVIS_MODEL_PATH"):
        ai["model_path"] = env["JARVIS_MODEL_PATH"]


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"  {location}: {err['msg']}")
    return "\n".join(lines)


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> JarvisConfig:
    """Load and validate configuration.

    Args:
        path: Explicit config file. Must exist if given.
        env: Environment mapping (defaults to ``os.environ``); injectable for tests.

    Raises:
        ConfigError: if the file is missing (when explicitly requested), unreadable,
            not valid TOML, or fails validation.
    """
    env = os.environ if env is None else env
    config_path, user_specified = resolve_config_path(path, env)

    data: dict[str, Any] = {}
    if config_path.is_file():
        data = _read_toml(config_path)
    elif user_specified:
        raise ConfigError(
            f"Config file not found: {config_path}",
            hint="Create one with `jarvis config init`, or unset JARVIS_CONFIG.",
        )

    _apply_env_overrides(data, env)
    try:
        return JarvisConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid configuration ({config_path}):\n{_format_validation_error(exc)}"
        ) from exc


def write_example_config(path: Path, *, overwrite: bool = False) -> Path:
    """Write :data:`EXAMPLE_CONFIG_TOML` to ``path`` without clobbering existing files."""
    if path.exists() and not overwrite:
        raise ConfigError(
            f"{path} already exists; not overwriting.",
            hint="Pass --force to replace it.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG_TOML, encoding="utf-8")
    return path
