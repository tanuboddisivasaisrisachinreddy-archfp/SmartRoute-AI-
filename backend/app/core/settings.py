"""Runtime configuration from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency).  Real env vars win."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env",
                      Path(__file__).resolve().parents[2] / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    app_name: str = "SmartRoute AI"
    version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_db: str = os.getenv("MONGODB_DB", "smartroute")
    tomtom_api_key: str = os.getenv("TOMTOM_API_KEY", "")
    frontend_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if o.strip()])
    enable_server_training: bool = _bool("ENABLE_SERVER_TRAINING", True)
    max_ql_episodes: int = int(os.getenv("MAX_TRAINING_EPISODES_QL", 1500))
    max_dqn_episodes: int = int(os.getenv("MAX_TRAINING_EPISODES_DQN", 1500))
    dqn_inference_backend: str = os.getenv("DQN_INFERENCE_BACKEND", "numpy")
    nominatim_enabled: bool = _bool("NOMINATIM_ENABLED", True)
    nominatim_url: str = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
    contact_email: str = os.getenv("CONTACT_EMAIL", "")
    ql_ondemand_episodes: int = int(os.getenv("QL_ONDEMAND_EPISODES", 600))
    ql_ondemand_budget_s: float = float(os.getenv("QL_ONDEMAND_BUDGET_S", 8))
    tomtom_sample_points: int = int(os.getenv("TOMTOM_SAMPLE_POINTS", 30))
    tomtom_min_refresh_s: int = int(os.getenv("TOMTOM_MIN_REFRESH_S", 300))


settings = Settings()
