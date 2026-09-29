import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_token: str
    allowed_users: frozenset[str]
    kb_url: str
    llm_url: str
    llm_model: str
    kb_token: str = ""
    llm_token: str = ""
    timeout: float = 120
    max_cases: int = 3
    docs_per_case: int = 3
    max_doc_chars: int = 12000
    max_total_chars: int = 48000
    batch_chars: int = 6000

    @classmethod
    def from_env(cls):
        result = cls(
            data_dir=Path(os.getenv("MRO_AGENT_DATA_DIR", "runtime")),
            api_token=os.getenv("MRO_AGENT_API_TOKEN", ""),
            allowed_users=frozenset(x.strip() for x in os.getenv("MRO_AGENT_ALLOWED_USERS", "").split(",") if x.strip()),
            kb_url=os.getenv("MRO_AGENT_KB_URL", "").rstrip("/"),
            llm_url=os.getenv("MRO_AGENT_LLM_URL", "").rstrip("/"),
            llm_model=os.getenv("MRO_AGENT_LLM_MODEL", ""),
            kb_token=os.getenv("MRO_AGENT_KB_TOKEN", ""),
            llm_token=os.getenv("MRO_AGENT_LLM_TOKEN", ""),
            timeout=float(os.getenv("MRO_AGENT_HTTP_TIMEOUT", "120")),
        )
        if len(result.api_token) < 24 or result.api_token.startswith("replace-"):
            raise ValueError("Set MRO_AGENT_API_TOKEN to a service token of at least 24 characters")
        if not result.allowed_users or not result.llm_model or result.llm_model.startswith("replace-"):
            raise ValueError("Configure allowed pilot user IDs and a local LLM model")
        for url in (result.kb_url, result.llm_url):
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Configure explicit HTTP(S) service base URLs without credentials/query")
        if result.timeout <= 0:
            raise ValueError("HTTP timeout must be positive")
        return result

