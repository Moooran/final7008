# agent/config.py
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_seconds: float = 1.0

@dataclass
class TimeoutConfig:
    step_timeout_seconds: float = 60.0
    overall_timeout_seconds: float = 1800.0

@dataclass
class AgentConfig:
    retry: RetryConfig = RetryConfig()
    timeout: TimeoutConfig = TimeoutConfig()
    log_dir: str = "logs"
    output_dir: str = "outputs"