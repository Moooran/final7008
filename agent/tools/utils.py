# agent/tools/utils.py
from typing import Callable, Awaitable, Any

async def with_retry(fn: Callable[[], Awaitable[Any]], max_retries: int = 3) -> Any:
    last_err = None
    for _ in range(max_retries):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err