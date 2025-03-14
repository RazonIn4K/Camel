import asyncio
from datetime import datetime, timedelta
from typing import Dict

from .exceptions import RateLimitError


class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[datetime, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquires a rate limit token or raises RateLimitError.

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        async with self._lock:
            now = datetime.now()
            self._cleanup_old_requests(now)

            current_requests = sum(self.requests.values())
            if current_requests >= self.max_requests:
                raise RateLimitError(
                    f"Rate limit of {self.max_requests} requests per {self.time_window} seconds exceeded"
                )

            self.requests[now] = self.requests.get(now, 0) + 1

    def _cleanup_old_requests(self, current_time: datetime) -> None:
        """Removes requests older than the time window."""
        cutoff = current_time - timedelta(seconds=self.time_window)
        self.requests = {
            ts: count for ts, count in self.requests.items() if ts > cutoff
        }
