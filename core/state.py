"""
TokenShield Proxy State
========================
Tracks per-user proxy statistics and token budget.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProxyState:
    """Per-user proxy state tracking."""
    user_id: str = ""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_saved: int = 0
    total_tokens_forwarded: int = 0
    providers_used: List[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def record_hit(self, tokens_saved: int):
        self.total_requests += 1
        self.cache_hits += 1
        self.total_tokens_saved += tokens_saved

    def record_miss(self, tokens_used: int, provider: str):
        self.total_requests += 1
        self.cache_misses += 1
        self.total_tokens_forwarded += tokens_used
        if provider not in self.providers_used:
            self.providers_used.append(provider)

    def to_public(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 3),
            "tokens_saved": self.total_tokens_saved,
            "tokens_forwarded": self.total_tokens_forwarded,
            "providers_used": self.providers_used,
        }
