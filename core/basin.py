"""
TokenShield Basin Cache
========================
Two-tier cache for proxy responses:

  Tier 1 — EXACT: Content hash lookup. O(1). Zero false positives.
  Tier 2 — NEAR:  Token Jaccard similarity, gated by signature proximity.
           Only fires if Jaccard >= 0.85 AND signature similarity >= 0.80.
           AGF principle: MISS is always safer than false HIT.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import time
import threading
from typing import Dict, List, Optional, Set, Tuple

from .physics import (
    compute_similarity,
    content_hash,
    tokenize,
    jaccard_similarity,
    JACCARD_THRESHOLD,
    SIGNATURE_GATE,
)


class BasinCache:
    """
    Two-tier hot cache for proxy responses.

    Tier 1: content_hash → record (exact match, O(1))
    Tier 2: iterate candidates, require BOTH high Jaccard AND signature gate

    Evicts LRU when full. Supports TTL expiration.
    """

    def __init__(self, max_size: int = 10_000, default_ttl: int = 86400):
        # Primary store: cache_id → record
        self._cache: Dict[str, Dict] = {}
        # Exact match index: content_hash → cache_id
        self._hash_index: Dict[str, str] = {}
        # LRU tracking
        self._access_order: List[str] = []
        self._max_size = max_size
        self._default_ttl = default_ttl  # seconds (default 24h)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._near_hits = 0
        self._total_tokens_saved = 0

    def get(self, cache_id: str) -> Optional[Dict]:
        """Get cached response by ID. Returns None if not in cache or expired."""
        with self._lock:
            if cache_id in self._cache:
                record = self._cache[cache_id]
                if self._is_expired(record):
                    self._evict(cache_id)
                    self._misses += 1
                    return None
                self._hits += 1
                record["hit_count"] = record.get("hit_count", 0) + 1
                record["last_hit_at"] = time.time()
                self._touch(cache_id)
                return record
            self._misses += 1
            return None

    def put(self, cache_id: str, record: Dict, ttl: Optional[int] = None):
        """
        Store a response in the cache.
        record MUST contain 'content_hash' and 'tokens' keys for Tier 2 matching.
        """
        with self._lock:
            record["cached_at"] = time.time()
            record["expires_at"] = time.time() + (ttl or self._default_ttl)
            record.setdefault("hit_count", 0)

            # Update existing
            if cache_id in self._cache:
                old_hash = self._cache[cache_id].get("content_hash")
                if old_hash and old_hash in self._hash_index:
                    del self._hash_index[old_hash]
                self._cache[cache_id] = record
                self._touch(cache_id)
            else:
                # Evict LRU if full
                while len(self._cache) >= self._max_size and self._access_order:
                    evict_id = self._access_order.pop(0)
                    evicted = self._cache.pop(evict_id, None)
                    if evicted:
                        eh = evicted.get("content_hash")
                        if eh and eh in self._hash_index:
                            del self._hash_index[eh]

                self._cache[cache_id] = record
                self._access_order.append(cache_id)

            # Index by content hash for O(1) exact match
            c_hash = record.get("content_hash")
            if c_hash:
                self._hash_index[c_hash] = cache_id

    def search_exact(self, query_hash: str, provider: str = "") -> Optional[Tuple[str, Dict]]:
        """
        Tier 1: Exact content hash match. O(1). Zero false positives.
        Returns (cache_id, record) or None.
        """
        with self._lock:
            cache_id = self._hash_index.get(query_hash)
            if cache_id is None:
                return None

            record = self._cache.get(cache_id)
            if record is None:
                # Stale index entry
                del self._hash_index[query_hash]
                return None

            if self._is_expired(record):
                self._evict(cache_id)
                return None

            # Provider filter
            if provider and record.get("provider") != provider:
                return None

            self._exact_hits += 1
            self._hits += 1
            record["hit_count"] = record.get("hit_count", 0) + 1
            record["last_hit_at"] = time.time()
            self._touch(cache_id)
            return cache_id, record

    def search_near(self, query_tokens: Set[str], query_sig: Dict[str, float],
                    provider: str = "", limit: int = 1) -> List[Tuple[str, Dict, float]]:
        """
        Tier 2: Near-match via token Jaccard + signature gate.
        Both must pass their thresholds.

        Returns list of (cache_id, record, jaccard_score) sorted by Jaccard descending.
        """
        with self._lock:
            scored = []
            for cid, record in self._cache.items():
                if self._is_expired(record):
                    continue
                if provider and record.get("provider") != provider:
                    continue

                # Gate 1: Signature similarity (fast pre-screen)
                sig = record.get("signature", {})
                sig_sim = compute_similarity(query_sig, sig)
                if sig_sim < SIGNATURE_GATE:
                    continue

                # Gate 2: Token Jaccard (actual content comparison)
                cached_tokens = record.get("tokens", set())
                if isinstance(cached_tokens, list):
                    cached_tokens = set(cached_tokens)
                jacc = jaccard_similarity(query_tokens, cached_tokens)
                if jacc < JACCARD_THRESHOLD:
                    continue

                scored.append((cid, record, jacc))

            scored.sort(key=lambda x: x[2], reverse=True)

            # Record near hits
            for cid, record, _ in scored[:limit]:
                self._near_hits += 1
                self._hits += 1
                record["hit_count"] = record.get("hit_count", 0) + 1
                record["last_hit_at"] = time.time()
                self._touch(cid)

            return scored[:limit]

    # Legacy interface for backward compatibility
    def search(self, query_sig: Dict[str, float], provider: str = "",
               limit: int = 1, threshold: float = 0.85) -> List[Tuple[str, Dict, float]]:
        """
        DEPRECATED: Use search_exact + search_near instead.
        Kept for backward compat but now requires content-level matching.
        """
        # Without content info, we can only do signature gate (not safe for cache matching)
        # Return empty — force callers to use the new two-tier API
        return []

    def clear_user(self, user_id: str) -> int:
        """Clear all cache entries for a user. Returns count removed."""
        with self._lock:
            to_remove = [
                cid for cid, rec in self._cache.items()
                if rec.get("user_id") == user_id
            ]
            for cid in to_remove:
                self._evict(cid)
            return len(to_remove)

    def record_tokens_saved(self, count: int):
        """Track cumulative tokens saved."""
        with self._lock:
            self._total_tokens_saved += count

    def stats(self) -> Dict:
        """Cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            # Clean expired entries
            expired = [
                cid for cid, rec in self._cache.items()
                if self._is_expired(rec)
            ]
            for cid in expired:
                self._evict(cid)

            return {
                "entries": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
                "hits": self._hits,
                "exact_hits": self._exact_hits,
                "near_hits": self._near_hits,
                "misses": self._misses,
                "total_tokens_saved": self._total_tokens_saved,
            }

    def _is_expired(self, record: Dict) -> bool:
        """Check if a record has expired."""
        expires = record.get("expires_at", 0)
        return expires > 0 and time.time() > expires

    def _evict(self, cache_id: str):
        """Remove a cache entry and its hash index."""
        record = self._cache.pop(cache_id, None)
        if record:
            c_hash = record.get("content_hash")
            if c_hash and c_hash in self._hash_index:
                del self._hash_index[c_hash]
        if cache_id in self._access_order:
            self._access_order.remove(cache_id)

    def _touch(self, cache_id: str):
        """Move entry to end of LRU list."""
        if cache_id in self._access_order:
            self._access_order.remove(cache_id)
        self._access_order.append(cache_id)
