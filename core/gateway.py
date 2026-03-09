"""
TokenShield Enforcement Gateway
=================================
End-to-end enforcement for proxy operations.
PHYSICS FIRST, STATE NOT PROMPT.

Cache Lookup Protocol (AGF-compliant):
  1. Thermosolve the normalized prompt → signature (physics enforcement)
  2. CBF check (all 8 schemes must pass)
  3. Content hash → exact match lookup (Tier 1, O(1))
  4. If no exact match → token Jaccard + signature gate (Tier 2)
  5. HIT → return cached response
  6. MISS → forward to provider API (caller handles this)
  7. On response → cache with content_hash + tokens + signature

AGF Principle: A MISS is always safer than a false HIT.
Never serve a wrong cached response. Err toward forwarding.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple

from .physics import thermosolve, content_hash, tokenize, compute_similarity, compute_quality_label
from .cbf import CBFEngine, CBFReport
from .basin import BasinCache
from .attestation import Attestation, AttestationBuilder


class ProxyResult:
    """Result of a proxy operation."""
    __slots__ = (
        "cache_hit", "response_body", "status_code",
        "input_tokens", "output_tokens", "attestation",
        "cache_id",
    )

    def __init__(self):
        self.cache_hit: bool = False
        self.response_body: dict = {}
        self.status_code: int = 200
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.attestation: Optional[Attestation] = None
        self.cache_id: str = ""


class EnforcementGateway:
    """
    Single entry point for ALL proxy operations.

    AGF Protocol:
      1. Thermosolve the normalized prompt → physics state
      2. Enforce CBFs on state (ALL 8 must pass)
      3. Tier 1: exact content hash match (O(1), zero false positives)
      4. Tier 2: token Jaccard + signature gate (near-match)
      5. On HIT → return cached response
      6. On MISS → forward to provider (caller handles this)
      7. Return with attestation
    """

    def __init__(self, cache: Optional[BasinCache] = None, max_cache: int = 10_000):
        self._cbf = CBFEngine()
        self._cache = cache or BasinCache(max_size=max_cache)

    def check_cache(
        self, normalized_prompt: str, provider: str, user_id: str = ""
    ) -> Tuple[Optional[Dict], str, Attestation]:
        """
        Physics-first cache check with two-tier matching.

        1. Thermosolve the prompt → signature (for CBF enforcement)
        2. CBF enforcement on physics state
        3. Tier 1: content_hash exact match
        4. Tier 2: token Jaccard + signature gate
        5. Return (cached_response or None, cache_id, attestation)
        """
        attest = AttestationBuilder(operation="proxy")

        # STEP 1: Physics solve FIRST
        t0 = time.perf_counter()
        signature = thermosolve(normalized_prompt)
        solve_ms = (time.perf_counter() - t0) * 1000
        attest.record_solve(solve_ms, jit=True)

        # STEP 2: CBF enforcement on STATE
        t1 = time.perf_counter()
        cbf_report = self._cbf.enforce(normalized_prompt, signature)
        cbf_ms = (time.perf_counter() - t1) * 1000
        attest.record_cbf(cbf_ms, cbf_report)

        if not cbf_report.all_safe:
            attest.record_blocked(cbf_report.unsafe_schemes)
            return None, "", attest.build()

        # STEP 3: Content-level cache lookup
        c_hash = content_hash(normalized_prompt)
        tokens = tokenize(normalized_prompt)
        cache_id = self._make_cache_id(provider, c_hash)

        t2 = time.perf_counter()

        # Tier 1: Exact content hash match (O(1), zero false positives)
        exact = self._cache.search_exact(c_hash, provider=provider)
        if exact:
            _, record = exact
            cache_ms = (time.perf_counter() - t2) * 1000
            attest.record_cache_hit(cache_ms, relevance=1.0)

            total_tokens = record.get("input_tokens", 0) + record.get("output_tokens", 0)
            attest.record_token_savings(total_tokens, state_size=0)
            attest.record_success()
            self._cache.record_tokens_saved(total_tokens)

            return record, cache_id, attest.build()

        # Tier 2: Near-match (token Jaccard + signature gate)
        near = self._cache.search_near(tokens, signature, provider=provider, limit=1)
        if near:
            _, record, jacc_score = near[0]
            cache_ms = (time.perf_counter() - t2) * 1000
            attest.record_cache_hit(cache_ms, relevance=jacc_score)

            total_tokens = record.get("input_tokens", 0) + record.get("output_tokens", 0)
            attest.record_token_savings(total_tokens, state_size=0)
            attest.record_success()
            self._cache.record_tokens_saved(total_tokens)

            return record, cache_id, attest.build()

        # Cache miss
        cache_ms = (time.perf_counter() - t2) * 1000
        attest.record_cache_miss(cache_ms)
        prompt_tokens = len(normalized_prompt.split())
        attest.record_token_savings(prompt_tokens, state_size=prompt_tokens)
        attest.record_success()

        return None, cache_id, attest.build()

    def cache_response(
        self, cache_id: str, provider: str, normalized_prompt: str,
        response_body: dict, input_tokens: int, output_tokens: int,
        user_id: str = "", ttl: Optional[int] = None,
    ):
        """
        Cache a provider response for future reuse.
        Called after a cache miss + successful forward.

        Stores content_hash, tokens, and signature for two-tier matching.
        """
        signature = thermosolve(normalized_prompt)
        c_hash = content_hash(normalized_prompt)
        tokens = list(tokenize(normalized_prompt))  # list for serialization

        record = {
            "provider": provider,
            "content_hash": c_hash,
            "tokens": tokens,
            "signature": signature,
            "response_body": response_body,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "user_id": user_id,
            "hit_count": 0,
        }

        self._cache.put(cache_id, record, ttl=ttl)

    def clear_user_cache(self, user_id: str) -> int:
        """Clear all cached responses for a user."""
        return self._cache.clear_user(user_id)

    def get_cache_stats(self) -> Dict:
        """Return cache statistics."""
        return self._cache.stats()

    @staticmethod
    def _make_cache_id(provider: str, c_hash: str) -> str:
        """Build a deterministic cache ID from provider + content hash."""
        return f"cache_{provider}_{c_hash[:16]}"
