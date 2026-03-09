"""
TokenShield Comprehensive Test Suite
======================================
Tests all core components + API endpoints via FastAPI TestClient.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import os
import sys
import json
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient
from main import app
from db.connection import init_db

# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Initialize DB before all tests."""
    init_db()
    yield

@pytest.fixture(scope="module")
def client():
    """FastAPI test client."""
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════
# 1. Core: Physics / Thermosolve
# ═══════════════════════════════════════════════════════════════════

class TestPhysics:

    def test_thermosolve_returns_signature(self):
        from core.physics import thermosolve
        sig = thermosolve("Hello, world!")
        assert isinstance(sig, dict)
        assert set(sig.keys()) == {"n", "S", "dS", "phi"}
        assert sig["n"] == 13

    def test_thermosolve_empty_string(self):
        from core.physics import thermosolve
        sig = thermosolve("")
        assert sig == {"n": 0, "S": 0.0, "dS": 0.0, "phi": 0.0}

    def test_thermosolve_deterministic(self):
        from core.physics import thermosolve
        a = thermosolve("Write a Python function to sort a list")
        b = thermosolve("Write a Python function to sort a list")
        assert a == b

    def test_content_hash_deterministic(self):
        from core.physics import content_hash
        h1 = content_hash("test prompt")
        h2 = content_hash("test prompt")
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_content_hash_case_insensitive(self):
        from core.physics import content_hash
        assert content_hash("Hello") == content_hash("hello")

    def test_tokenize_removes_stopwords(self):
        from core.physics import tokenize
        tokens = tokenize("user: Write a Python function to sort the list")
        assert "user:" not in tokens
        assert "the" not in tokens
        assert "python" in tokens
        assert "function" in tokens

    def test_jaccard_identical(self):
        from core.physics import jaccard_similarity
        a = {"python", "function", "sort"}
        assert jaccard_similarity(a, a) == 1.0

    def test_jaccard_disjoint(self):
        from core.physics import jaccard_similarity
        a = {"python", "function"}
        b = {"javascript", "class"}
        assert jaccard_similarity(a, b) == 0.0

    def test_jaccard_partial(self):
        from core.physics import jaccard_similarity
        a = {"python", "function", "sort", "list"}
        b = {"python", "function", "filter", "list"}
        score = jaccard_similarity(a, b)
        assert 0.0 < score < 1.0

    def test_compute_similarity_identical(self):
        from core.physics import compute_similarity, thermosolve
        sig = thermosolve("Write a Python function")
        score = compute_similarity(sig, sig)
        assert score == 1.0

    def test_compute_similarity_different(self):
        from core.physics import compute_similarity, thermosolve
        a = thermosolve("Short")
        b = thermosolve("A very long text with many different words and sentences about various topics in computer science and mathematics")
        score = compute_similarity(a, b)
        assert score < 0.9

    def test_entropy_positive(self):
        from core.physics import thermosolve
        sig = thermosolve("Write a Python function to calculate entropy")
        assert sig["S"] > 0.0

    def test_ds_nonpositive(self):
        from core.physics import thermosolve
        sig = thermosolve("Any reasonable text should have dS <= 0")
        assert sig["dS"] <= 0.0


# ═══════════════════════════════════════════════════════════════════
# 2. Core: CBF Engine
# ═══════════════════════════════════════════════════════════════════

class TestCBF:

    def test_cbf_all_safe_for_normal_text(self):
        from core.cbf import CBFEngine
        from core.physics import thermosolve
        engine = CBFEngine()
        sig = thermosolve("Write a Python function to reverse a string")
        report = engine.enforce("Write a Python function to reverse a string", sig)
        assert report.all_safe is True
        assert len(report.results) == 8

    def test_cbf_public_sanitized(self):
        from core.cbf import CBFEngine
        from core.physics import thermosolve
        engine = CBFEngine()
        sig = thermosolve("test")
        report = engine.enforce("test", sig)
        public = report.to_public()
        assert "passed" in public
        assert "checks_run" in public
        # No internal scheme names exposed
        assert "BNR" not in str(public)


# ═══════════════════════════════════════════════════════════════════
# 3. Core: Basin Cache
# ═══════════════════════════════════════════════════════════════════

class TestBasinCache:

    def test_put_and_get(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        record = {
            "content_hash": "abc123",
            "tokens": ["python", "function"],
            "signature": {"n": 10, "S": 2.5, "dS": -0.1, "phi": 0.7},
            "response_body": {"content": "test"},
            "input_tokens": 100,
            "output_tokens": 50,
        }
        cache.put("test_id", record)
        result = cache.get("test_id")
        assert result is not None
        assert result["content_hash"] == "abc123"

    def test_exact_search(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        record = {
            "content_hash": "exact_hash_123",
            "tokens": ["python", "function"],
            "signature": {"n": 10, "S": 2.5, "dS": -0.1, "phi": 0.7},
            "response_body": {"content": "cached response"},
            "input_tokens": 100,
            "output_tokens": 50,
            "provider": "anthropic",
        }
        cache.put("test_exact", record)
        result = cache.search_exact("exact_hash_123", provider="anthropic")
        assert result is not None
        cache_id, found_record = result
        assert cache_id == "test_exact"

    def test_exact_search_miss(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        result = cache.search_exact("nonexistent_hash")
        assert result is None

    def test_near_search(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        # Put a record with specific tokens
        tokens = {"python", "function", "sort", "list", "algorithm", "data", "structure"}
        record = {
            "content_hash": "near_hash_1",
            "tokens": list(tokens),
            "signature": {"n": 50, "S": 3.0, "dS": -0.05, "phi": 0.65},
            "response_body": {"content": "cached near response"},
            "input_tokens": 100,
            "output_tokens": 50,
            "provider": "anthropic",
        }
        cache.put("near_test", record)

        # Search with very similar tokens (Jaccard >= 0.85)
        query_tokens = {"python", "function", "sort", "list", "algorithm", "data", "structure"}
        query_sig = {"n": 50, "S": 3.0, "dS": -0.05, "phi": 0.65}
        results = cache.search_near(query_tokens, query_sig, provider="anthropic")
        assert len(results) > 0

    def test_near_search_rejects_dissimilar(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        record = {
            "content_hash": "far_hash_1",
            "tokens": ["python", "function", "sort"],
            "signature": {"n": 50, "S": 3.0, "dS": -0.05, "phi": 0.65},
            "response_body": {"content": "cached"},
            "input_tokens": 100,
            "output_tokens": 50,
            "provider": "anthropic",
        }
        cache.put("far_test", record)

        # Search with completely different tokens
        query_tokens = {"javascript", "class", "react", "component", "render", "state", "hooks"}
        query_sig = {"n": 200, "S": 4.0, "dS": -0.2, "phi": 0.3}
        results = cache.search_near(query_tokens, query_sig, provider="anthropic")
        assert len(results) == 0

    def test_stats(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        stats = cache.stats()
        assert "entries" in stats
        assert "hit_rate" in stats
        assert "exact_hits" in stats
        assert "near_hits" in stats

    def test_lru_eviction(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=3)
        for i in range(5):
            cache.put(f"id_{i}", {
                "content_hash": f"hash_{i}",
                "tokens": [],
                "response_body": {},
            })
        assert cache.stats()["entries"] == 3

    def test_clear_user(self):
        from core.basin import BasinCache
        cache = BasinCache(max_size=100)
        for i in range(3):
            cache.put(f"user_cache_{i}", {
                "content_hash": f"uh_{i}",
                "tokens": [],
                "response_body": {},
                "user_id": "user@test.com",
            })
        removed = cache.clear_user("user@test.com")
        assert removed == 3


# ═══════════════════════════════════════════════════════════════════
# 4. Core: Gateway
# ═══════════════════════════════════════════════════════════════════

class TestGateway:

    def test_check_cache_miss(self):
        from core.gateway import EnforcementGateway
        gw = EnforcementGateway()
        cached, cache_id, attest = gw.check_cache(
            "Write a Python function to calculate fibonacci",
            provider="anthropic",
            user_id="test@test.com",
        )
        assert cached is None
        assert cache_id != ""
        assert attest.blocked is False
        assert attest.cbf_passed is True

    def test_cache_and_retrieve(self):
        from core.gateway import EnforcementGateway
        gw = EnforcementGateway()
        prompt = "Write a function to calculate the sum of two numbers"

        # First call: miss
        cached, cache_id, _ = gw.check_cache(prompt, "anthropic", "user1")
        assert cached is None

        # Cache a response
        gw.cache_response(
            cache_id=cache_id,
            provider="anthropic",
            normalized_prompt=prompt,
            response_body={"content": [{"type": "text", "text": "def add(a, b): return a + b"}]},
            input_tokens=50,
            output_tokens=30,
            user_id="user1",
        )

        # Second call: hit
        cached, _, attest = gw.check_cache(prompt, "anthropic", "user1")
        assert cached is not None
        assert cached["response_body"]["content"][0]["text"] == "def add(a, b): return a + b"


# ═══════════════════════════════════════════════════════════════════
# 5. Core: Provider Adapters
# ═══════════════════════════════════════════════════════════════════

class TestProviders:

    def test_anthropic_extract_prompt(self):
        from core.providers import AnthropicAdapter
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, Claude!"}
            ]
        }
        prompt = AnthropicAdapter.extract_prompt(body)
        assert "Hello, Claude!" in prompt

    def test_anthropic_extract_prompt_with_system(self):
        from core.providers import AnthropicAdapter
        body = {
            "model": "claude-sonnet-4-20250514",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }
        prompt = AnthropicAdapter.extract_prompt(body)
        assert "helpful assistant" in prompt
        assert "Hi" in prompt

    def test_anthropic_extract_auth(self):
        from core.providers import AnthropicAdapter
        headers = {"x-api-key": "sk-ant-test123"}
        assert AnthropicAdapter.extract_auth_key(headers) == "sk-ant-test123"

    def test_openai_extract_prompt(self):
        from core.providers import OpenAIAdapter
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
        }
        prompt = OpenAIAdapter.extract_prompt(body)
        assert "helpful" in prompt
        assert "Hello!" in prompt

    def test_openai_extract_auth(self):
        from core.providers import OpenAIAdapter
        headers = {"authorization": "Bearer sk-openai-test123"}
        assert OpenAIAdapter.extract_auth_key(headers) == "sk-openai-test123"

    def test_google_extract_prompt(self):
        from core.providers import GoogleAdapter
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello, Gemini!"}]
                }
            ]
        }
        prompt = GoogleAdapter.extract_prompt(body)
        assert "Hello, Gemini!" in prompt

    def test_google_extract_auth(self):
        from core.providers import GoogleAdapter
        headers = {"x-goog-api-key": "google-key-123"}
        assert GoogleAdapter.extract_auth_key(headers) == "google-key-123"


# ═══════════════════════════════════════════════════════════════════
# 6. API Endpoints
# ═══════════════════════════════════════════════════════════════════

class TestAPI:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_landing_page(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_signup(self, client):
        r = client.post("/v1/signup", json={"email": "test@tokenshield.com"})
        assert r.status_code == 200
        data = r.json()
        assert "api_key" in data
        assert data["api_key"].startswith("sk-ts-")
        assert data["tier"] == "free"

    def test_signup_duplicate(self, client):
        # First signup
        client.post("/v1/signup", json={"email": "dup@tokenshield.com"})
        # Duplicate
        r = client.post("/v1/signup", json={"email": "dup@tokenshield.com"})
        assert r.status_code == 409

    def test_status_requires_auth(self, client):
        r = client.get("/v1/status")
        assert r.status_code == 401

    def test_status_with_key(self, client):
        # Signup first
        signup = client.post("/v1/signup", json={"email": "status@tokenshield.com"})
        key = signup.json()["api_key"]

        r = client.get("/v1/status", headers={"X-Proxy-Key": key})
        assert r.status_code == 200
        data = r.json()
        assert "account" in data
        assert "cache" in data
        assert data["account"]["tier"] == "free"

    def test_savings_with_key(self, client):
        signup = client.post("/v1/signup", json={"email": "savings@tokenshield.com"})
        key = signup.json()["api_key"]

        r = client.get("/v1/savings", headers={"X-Proxy-Key": key})
        assert r.status_code == 200
        data = r.json()
        assert "total_tokens_saved" in data
        assert "by_provider" in data

    def test_proxy_requires_auth(self, client):
        r = client.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401

    def test_proxy_invalid_key(self, client):
        r = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Proxy-Key": "sk-ts-invalid"}
        )
        assert r.status_code == 401

    def test_proxy_missing_provider_key(self, client):
        signup = client.post("/v1/signup", json={"email": "proxy1@tokenshield.com"})
        key = signup.json()["api_key"]

        r = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Proxy-Key": key}
        )
        # Should fail because no x-api-key (provider key)
        assert r.status_code == 401

    def test_cache_clear(self, client):
        signup = client.post("/v1/signup", json={"email": "clear@tokenshield.com"})
        key = signup.json()["api_key"]

        r = client.delete("/v1/cache/clear", headers={"X-Proxy-Key": key})
        assert r.status_code == 200
        assert "cleared" in r.json()

    def test_docs_page(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openai_endpoint_exists(self, client):
        # Should return 401 (not 404) - endpoint exists but auth required
        r = client.post("/v1/chat/completions", json={})
        assert r.status_code == 401

    def test_google_endpoint_exists(self, client):
        r = client.post("/v1beta/models/gemini-2.5-flash:generateContent", json={})
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════
# 7. End-to-End: Cache Hit/Miss Flow
# ═══════════════════════════════════════════════════════════════════

class TestE2ECacheFlow:
    """Test the full cache flow without hitting real provider APIs."""

    def test_gateway_cache_roundtrip(self):
        """Verify: miss → cache → hit → same response."""
        from core.gateway import EnforcementGateway
        gw = EnforcementGateway()

        prompt = "Explain the theory of relativity in simple terms"
        provider = "anthropic"

        # Miss
        cached, cache_id, attest1 = gw.check_cache(prompt, provider)
        assert cached is None
        assert attest1.blocked is False

        # Cache
        response_body = {
            "id": "msg_test",
            "type": "message",
            "content": [{"type": "text", "text": "E=mc^2 means energy equals mass times speed of light squared."}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 20, "output_tokens": 30},
        }
        gw.cache_response(cache_id, provider, prompt, response_body, 20, 30)

        # Hit
        cached, _, attest2 = gw.check_cache(prompt, provider)
        assert cached is not None
        assert cached["response_body"]["content"][0]["text"].startswith("E=mc^2")

    def test_different_prompts_no_false_hit(self):
        """Different prompts must NOT match."""
        from core.gateway import EnforcementGateway
        gw = EnforcementGateway()

        # Cache prompt A
        prompt_a = "How do I make a REST API in Python with Flask"
        cached, cid, _ = gw.check_cache(prompt_a, "openai")
        gw.cache_response(cid, "openai", prompt_a, {"result": "Flask tutorial"}, 30, 40)

        # Query prompt B (completely different)
        prompt_b = "What is the capital of France"
        cached, _, _ = gw.check_cache(prompt_b, "openai")
        assert cached is None  # Must NOT return Flask tutorial


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
