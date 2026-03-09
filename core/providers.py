"""
TokenShield Provider Adapters
==============================
Parse, normalize, and forward requests to Claude/OpenAI/Google APIs.

Each adapter:
1. Extracts messages from provider-specific request format
2. Produces a normalized prompt string for thermosolve
3. Forwards the original request to the real API (on cache miss)
4. Returns the response body + token counts

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx

# Real API base URLs
_ANTHROPIC_BASE = "https://api.anthropic.com"
_OPENAI_BASE = "https://api.openai.com"
_GOOGLE_BASE = "https://generativelanguage.googleapis.com"

# Timeout for forwarding requests (seconds)
_FORWARD_TIMEOUT = 120.0


@dataclass
class ProviderResult:
    """Result of forwarding a request to a provider."""
    status_code: int
    body: dict
    input_tokens: int = 0
    output_tokens: int = 0
    raw_headers: dict = None

    def __post_init__(self):
        if self.raw_headers is None:
            self.raw_headers = {}


class AnthropicAdapter:
    """Adapter for Anthropic Claude API (POST /v1/messages)."""

    provider = "anthropic"

    @staticmethod
    def extract_prompt(request_body: dict) -> str:
        """Extract normalized prompt text from Claude request format."""
        parts = []

        # System prompt
        system = request_body.get("system")
        if system:
            if isinstance(system, str):
                parts.append(system)
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block["text"])

        # Messages
        messages = request_body.get("messages", [])
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(f"{role}: {block['text']}")

        return "\n".join(parts)

    @staticmethod
    def extract_auth_key(headers: dict) -> Optional[str]:
        """Extract the provider API key from request headers."""
        return headers.get("x-api-key")

    @staticmethod
    def extract_token_counts(response_body: dict) -> Tuple[int, int]:
        """Extract input/output token counts from response."""
        usage = response_body.get("usage", {})
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    @staticmethod
    async def forward(request_body: dict, provider_key: str,
                      extra_headers: dict = None) -> ProviderResult:
        """Forward request to Anthropic API."""
        headers = {
            "x-api-key": provider_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if extra_headers:
            # Pass through anthropic-specific headers
            for k, v in extra_headers.items():
                if k.startswith("anthropic-"):
                    headers[k] = v

        async with httpx.AsyncClient(timeout=_FORWARD_TIMEOUT) as client:
            resp = await client.post(
                f"{_ANTHROPIC_BASE}/v1/messages",
                headers=headers,
                json=request_body,
            )
            body = resp.json()
            inp, out = AnthropicAdapter.extract_token_counts(body)
            return ProviderResult(
                status_code=resp.status_code,
                body=body,
                input_tokens=inp,
                output_tokens=out,
            )


class OpenAIAdapter:
    """Adapter for OpenAI API (POST /v1/chat/completions)."""

    provider = "openai"

    @staticmethod
    def extract_prompt(request_body: dict) -> str:
        """Extract normalized prompt text from OpenAI request format."""
        parts = []
        messages = request_body.get("messages", [])
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(f"{role}: {block['text']}")
        return "\n".join(parts)

    @staticmethod
    def extract_auth_key(headers: dict) -> Optional[str]:
        """Extract the provider API key from Authorization header."""
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None

    @staticmethod
    def extract_token_counts(response_body: dict) -> Tuple[int, int]:
        """Extract input/output token counts from response."""
        usage = response_body.get("usage", {})
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    @staticmethod
    async def forward(request_body: dict, provider_key: str,
                      extra_headers: dict = None) -> ProviderResult:
        """Forward request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {provider_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            for k, v in extra_headers.items():
                if k.lower().startswith("openai-"):
                    headers[k] = v

        async with httpx.AsyncClient(timeout=_FORWARD_TIMEOUT) as client:
            resp = await client.post(
                f"{_OPENAI_BASE}/v1/chat/completions",
                headers=headers,
                json=request_body,
            )
            body = resp.json()
            inp, out = OpenAIAdapter.extract_token_counts(body)
            return ProviderResult(
                status_code=resp.status_code,
                body=body,
                input_tokens=inp,
                output_tokens=out,
            )


class GoogleAdapter:
    """Adapter for Google Gemini API (POST /v1beta/models/{model}:generateContent)."""

    provider = "google"

    @staticmethod
    def extract_prompt(request_body: dict) -> str:
        """Extract normalized prompt text from Google request format."""
        parts = []

        # System instruction
        sys_inst = request_body.get("systemInstruction", {})
        if sys_inst:
            sys_parts = sys_inst.get("parts", [])
            if isinstance(sys_parts, dict):
                sys_parts = [sys_parts]
            for p in sys_parts:
                if isinstance(p, dict) and "text" in p:
                    parts.append(f"system: {p['text']}")

        # Contents
        contents = request_body.get("contents", [])
        for content in contents:
            role = content.get("role", "user")
            for p in content.get("parts", []):
                if isinstance(p, dict) and "text" in p:
                    parts.append(f"{role}: {p['text']}")

        return "\n".join(parts)

    @staticmethod
    def extract_auth_key(headers: dict) -> Optional[str]:
        """Extract the provider API key from x-goog-api-key header."""
        return headers.get("x-goog-api-key")

    @staticmethod
    def extract_token_counts(response_body: dict) -> Tuple[int, int]:
        """Extract input/output token counts from response."""
        usage = response_body.get("usageMetadata", {})
        return (
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
        )

    @staticmethod
    async def forward(request_body: dict, provider_key: str,
                      model: str = "gemini-2.5-flash",
                      extra_headers: dict = None) -> ProviderResult:
        """Forward request to Google Gemini API."""
        headers = {
            "x-goog-api-key": provider_key,
            "Content-Type": "application/json",
        }

        url = f"{_GOOGLE_BASE}/v1beta/models/{model}:generateContent"

        async with httpx.AsyncClient(timeout=_FORWARD_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=request_body)
            body = resp.json()
            inp, out = GoogleAdapter.extract_token_counts(body)
            return ProviderResult(
                status_code=resp.status_code,
                body=body,
                input_tokens=inp,
                output_tokens=out,
            )


# Registry for lookup
ADAPTERS = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "google": GoogleAdapter,
}
