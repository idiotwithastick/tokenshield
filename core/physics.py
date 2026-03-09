"""
TokenShield Thermosolve Engine
================================
IEEE 2847-2025 compliant thermosolve in NATS (natural logarithm).
Zero external dependencies — pure Python stdlib.

Computes thermodynamic signatures {n, S, dS, phi} for any text input.
Signatures are used for PHYSICS ENFORCEMENT (CBF gating, quality labels).
Cache matching uses content_hash (exact) + token Jaccard (near-match).

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import hashlib
import math
from typing import Dict, List, Set, Tuple

# Coherence component weights (sum to 1.0)
_W_LEXICAL = 0.4
_W_SENTENCE = 0.3
_W_ENTROPY = 0.3

# Ideal targets for coherence scoring
_IDEAL_SENTENCE_LEN = 15
_IDEAL_ENTROPY_NATS = 3.5

# Near-match thresholds (AGF: err toward MISS over false HIT)
JACCARD_THRESHOLD = 0.85       # Token overlap must be >= 85%
SIGNATURE_GATE = 0.80          # Signature must be >= 80% similar (pre-screen only)


def thermosolve(content: str) -> Dict[str, float]:
    """
    Compute thermosolve signature {n, S, dS, phi}.

    Args:
        content: Input text.

    Returns:
        Dict with n (length), S (Shannon entropy in nats),
        dS (normalized entropy change, <= 0), phi (coherence, 0-1).
    """
    n = len(content)
    if n == 0:
        return {"n": 0, "S": 0.0, "dS": 0.0, "phi": 0.0}

    # Shannon entropy in NATS
    freq: Dict[str, int] = {}
    for c in content:
        freq[c] = freq.get(c, 0) + 1

    S = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            S -= p * math.log(p)

    # Baseline entropy (uniform distribution)
    alphabet_size = min(n, 256)
    S_baseline = math.log(max(alphabet_size, 1))

    # Normalized entropy change
    normalizer = math.log(n + 1) if n > 0 else 1.0
    dS = (S - S_baseline) / normalizer if normalizer > 0 else 0.0
    dS = min(dS, 0.0)

    # Coherence (phi): lexical diversity + sentence structure + entropy score
    words = content.lower().split()
    if words:
        lexical_diversity = len(set(words)) / len(words)
    else:
        lexical_diversity = 0.0

    sentences = [
        s.strip()
        for s in content.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    avg_sentence_len = (
        sum(len(s.split()) for s in sentences) / len(sentences)
        if sentences else 0
    )
    sentence_score = 1.0 - abs(avg_sentence_len - _IDEAL_SENTENCE_LEN) / _IDEAL_SENTENCE_LEN
    sentence_score = max(0.0, min(1.0, sentence_score))

    if S > 0:
        entropy_score = 1.0 - abs(S - _IDEAL_ENTROPY_NATS) / _IDEAL_ENTROPY_NATS
        entropy_score = max(0.0, min(1.0, entropy_score))
    else:
        entropy_score = 0.0

    phi = (
        _W_LEXICAL * lexical_diversity
        + _W_SENTENCE * sentence_score
        + _W_ENTROPY * entropy_score
    )
    phi = max(0.0, min(1.0, phi))

    return {
        "n": n,
        "S": round(S, 6),
        "dS": round(dS, 6),
        "phi": round(phi, 6),
    }


def content_hash(text: str) -> str:
    """
    SHA256 hash of normalized text. Primary cache key.
    Exact match only — zero false positives.
    """
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def tokenize(text: str) -> Set[str]:
    """
    Tokenize text into lowercase word set for Jaccard comparison.
    Strips punctuation so 'React.' and 'React' match.
    Filters out stopwords and role prefixes that add noise.
    """
    _STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
             "in", "on", "at", "to", "for", "of", "and", "or", "but",
             "it", "its", "this", "that", "with", "from", "by", "as",
             "i", "me", "my", "you", "your", "we", "our", "they", "them",
             "user:", "system:", "assistant:", "user", "system", "assistant"}
    words = text.lower().split()
    result = set()
    for w in words:
        # Strip leading/trailing punctuation
        clean = w.strip(".,;:!?\"'()[]{}…–—-")
        if clean and len(clean) > 1 and clean not in _STOP:
            result.add(clean)
    return result


def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|.
    Measures actual content overlap, not statistical properties.
    Range: 0.0 (no overlap) to 1.0 (identical token sets).
    """
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def compute_similarity(sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
    """
    Signature-space similarity. Used ONLY as a pre-screen gate.
    NOT sufficient for cache matching on its own.

    Uses weighted distance in (S, dS, phi) space with exponential decay.
    Range: 0.0 (completely different) to 1.0 (identical).
    """
    if not sig_a or not sig_b:
        return 0.0

    # Weighted Euclidean distance in signature space
    w_S, w_dS, w_phi = 0.3, 0.3, 0.4
    d_S = (sig_a.get("S", 0) - sig_b.get("S", 0)) ** 2
    d_dS = (sig_a.get("dS", 0) - sig_b.get("dS", 0)) ** 2
    d_phi = (sig_a.get("phi", 0) - sig_b.get("phi", 0)) ** 2

    distance = math.sqrt(w_S * d_S + w_dS * d_dS + w_phi * d_phi)

    # Exponential decay: close signatures → high similarity
    return math.exp(-distance * 2.0)


def compute_quality_label(sig: Dict[str, float]) -> str:
    """Map signature to user-facing quality label."""
    phi = sig.get("phi", 0)
    dS = sig.get("dS", 0)

    if phi >= 0.7 and dS <= -0.05:
        return "HIGH"
    elif phi >= 0.4 and dS <= -0.02:
        return "MEDIUM"
    else:
        return "LOW"
