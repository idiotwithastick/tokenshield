"""
CPUagen Control Barrier Functions
==================================
8 CBF schemes enforced on EVERY operation. Internal only.
Users see quality labels (HIGH/MEDIUM/LOW), never CBF internals.

If ANY scheme is UNSAFE -> operation BLOCKED.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CBFResult:
    """Result of a single CBF scheme check."""
    scheme: str
    safe: bool
    value: float
    threshold: float
    margin: float = 0.0

    def __post_init__(self):
        self.margin = self.value - self.threshold if self.safe else self.threshold - self.value


@dataclass
class CBFReport:
    """Report of all 8 CBF checks."""
    results: Dict[str, CBFResult] = field(default_factory=dict)
    all_safe: bool = False
    unsafe_schemes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.results:
            self.all_safe = all(r.safe for r in self.results.values())
            self.unsafe_schemes = [name for name, r in self.results.items() if not r.safe]

    def to_public(self) -> Dict:
        """Return sanitized report (no internal scheme names)."""
        return {
            "passed": self.all_safe,
            "checks_run": len(self.results),
            "checks_passed": sum(1 for r in self.results.values() if r.safe),
        }


# Default thresholds per Constitution Article IV
_THRESHOLDS = {
    "BNR": ("I_truth", 0.3, "gte"),
    "BNN": ("naturality", 0.2, "gte"),
    "BNA": ("energy", 1e5, "lte"),
    "TSE": ("beta_T", 0.5, "gte"),
    "PCD": ("coherence", 0.1, "gte"),
    "OGP": ("error_count", 100, "lte"),
    "ECM": ("quality_factor", 500, "lte"),
    "SPC": ("synergy", 0.5, "gte"),
}


class CBFEngine:
    """
    Control Barrier Function engine.
    Enforces 8 safety schemes on every operation.
    """

    def check_all(self, state: Dict[str, float]) -> CBFReport:
        """
        Check all 8 CBF schemes against current state.

        Args:
            state: Dict with keys matching threshold field names.

        Returns:
            CBFReport with all results.
        """
        results = {}
        for scheme, (field_name, threshold, direction) in _THRESHOLDS.items():
            value = state.get(field_name, self._default_value(field_name))

            if direction == "gte":
                safe = value >= threshold
            else:
                safe = value <= threshold

            results[scheme] = CBFResult(
                scheme=scheme,
                safe=safe,
                value=value,
                threshold=threshold,
            )

        return CBFReport(results=results)

    def enforce(self, content: str, signature: Dict[str, float]) -> CBFReport:
        """
        Enforce CBFs on a thermosolve signature.
        Maps signature fields to CBF state for checking.

        Args:
            content: The text being processed.
            signature: Thermosolve output {n, S, dS, phi}.

        Returns:
            CBFReport.
        """
        phi = signature.get("phi", 0.0)
        dS = signature.get("dS", 0.0)
        n = signature.get("n", 0)

        # Map signature to CBF state
        state = {
            "I_truth": min(1.0, phi * 1.2),           # Truth from coherence
            "naturality": phi,                          # Direct coherence
            "energy": max(10.0, n * 0.1),              # Energy from content length
            "beta_T": 1.0 if abs(dS) < 1.0 else 0.3,  # Thermal equilibrium
            "coherence": phi,                           # Direct coherence
            "error_count": 0,                           # No errors in text processing
            "quality_factor": max(1.0, abs(dS) * 100),  # Quality from entropy change
            "synergy": phi * 0.8 + 0.2,                # Synergy from coherence
        }

        return self.check_all(state)

    @staticmethod
    def _default_value(field_name: str) -> float:
        """Provide safe defaults for missing state fields."""
        defaults = {
            "I_truth": 0.5,
            "naturality": 0.5,
            "energy": 100.0,
            "beta_T": 1.0,
            "coherence": 0.7,
            "error_count": 0,
            "quality_factor": 10.0,
            "synergy": 1.0,
        }
        return defaults.get(field_name, 0.0)
