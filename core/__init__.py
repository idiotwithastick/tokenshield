"""TokenShield Core — Portable SSD-RCI Engine (IP Protected)"""
from .physics import thermosolve, compute_similarity, compute_quality_label, content_hash, tokenize, jaccard_similarity
from .cbf import CBFEngine, CBFReport
from .state import ProxyState
from .basin import BasinCache
from .attestation import Attestation, AttestationBuilder
from .gateway import EnforcementGateway
from .providers import AnthropicAdapter, OpenAIAdapter, GoogleAdapter, ADAPTERS
