"""Composable, auditable evidence-family admission for Agent 1 candidates."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Signal:
    name: str
    family: str
    match: str = "exact"
    witness_start: int | None = None
    witness_end: int | None = None


@dataclass
class Candidate:
    start: int
    end: int
    surface: str
    signals: set[Signal] = field(default_factory=set)
    missing_signals: set[str] = field(default_factory=set)
    soft_conflicts: set[str] = field(default_factory=set)
    vetoes: set[str] = field(default_factory=set)

    def add(
        self,
        name: str,
        family: str,
        match: str = "exact",
        witness_start: int | None = None,
        witness_end: int | None = None,
    ) -> None:
        self.signals.add(Signal(
            name=name,
            family=family,
            match=match,
            witness_start=witness_start,
            witness_end=witness_end,
        ))

    def veto(self, name: str) -> None:
        self.vetoes.add(name)

    def missing(self, name: str) -> None:
        self.missing_signals.add(name)

    def conflict(self, name: str) -> None:
        self.soft_conflicts.add(name)


@dataclass(frozen=True)
class AdmissionPolicy:
    name: str
    required_signals: frozenset[str]
    prerequisite_signals: frozenset[str] = frozenset()
    allowed_soft_conflicts: frozenset[str] = frozenset()

    def matches(self, candidate: Candidate) -> bool:
        if candidate.vetoes:
            return False
        if not candidate.soft_conflicts <= self.allowed_soft_conflicts:
            return False
        by_name = {signal.name: signal for signal in candidate.signals}
        if not (
            self.required_signals | self.prerequisite_signals
        ) <= by_name.keys():
            return False
        families = {
            by_name[signal_name].family
            for signal_name in self.required_signals
        }
        return len(families) == len(self.required_signals)


def fuzzy_relation(
    candidate_start: int,
    candidate_end: int,
    witness_start: int,
    witness_end: int,
    max_boundary_delta: int = 1,
) -> str | None:
    """Return a tight span relation; adjacency without overlap is never support."""
    if candidate_start == witness_start and candidate_end == witness_end:
        return "exact"
    if not (
        candidate_start < witness_end
        and witness_start < candidate_end
    ):
        return None
    if (
        witness_start <= candidate_start
        and candidate_end <= witness_end
        and candidate_start - witness_start <= max_boundary_delta
        and witness_end - candidate_end <= max_boundary_delta
    ):
        return "witness_contains"
    if (
        candidate_start <= witness_start
        and witness_end <= candidate_end
        and witness_start - candidate_start <= max_boundary_delta
        and candidate_end - witness_end <= max_boundary_delta
    ):
        return "candidate_contains"
    if (
        abs(candidate_start - witness_start) <= max_boundary_delta
        and abs(candidate_end - witness_end) <= max_boundary_delta
    ):
        return "boundary_shift"
    return None


def decide(
    candidate: Candidate,
    policies: tuple[AdmissionPolicy, ...],
) -> AdmissionPolicy | None:
    return next((policy for policy in policies if policy.matches(candidate)), None)


def audit_metadata(candidate: Candidate, policy: AdmissionPolicy) -> dict:
    return {
        "evidence_policy": policy.name,
        "evidence_families": sorted({signal.family for signal in candidate.signals}),
        "evidence_signals": sorted(signal.name for signal in candidate.signals),
        "evidence_missing": sorted(candidate.missing_signals),
        "evidence_soft_conflicts": sorted(candidate.soft_conflicts),
        "evidence_witnesses": sorted(
            (
                {
                    "signal": signal.name,
                    "family": signal.family,
                    "match": signal.match,
                    "start": signal.witness_start,
                    "end": signal.witness_end,
                }
                for signal in candidate.signals
                if signal.witness_start is not None
            ),
            key=lambda row: (
                row["signal"],
                row["start"],
                row["end"],
            ),
        ),
    }
