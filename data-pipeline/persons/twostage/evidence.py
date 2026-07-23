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


@dataclass(frozen=True)
class FamilySupport:
    family: str
    weight: int
    any_signals: frozenset[str] = frozenset()
    all_signals: frozenset[str] = frozenset()

    def matches(self, signal_names: set[str]) -> bool:
        return (
            (not self.any_signals or bool(self.any_signals & signal_names))
            and self.all_signals <= signal_names
        )


@dataclass(frozen=True)
class CumulativeAdmissionPolicy:
    name: str
    family_support: tuple[FamilySupport, ...]
    minimum_score: int
    minimum_families: int
    prerequisite_signals: frozenset[str] = frozenset()
    conflict_penalties: tuple[tuple[str, int], ...] = ()

    @property
    def allowed_soft_conflicts(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.conflict_penalties)

    def score(self, candidate: Candidate) -> tuple[int, frozenset[str]]:
        signal_names = {signal.name for signal in candidate.signals}
        family_weights: dict[str, int] = {}
        for support in self.family_support:
            if support.matches(signal_names):
                family_weights[support.family] = max(
                    family_weights.get(support.family, 0),
                    support.weight,
                )
        penalties = dict(self.conflict_penalties)
        score = sum(family_weights.values()) - sum(
            penalties.get(conflict, 0)
            for conflict in candidate.soft_conflicts
        )
        return score, frozenset(family_weights)

    def matches(self, candidate: Candidate) -> bool:
        if candidate.vetoes:
            return False
        if not candidate.soft_conflicts <= self.allowed_soft_conflicts:
            return False
        signal_names = {signal.name for signal in candidate.signals}
        if not self.prerequisite_signals <= signal_names:
            return False
        score, families = self.score(candidate)
        return score >= self.minimum_score and len(families) >= self.minimum_families


def policy_diagnostics(
    candidate: Candidate,
    policy: AdmissionPolicy | CumulativeAdmissionPolicy,
) -> dict:
    """Explain exactly why a candidate does or does not satisfy a policy."""
    signal_names = {signal.name for signal in candidate.signals}
    if isinstance(policy, CumulativeAdmissionPolicy):
        score, families = policy.score(candidate)
        return {
            "policy": policy.name,
            "matched_required": [],
            "missing_required": [],
            "missing_prerequisites": sorted(
                policy.prerequisite_signals - signal_names
            ),
            "unallowed_soft_conflicts": sorted(
                candidate.soft_conflicts - policy.allowed_soft_conflicts
            ),
            "vetoes": sorted(candidate.vetoes),
            "score": score,
            "minimum_score": policy.minimum_score,
            "supporting_families": sorted(families),
            "minimum_families": policy.minimum_families,
        }
    return {
        "policy": policy.name,
        "matched_required": sorted(policy.required_signals & signal_names),
        "missing_required": sorted(policy.required_signals - signal_names),
        "missing_prerequisites": sorted(
            policy.prerequisite_signals - signal_names
        ),
        "unallowed_soft_conflicts": sorted(
            candidate.soft_conflicts - policy.allowed_soft_conflicts
        ),
        "vetoes": sorted(candidate.vetoes),
    }


def nearest_policy(
    candidate: Candidate,
    policies: tuple[AdmissionPolicy | CumulativeAdmissionPolicy, ...],
) -> dict | None:
    """Return the policy with the fewest explicit blockers."""
    diagnostics = [policy_diagnostics(candidate, policy) for policy in policies]
    if not diagnostics:
        return None
    return min(
        diagnostics,
        key=lambda row: (
            bool(row["vetoes"]),
            len(row["unallowed_soft_conflicts"]),
            len(row["missing_prerequisites"]),
            len(row["missing_required"]),
            -len(row["matched_required"]),
            row["policy"],
        ),
    )


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
    policies: tuple[AdmissionPolicy | CumulativeAdmissionPolicy, ...],
) -> AdmissionPolicy | CumulativeAdmissionPolicy | None:
    return next((policy for policy in policies if policy.matches(candidate)), None)


def audit_metadata(
    candidate: Candidate,
    policy: AdmissionPolicy | CumulativeAdmissionPolicy,
) -> dict:
    metadata = {
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
    if isinstance(policy, CumulativeAdmissionPolicy):
        diagnostics = policy_diagnostics(candidate, policy)
        metadata["evidence_score"] = diagnostics["score"]
        metadata["evidence_support_families"] = diagnostics[
            "supporting_families"
        ]
    return metadata


def candidate_audit_metadata(
    candidate: Candidate,
    policies: tuple[AdmissionPolicy | CumulativeAdmissionPolicy, ...],
    policy: AdmissionPolicy | CumulativeAdmissionPolicy | None,
) -> dict:
    """Serialize a candidate whether admitted or rejected."""
    return {
        "start": candidate.start,
        "end": candidate.end,
        "surface": candidate.surface,
        "admitted": policy is not None,
        "evidence_policy": policy.name if policy is not None else None,
        "evidence_families": sorted({
            signal.family for signal in candidate.signals
        }),
        "evidence_support_families": sorted({
            signal.family
            for signal in candidate.signals
            if (
                signal.name not in candidate.soft_conflicts
                and signal.family != "span_shape"
            )
        }),
        "evidence_signals": sorted(
            signal.name for signal in candidate.signals
        ),
        "evidence_missing": sorted(candidate.missing_signals),
        "evidence_soft_conflicts": sorted(candidate.soft_conflicts),
        "evidence_vetoes": sorted(candidate.vetoes),
        "policy_diagnostics": (
            policy_diagnostics(candidate, policy)
            if policy is not None else None
        ),
        "nearest_policy": nearest_policy(candidate, policies),
    }
