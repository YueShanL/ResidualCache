"""Standalone probabilistic hierarchical retrieval memory.

This module intentionally has no model or ANN dependency.  It implements the
write, retention, split/merge, and exact-reranking policies described in
``probabilistic_hierarchical_retrieval_memory.md`` with exhaustive searches.
The exhaustive implementation is useful as a correctness oracle before an ANN
index or a Transformer integration is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Hashable, Iterable, Literal, Sequence


Vector = tuple[float, ...]
TemporalPolicy = Literal["current", "historical", "all"]


def _vector(values: Sequence[float], *, name: str, allow_zero: bool = True) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    if not allow_zero and _norm(result) <= 0.0:
        raise ValueError(f"{name} must have non-zero norm.")
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}.")
    return math.fsum(a * b for a, b in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(max(0.0, _dot(vector, vector)))


def _normalize(vector: Sequence[float], *, name: str = "vector") -> Vector:
    converted = _vector(vector, name=name, allow_zero=False)
    length = _norm(converted)
    return tuple(value / length for value in converted)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity and reject zero vectors explicitly."""

    left_normalized = _normalize(left, name="left")
    right_normalized = _normalize(right, name="right")
    return max(-1.0, min(1.0, _dot(left_normalized, right_normalized)))


def _weighted_sum(vectors: Iterable[tuple[float, Sequence[float]]], dimension: int) -> Vector:
    total = [0.0] * dimension
    for weight, vector in vectors:
        if len(vector) != dimension:
            raise ValueError("Cannot aggregate vectors with different dimensions.")
        for index, value in enumerate(vector):
            total[index] += weight * value
    return tuple(total)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _uniform_sphere_log_density(dimension: int) -> float:
    # Surface area S_(d-1) = 2*pi^(d/2)/Gamma(d/2).
    return math.lgamma(dimension / 2.0) - math.log(2.0) - dimension / 2.0 * math.log(math.pi)


def _log_bessel_iv(order: float, argument: float) -> float:
    """Compute log(I_order(argument)) without SciPy.

    The positive-term series is stable in log space for the concentration
    range normally produced by the moment estimator.  A standard large-x
    asymptotic expansion avoids thousands of terms for very concentrated
    clusters.
    """

    if order < -0.5 or argument < 0.0:
        raise ValueError("Bessel order and argument are outside the supported range.")
    if argument == 0.0:
        return 0.0 if order == 0.0 else -math.inf

    asymptotic_boundary = max(50.0, 2.0 * (order * order + 1.0))
    if argument > asymptotic_boundary:
        mu = 4.0 * order * order
        correction = 1.0
        product = 1.0
        for term_index in range(1, 6):
            product *= mu - (2 * term_index - 1) ** 2
            correction += ((-1.0) ** term_index) * product / (
                math.factorial(term_index) * (8.0 * argument) ** term_index
            )
        if correction > 0.0:
            return (
                argument
                - 0.5 * math.log(2.0 * math.pi * argument)
                + math.log(correction)
            )

    log_half = math.log(argument / 2.0)
    log_term = order * log_half - math.lgamma(order + 1.0)
    terms = [log_term]
    maximum = log_term
    passed_peak = False
    for term_index in range(1, 20001):
        log_term += (
            2.0 * log_half
            - math.log(term_index)
            - math.log(term_index + order)
        )
        terms.append(log_term)
        if log_term <= maximum:
            passed_peak = True
        else:
            maximum = log_term
            passed_peak = False
        if passed_peak and log_term < maximum - 45.0:
            break
    else:
        raise ArithmeticError("vMF normalizer series did not converge.")
    return _logsumexp(terms)


def vmf_log_normalizer(dimension: int, concentration: float) -> float:
    """Return log C_d(kappa) for the von Mises-Fisher distribution."""

    if dimension < 2:
        raise ValueError("vMF vectors must have dimension >= 2.")
    if concentration < 0.0 or not math.isfinite(concentration):
        raise ValueError("concentration must be finite and non-negative.")
    if concentration == 0.0:
        return _uniform_sphere_log_density(dimension)
    order = dimension / 2.0 - 1.0
    return (
        order * math.log(concentration)
        - dimension / 2.0 * math.log(2.0 * math.pi)
        - _log_bessel_iv(order, concentration)
    )


def vmf_log_density(
    vector: Sequence[float],
    mean_direction: Sequence[float],
    concentration: float,
) -> float:
    normalized = _normalize(vector)
    mean = _normalize(mean_direction, name="mean_direction")
    return vmf_log_normalizer(len(normalized), concentration) + concentration * _dot(
        normalized, mean
    )


def estimate_vmf_concentration(
    resultant_length: float,
    dimension: int,
    *,
    minimum: float = 0.0,
    maximum: float = 1_000.0,
) -> float:
    """Moment estimate of kappa from the mean resultant length.

    This is the standard Banerjee et al. approximation.  Callers should shrink
    the observed resultant length with prior mass before calling it; doing so
    prevents a one-record cluster from receiving infinite confidence.
    """

    if dimension < 2:
        raise ValueError("dimension must be >= 2.")
    if not 0.0 <= resultant_length <= 1.0 + 1e-12:
        raise ValueError("resultant_length must be in [0, 1].")
    clipped = min(1.0 - 1e-9, max(0.0, resultant_length))
    if clipped == 0.0:
        estimate = 0.0
    else:
        estimate = clipped * (dimension - clipped * clipped) / (
            1.0 - clipped * clipped
        )
    return min(maximum, max(minimum, estimate))


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration shared by the probabilistic memory variants."""

    alpha: float = 0.1
    tau_new: float = 0.5
    count_exponent: float = 0.5
    count_epsilon: float = 1e-6
    concentration_prior_mass: float = 1.0
    minimum_concentration: float = 0.0
    maximum_concentration: float = 1_000.0

    # A learned router key is an opaque read index.  These parameters affect
    # only ``retrieve_router_clusters``; they are deliberately separate from
    # the native K/V posterior and never participate in write assignment,
    # retention, split, or merge decisions.
    router_count_exponent: float = 0.5
    router_count_epsilon: float = 1e-6
    router_concentration_prior_mass: float = 1.0
    router_minimum_concentration: float = 0.0
    router_maximum_concentration: float = 1_000.0

    use_temporal_weights: bool = True
    age_decay: float = 0.0
    inactivity_decay: float = 0.0
    gain_exponent: float = 0.0
    weight_epsilon: float = 1e-6
    minimum_effective_weight: float = 1e-9
    new_record_protection: float = 0.0
    memory_budget_bytes: int | None = None
    memory_cost_lambda: float = 0.0
    budget_step_size: float = 1e-6

    route_top_k: int = 4
    child_top_k: int = 2
    record_top_k: int = 4
    route_recency_bias: float = 0.0
    route_utility_bias: float = 0.0
    native_time_bias: float = 0.0
    authority_bias: float = 0.0
    minimum_native_score: float = -math.inf

    enable_split_merge: bool = True
    minimum_child_mass: float = 2.0
    split_minimum_resultant: float = 0.75
    split_minimum_gain: float = 0.2
    split_maximum_regret: float = 0.2
    split_maximum_conflict: float = 0.5
    split_patience: int = 2
    split_cooldown: float = 10.0
    merge_minimum_similarity: float = 0.97
    merge_maximum_conflict: float = 0.15
    merge_maximum_regret: float = 0.15
    merge_cooldown: float = 10.0
    slot_penalty: float = 0.05
    split_value_weight: float = 0.0
    split_time_weight: float = 0.0
    statistic_ema_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= self.tau_new <= 1.0:
            raise ValueError("tau_new must be in [0, 1].")
        if not 0.0 <= self.count_exponent < 1.0:
            raise ValueError("count_exponent must be in [0, 1).")
        if self.count_epsilon <= 0.0 or self.weight_epsilon <= 0.0:
            raise ValueError("epsilon values must be positive.")
        if self.concentration_prior_mass < 0.0:
            raise ValueError("concentration_prior_mass cannot be negative.")
        if not 0.0 <= self.minimum_concentration <= self.maximum_concentration:
            raise ValueError("Invalid concentration bounds.")
        if not 0.0 <= self.router_count_exponent < 1.0:
            raise ValueError("router_count_exponent must be in [0, 1).")
        if self.router_count_epsilon <= 0.0:
            raise ValueError("router_count_epsilon must be positive.")
        if self.router_concentration_prior_mass < 0.0:
            raise ValueError("router_concentration_prior_mass cannot be negative.")
        if not (
            0.0
            <= self.router_minimum_concentration
            <= self.router_maximum_concentration
        ):
            raise ValueError("Invalid router concentration bounds.")
        if self.age_decay < 0.0 or self.inactivity_decay < 0.0:
            raise ValueError("Decay rates cannot be negative.")
        if self.gain_exponent < 0.0:
            raise ValueError("gain_exponent cannot be negative.")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive.")
        if self.route_top_k <= 0 or self.child_top_k <= 0 or self.record_top_k <= 0:
            raise ValueError("Retrieval widths must be positive.")
        if self.minimum_child_mass <= 0.0:
            raise ValueError("minimum_child_mass must be positive.")
        if self.split_patience <= 0:
            raise ValueError("split_patience must be positive.")
        if not 0.0 <= self.statistic_ema_rate <= 1.0:
            raise ValueError("statistic_ema_rate must be in [0, 1].")


@dataclass
class MemoryRecord:
    id: str
    layer: int
    head_or_kv_group: Hashable
    source_token_or_span: Hashable | None
    index_vector: Vector
    original_key: Vector
    original_value: Vector
    write_time: float
    sequence_order: int
    last_retrieval_time: float | None = None
    retrieval_count: int = 0
    attention_contribution_ema: float = 0.0
    counterfactual_gain_ema: float = 1.0
    active_weight: float = 1.0
    superseded_by: str | None = None
    superseded_probability: float = 0.0
    conflict_group: Hashable | None = None
    source_authority: float = 0.0
    multiplicity: int = 1
    router_key: Vector | None = None
    router_block_id: Hashable | None = None
    router_block_size: int | None = None
    router_weight: float = 0.0

    def approximate_bytes(self) -> int:
        vector_bytes = 8 * (
            len(self.index_vector) + len(self.original_key) + len(self.original_value)
        )
        return 160 + vector_bytes


@dataclass
class LeafSlot:
    id: str
    layer: int
    head_or_kv_group: Hashable
    record_ids: list[str] = field(default_factory=list)
    centroid: Vector = field(default_factory=tuple)
    effective_count: float = 0.0
    resultant_length: float = 0.0
    concentration: float = 0.0
    weighted_scatter: float = 0.0
    ordered_time_range: tuple[float, float] = (0.0, 0.0)
    usage_ema: float = 0.0
    routing_regret_ema: float = 0.0
    value_conflict_score: float = 0.0
    pending_split_observations: int = 0
    parent_id: str | None = None
    cooldown_until: float = -math.inf
    # Derived exclusively from router metadata on the records currently in
    # ``record_ids``.  This is cached inside the leaf so eviction and topology
    # changes cannot leave a detached adapter-side distribution behind.
    router_dimension: int = 0
    router_record_count: int = 0
    router_block_count: int = 0
    router_mass: float = 0.0
    router_resultant: Vector = field(default_factory=tuple)
    router_mean_direction: Vector = field(default_factory=tuple)
    router_resultant_length: float = 0.0
    router_concentration: float = 0.0
    router_block_weights: tuple[tuple[Hashable, float], ...] = field(
        default_factory=tuple
    )


@dataclass
class ParentSlot:
    id: str
    layer: int
    head_or_kv_group: Hashable
    child_ids: list[str]
    centroid: Vector
    effective_count: float
    ordered_time_range: tuple[float, float]
    parent_id: str | None = None
    routing_usage: float = 0.0
    cooldown_until: float = -math.inf


@dataclass(frozen=True)
class WriteDecision:
    record_id: str
    slot_id: str
    created_new_slot: bool
    probability_new: float
    existing_probabilities: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RetrievalResult:
    record_id: str
    slot_id: str
    score: float
    original_key: Vector
    original_value: Vector
    source_token_or_span: Hashable | None
    write_time: float
    source_authority: float
    multiplicity: int = 1


@dataclass(frozen=True)
class RouterClusterResult:
    """One leaf posterior under the opaque learned-router read index."""

    slot_id: str
    layer: int
    head_or_kv_group: Hashable
    probability: float
    log_score: float
    record_ids: tuple[str, ...]
    router_record_count: int
    router_block_count: int
    router_mass: float
    mean_direction: Vector
    concentration: float


@dataclass(frozen=True)
class CompressedSources:
    """Source identities represented by one averaged baseline slot."""

    sources: frozenset[Hashable | None]


@dataclass(frozen=True)
class MaintenanceReport:
    evicted_record_ids: tuple[str, ...]
    split_slot_ids: tuple[str, ...]
    merged_slot_pairs: tuple[tuple[str, str], ...]
    memory_bytes: int
    memory_cost_lambda: float


class ProbabilisticHierarchicalMemory:
    """Exact standalone implementation of temporal hierarchical vMF memory."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self.records: dict[str, MemoryRecord] = {}
        self.leaves: dict[str, LeafSlot] = {}
        self.parents: dict[str, ParentSlot] = {}
        self._record_counter = 0
        self._slot_counter = 0
        self._sequence_counter = 0
        self._clock = 0.0
        self._memory_cost_lambda = self.config.memory_cost_lambda

    @property
    def slot_count(self) -> int:
        return len(self.leaves) + len(self.parents)

    @property
    def leaf_count(self) -> int:
        return len(self.leaves)

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def memory_bytes(self) -> int:
        record_bytes = sum(record.approximate_bytes() for record in self.records.values())
        slot_bytes = 0
        for slot in self.leaves.values():
            slot_bytes += 192 + 8 * len(slot.centroid) + 8 * len(slot.record_ids)
        for parent in self.parents.values():
            slot_bytes += 160 + 8 * len(parent.centroid) + 8 * len(parent.child_ids)
        return record_bytes + slot_bytes

    def _next_time(self, time: float | None) -> float:
        if time is None:
            self._clock += 1.0
            return self._clock
        value = float(time)
        if not math.isfinite(value):
            raise ValueError("time must be finite.")
        if value < self._clock:
            raise ValueError("Online updates must have non-decreasing time.")
        self._clock = value
        return value

    def _new_record_id(self) -> str:
        self._record_counter += 1
        return f"record-{self._record_counter:08d}"

    def _new_slot_id(self) -> str:
        self._slot_counter += 1
        return f"slot-{self._slot_counter:08d}"

    def _record_weight(
        self,
        record: MemoryRecord,
        time: float,
        *,
        supersession_as_of: float | None = None,
        ignore_supersession: bool = False,
    ) -> float:
        if not self.config.use_temporal_weights:
            return record.active_weight
        age = max(0.0, time - record.write_time)
        last_use = record.last_retrieval_time
        inactivity = max(0.0, time - (last_use if last_use is not None else record.write_time))
        gain = max(0.0, record.counterfactual_gain_ema)
        superseded_probability = record.superseded_probability
        if ignore_supersession:
            superseded_probability = 0.0
        elif supersession_as_of is not None and record.superseded_by is not None:
            superseder = self.records.get(record.superseded_by)
            if superseder is None or superseder.write_time > supersession_as_of:
                superseded_probability = 0.0
        return (
            record.active_weight
            * math.exp(-self.config.age_decay * age)
            * math.exp(-self.config.inactivity_decay * inactivity)
            * (self.config.weight_epsilon + gain) ** self.config.gain_exponent
            * max(0.0, 1.0 - superseded_probability)
        )

    def _refresh_leaf(
        self, slot: LeafSlot, time: float, *, as_of: float | None = None
    ) -> None:
        all_members = [
            self.records[record_id]
            for record_id in slot.record_ids
            if record_id in self.records
        ]
        slot.record_ids = [record.id for record in all_members]
        self._refresh_leaf_router_index(slot, all_members)
        members = [
            record
            for record in all_members
            if as_of is None or record.write_time <= as_of
        ]
        if not members:
            slot.effective_count = 0.0
            slot.resultant_length = 0.0
            slot.concentration = 0.0
            slot.weighted_scatter = 0.0
            slot.value_conflict_score = 0.0
            return

        weights = [
            self._record_weight(record, time, supersession_as_of=as_of)
            for record in members
        ]
        dimension = len(members[0].index_vector)
        resultant = _weighted_sum(
            zip(weights, (record.index_vector for record in members)), dimension
        )
        mass = math.fsum(weights)
        resultant_norm = _norm(resultant)
        if resultant_norm > 1e-15:
            centroid = tuple(value / resultant_norm for value in resultant)
        elif slot.centroid:
            centroid = slot.centroid
        else:
            centroid = members[0].index_vector
        slot.centroid = centroid
        slot.effective_count = mass
        slot.resultant_length = resultant_norm / mass if mass > 0.0 else 0.0
        shrunken_resultant = (
            resultant_norm / (mass + self.config.concentration_prior_mass)
            if mass + self.config.concentration_prior_mass > 0.0
            else 0.0
        )
        slot.concentration = estimate_vmf_concentration(
            min(1.0, shrunken_resultant),
            dimension,
            minimum=self.config.minimum_concentration,
            maximum=self.config.maximum_concentration,
        )
        slot.weighted_scatter = math.fsum(
            weight * (1.0 - _dot(record.index_vector, centroid))
            for weight, record in zip(weights, members)
        )
        slot.ordered_time_range = (
            min(record.write_time for record in members),
            max(record.write_time for record in members),
        )
        slot.value_conflict_score = self._value_conflict(members, weights)

    def _refresh_leaf_router_index(
        self, slot: LeafSlot, members: Sequence[MemoryRecord]
    ) -> None:
        """Rebuild the learned-router vMF from physically retained members.

        Router weights are intentionally independent of temporal/native record
        weights.  A record contributes exactly while it is a member of this
        leaf, and contributes ``1 / router_block_size`` so repeating one block
        key for every token does not multiply that block's total evidence.
        """

        routed = [record for record in members if record.router_key is not None]
        if not routed:
            slot.router_dimension = 0
            slot.router_record_count = 0
            slot.router_block_count = 0
            slot.router_mass = 0.0
            slot.router_resultant = ()
            slot.router_mean_direction = ()
            slot.router_resultant_length = 0.0
            slot.router_concentration = 0.0
            slot.router_block_weights = ()
            return

        dimension = len(routed[0].router_key or ())
        if dimension < 2 or any(
            len(record.router_key or ()) != dimension for record in routed
        ):
            raise ValueError("router key dimensions must be constant within a leaf.")
        block_weights: dict[Hashable, float] = {}
        weighted_keys: list[tuple[float, Vector]] = []
        for record in routed:
            assert record.router_key is not None
            assert record.router_block_id is not None
            weight = record.router_weight
            if weight <= 0.0 or not math.isfinite(weight):
                raise ValueError("router record weights must be finite and positive.")
            weighted_keys.append((weight, record.router_key))
            block_weights[record.router_block_id] = (
                block_weights.get(record.router_block_id, 0.0) + weight
            )

        resultant = _weighted_sum(weighted_keys, dimension)
        mass = math.fsum(weight for weight, _key in weighted_keys)
        resultant_norm = _norm(resultant)
        mean_direction = (
            tuple(value / resultant_norm for value in resultant)
            if resultant_norm > 1e-15
            else tuple(0.0 for _ in range(dimension))
        )
        shrunken_resultant = resultant_norm / (
            mass + self.config.router_concentration_prior_mass
        )
        concentration = 0.0
        if resultant_norm > 1e-15:
            concentration = estimate_vmf_concentration(
                min(1.0, shrunken_resultant),
                dimension,
                minimum=self.config.router_minimum_concentration,
                maximum=self.config.router_maximum_concentration,
            )
        slot.router_dimension = dimension
        slot.router_record_count = len(routed)
        slot.router_block_count = len(block_weights)
        slot.router_mass = mass
        slot.router_resultant = resultant
        slot.router_mean_direction = mean_direction
        slot.router_resultant_length = min(1.0, resultant_norm / mass)
        slot.router_concentration = concentration
        slot.router_block_weights = tuple(
            sorted(block_weights.items(), key=lambda item: repr(item[0]))
        )

    @staticmethod
    def _value_conflict(records: Sequence[MemoryRecord], weights: Sequence[float]) -> float:
        non_zero = [
            (weight, _normalize(record.original_value, name="original_value"))
            for weight, record in zip(weights, records)
            if weight > 0.0 and _norm(record.original_value) > 0.0
        ]
        mass = math.fsum(weight for weight, _value in non_zero)
        if mass <= 0.0:
            return 0.0
        resultant = _weighted_sum(non_zero, len(non_zero[0][1]))
        return max(0.0, min(1.0, 1.0 - _norm(resultant) / mass))

    def _refresh_parent(self, parent: ParentSlot) -> None:
        children = [self._node(child_id) for child_id in parent.child_ids if self._node(child_id)]
        parent.child_ids = [child.id for child in children]
        if not children:
            parent.effective_count = 0.0
            return
        dimension = len(children[0].centroid)
        resultant = _weighted_sum(
            ((child.effective_count, child.centroid) for child in children), dimension
        )
        resultant_norm = _norm(resultant)
        if resultant_norm > 1e-15:
            parent.centroid = tuple(value / resultant_norm for value in resultant)
        parent.effective_count = math.fsum(child.effective_count for child in children)
        parent.ordered_time_range = (
            min(child.ordered_time_range[0] for child in children),
            max(child.ordered_time_range[1] for child in children),
        )

    def _refresh_all(self, time: float, *, as_of: float | None = None) -> None:
        for leaf in self.leaves.values():
            self._refresh_leaf(leaf, time, as_of=as_of)
        # Parents are shallow in typical use, but repeated passes also support
        # parents whose children are themselves parents.
        for _ in range(max(1, len(self.parents))):
            for parent in self.parents.values():
                self._refresh_parent(parent)

    def _node(self, node_id: str) -> LeafSlot | ParentSlot | None:
        return self.leaves.get(node_id) or self.parents.get(node_id)

    @staticmethod
    def _compatible(
        node: LeafSlot | ParentSlot, layer: int, head_or_kv_group: Hashable
    ) -> bool:
        return node.layer == layer and node.head_or_kv_group == head_or_kv_group

    def _existing_log_score(self, slot: LeafSlot, vector: Vector) -> float:
        return (
            self.config.count_exponent
            * math.log(slot.effective_count + self.config.count_epsilon)
            + vmf_log_density(vector, slot.centroid, slot.concentration)
        )

    def _assignment_probabilities(
        self, candidates: Sequence[LeafSlot], vector: Vector
    ) -> tuple[float, list[tuple[LeafSlot, float]]]:
        new_log_score = math.log(self.config.alpha) + _uniform_sphere_log_density(
            len(vector)
        )
        existing_log_scores = [self._existing_log_score(slot, vector) for slot in candidates]
        normalizer = _logsumexp([new_log_score, *existing_log_scores])
        probability_new = math.exp(new_log_score - normalizer)
        existing = [
            (slot, math.exp(score - normalizer))
            for slot, score in zip(candidates, existing_log_scores)
        ]
        return probability_new, existing

    def write(
        self,
        index_vector: Sequence[float],
        original_key: Sequence[float],
        original_value: Sequence[float],
        *,
        layer: int = 0,
        head_or_kv_group: Hashable = 0,
        source_token_or_span: Hashable | None = None,
        time: float | None = None,
        conflict_group: Hashable | None = None,
        supersedes: str | None = None,
        source_authority: float = 0.0,
        counterfactual_gain: float = 1.0,
        active_weight: float = 1.0,
        multiplicity: int = 1,
        router_key: Sequence[float] | None = None,
        router_block_id: Hashable | None = None,
        router_block_size: int | None = None,
    ) -> WriteDecision:
        """Write one historical record and return the native posterior decision.

        ``router_*`` fields are opaque read-index metadata stored on the record.
        They are validated before the write but never used by the assignment
        posterior or any native memory maintenance policy.
        """

        now = self._next_time(time)
        index = _normalize(index_vector, name="index_vector")
        key = _vector(original_key, name="original_key", allow_zero=False)
        value = _vector(original_value, name="original_value")
        if len(key) != len(index):
            raise ValueError("The standalone oracle requires index and original key dimensions to match.")
        if active_weight < 0.0 or not math.isfinite(active_weight):
            raise ValueError("active_weight must be finite and non-negative.")
        if not math.isfinite(source_authority) or not math.isfinite(counterfactual_gain):
            raise ValueError("source_authority and counterfactual_gain must be finite.")
        if multiplicity <= 0:
            raise ValueError("multiplicity must be positive.")
        if supersedes is not None and supersedes not in self.records:
            raise KeyError(f"Unknown superseded record {supersedes!r}.")
        router_direction: Vector | None = None
        router_weight = 0.0
        if router_key is None:
            if router_block_id is not None or router_block_size is not None:
                raise ValueError(
                    "router_block_id and router_block_size require router_key."
                )
        else:
            router_direction = _normalize(router_key, name="router_key")
            if len(router_direction) < 2:
                raise ValueError("router_key dimension must be at least 2.")
            if router_block_id is None:
                raise ValueError("router_block_id is required with router_key.")
            hash(router_block_id)
            if (
                not isinstance(router_block_size, int)
                or isinstance(router_block_size, bool)
                or router_block_size <= 0
            ):
                raise ValueError("router_block_size must be a positive integer.")
            for existing_record in self.records.values():
                if (
                    existing_record.layer == layer
                    and existing_record.head_or_kv_group == head_or_kv_group
                    and existing_record.router_key is not None
                    and len(existing_record.router_key) != len(router_direction)
                ):
                    raise ValueError(
                        "router_key dimension must be constant within a layer/group."
                    )
            router_weight = 1.0 / router_block_size

        self._refresh_all(now)
        candidates = [
            slot
            for slot in self.leaves.values()
            if self._compatible(slot, layer, head_or_kv_group)
            and len(slot.centroid) == len(index)
        ]
        for slot in candidates:
            exemplar = self.records[slot.record_ids[0]]
            if len(exemplar.original_value) != len(value):
                raise ValueError(
                    "original_value dimension must be constant within a layer/group."
                )
        if candidates:
            probability_new, existing = self._assignment_probabilities(candidates, index)
            create_new = probability_new > self.config.tau_new
            selected = max(existing, key=lambda item: item[1])[0] if not create_new else None
        else:
            probability_new, existing, create_new, selected = 1.0, [], True, None

        record_id = self._new_record_id()
        self._sequence_counter += 1
        if supersedes is not None:
            previous = self.records[supersedes]
            previous.superseded_by = record_id
            previous.superseded_probability = 1.0
            if conflict_group is None:
                conflict_group = previous.conflict_group
        record = MemoryRecord(
            id=record_id,
            layer=layer,
            head_or_kv_group=head_or_kv_group,
            source_token_or_span=source_token_or_span,
            index_vector=index,
            original_key=key,
            original_value=value,
            write_time=now,
            sequence_order=self._sequence_counter,
            counterfactual_gain_ema=float(counterfactual_gain),
            active_weight=float(active_weight),
            conflict_group=conflict_group,
            source_authority=float(source_authority),
            multiplicity=multiplicity,
            router_key=router_direction,
            router_block_id=router_block_id,
            router_block_size=router_block_size,
            router_weight=router_weight,
        )
        self.records[record_id] = record

        if create_new:
            slot_id = self._new_slot_id()
            selected = LeafSlot(
                id=slot_id,
                layer=layer,
                head_or_kv_group=head_or_kv_group,
                record_ids=[record_id],
                centroid=index,
            )
            self.leaves[slot_id] = selected
        else:
            assert selected is not None
            selected.record_ids.append(record_id)
            slot_id = selected.id
        self._refresh_leaf(selected, now)
        self._refresh_all(now)
        return WriteDecision(
            record_id=record_id,
            slot_id=slot_id,
            created_new_slot=create_new,
            probability_new=probability_new,
            existing_probabilities=tuple(
                sorted(
                    ((slot.id, probability) for slot, probability in existing),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
        )

    def retrieve_router_clusters(
        self,
        query_router_key: Sequence[float],
        *,
        layer: int = 0,
        head_or_kv_group: Hashable = 0,
        top_n: int = 4,
    ) -> list[RouterClusterResult]:
        """Rank current leaves by the learned block-router posterior.

        This operation is a pure read over leaf-local cached statistics.  It
        neither advances the memory clock nor updates native usage, routing,
        retention, split, or merge state.  Returned ``record_ids`` include all
        currently retained records in a selected leaf so callers can replay
        the complete cluster, including records without router metadata.
        """

        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        query = _normalize(query_router_key, name="query_router_key")
        if len(query) < 2:
            raise ValueError("query_router_key dimension must be at least 2.")
        candidates = [
            slot
            for slot in self.leaves.values()
            if self._compatible(slot, layer, head_or_kv_group)
            and slot.router_record_count > 0
        ]
        if not candidates:
            return []
        dimensions = {slot.router_dimension for slot in candidates}
        if dimensions != {len(query)}:
            raise ValueError(
                "query_router_key dimension differs from retained router keys."
            )

        scored: list[tuple[float, LeafSlot]] = []
        for slot in candidates:
            density = vmf_log_normalizer(
                slot.router_dimension, slot.router_concentration
            )
            if slot.router_concentration > 0.0:
                density += slot.router_concentration * _dot(
                    query, slot.router_mean_direction
                )
            log_score = (
                self.config.router_count_exponent
                * math.log(slot.router_mass + self.config.router_count_epsilon)
                + density
            )
            scored.append((log_score, slot))

        normalizer = _logsumexp([score for score, _slot in scored])
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            RouterClusterResult(
                slot_id=slot.id,
                layer=slot.layer,
                head_or_kv_group=slot.head_or_kv_group,
                probability=math.exp(score - normalizer),
                log_score=score,
                record_ids=tuple(
                    record_id
                    for record_id in slot.record_ids
                    if record_id in self.records
                ),
                router_record_count=slot.router_record_count,
                router_block_count=slot.router_block_count,
                router_mass=slot.router_mass,
                mean_direction=slot.router_mean_direction,
                concentration=slot.router_concentration,
            )
            for score, slot in scored[:top_n]
        ]

    def _root_nodes(self, layer: int, group: Hashable) -> list[LeafSlot | ParentSlot]:
        nodes: list[LeafSlot | ParentSlot] = [*self.leaves.values(), *self.parents.values()]
        return [
            node
            for node in nodes
            if node.parent_id is None and self._compatible(node, layer, group)
        ]

    def _route_score(
        self, query: Vector, node: LeafSlot | ParentSlot, time: float
    ) -> float:
        score = _dot(query, node.centroid)
        if self.config.route_recency_bias:
            age = max(0.0, time - node.ordered_time_range[1])
            score += self.config.route_recency_bias / (1.0 + age)
        utility = node.usage_ema if isinstance(node, LeafSlot) else node.routing_usage
        score += self.config.route_utility_bias * utility
        return score

    def _descend(
        self, node: LeafSlot | ParentSlot, query: Vector, time: float
    ) -> list[LeafSlot]:
        if isinstance(node, LeafSlot):
            return [node]
        children = [self._node(child_id) for child_id in node.child_ids]
        ranked = sorted(
            (child for child in children if child is not None),
            key=lambda child: self._route_score(query, child, time),
            reverse=True,
        )[: self.config.child_top_k]
        leaves: list[LeafSlot] = []
        for child in ranked:
            leaves.extend(self._descend(child, query, time))
        return leaves

    def _visible(
        self, record: MemoryRecord, policy: TemporalPolicy, as_of: float | None
    ) -> bool:
        if policy == "all":
            return True
        if policy == "current":
            return record.superseded_by is None
        if policy != "historical":
            raise ValueError(f"Unknown temporal policy {policy!r}.")
        if as_of is None:
            raise ValueError("historical retrieval requires as_of.")
        if record.write_time > as_of:
            return False
        if record.superseded_by is None:
            return True
        superseder = self.records.get(record.superseded_by)
        return superseder is None or superseder.write_time > as_of

    def _native_score(
        self,
        query_key: Vector,
        record: MemoryRecord,
        *,
        time: float,
        policy: TemporalPolicy,
        as_of: float | None,
    ) -> float:
        score = _dot(query_key, record.original_key) / math.sqrt(len(query_key))
        if self.config.native_time_bias:
            target_time = as_of if policy == "historical" and as_of is not None else time
            score += self.config.native_time_bias / (
                1.0 + abs(target_time - record.write_time)
            )
        score += self.config.authority_bias * record.source_authority
        if record.multiplicity > 1:
            score += math.log(record.multiplicity)
        return score

    def retrieve(
        self,
        query_index: Sequence[float],
        *,
        query_key: Sequence[float] | None = None,
        layer: int = 0,
        head_or_kv_group: Hashable = 0,
        time: float | None = None,
        temporal_policy: TemporalPolicy = "current",
        as_of: float | None = None,
        top_k: int | None = None,
        update_usage: bool = True,
    ) -> list[RetrievalResult]:
        """Hierarchically route, then rerank original historical keys exactly."""

        now = self._next_time(time)
        index = _normalize(query_index, name="query_index")
        key = _vector(query_key if query_key is not None else query_index, name="query_key")
        if len(key) != len(index):
            raise ValueError("query_index and query_key dimensions must match.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive.")
        if temporal_policy == "historical":
            if as_of is None:
                raise ValueError("historical retrieval requires as_of.")
            routing_time = as_of
            self._refresh_all(as_of, as_of=as_of)
        else:
            routing_time = now
            self._refresh_all(now)
        roots = self._root_nodes(layer, head_or_kv_group)
        routed = sorted(
            roots,
            key=lambda node: self._route_score(index, node, routing_time),
            reverse=True,
        )[: self.config.route_top_k]
        selected_leaves: dict[str, LeafSlot] = {}
        for node in routed:
            for leaf in self._descend(node, index, routing_time):
                selected_leaves[leaf.id] = leaf

        scored: list[tuple[float, MemoryRecord, str]] = []
        for leaf in selected_leaves.values():
            active_records = []
            for record_id in leaf.record_ids:
                if record_id not in self.records:
                    continue
                record = self.records[record_id]
                if not self._visible(record, temporal_policy, as_of):
                    continue
                weight_time = as_of if temporal_policy == "historical" else now
                weight = self._record_weight(
                    record,
                    weight_time,
                    supersession_as_of=as_of if temporal_policy == "historical" else None,
                    ignore_supersession=temporal_policy == "all",
                )
                if weight >= self.config.minimum_effective_weight:
                    active_records.append(record)
            if active_records:
                best_index_similarity = max(
                    _dot(index, record.index_vector) for record in active_records
                )
                regret = max(0.0, best_index_similarity - _dot(index, leaf.centroid))
                rate = self.config.statistic_ema_rate
                leaf.routing_regret_ema = (
                    (1.0 - rate) * leaf.routing_regret_ema + rate * regret
                )
            for record in active_records:
                score = self._native_score(
                    key, record, time=now, policy=temporal_policy, as_of=as_of
                )
                if score >= self.config.minimum_native_score:
                    scored.append((score, record, leaf.id))
        scored.sort(key=lambda item: (item[0], item[1].sequence_order), reverse=True)
        width = top_k if top_k is not None else self.config.record_top_k
        selected = scored[:width]
        if update_usage:
            rate = self.config.statistic_ema_rate
            selected_ids = {record.id for _score, record, _slot_id in selected}
            for leaf in selected_leaves.values():
                used = any(record_id in selected_ids for record_id in leaf.record_ids)
                leaf.usage_ema = (1.0 - rate) * leaf.usage_ema + rate * float(used)
            for _score, record, _slot_id in selected:
                record.retrieval_count += 1
                record.last_retrieval_time = now
        results = [
            RetrievalResult(
                record_id=record.id,
                slot_id=slot_id,
                score=score,
                original_key=record.original_key,
                original_value=record.original_value,
                source_token_or_span=record.source_token_or_span,
                write_time=record.write_time,
                source_authority=record.source_authority,
                multiplicity=record.multiplicity,
            )
            for score, record, slot_id in selected
        ]
        if temporal_policy == "historical":
            self._refresh_all(now)
        return results

    def update_utility(
        self,
        record_id: str,
        *,
        counterfactual_gain: float | None = None,
        attention_contribution: float | None = None,
        ema_rate: float = 0.1,
    ) -> None:
        if record_id not in self.records:
            raise KeyError(record_id)
        if not 0.0 <= ema_rate <= 1.0:
            raise ValueError("ema_rate must be in [0, 1].")
        record = self.records[record_id]
        if counterfactual_gain is not None:
            record.counterfactual_gain_ema = (
                (1.0 - ema_rate) * record.counterfactual_gain_ema
                + ema_rate * float(counterfactual_gain)
            )
        if attention_contribution is not None:
            record.attention_contribution_ema = (
                (1.0 - ema_rate) * record.attention_contribution_ema
                + ema_rate * float(attention_contribution)
            )

    def _eviction_priority(self, record: MemoryRecord, time: float) -> float:
        inactivity = max(
            0.0,
            time
            - (
                record.last_retrieval_time
                if record.last_retrieval_time is not None
                else record.write_time
            ),
        )
        age = max(0.0, time - record.write_time)
        reuse = (
            (record.retrieval_count + 1.0)
            / (age + record.retrieval_count + 1.0)
            * math.exp(-self.config.inactivity_decay * inactivity)
        )
        gain = max(0.0, record.counterfactual_gain_ema)
        return reuse * gain / max(1, record.approximate_bytes())

    def _remove_record(self, record_id: str) -> None:
        self.records.pop(record_id, None)
        for leaf in self.leaves.values():
            if record_id in leaf.record_ids:
                leaf.record_ids.remove(record_id)

    def _prune_empty_nodes(self) -> None:
        empty_leaves = [slot_id for slot_id, slot in self.leaves.items() if not slot.record_ids]
        for slot_id in empty_leaves:
            parent_id = self.leaves[slot_id].parent_id
            self.leaves.pop(slot_id)
            if parent_id in self.parents and slot_id in self.parents[parent_id].child_ids:
                self.parents[parent_id].child_ids.remove(slot_id)
        changed = True
        while changed:
            changed = False
            for parent_id, parent in list(self.parents.items()):
                parent.child_ids = [
                    child_id for child_id in parent.child_ids if self._node(child_id) is not None
                ]
                if not parent.child_ids:
                    grandparent = parent.parent_id
                    self.parents.pop(parent_id)
                    if grandparent in self.parents and parent_id in self.parents[grandparent].child_ids:
                        self.parents[grandparent].child_ids.remove(parent_id)
                    changed = True

    def _two_means(
        self, slot: LeafSlot, time: float
    ) -> tuple[list[str], list[str], float, float, float, float] | None:
        records = [self.records[record_id] for record_id in slot.record_ids]
        if len(records) < 2:
            return None
        weights = [self._record_weight(record, time) for record in records]
        farthest = min(
            (
                (_dot(left.index_vector, right.index_vector), left_index, right_index)
                for left_index, left in enumerate(records)
                for right_index, right in enumerate(records)
                if left_index < right_index
            ),
            default=None,
        )
        if farthest is None:
            return None
        center_a = records[farthest[1]].index_vector
        center_b = records[farthest[2]].index_vector
        assignments: list[int] = []
        for _iteration in range(50):
            new_assignments = []
            for record in records:
                distance_a = self._split_distance(record, center_a, records[farthest[1]], time)
                distance_b = self._split_distance(record, center_b, records[farthest[2]], time)
                new_assignments.append(0 if distance_a <= distance_b else 1)
            if not any(value == 0 for value in new_assignments) or not any(
                value == 1 for value in new_assignments
            ):
                return None
            if new_assignments == assignments:
                break
            assignments = new_assignments
            center_a = self._cluster_center(records, weights, assignments, 0)
            center_b = self._cluster_center(records, weights, assignments, 1)
        left_ids = [record.id for record, assignment in zip(records, assignments) if assignment == 0]
        right_ids = [record.id for record, assignment in zip(records, assignments) if assignment == 1]
        mass_a = math.fsum(weight for weight, assignment in zip(weights, assignments) if assignment == 0)
        mass_b = math.fsum(weight for weight, assignment in zip(weights, assignments) if assignment == 1)
        j1 = slot.weighted_scatter
        j2 = math.fsum(
            weight
            * (
                1.0
                - _dot(
                    record.index_vector,
                    center_a if assignment == 0 else center_b,
                )
            )
            for record, weight, assignment in zip(records, weights, assignments)
        )
        return left_ids, right_ids, mass_a, mass_b, j1, j2

    def _split_distance(
        self,
        record: MemoryRecord,
        center: Vector,
        representative: MemoryRecord,
        time: float,
    ) -> float:
        distance = 1.0 - _dot(record.index_vector, center)
        if self.config.split_value_weight:
            if _norm(record.original_value) > 0.0 and _norm(representative.original_value) > 0.0:
                distance += self.config.split_value_weight * (
                    1.0 - cosine_similarity(record.original_value, representative.original_value)
                )
        if self.config.split_time_weight:
            scale = max(1.0, time - min(record.write_time, representative.write_time))
            distance += self.config.split_time_weight * abs(
                record.write_time - representative.write_time
            ) / scale
        return distance

    @staticmethod
    def _cluster_center(
        records: Sequence[MemoryRecord],
        weights: Sequence[float],
        assignments: Sequence[int],
        target: int,
    ) -> Vector:
        dimension = len(records[0].index_vector)
        resultant = _weighted_sum(
            (
                (weight, record.index_vector)
                for record, weight, assignment in zip(records, weights, assignments)
                if assignment == target
            ),
            dimension,
        )
        if _norm(resultant) <= 1e-15:
            return next(
                record.index_vector
                for record, assignment in zip(records, assignments)
                if assignment == target
            )
        return _normalize(resultant)

    def _empirical_regret(
        self, record_ids: Sequence[str], centroids: Sequence[Vector]
    ) -> float:
        if not record_ids:
            return 0.0
        return math.fsum(
            1.0 - max(_dot(self.records[record_id].index_vector, center) for center in centroids)
            for record_id in record_ids
        ) / len(record_ids)

    def _try_split(self, slot: LeafSlot, time: float) -> bool:
        if time < slot.cooldown_until:
            return False
        candidate = self._two_means(slot, time)
        if candidate is None:
            slot.pending_split_observations = 0
            return False
        left_ids, right_ids, mass_a, mass_b, j1, j2 = candidate
        gain = (j1 - j2) / j1 if j1 > 1e-15 else 0.0
        triggered = (
            slot.resultant_length < self.config.split_minimum_resultant
            or gain > self.config.split_minimum_gain
            or slot.routing_regret_ema > self.config.split_maximum_regret
            or slot.value_conflict_score > self.config.split_maximum_conflict
        )
        valid = (
            triggered
            and mass_a >= self.config.minimum_child_mass
            and mass_b >= self.config.minimum_child_mass
            and j1 - j2 > self.config.slot_penalty
        )
        slot.pending_split_observations = (
            slot.pending_split_observations + 1 if valid else 0
        )
        if slot.pending_split_observations < self.config.split_patience:
            return False
        old_regret = self._empirical_regret(slot.record_ids, [slot.centroid])
        child_centers = [
            _normalize(
                _weighted_sum(
                    (
                        (self._record_weight(self.records[record_id], time), self.records[record_id].index_vector)
                        for record_id in ids
                    ),
                    len(slot.centroid),
                )
            )
            for ids in (left_ids, right_ids)
        ]
        new_regret = (
            self._empirical_regret(left_ids, [child_centers[0]])
            * len(left_ids)
            + self._empirical_regret(right_ids, [child_centers[1]])
            * len(right_ids)
        ) / len(slot.record_ids)
        if new_regret >= old_regret - 1e-12:
            return False

        parent_id = slot.id
        child_a_id = self._new_slot_id()
        child_b_id = self._new_slot_id()
        child_a = LeafSlot(
            id=child_a_id,
            layer=slot.layer,
            head_or_kv_group=slot.head_or_kv_group,
            record_ids=left_ids,
            parent_id=parent_id,
            cooldown_until=time + self.config.split_cooldown,
        )
        child_b = LeafSlot(
            id=child_b_id,
            layer=slot.layer,
            head_or_kv_group=slot.head_or_kv_group,
            record_ids=right_ids,
            parent_id=parent_id,
            cooldown_until=time + self.config.split_cooldown,
        )
        parent = ParentSlot(
            id=parent_id,
            layer=slot.layer,
            head_or_kv_group=slot.head_or_kv_group,
            child_ids=[child_a_id, child_b_id],
            centroid=slot.centroid,
            effective_count=slot.effective_count,
            ordered_time_range=slot.ordered_time_range,
            parent_id=slot.parent_id,
            cooldown_until=time + self.config.split_cooldown,
        )
        self.leaves.pop(parent_id)
        self.parents[parent_id] = parent
        self.leaves[child_a_id] = child_a
        self.leaves[child_b_id] = child_b
        self._refresh_leaf(child_a, time)
        self._refresh_leaf(child_b, time)
        self._refresh_parent(parent)
        return True

    def _cross_supersession(self, left: LeafSlot, right: LeafSlot) -> bool:
        left_ids = set(left.record_ids)
        right_ids = set(right.record_ids)
        return any(
            (self.records[record_id].superseded_by in right_ids)
            for record_id in left_ids
        ) or any(
            (self.records[record_id].superseded_by in left_ids)
            for record_id in right_ids
        )

    def _merge_candidate_pairs(self) -> list[tuple[LeafSlot, LeafSlot]]:
        leaves = list(self.leaves.values())
        pairs = []
        for left_index, left in enumerate(leaves):
            for right in leaves[left_index + 1 :]:
                if (
                    left.parent_id == right.parent_id
                    and self._compatible(right, left.layer, left.head_or_kv_group)
                ):
                    pairs.append((left, right))
        return pairs

    def _try_merge(self, left: LeafSlot, right: LeafSlot, time: float) -> bool:
        if time < left.cooldown_until or time < right.cooldown_until:
            return False
        if _dot(left.centroid, right.centroid) < self.config.merge_minimum_similarity:
            return False
        if max(left.value_conflict_score, right.value_conflict_score) > self.config.merge_maximum_conflict:
            return False
        if max(left.routing_regret_ema, right.routing_regret_ema) > self.config.merge_maximum_regret:
            return False
        if self._cross_supersession(left, right):
            return False
        combined_ids = [*left.record_ids, *right.record_ids]
        dimension = len(left.centroid)
        weights = [self._record_weight(self.records[record_id], time) for record_id in combined_ids]
        resultant = _weighted_sum(
            zip(weights, (self.records[record_id].index_vector for record_id in combined_ids)),
            dimension,
        )
        if _norm(resultant) <= 1e-15:
            return False
        center = _normalize(resultant)
        merged_scatter = math.fsum(
            weight * (1.0 - _dot(self.records[record_id].index_vector, center))
            for weight, record_id in zip(weights, combined_ids)
        )
        added_distortion = merged_scatter - left.weighted_scatter - right.weighted_scatter
        if added_distortion >= self.config.slot_penalty:
            return False

        parent_id = left.parent_id
        merged_id = self._new_slot_id()
        merged = LeafSlot(
            id=merged_id,
            layer=left.layer,
            head_or_kv_group=left.head_or_kv_group,
            record_ids=combined_ids,
            parent_id=parent_id,
            cooldown_until=time + self.config.merge_cooldown,
        )
        self.leaves.pop(left.id)
        self.leaves.pop(right.id)
        self.leaves[merged_id] = merged
        if parent_id in self.parents:
            parent = self.parents[parent_id]
            parent.child_ids = [
                child_id
                for child_id in parent.child_ids
                if child_id not in {left.id, right.id}
            ]
            parent.child_ids.append(merged_id)
            if len(parent.child_ids) == 1:
                grandparent_id = parent.parent_id
                collapsed_id = parent.id
                self.leaves.pop(merged_id)
                merged.id = collapsed_id
                merged.parent_id = grandparent_id
                self.parents.pop(collapsed_id)
                self.leaves[collapsed_id] = merged
        self._refresh_all(time)
        return True

    def maintain(self, *, time: float | None = None) -> MaintenanceReport:
        """Apply exact expiry, budget control, split, and merge maintenance."""

        now = self._next_time(time)
        self._refresh_all(now)
        evicted: list[str] = []
        for record in list(self.records.values()):
            age = now - record.write_time
            if (
                age >= self.config.new_record_protection
                and self._record_weight(record, now) < self.config.minimum_effective_weight
            ):
                evicted.append(record.id)
                self._remove_record(record.id)

        budget = self.config.memory_budget_bytes
        if budget is not None:
            self._memory_cost_lambda = max(
                0.0,
                self._memory_cost_lambda
                + self.config.budget_step_size * (self.memory_bytes - budget),
            )
            eligible = [
                record
                for record in self.records.values()
                if now - record.write_time >= self.config.new_record_protection
            ]
            eligible.sort(key=lambda record: self._eviction_priority(record, now))
            for record in eligible:
                expected_gain = self._eviction_priority(record, now) * record.approximate_bytes()
                storage_cost = self._memory_cost_lambda * record.approximate_bytes()
                if self.memory_bytes <= budget and expected_gain > storage_cost:
                    break
                evicted.append(record.id)
                self._remove_record(record.id)

        self._prune_empty_nodes()
        self._refresh_all(now)
        split_ids: list[str] = []
        if self.config.enable_split_merge:
            for slot in list(self.leaves.values()):
                if slot.id in self.leaves and self._try_split(slot, now):
                    split_ids.append(slot.id)
            self._refresh_all(now)

        merged_pairs: list[tuple[str, str]] = []
        if self.config.enable_split_merge:
            candidates = sorted(
                self._merge_candidate_pairs(),
                key=lambda pair: _dot(pair[0].centroid, pair[1].centroid),
                reverse=True,
            )
            used: set[str] = set()
            for left, right in candidates:
                if (
                    left.id in used
                    or right.id in used
                    or left.id not in self.leaves
                    or right.id not in self.leaves
                ):
                    continue
                if self._try_merge(left, right, now):
                    merged_pairs.append((left.id, right.id))
                    used.update((left.id, right.id))
        self._prune_empty_nodes()
        self._refresh_all(now)
        return MaintenanceReport(
            evicted_record_ids=tuple(evicted),
            split_slot_ids=tuple(split_ids),
            merged_slot_pairs=tuple(merged_pairs),
            memory_bytes=self.memory_bytes,
            memory_cost_lambda=self._memory_cost_lambda,
        )

    def snapshot(self) -> dict:
        """Return JSON-compatible research diagnostics."""

        return {
            "record_count": self.record_count,
            "leaf_count": self.leaf_count,
            "parent_count": len(self.parents),
            "slot_count": self.slot_count,
            "memory_bytes": self.memory_bytes,
            "memory_cost_lambda": self._memory_cost_lambda,
            "leaves": [
                {
                    "id": slot.id,
                    "parent_id": slot.parent_id,
                    "layer": slot.layer,
                    "group": str(slot.head_or_kv_group),
                    "record_count": len(slot.record_ids),
                    "effective_count": slot.effective_count,
                    "resultant_length": slot.resultant_length,
                    "concentration": slot.concentration,
                    "weighted_scatter": slot.weighted_scatter,
                    "routing_regret_ema": slot.routing_regret_ema,
                    "value_conflict_score": slot.value_conflict_score,
                    "router_record_count": slot.router_record_count,
                    "router_block_count": slot.router_block_count,
                    "router_mass": slot.router_mass,
                    "router_resultant_length": slot.router_resultant_length,
                    "router_concentration": slot.router_concentration,
                }
                for slot in sorted(self.leaves.values(), key=lambda item: item.id)
            ],
        }


class VMFPosteriorMemory(ProbabilisticHierarchicalMemory):
    """Phase-2 ablation: posterior writes with averaged centroid K/V payload.

    Ordered original payloads and temporal conflict handling are deliberately
    absent here so that the C -> D comparison isolates the Phase-3 change.
    """

    def __init__(self, config: MemoryConfig | None = None):
        base = config or MemoryConfig()
        super().__init__(
            replace(base, use_temporal_weights=False, enable_split_merge=False)
        )

    def retrieve(
        self,
        query_index: Sequence[float],
        *,
        query_key: Sequence[float] | None = None,
        layer: int = 0,
        head_or_kv_group: Hashable = 0,
        time: float | None = None,
        top_k: int | None = None,
        **_ignored: object,
    ) -> list[RetrievalResult]:
        now = self._next_time(time)
        index = _normalize(query_index, name="query_index")
        key = _vector(query_key if query_key is not None else query_index, name="query_key")
        if len(key) != len(index):
            raise ValueError("query_index and query_key dimensions must match.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive.")
        self._refresh_all(now)
        routed = sorted(
            (
                slot
                for slot in self.leaves.values()
                if self._compatible(slot, layer, head_or_kv_group)
            ),
            key=lambda slot: self._route_score(index, slot, now),
            reverse=True,
        )[: self.config.route_top_k]
        scored: list[tuple[float, LeafSlot, Vector, Vector, int, CompressedSources]] = []
        for slot in routed:
            records = [self.records[record_id] for record_id in slot.record_ids]
            if not records:
                continue
            total_count = sum(record.multiplicity for record in records)
            mean_key = tuple(
                math.fsum(
                    record.multiplicity * record.original_key[position]
                    for record in records
                )
                / total_count
                for position in range(len(records[0].original_key))
            )
            mean_value = tuple(
                math.fsum(
                    record.multiplicity * record.original_value[position]
                    for record in records
                )
                / total_count
                for position in range(len(records[0].original_value))
            )
            exact_duplicate = all(
                record.original_key == records[0].original_key
                and record.original_value == records[0].original_value
                for record in records[1:]
            )
            multiplicity = total_count if exact_duplicate else 1
            score = _dot(key, mean_key) / math.sqrt(len(key))
            if multiplicity > 1:
                score += math.log(multiplicity)
            sources = CompressedSources(
                frozenset(record.source_token_or_span for record in records)
            )
            scored.append((score, slot, mean_key, mean_value, multiplicity, sources))
        scored.sort(key=lambda item: item[0], reverse=True)
        width = top_k if top_k is not None else self.config.record_top_k
        return [
            RetrievalResult(
                record_id=slot.id,
                slot_id=slot.id,
                score=score,
                original_key=mean_key,
                original_value=mean_value,
                source_token_or_span=sources,
                write_time=slot.ordered_time_range[1],
                source_authority=0.0,
                multiplicity=multiplicity,
            )
            for score, slot, mean_key, mean_value, multiplicity, sources in scored[:width]
        ]


class TemporalVMFMemory(ProbabilisticHierarchicalMemory):
    """Phase-3 ablation: posterior writes and temporal payload retention."""

    def __init__(self, config: MemoryConfig | None = None):
        base = config or MemoryConfig()
        super().__init__(replace(base, use_temporal_weights=True, enable_split_merge=False))


class HierarchicalVMFMemory(ProbabilisticHierarchicalMemory):
    """Named Phase-4 variant used by fair benchmark sweeps."""


@dataclass
class _CentroidRecord:
    id: str
    layer: int
    group: Hashable
    centroid_sum: list[float]
    key_sum: list[float]
    value_sum: list[float]
    count: int
    write_time: float
    source_tokens: list[Hashable | None]
    exact_duplicate: bool
    first_index: Vector
    first_key: Vector
    first_value: Vector

    @property
    def centroid(self) -> Vector:
        return _normalize(self.centroid_sum)

    @property
    def mean_key(self) -> Vector:
        return tuple(value / self.count for value in self.key_sum)

    @property
    def mean_value(self) -> Vector:
        return tuple(value / self.count for value in self.value_sum)


class CamelotHardMemory:
    """CAMELoT-style fixed cosine-threshold, averaged-K/V baseline."""

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        record_top_k: int = 4,
        exact_duplicate_count_correction: bool = True,
    ):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [-1, 1].")
        if record_top_k <= 0:
            raise ValueError("record_top_k must be positive.")
        self.fixed_threshold = float(threshold)
        self.record_top_k = record_top_k
        self.exact_duplicate_count_correction = exact_duplicate_count_correction
        self.slots: dict[str, _CentroidRecord] = {}
        self._counter = 0
        self._clock = 0.0

    @property
    def threshold(self) -> float:
        return self.fixed_threshold

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    @property
    def record_count(self) -> int:
        return sum(slot.count for slot in self.slots.values())

    @property
    def memory_bytes(self) -> int:
        return sum(
            144 + 8 * (len(slot.centroid_sum) + len(slot.key_sum) + len(slot.value_sum))
            for slot in self.slots.values()
        )

    def _time(self, time: float | None) -> float:
        if time is None:
            self._clock += 1.0
        else:
            value = float(time)
            if value < self._clock:
                raise ValueError("Online updates must have non-decreasing time.")
            self._clock = value
        return self._clock

    def _compatible_slots(
        self, layer: int, group: Hashable, dimension: int
    ) -> list[_CentroidRecord]:
        return [
            slot
            for slot in self.slots.values()
            if slot.layer == layer
            and slot.group == group
            and len(slot.centroid_sum) == dimension
        ]

    def _observe_nearest_similarity(self, similarity: float) -> None:
        del similarity

    def write(
        self,
        index_vector: Sequence[float],
        original_key: Sequence[float],
        original_value: Sequence[float],
        *,
        layer: int = 0,
        head_or_kv_group: Hashable = 0,
        source_token_or_span: Hashable | None = None,
        time: float | None = None,
        **_ignored: object,
    ) -> WriteDecision:
        now = self._time(time)
        index = _normalize(index_vector, name="index_vector")
        key = _vector(original_key, name="original_key", allow_zero=False)
        value = _vector(original_value, name="original_value")
        if len(index) != len(key):
            raise ValueError("index and key dimensions must match.")
        candidates = self._compatible_slots(layer, head_or_kv_group, len(index))
        scored = [(cosine_similarity(index, slot.centroid), slot) for slot in candidates]
        nearest_similarity, nearest = max(scored, default=(-math.inf, None), key=lambda item: item[0])
        decision_threshold = self.threshold
        create_new = nearest is None or nearest_similarity < decision_threshold
        self._observe_nearest_similarity(nearest_similarity)
        if create_new:
            self._counter += 1
            slot_id = f"camelot-slot-{self._counter:08d}"
            nearest = _CentroidRecord(
                id=slot_id,
                layer=layer,
                group=head_or_kv_group,
                centroid_sum=list(index),
                key_sum=list(key),
                value_sum=list(value),
                count=1,
                write_time=now,
                source_tokens=[source_token_or_span],
                exact_duplicate=True,
                first_index=index,
                first_key=key,
                first_value=value,
            )
            self.slots[slot_id] = nearest
        else:
            assert nearest is not None
            nearest.exact_duplicate = nearest.exact_duplicate and (
                index == nearest.first_index
                and key == nearest.first_key
                and value == nearest.first_value
            )
            for target, source in (
                (nearest.centroid_sum, index),
                (nearest.key_sum, key),
                (nearest.value_sum, value),
            ):
                for position, item in enumerate(source):
                    target[position] += item
            nearest.count += 1
            nearest.write_time = now
            nearest.source_tokens.append(source_token_or_span)
            slot_id = nearest.id
        probability_new = 1.0 if create_new else 0.0
        return WriteDecision(
            record_id=slot_id,
            slot_id=slot_id,
            created_new_slot=create_new,
            probability_new=probability_new,
            existing_probabilities=tuple(
                (
                    slot.id,
                    float(not create_new and nearest is not None and slot.id == nearest.id),
                )
                for _similarity, slot in sorted(
                    scored, key=lambda item: item[0], reverse=True
                )
            ),
        )

    def retrieve(
        self,
        query_index: Sequence[float],
        *,
        query_key: Sequence[float] | None = None,
        layer: int = 0,
        head_or_kv_group: Hashable = 0,
        time: float | None = None,
        top_k: int | None = None,
        **_ignored: object,
    ) -> list[RetrievalResult]:
        now = self._time(time)
        index = _normalize(query_index, name="query_index")
        key = _vector(query_key if query_key is not None else query_index, name="query_key")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive.")
        candidates = self._compatible_slots(layer, head_or_kv_group, len(index))
        scored = []
        for slot in candidates:
            score = _dot(key, slot.mean_key) / math.sqrt(len(key))
            multiplicity = (
                slot.count
                if self.exact_duplicate_count_correction and slot.exact_duplicate
                else 1
            )
            if multiplicity > 1:
                score += math.log(multiplicity)
            scored.append((score, slot, multiplicity))
        scored.sort(key=lambda item: item[0], reverse=True)
        width = top_k if top_k is not None else self.record_top_k
        return [
            RetrievalResult(
                record_id=slot.id,
                slot_id=slot.id,
                score=score,
                original_key=slot.mean_key,
                original_value=slot.mean_value,
                source_token_or_span=CompressedSources(frozenset(slot.source_tokens)),
                write_time=slot.write_time,
                source_authority=0.0,
                multiplicity=multiplicity,
            )
            for score, slot, multiplicity in scored[:width]
        ]

    def snapshot(self) -> dict:
        return {
            "record_count": self.record_count,
            "slot_count": self.slot_count,
            "memory_bytes": self.memory_bytes,
            "threshold": self.threshold,
        }


class AdaptiveQuantileMemory(CamelotHardMemory):
    """Threshold baseline driven by a rolling nearest-similarity quantile."""

    def __init__(
        self,
        *,
        initial_threshold: float = 0.9,
        quantile: float = 0.1,
        history_window: int = 256,
        minimum_history: int = 8,
        record_top_k: int = 4,
        exact_duplicate_count_correction: bool = True,
    ):
        super().__init__(
            initial_threshold,
            record_top_k=record_top_k,
            exact_duplicate_count_correction=exact_duplicate_count_correction,
        )
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1].")
        if history_window <= 0 or minimum_history <= 0:
            raise ValueError("history sizes must be positive.")
        self.quantile = quantile
        self.history_window = history_window
        self.minimum_history = minimum_history
        self._similarity_history: list[float] = []

    @property
    def threshold(self) -> float:
        if len(self._similarity_history) < self.minimum_history:
            return self.fixed_threshold
        ordered = sorted(self._similarity_history)
        position = self.quantile * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def _observe_nearest_similarity(self, similarity: float) -> None:
        if math.isfinite(similarity):
            self._similarity_history.append(similarity)
            if len(self._similarity_history) > self.history_window:
                self._similarity_history.pop(0)


def attention_readout(
    query_key: Sequence[float],
    memory_results: Sequence[RetrievalResult],
    *,
    window_keys: Sequence[Sequence[float]] = (),
    window_values: Sequence[Sequence[float]] = (),
) -> tuple[Vector, tuple[float, ...]]:
    """Apply native attention to memory and current-window K/V.

    ``RetrievalResult.score`` is deliberately not reused: final attention is
    recomputed from the original key.  Multiplicity contributes ``log(n)`` only
    for an explicitly compressed exact duplicate.
    """

    query = _vector(query_key, name="query_key")
    if len(window_keys) != len(window_values):
        raise ValueError("window_keys and window_values must have equal length.")
    keys = [result.original_key for result in memory_results] + [
        _vector(key, name="window_key", allow_zero=False) for key in window_keys
    ]
    values = [result.original_value for result in memory_results] + [
        _vector(value, name="window_value") for value in window_values
    ]
    counts = [result.multiplicity for result in memory_results] + [1] * len(window_keys)
    if not keys:
        raise ValueError("attention_readout requires at least one key/value pair.")
    if any(len(key) != len(query) for key in keys):
        raise ValueError("All keys must match query dimension.")
    value_dimension = len(values[0])
    if any(len(value) != value_dimension for value in values):
        raise ValueError("All values must have equal dimension.")
    logits = [
        _dot(query, key) / math.sqrt(len(query)) + math.log(count)
        for key, count in zip(keys, counts)
    ]
    normalizer = _logsumexp(logits)
    weights = tuple(math.exp(logit - normalizer) for logit in logits)
    output = _weighted_sum(zip(weights, values), value_dimension)
    return output, weights


__all__ = [
    "AdaptiveQuantileMemory",
    "CamelotHardMemory",
    "CompressedSources",
    "HierarchicalVMFMemory",
    "MaintenanceReport",
    "MemoryConfig",
    "MemoryRecord",
    "ProbabilisticHierarchicalMemory",
    "RetrievalResult",
    "TemporalVMFMemory",
    "VMFPosteriorMemory",
    "WriteDecision",
    "attention_readout",
    "cosine_similarity",
    "estimate_vmf_concentration",
    "vmf_log_density",
    "vmf_log_normalizer",
]
