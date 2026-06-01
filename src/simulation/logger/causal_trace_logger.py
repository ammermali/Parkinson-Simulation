from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


PHASE_INDEX = {"0_pre_state": 0, "1_perception": 1, "2_state_update": 2, "3_action_selection": 3, "4_effect_buffer": 4, "5_commit": 5}
NODE_KINDS = {"agent_state", "env_field", "internal_field", "action", "aggregate", "buffer"}
LEVELS = {"extracellular", "intracellular", "macro", "environment"}
RELATIONS = {"state_transition", "threshold_trigger", "action_selection", "field_effect", "internal_field_effect", "agent_to_agent", "aggregation", "degradation", "target_assignment", "buffer_commit", "lifecycle", "structural"}


@dataclass(frozen=True)
class CausalNode:
    """One temporal G0 node emitted by the runtime causal logger."""

    run_id: str
    tick: int
    phase: str
    rank: int
    node_id: str
    kind: str
    uid: Optional[str]
    agent_type: Optional[str]
    state: Optional[str]
    field: Optional[str]
    value: Optional[float | int | str | bool]
    level: str
    owner_uid: Optional[str]
    compartment: Optional[str]
    g1_key: str
    g2_key: str


@dataclass(frozen=True)
class CausalEdge:
    """One directed causal relation between two G0 nodes."""

    run_id: str
    tick: int
    phase_from: str
    phase_to: str
    rank: int
    edge_id: str
    source_node_id: str
    target_node_id: str
    source_kind: str
    target_kind: str
    source_uid: Optional[str]
    target_uid: Optional[str]
    source_type: Optional[str]
    target_type: Optional[str]
    source_state: Optional[str]
    target_state: Optional[str]
    source_field: Optional[str]
    target_field: Optional[str]
    relation: str
    mechanism: str
    rule_id: Optional[str]
    effect_value: Optional[float]
    effect_sign: Optional[str]
    effect_unit: Optional[str]
    predicate: Optional[str]
    probability: Optional[float]
    rng_value: Optional[float]
    outcome: Optional[str]
    compartment: Optional[str]
    owner_uid: Optional[str]
    g1_source_key: str
    g1_target_key: str
    g2_source_key: str
    g2_target_key: str
    valid: bool


class CausalTraceLogger:
    """Compact runtime logger for G0 temporal causal graph construction.
    This logger stores typed nodes and edges only. It intentionally omits raw
    perceptions, positions, full configs and failed stochastic checks unless a
    successful causal outcome must carry probability data."""
    schema_version = "2.0-json"
    def __init__(self, run_id: str, rank: int, comm=None, output_dir: Path | str = "output/simulation/logs", enabled: bool = False, agent_type_map: Optional[dict[Any, str]] = None, params: Optional[dict[str, Any]] = None, model_version: Optional[str] = None):
        self.run_id = run_id
        self.rank = rank
        self.comm = comm
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.current_tick = 0
        self._edge_index = 0
        self._seen_nodes: set[str] = set()
        self.agent_type_map = {
            str(key): value
            for key, value in (agent_type_map or {}).items()
        }
        self.params_hash = self._hash_params(params)
        self.model_version = model_version
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_path.write_text("", encoding="utf-8")
        self.edges_path.write_text("", encoding="utf-8")
        if self.rank == 0:
            self.merged_nodes_path.write_text("", encoding="utf-8")
            self.merged_edges_path.write_text("", encoding="utf-8")
            self._write_metadata()
        self._barrier()

    @property
    def nodes_path(self) -> Path:
        return self.output_dir / f"g0_nodes_rank{self.rank}.jsonl"

    @property
    def edges_path(self) -> Path:
        return self.output_dir / f"g0_edges_rank{self.rank}.jsonl"

    @property
    def merged_nodes_path(self) -> Path:
        return self.output_dir / "g0_nodes.jsonl"

    @property
    def merged_edges_path(self) -> Path:
        return self.output_dir / "g0_edges.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "run_metadata.json"

    def set_tick(self, tick: int) -> None:
        self.current_tick = tick

    def node(
        self,
        kind: str,
        phase: str,
        uid: Optional[str] = None,
        agent_type: Optional[str] = None,
        state: Optional[Any] = None,
        field: Optional[str] = None,
        value: Optional[float | int | str | bool] = None,
        level: str = "extracellular",
        owner_uid: Optional[str] = None,
        compartment: Optional[Any] = None,
        label: Optional[str] = None,
        tick: Optional[int] = None,
    ) -> CausalNode:
        """Create and write one causal node, de-duplicated by node id."""
        if kind not in NODE_KINDS:
            raise ValueError(f"Invalid causal node kind: {kind}")
        if level not in LEVELS:
            raise ValueError(f"Invalid causal node level: {level}")
        tick_value = self.current_tick if tick is None else tick
        node_id = self.node_id(kind, phase, uid, agent_type, state, field, label, tick_value)
        g1_key, g2_key = self.contraction_keys(kind, uid, agent_type, state, field, label)
        node = CausalNode(
            run_id=self.run_id,
            tick=tick_value,
            phase=phase,
            rank=self.rank,
            node_id=node_id,
            kind=kind,
            uid=uid,
            agent_type=agent_type,
            state=value_of(state),
            field=field,
            value=value,
            level=level,
            owner_uid=owner_uid,
            compartment=value_of(compartment),
            g1_key=g1_key,
            g2_key=g2_key
        )
        self._write_node(node)
        return node

    def edge(
        self,
        source: CausalNode,
        target: CausalNode,
        relation: str,
        mechanism: str,
        rule_id: Optional[str] = None,
        effect_value: Optional[float] = None,
        effect_sign: Optional[str] = None,
        effect_unit: Optional[str] = None,
        predicate: Optional[str] = None,
        probability: Optional[float] = None,
        rng_value: Optional[float] = None,
        outcome: Optional[str] = None,
        compartment: Optional[Any] = None,
        owner_uid: Optional[str] = None,
    ) -> Optional[CausalEdge]:
        """Create one typed causal edge."""
        if relation not in RELATIONS:
            raise ValueError(f"Invalid causal relation: {relation}")
        valid = self._phase_order_valid(source.phase, target.phase)
        edge = CausalEdge(
            run_id=self.run_id,
            tick=self.current_tick,
            phase_from=source.phase,
            phase_to=target.phase,
            rank=self.rank,
            edge_id=self.edge_id(),
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            source_kind=source.kind,
            target_kind=target.kind,
            source_uid=source.uid,
            target_uid=target.uid,
            source_type=source.agent_type,
            target_type=target.agent_type,
            source_state=source.state,
            target_state=target.state,
            source_field=source.field,
            target_field=target.field,
            relation=relation,
            mechanism=mechanism,
            rule_id=rule_id,
            effect_value=effect_value,
            effect_sign=effect_sign,
            effect_unit=effect_unit,
            predicate=predicate,
            probability=probability,
            rng_value=rng_value,
            outcome=outcome,
            compartment=value_of(compartment),
            owner_uid=owner_uid,
            g1_source_key=source.g1_key,
            g1_target_key=target.g1_key,
            g2_source_key=source.g2_key,
            g2_target_key=target.g2_key,
            valid=valid
        )
        self._write_edge(edge)
        return edge

    def state_transition(self, agent, from_state, to_state, mechanism: str, rule_id: Optional[str] = None, probability: Optional[float] = None, rng_value: Optional[float] = None, outcome: str = "transitioned", owner=None, compartment=None) -> None:
        """Log a causal state transition for one agent."""

        if value_of(from_state) == value_of(to_state):
            return
        source = self.agent_state_node(agent, from_state, "0_pre_state", owner=owner, compartment=compartment)
        target = self.agent_state_node(agent, to_state, "2_state_update", owner=owner, compartment=compartment)
        self.edge(source, target, "state_transition", mechanism, rule_id=rule_id, probability=probability, rng_value=rng_value, outcome=outcome, compartment=compartment, owner_uid=uid_of(owner))

    def threshold_trigger(self, source_node: CausalNode, target_agent, target_state, mechanism: str, rule_id: str, predicate: str, owner=None, compartment=None) -> None:
        """Log a perceived field or condition that triggered a transition."""

        target = self.agent_state_node(target_agent, target_state, "2_state_update", owner=owner, compartment=compartment)
        self.edge(source_node, target, "threshold_trigger", mechanism, rule_id=rule_id, predicate=predicate, outcome="triggered", compartment=compartment, owner_uid=uid_of(owner))

    def action_selection(self, agent, action, mechanism: str, rule_id: Optional[str] = None, owner=None, compartment=None) -> CausalNode:
        """Log the action selected by an agent and return its action node."""

        source = self.agent_state_node(agent, getattr(agent, "state", None), "2_state_update", owner=owner, compartment=compartment)
        target = self.action_node(agent, action, "3_action_selection", owner=owner, compartment=compartment)
        self.edge(source, target, "action_selection", mechanism, rule_id=rule_id, outcome="selected", compartment=compartment, owner_uid=uid_of(owner))
        return target

    def field_effect(self, agent, action, field: str, effect_value: float, effect_sign: str, mechanism: str, rule_id: Optional[str] = None, unit: Optional[str] = None) -> None:
        """Log an extracellular field effect buffered by an action."""

        source = self.action_node(agent, action, "3_action_selection")
        target = self.env_field_node(f"SN.{field}_buffer", field=f"{field}_buffer", phase="4_effect_buffer", value=effect_value)
        self.edge(source, target, "field_effect", mechanism, rule_id=rule_id, effect_value=effect_value, effect_sign=effect_sign, effect_unit=unit, outcome="buffered")

    def internal_field_effect(self, agent, owner, field: str, effect_value: float, effect_sign: str, mechanism: str, action=None, rule_id: Optional[str] = None, unit: Optional[str] = None) -> None:
        """Log an intracellular field effect buffered inside a neuron."""

        source = self.action_node(agent, action or getattr(agent, "pending_action", None), "3_action_selection", owner=owner, compartment="Intracellular")
        target = self.internal_field_node(owner, field=f"{field}_buffer", phase="4_effect_buffer", value=effect_value)
        self.edge(source, target, "internal_field_effect", mechanism, rule_id=rule_id, effect_value=effect_value, effect_sign=effect_sign, effect_unit=unit, outcome="buffered", compartment="Intracellular", owner_uid=uid_of(owner))

    def target_assignment(self, source_agent, target_agent, mechanism: str, rule_id: Optional[str] = None, owner=None, outcome: str = "assigned") -> None:
        """Log an agent-to-agent target assignment such as lysosome targeting."""

        source = self.agent_state_node(source_agent, getattr(source_agent, "state", None), "3_action_selection", owner=owner, compartment="Intracellular")
        target = self.agent_state_node(target_agent, getattr(target_agent, "state", None), "4_effect_buffer", owner=owner, compartment="Intracellular")
        self.edge(source, target, "target_assignment", mechanism, rule_id=rule_id, outcome=outcome, compartment="Intracellular", owner_uid=uid_of(owner))

    def aggregation(self, source_agent, aggregate_agent, mechanism: str, aggregate_id: Optional[int] = None, owner=None, outcome: str = "aggregated") -> None:
        """Log alpha-synuclein contribution to an aggregate."""

        source = self.agent_state_node(source_agent, getattr(source_agent, "state", None), "2_state_update", owner=owner, compartment="Intracellular")
        target = self.aggregate_snapshot(aggregate_agent, aggregate_id=aggregate_id, owner=owner)
        self.edge(source, target, "aggregation", mechanism, rule_id="ALPHA_AGGREGATION", outcome=outcome, compartment="Intracellular", owner_uid=uid_of(owner))

    def aggregate_snapshot(self, aggregate_agent, aggregate_id: Optional[int] = None, owner=None, phase: str = "4_effect_buffer") -> CausalNode:
        """Create a G0 node representing one aggregate at the current tick."""
        resolved_id = aggregate_id or getattr(aggregate_agent, "aggregate_id", "")
        owner_uid = uid_of(owner)
        aggregate_identity = f"{owner_uid or 'None'}::Aggregate_{resolved_id}"
        state = getattr(aggregate_agent, "state", None)
        state_value = value_of(state) or "unknown"
        return self.node(
            "aggregate",
            phase,
            uid=aggregate_identity,
            agent_type=type_name(aggregate_agent),
            state=state,
            value=getattr(aggregate_agent, "size", None),
            level="intracellular",
            owner_uid=owner_uid,
            compartment="Intracellular",
            label=f"{aggregate_identity}.{state_value}"
        )

    def degradation(self, lysosome, target_agent, mechanism: str, outcome: str, probability: Optional[float] = None, rng_value: Optional[float] = None, owner=None) -> None:
        """Log a lysosome degradation attempt or outcome."""

        source = self.action_node(lysosome, getattr(lysosome, "pending_action", None), "3_action_selection", owner=owner, compartment="Intracellular")
        target = self.agent_state_node(target_agent, getattr(target_agent, "state", None), "4_effect_buffer", owner=owner, compartment="Intracellular")
        self.edge(source, target, "degradation", mechanism, rule_id="LYSOSOME_DEGRADATION", probability=probability, rng_value=rng_value, outcome=outcome, compartment="Intracellular", owner_uid=uid_of(owner))

    def buffer_commit(self, source_field: str, target_field: str, effect_value: float, effect_sign: str, mechanism: str, level: str = "environment", owner=None) -> None:
        """Log the commit of a buffered scalar effect."""

        if effect_value == 0:
            return
        if level == "environment":
            source = self.env_field_node(f"SN.{source_field}", source_field, "4_effect_buffer", effect_value)
            target = self.env_field_node(f"SN.{target_field}", target_field, "5_commit")
        else:
            source = self.internal_field_node(owner, source_field, "4_effect_buffer", effect_value)
            target = self.internal_field_node(owner, target_field, "5_commit")
        self.edge(source, target, "buffer_commit", mechanism, effect_value=effect_value, effect_sign=effect_sign, outcome="committed", owner_uid=uid_of(owner))

    def snapshot_field(self, field: str, value: float, level: str = "environment", owner=None) -> CausalNode:
        """Log a committed scalar field value as a G0 node."""

        if level == "environment":
            return self.env_field_node(f"SN.{field}", field, "5_commit", value)
        return self.internal_field_node(owner, field, "5_commit", value)

    def agent_state_node(self, agent, state, phase: str, owner=None, compartment=None) -> CausalNode:
        """Create a G0 node for an agent state."""

        level = "macro" if type_name(agent) == "Neuron" else "intracellular" if owner is not None else "extracellular"
        return self.node("agent_state", phase, uid=uid_of(agent), agent_type=type_name(agent), state=state, level=level, owner_uid=uid_of(owner), compartment=compartment or getattr(agent, "compartment", None))

    def action_node(self, agent, action, phase: str, owner=None, compartment=None) -> CausalNode:
        """Create a G0 node for an action selection."""

        action_name = value_of(action) or "action"
        return self.node("action", phase, uid=uid_of(agent), agent_type=type_name(agent), state=action_name, level="intracellular" if owner is not None else "extracellular", owner_uid=uid_of(owner), compartment=compartment or getattr(agent, "compartment", None), label=f"{type_name(agent)}_{short_uid(agent)}.{action_name}")

    def env_field_node(self, label: str, field: str, phase: str, value: Optional[float] = None) -> CausalNode:
        """Create a G0 node for a Substantia Nigra field or buffer."""

        return self.node("env_field" if not field.endswith("_buffer") else "buffer", phase, uid="SN", agent_type="SubstantiaNigra", field=field, value=value, level="environment", label=label)

    def internal_field_node(self, owner, field: str, phase: str, value: Optional[float] = None) -> CausalNode:
        """Create a G0 node for an intracellular neuron field or buffer."""

        owner_uid = uid_of(owner) or "Neuron"
        return self.node("internal_field" if not field.endswith("_buffer") else "buffer", phase, uid=owner_uid, agent_type="Neuron", field=field, value=value, level="macro", owner_uid=owner_uid, label=f"Neuron_{short_uid(owner)}.internal.{field}")

    def node_id(self, kind: str, phase: str, uid: Optional[str], agent_type: Optional[str], state, field: Optional[str], label: Optional[str], tick: int) -> str:
        """Build a deterministic temporal node id."""

        suffix = PHASE_INDEX[phase]
        if label:
            base = label
        elif kind in ("env_field", "internal_field", "buffer"):
            base = field or "field"
        else:
            base = f"{agent_type}_{uid}.{value_of(state)}"
        return f"{base}@{tick}.{suffix}"

    def contraction_keys(self, kind: str, uid: Optional[str], agent_type: Optional[str], state, field: Optional[str], label: Optional[str]) -> tuple[str, str]:
        """Return G1 and G2 grouping keys stored on logged rows."""

        if kind in ("env_field", "buffer") and uid == "SN":
            key = f"SN.{field}"
            return key, key
        if kind == "internal_field":
            key = label or f"{uid}.internal.{field}"
            return key, key
        if label:
            parts = label.split(".")
            g2 = f"{agent_type}.{parts[-1]}" if agent_type else label
            return label, g2
        state_value = value_of(state)
        short = uid.split(":")[0] if uid else ""
        return f"{agent_type}_{short}.{state_value}", f"{agent_type}.{state_value}"

    def edge_id(self) -> str:
        """Return the next rank-local edge id."""

        self._edge_index += 1
        return f"edge_{self.rank}_{self._edge_index:08d}"

    def close(self) -> None:
        """Flush rank-local logs and merge them on rank zero."""

        if not self.enabled:
            return
        self._barrier()
        if self.rank == 0:
            self._merge_rank_files("g0_nodes_rank*.jsonl", self.merged_nodes_path)
            self._merge_rank_files("g0_edges_rank*.jsonl", self.merged_edges_path)
            self._write_metadata()
        self._barrier()

    def _write_node(self, node: CausalNode) -> None:
        if not self.enabled or node.node_id in self._seen_nodes:
            return
        self._seen_nodes.add(node.node_id)
        self._append_jsonl(self.nodes_path, asdict(node))

    def _write_edge(self, edge: CausalEdge) -> None:
        if not self.enabled:
            return
        self._append_jsonl(self.edges_path, asdict(edge))

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "params_hash": self.params_hash,
            "ranks": self._comm_size(),
            "logger_schema_version": self.schema_version,
            "agent_type_map": self.agent_type_map,
            "rule_map": rule_map(),
            "thresholds": {
                "stored_once": True,
                "description": "Full configs are stored in initialization logs; runtime edges keep rule_id and compact predicates."
            },
            "notes": [
                "Runtime causal logs intentionally omit positions and raw perceptions.",
                "Full initial positions and configurations are stored by InitializationLogger.",
                "Initial AlphaSynuclein state nodes are written at tick 0 so non-reactive proteins remain visible in G0 analyses.",
                "Aggregate nodes carry the current aggregate state and member count in value, enabling LewyBody size summaries.",
                "MPI ranks write rank-local JSONL files first; rank 0 merges them at close to keep JSONL rows atomic."
            ],
        }
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _merge_rank_files(self, pattern: str, destination: Path) -> None:
        """Concatenate rank-local JSONL files into one analysis-friendly file."""
        with destination.open("w", encoding="utf-8") as output:
            for path in sorted(self.output_dir.glob(pattern), key=_rank_file_sort_key):
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            output.write(line if line.endswith("\n") else line + "\n")

    def _phase_order_valid(self, phase_from: str, phase_to: str) -> bool:
        return PHASE_INDEX[phase_from] <= PHASE_INDEX[phase_to]

    def _barrier(self) -> None:
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier):
            barrier()

    def _comm_size(self) -> int:
        get_size = getattr(self.comm, "Get_size", None)
        if callable(get_size):
            return get_size()
        return 1

    def _hash_params(self, params: Optional[dict[str, Any]]) -> Optional[str]:
        if params is None:
            return None
        payload = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bind_causal_logger(agent, model):
    """Attach the model logger to an agent when available."""

    logger = causal_logger_from(model)
    try:
        agent.causal_logger = logger
    except Exception:
        pass
    return logger


def causal_logger_from(source) -> Optional[CausalTraceLogger]:
    """Return a CausalTraceLogger from a model, agent or logger object."""

    if source is None:
        return None
    logger = getattr(source, "causal_logger", None)
    if logger is not None:
        return logger
    return source if isinstance(source, CausalTraceLogger) else None


def uid_of(agent) -> Optional[str]:
    """Return a stable string uid for Repast and test agents."""

    if agent is None:
        return None
    uid = getattr(agent, "uid", None)
    if uid is not None:
        if isinstance(uid, tuple):
            return ":".join(str(item) for item in uid)
        return str(uid)
    local_id = getattr(agent, "local_id", getattr(agent, "id", ""))
    ptype = getattr(agent, "ptype", getattr(agent, "type_id", ""))
    rank = getattr(agent, "rank", "")
    return f"{local_id}:{ptype}:{rank}"


def short_uid(agent) -> str:
    """Return the local-id prefix of an agent uid."""

    uid = uid_of(agent)
    if not uid:
        return ""
    return uid.split(":")[0]


def type_name(agent) -> Optional[str]:
    """Return the concrete class name of an agent-like object."""

    if agent is None:
        return None
    return type(agent).__name__


def value_of(value) -> Optional[str]:
    """Return enum values and other objects as stable strings."""

    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def effect_sign(value: float) -> Optional[str]:
    """Return a sign label for a numeric effect."""

    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _rank_file_sort_key(path: Path) -> tuple[int, str]:
    """Sort rank-local log files by numeric rank when possible."""
    stem = path.stem
    marker = "_rank"
    if marker not in stem:
        return (0, path.name)
    suffix = stem.split(marker, 1)[1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (0, path.name)


def rule_map() -> dict[str, dict[str, str]]:
    """Return human-readable descriptions for logged rule ids."""

    return {
        "MICROGLIA_ACTIVATION_INFLAMMATION_HIGH": {
            "description": "Microglia can become Activated with probability proportional to inflammation pressure between its low and high thresholds.",
            "source": "Microglia.next"
        },
        "MICROGLIA_ACTIVATION_ALPHA_HIGH": {
            "description": "Microglia can become Activated with probability proportional to nearby alpha pressure between its low and high thresholds.",
            "source": "Microglia.next"
        },
        "MICROGLIA_CLEARING_DEBRIS_PRESSURE": {
            "description": "Microglia can become Clearing with probability proportional to extracellular debris pressure between its low and high thresholds.",
            "source": "Microglia.next"
        },
        "ASTROCYTE_REACTIVE_STRESS_HIGH": {
            "description": "Astrocyte becomes Reactive when inflammation or debris exceeds high thresholds.",
            "source": "Astrocyte.next"
        },
        "NEURON_DAMAGE_ACCUMULATION": {
            "description": "Neuron damage state changes after external and internal stress are combined.",
            "source": "Neuron.next"
        },
        "ALPHA_MISFOLDING": {
            "description": "Alpha-synuclein misfolds from basal, oxidative-stress, or aggregate-seeded pressure when the stochastic draw succeeds.",
            "source": "AlphaSynuclein.next"
        },
        "ALPHA_AGGREGATION": {
            "description": "AggregateRegistry creates or grows alpha-synuclein aggregates.",
            "source": "AggregateRegistry"
        },
        "ALPHA_OLIGOMERIZATION_INTENTION": {
            "description": "Misfolded alpha-synuclein becomes willing to oligomerize after a successful stochastic check.",
            "source": "AlphaSynuclein.next"
        },
        "AGGREGATE_MATURES_LEWY_BODY": {
            "description": "An oligomer aggregate matures into a Lewy body.",
            "source": "AggregateRegistry.mature_to_lewy_body"
        },
        "LYSOSOME_TARGET_ASSIGNMENT": {
            "description": "Lysosome claims a registered degradation target.",
            "source": "Lysosome._select_target"
        },
        "LYSOSOME_DEGRADATION": {
            "description": "Lysosome attempts to degrade or repair its assigned target.",
            "source": "Lysosome._resolve_degradation_attempt"
        },
        "NEURON_REGISTER_DEGRADATION_TARGET": {
            "description": "Neuron exposes an intracellular agent to lysosomal targeting.",
            "source": "Neuron.register_degradation_target"
        }}
