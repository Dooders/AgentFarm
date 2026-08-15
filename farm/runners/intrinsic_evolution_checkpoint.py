"""Mid-run checkpoints for intrinsic-evolution Spot / preemptible runs.

Persists a lean, reconstructible snapshot of evolutionary state so an
interrupted run can continue from the last durable step instead of restarting
from scratch.  Full ``Environment`` pickles are avoided (weakrefs + large
replay buffers); instead we store chromosomes, agent vitals, optional policy
weights, resources, and RNG bookkeeping.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from farm.core.hyperparameter_chromosome import HyperparameterChromosome
from farm.utils.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_VERSION = 1
CHECKPOINT_PAYLOAD_FILENAME = "intrinsic_evolution_checkpoint.pkl"
CHECKPOINT_META_FILENAME = "intrinsic_evolution_checkpoint.json"


@dataclass(frozen=True)
class CheckpointMeta:
    """Lightweight marker written beside the pickle payload."""

    version: int
    logical_step: int
    num_steps_configured: int
    total_sim_steps: int
    effective_warmup: int
    status: str = "partial"
    payload_file: str = CHECKPOINT_PAYLOAD_FILENAME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "logical_step": self.logical_step,
            "num_steps_configured": self.num_steps_configured,
            "total_sim_steps": self.total_sim_steps,
            "effective_warmup": self.effective_warmup,
            "status": self.status,
            "payload_file": self.payload_file,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMeta":
        return cls(
            version=int(data.get("version", CHECKPOINT_VERSION)),
            logical_step=int(data["logical_step"]),
            num_steps_configured=int(data["num_steps_configured"]),
            total_sim_steps=int(data["total_sim_steps"]),
            effective_warmup=int(data.get("effective_warmup", 0)),
            status=str(data.get("status", "partial")),
            payload_file=str(data.get("payload_file", CHECKPOINT_PAYLOAD_FILENAME)),
        )


def checkpoint_meta_path(output_dir: str) -> str:
    return os.path.join(output_dir, CHECKPOINT_META_FILENAME)


def checkpoint_payload_path(output_dir: str, meta: Optional[CheckpointMeta] = None) -> str:
    name = meta.payload_file if meta is not None else CHECKPOINT_PAYLOAD_FILENAME
    return os.path.join(output_dir, name)


def read_checkpoint_meta(output_dir: str) -> Optional[CheckpointMeta]:
    path = checkpoint_meta_path(output_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return CheckpointMeta.from_dict(json.load(handle))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def has_resumable_checkpoint(output_dir: str, num_steps: int) -> bool:
    """True when a partial checkpoint exists with progress below ``num_steps``.

    ``num_steps`` is the post-warmup configured horizon (CLI ``--num-steps``).
    Checkpoint ``logical_step`` counts from simulation start (includes warmup).
    """
    meta = read_checkpoint_meta(output_dir)
    if meta is None or meta.status != "partial":
        return False
    if meta.version != CHECKPOINT_VERSION:
        return False
    if not os.path.isfile(checkpoint_payload_path(output_dir, meta)):
        return False
    post_warmup = max(0, meta.logical_step - meta.effective_warmup)
    return post_warmup < int(num_steps)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(path, raw)


def _serialize_resource(resource: Any) -> Dict[str, Any]:
    return {
        "resource_id": getattr(resource, "resource_id", None),
        "position": tuple(resource.position),
        "amount": float(resource.amount),
        "max_amount": float(getattr(resource, "max_amount", resource.amount)),
        "regeneration_rate": float(getattr(resource, "regeneration_rate", 0.1)),
    }


def _serialize_agent(agent: Any) -> Dict[str, Any]:
    chromosome = getattr(agent, "hyperparameter_chromosome", None)
    chromosome_payload = chromosome.to_dict() if chromosome is not None else None

    model_state = None
    decision_module = getattr(getattr(agent, "behavior", None), "decision_module", None)
    if decision_module is not None:
        algorithm = getattr(decision_module, "algorithm", None)
        getter = getattr(algorithm, "get_model_state", None) if algorithm is not None else None
        if callable(getter):
            try:
                model_state = getter()
            except Exception as exc:
                logger.warning(
                    "checkpoint_model_state_export_failed",
                    agent_id=getattr(agent, "agent_id", None),
                    error=str(exc),
                )

    resource_comp = getattr(agent, "resource_component", None) or getattr(agent, "resource", None)
    # Prefer public resource level accessors used across AgentCore.
    resources_value = getattr(agent, "resource_level", None)
    if resources_value is None and resource_comp is not None:
        resources_value = getattr(resource_comp, "resource_level", None)
    if resources_value is None:
        resources_value = getattr(agent, "resources", 0.0)

    health_value = getattr(agent, "current_health", None)
    if health_value is None:
        health_value = getattr(agent, "health", None)

    return {
        "agent_id": str(agent.agent_id),
        "agent_type": str(getattr(agent, "agent_type", "AgentCore")),
        "position": tuple(agent.position),
        "resources": float(resources_value or 0.0),
        "health": float(health_value) if health_value is not None else None,
        "generation": int(getattr(agent, "generation", 0) or 0),
        "genome_id": getattr(agent, "genome_id", None),
        "parent_ids": list(getattr(agent, "parent_ids", []) or []),
        "alive": bool(getattr(agent, "alive", True)),
        "chromosome": chromosome_payload,
        "model_state": model_state,
        "decision_is_trained": bool(getattr(decision_module, "_is_trained", False))
        if decision_module is not None
        else False,
    }


def _capture_rng_states(policy_rng: Optional[random.Random]) -> Dict[str, Any]:
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "policy_rng": policy_rng.getstate() if policy_rng is not None else None,
    }
    try:
        import numpy as np

        states["numpy"] = np.random.get_state()
    except Exception:
        states["numpy"] = None
    try:
        import torch

        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        states["torch"] = None
        states["torch_cuda"] = None
    return states


def _restore_rng_states(states: Dict[str, Any], policy_rng: Optional[random.Random]) -> None:
    py_state = states.get("python")
    if py_state is not None:
        random.setstate(py_state)
    if policy_rng is not None and states.get("policy_rng") is not None:
        policy_rng.setstate(states["policy_rng"])
    try:
        import numpy as np

        if states.get("numpy") is not None:
            np.random.set_state(states["numpy"])
    except Exception:
        pass
    try:
        import torch

        if states.get("torch") is not None:
            torch.set_rng_state(states["torch"])
        if states.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])
    except Exception:
        pass


def build_checkpoint_payload(
    *,
    environment: Any,
    logical_step: int,
    num_steps_configured: int,
    total_sim_steps: int,
    effective_warmup: int,
    simulation_id: str,
    policy_rng: Optional[random.Random],
    gene_logger_state: Optional[Dict[str, Any]] = None,
    transient_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble an in-memory checkpoint dict from the live environment."""
    alive_agents = [a for a in environment.alive_agent_objects if getattr(a, "alive", True)]
    return {
        "version": CHECKPOINT_VERSION,
        "logical_step": int(logical_step),
        "num_steps_configured": int(num_steps_configured),
        "total_sim_steps": int(total_sim_steps),
        "effective_warmup": int(effective_warmup),
        "simulation_id": simulation_id,
        "environment_time": int(getattr(environment, "time", logical_step)),
        "agents": [_serialize_agent(agent) for agent in alive_agents],
        "resources": [_serialize_resource(resource) for resource in list(environment.resources)],
        "rng": _capture_rng_states(policy_rng),
        "gene_logger_state": gene_logger_state or {},
        "transient_state": transient_state or {},
        "inheritance_telemetry": (
            environment.inheritance_telemetry.to_dict()
            if getattr(environment, "inheritance_telemetry", None) is not None
            and hasattr(environment.inheritance_telemetry, "to_dict")
            else None
        ),
    }


def save_checkpoint(output_dir: str, payload: Dict[str, Any]) -> CheckpointMeta:
    """Atomically write checkpoint payload + JSON meta marker."""
    meta = CheckpointMeta(
        version=int(payload.get("version", CHECKPOINT_VERSION)),
        logical_step=int(payload["logical_step"]),
        num_steps_configured=int(payload["num_steps_configured"]),
        total_sim_steps=int(payload["total_sim_steps"]),
        effective_warmup=int(payload.get("effective_warmup", 0)),
    )
    os.makedirs(output_dir, exist_ok=True)
    payload_path = checkpoint_payload_path(output_dir, meta)
    _atomic_write_bytes(payload_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    _atomic_write_json(checkpoint_meta_path(output_dir), meta.to_dict())
    logger.info(
        "intrinsic_evolution_checkpoint_saved",
        output_dir=output_dir,
        logical_step=meta.logical_step,
        agents=len(payload.get("agents") or []),
    )
    return meta


def load_checkpoint_payload(output_dir: str) -> Optional[Dict[str, Any]]:
    meta = read_checkpoint_meta(output_dir)
    if meta is None:
        return None
    path = checkpoint_payload_path(output_dir, meta)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
            return None
        return payload
    except (OSError, pickle.PickleError, TypeError, ValueError, KeyError):
        return None


def clear_checkpoint(output_dir: str) -> None:
    """Remove checkpoint artifacts after a successful full run."""
    meta = read_checkpoint_meta(output_dir)
    paths = [checkpoint_meta_path(output_dir)]
    if meta is not None:
        paths.append(checkpoint_payload_path(output_dir, meta))
    else:
        paths.append(checkpoint_payload_path(output_dir))
    for path in paths:
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


def restore_environment_from_checkpoint(
    payload: Dict[str, Any],
    *,
    config: Any,
    path: Optional[str],
    policy: Any,
    policy_rng: random.Random,
) -> Any:
    """Rebuild a runnable ``Environment`` from a checkpoint payload.

    Creates a fresh environment (new DB handle), restores resources and agents
    with saved chromosomes / optional policy weights, and rebinds spatial
    indexes.  Replay buffers are not restored.
    """
    from farm.core.agent.config.component_configs import AgentComponentConfig
    from farm.core.agent.factory import AgentFactory
    from farm.core.environment import Environment
    from farm.core.resources import Resource
    from farm.core.simulation import create_services_from_environment
    from farm.core.inheritance_telemetry import InheritanceTelemetry
    from farm.database.database import InMemorySimulationDatabase

    simulation_id = str(payload.get("simulation_id") or "intrinsic-resume")
    use_memory = bool(getattr(getattr(config, "database", None), "use_in_memory_db", False))
    db_path = ""
    if path is not None and not use_memory:
        os.makedirs(path, exist_ok=True)
        db_path = os.path.join(path, f"simulation_{simulation_id}_resume.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass

    environment = Environment(
        width=config.environment.width,
        height=config.environment.height,
        resource_distribution={"type": "random", "amount": 0},
        db_path=db_path if db_path else "simulation_resume.db",
        config=config,
        simulation_id=simulation_id,
        seed=config.seed,
    )

    if use_memory:
        if environment.db is not None:
            try:
                environment.db.close()
            except Exception:
                pass
        environment.db = InMemorySimulationDatabase(
            memory_limit_mb=getattr(config.database, "in_memory_db_memory_limit_mb", None),
            simulation_id=simulation_id,
        )
        try:
            environment.db.add_simulation_record(
                simulation_id=simulation_id,
                start_time=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ),
                status="running",
                parameters=config.to_dict(),
            )
            environment.db.logger.flush_all_buffers()
        except Exception:
            pass

    # Replace default resources with checkpointed nodes.
    restored_resources: List[Resource] = []
    for index, raw in enumerate(payload.get("resources") or []):
        restored_resources.append(
            Resource(
                resource_id=raw.get("resource_id", index),
                position=tuple(raw["position"]),
                amount=float(raw["amount"]),
                max_amount=float(raw.get("max_amount", raw["amount"])),
                regeneration_rate=float(raw.get("regeneration_rate", 0.1)),
            )
        )
    environment.resources = restored_resources
    if getattr(environment, "resource_manager", None) is not None:
        environment.resource_manager.resources = restored_resources

    environment.intrinsic_evolution_policy = policy
    environment.intrinsic_evolution_rng = policy_rng
    telemetry = InheritanceTelemetry()
    telemetry_raw = payload.get("inheritance_telemetry") or {}
    if isinstance(telemetry_raw, dict):
        telemetry.warmstart_applied = int(telemetry_raw.get("warmstart_applied", 0) or 0)
        telemetry.warmstart_skipped = int(telemetry_raw.get("warmstart_skipped", 0) or 0)
        skipped_reasons = telemetry_raw.get("warmstart_skipped_reasons") or {}
        if isinstance(skipped_reasons, dict):
            telemetry.warmstart_skipped_reasons.update(
                {str(k): int(v) for k, v in skipped_reasons.items()}
            )
        if telemetry_raw.get("blend_alpha") is not None:
            telemetry.blend_alpha = float(telemetry_raw["blend_alpha"])
        telemetry.decide_action_failures = int(
            telemetry_raw.get("decide_action_failures", 0) or 0
        )
        failure_reasons = telemetry_raw.get("decide_action_failure_reasons") or {}
        if isinstance(failure_reasons, dict):
            telemetry.decide_action_failure_reasons.update(
                {str(k): int(v) for k, v in failure_reasons.items()}
            )
    environment.inheritance_telemetry = telemetry

    services = create_services_from_environment(environment)
    factory = AgentFactory(services)
    agent_config = AgentComponentConfig.from_simulation_config(config)

    for raw_agent in payload.get("agents") or []:
        if not raw_agent.get("alive", True):
            continue
        agent = factory.create_learning_agent(
            agent_id=str(raw_agent["agent_id"]),
            position=tuple(raw_agent["position"]),
            initial_resources=float(raw_agent.get("resources", 0.0)),
            config=agent_config,
            environment=environment,
            agent_type=str(raw_agent.get("agent_type", "AgentCore")),
        )
        agent.generation = int(raw_agent.get("generation", 0) or 0)
        # genome_id / parent_ids are typically read-only properties on AgentCore;
        # skip restore when no writable backing field is available.
        agent.resource_level = float(raw_agent.get("resources", 0.0))
        if raw_agent.get("health") is not None:
            combat_comp = agent.get_component("combat")
            if combat_comp is not None and hasattr(combat_comp, "health"):
                combat_comp.health = float(raw_agent["health"])

        chromosome_data = raw_agent.get("chromosome")
        if chromosome_data is not None:
            agent.hyperparameter_chromosome = HyperparameterChromosome.from_dict(chromosome_data)

        decision_module = getattr(getattr(agent, "behavior", None), "decision_module", None)
        model_state = raw_agent.get("model_state")
        if decision_module is not None and model_state is not None:
            algorithm = getattr(decision_module, "algorithm", None)
            loader = getattr(algorithm, "load_model_state", None) if algorithm is not None else None
            if callable(loader):
                try:
                    loader(model_state)
                    decision_module._is_trained = bool(raw_agent.get("decision_is_trained", True))
                except Exception as exc:
                    logger.warning(
                        "checkpoint_model_state_load_failed",
                        agent_id=agent.agent_id,
                        error=str(exc),
                    )

        environment.add_agent(agent, flush_immediately=False, defer_spatial_update=True)

    # Rebuild spatial indexes / alive sets after bulk add.
    if getattr(environment, "spatial_index", None) is not None:
        environment.spatial_index.set_references(
            list(environment._agent_objects.values()),
            environment.resources,
        )
        environment.spatial_index.update()
    environment._alive_agents = {
        a.agent_id for a in environment._agent_objects.values() if getattr(a, "alive", True)
    }
    environment.agents = [
        a.agent_id for a in environment._agent_objects.values() if a.agent_id in environment._alive_agents
    ]

    environment.time = int(payload.get("environment_time", payload.get("logical_step", 0)))
    _restore_rng_states(payload.get("rng") or {}, policy_rng)

    # Detach DB after restore. Checkpoint fidelity is chromosomes / vitals /
    # policy weights + gene JSONL; a fresh incomplete SQLite schema on resume
    # is more trouble than it's worth for Spot preemption recovery.
    if environment.db is not None:
        try:
            environment.db.close()
        except Exception:
            pass
        environment.db = None

    logger.info(
        "intrinsic_evolution_checkpoint_restored",
        simulation_id=simulation_id,
        logical_step=payload.get("logical_step"),
        agents=len(environment.alive_agent_objects),
        resources=len(environment.resources),
    )
    return environment
