"""
Benchmark experiment for evaluating research assistant tool usage against tiered questions.

The experiment replays pre-defined question cases (or executes real agent runs when
integrated) and produces normalized metrics about tool usage efficiency, coverage,
and answer quality. It is designed to support future integration with live agents
while remaining runnable with deterministic fixture data for CI baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.registry import register_experiment

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass
class ToolCall:
    """Normalized representation of a single tool invocation."""

    name: str
    relevant: bool = True
    latency_s: Optional[float] = None


@register_experiment(
    "research_assistant_tool_usage",
    summary="Evaluate research assistant tool orchestration across tiered question cases.",
    tags=["agent", "tools", "evaluation"],
)
class ResearchAssistantToolUsageExperiment(Experiment):
    """
    Prototype experiment that evaluates a research assistant's behaviour on curated question tiers.

    The experiment expects either inline case definitions via `cases` or a `cases_path` pointing
    to a JSON/YAML file. Each case should minimally include:

    ```json
    {
      "id": "tier0_q1",
      "tier": "tier_0",
      "question": "Where is the project README located?",
      "expected_tools": ["repo_search"],
      "expected_answer": "README.md in repository root",
      "simulated_trace": {
        "answer": "README.md in repository root",
        "answer_correct": true,
        "tools_used": [
          {"name": "repo_search", "relevant": true, "latency_s": 0.8}
        ],
        "latency_total": 0.8,
        "metrics": {"tool_latency_mean": 0.8}
      }
    }
    ```

    When integrated with a live agent, the `simulated_trace` key can be omitted and replaced by a
    custom runner supplied via `agent_adapter`. The adapter should implement a callable that accepts
    `(case: Dict[str, Any], context: ExperimentContext)` and returns a trace dictionary with the
    same structure as shown above.
    """

    param_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Inline case definitions evaluated sequentially.",
            },
            "cases_path": {
                "type": "string",
                "description": "Path to JSON/YAML file containing case definitions.",
            },
            "agent_adapter": {
                "type": "object",
                "description": "Future hook for live agent execution (module:function spec).",
            },
            "emit_tool_trace": {
                "type": "boolean",
                "default": True,
                "description": "Include normalized tool trace in the result payload.",
            },
        },
        "anyOf": [
            {"required": ["cases"]},
            {"required": ["cases_path"]},
        ],
    }

    def __init__(
        self,
        cases: Optional[Sequence[Dict[str, Any]]] = None,
        cases_path: Optional[str] = None,
        agent_adapter: Optional[Dict[str, Any]] = None,
        emit_tool_trace: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parameters or {})

        # Persist core configuration
        self.params.update(
            {
                "cases": list(cases) if cases is not None else None,
                "cases_path": cases_path,
                "agent_adapter": agent_adapter,
                "emit_tool_trace": emit_tool_trace,
            }
        )

        self._cases: List[Dict[str, Any]] = []
        self._case_index: int = 0
        self._emit_tool_trace = bool(emit_tool_trace)
        self._agent_runner = None  # Callable placeholder for future integration

    # --------------------------------------------------------------------- #
    # Experiment lifecycle
    # --------------------------------------------------------------------- #

    def setup(self, context: ExperimentContext) -> None:
        """Load case definitions and prepare optional agent runner."""
        self._cases = self._resolve_cases()
        self._case_index = 0

        adapter_cfg = self.params.get("agent_adapter")
        if adapter_cfg:
            self._agent_runner = self._import_adapter(adapter_cfg)

    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        """Execute the next case in the list and emit normalized metrics."""
        if not self._cases:
            raise RuntimeError("No cases available for research_assistant_tool_usage experiment.")

        case = self._cases[self._case_index % len(self._cases)]
        self._case_index += 1

        raw_trace = self._run_case(case, context)
        metrics, tool_trace = self._score_case(case, raw_trace)

        result: Dict[str, Any] = {
            "tier": case["tier"],
            "question_id": case["id"],
            "question": case.get("question"),
            "pass": bool(metrics.get("answer_correct", False))
            and not metrics.get("unused_expected_tools"),
            "metrics": metrics,
        }

        if self._emit_tool_trace:
            result["tool_trace"] = [tool.__dict__ for tool in tool_trace]
        return result

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _resolve_cases(self) -> List[Dict[str, Any]]:
        inline_cases = self.params.get("cases")
        path_value = self.params.get("cases_path")

        cases: List[Dict[str, Any]] = []
        if inline_cases:
            cases.extend(self._normalize_case(entry) for entry in inline_cases)

        if path_value:
            path = Path(path_value)
            if not path.is_absolute():
                # Resolve relative to repository root (workspace)
                path = Path.cwd() / path
            if not path.exists():
                raise FileNotFoundError(f"cases_path does not exist: {path}")

            loaded = self._load_cases_from_file(path)
            cases.extend(self._normalize_case(entry) for entry in loaded)

        if not cases:
            raise ValueError("No case definitions supplied via `cases` or `cases_path`.")

        return cases

    def _load_cases_from_file(self, path: Path) -> Sequence[Dict[str, Any]]:
        if path.suffix.lower() in {".json"}:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML case files.")
            with path.open("r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh)
        else:
            raise ValueError(f"Unsupported cases file extension: {path.suffix}")

        if isinstance(payload, dict) and "cases" in payload:
            candidate = payload["cases"]
        else:
            candidate = payload

        if not isinstance(candidate, Sequence):
            raise ValueError("Cases file must contain a list of case objects.")
        return candidate  # type: ignore[return-value]

    def _normalize_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("id", "tier", "question"):
            if key not in case:
                raise ValueError(f"Case missing required key '{key}': {case}")

        normalized = dict(case)
        normalized.setdefault("expected_tools", [])
        normalized.setdefault("expected_answer", None)

        if not isinstance(normalized["expected_tools"], Sequence):
            raise ValueError(
                f"expected_tools must be a sequence for case '{normalized['id']}'"
            )

        return normalized

    def _import_adapter(self, adapter_cfg: Dict[str, Any]):
        """
        Dynamically import an adapter callable specified as {"module": "...", "callable": "..."}.
        """
        module_name = adapter_cfg.get("module")
        callable_name = adapter_cfg.get("callable")
        if not module_name or not callable_name:
            raise ValueError("agent_adapter requires 'module' and 'callable' keys.")

        module = __import__(module_name, fromlist=[callable_name])
        runner = getattr(module, callable_name, None)
        if runner is None or not callable(runner):
            raise ValueError(
                f"agent_adapter callable '{callable_name}' not found in module '{module_name}'."
            )
        return runner

    def _run_case(self, case: Dict[str, Any], context: ExperimentContext) -> Dict[str, Any]:
        if self._agent_runner:
            return self._agent_runner(case, context)

        simulated_trace = case.get("simulated_trace")
        if not simulated_trace:
            raise ValueError(
                f"Case '{case['id']}' is missing `simulated_trace` and no agent_adapter is configured."
            )
        if not isinstance(simulated_trace, dict):
            raise ValueError(
                f"`simulated_trace` must be a dict for case '{case['id']}'."
            )
        return simulated_trace

    def _score_case(
        self, case: Dict[str, Any], trace: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[ToolCall]]:
        tool_calls = self._extract_tool_calls(trace)

        expected_tools = set(case.get("expected_tools", []))
        used_tools = {call.name for call in tool_calls}

        unused_expected = sorted(expected_tools.difference(used_tools))
        unexpected_tools = sorted(used_tools.difference(expected_tools))

        answer_correct = self._determine_answer_correctness(case, trace)

        relevant_calls = sum(1 for call in tool_calls if call.relevant)
        irrelevant_calls = len(tool_calls) - relevant_calls

        total_latency = self._determine_total_latency(tool_calls, trace)
        latency_mean = total_latency / len(tool_calls) if tool_calls else 0.0

        metrics: Dict[str, Any] = {
            "tier": case["tier"],
            "question_id": case["id"],
            "answer": trace.get("answer"),
            "answer_correct": answer_correct,
            "expected_tool_count": len(expected_tools),
            "unused_expected_tools": unused_expected,
            "unexpected_tools": unexpected_tools,
            "tool_calls_total": len(tool_calls),
            "relevant_tool_calls": relevant_calls,
            "irrelevant_call_rate": (
                irrelevant_calls / len(tool_calls) if tool_calls else 0.0
            ),
            "tool_latency_total_s": total_latency,
            "tool_latency_mean_s": latency_mean,
            "tool_efficiency": (
                relevant_calls / len(tool_calls) if tool_calls else 0.0
            ),
            "notes": trace.get("notes"),
        }

        additional_metrics = trace.get("metrics", {})
        if isinstance(additional_metrics, dict):
            metrics.update(additional_metrics)

        return metrics, tool_calls

    # ------------------------------------------------------------------ #
    # Utility extraction helpers
    # ------------------------------------------------------------------ #

    def _extract_tool_calls(self, trace: Dict[str, Any]) -> List[ToolCall]:
        raw_calls = trace.get("tool_calls") or trace.get("tools_used") or []
        if not isinstance(raw_calls, Iterable) or isinstance(raw_calls, (str, bytes)):
            raise ValueError("tool_calls/tools_used must be a list of call entries.")

        normalized: List[ToolCall] = []
        for entry in raw_calls:
            if isinstance(entry, ToolCall):
                normalized.append(entry)
                continue

            if isinstance(entry, str):
                normalized.append(ToolCall(name=entry))
                continue

            if not isinstance(entry, dict):
                raise ValueError(f"Unsupported tool call representation: {entry}")

            name = entry.get("name") or entry.get("tool")
            if not name:
                raise ValueError(f"Tool call entry missing 'name': {entry}")

            normalized.append(
                ToolCall(
                    name=name,
                    relevant=bool(entry.get("relevant", True)),
                    latency_s=self._coerce_latency(entry),
                )
            )

        return normalized

    def _coerce_latency(self, entry: Dict[str, Any]) -> Optional[float]:
        latency = entry.get("latency_s")
        if latency is not None:
            try:
                return float(latency)
            except (TypeError, ValueError):
                pass
        return None

    def _determine_total_latency(
        self, tool_calls: Sequence[ToolCall], trace: Dict[str, Any]
    ) -> float:
        explicit = trace.get("latency_total") or trace.get("latency_total_s")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass

        return float(
            sum(call.latency_s for call in tool_calls if call.latency_s is not None)
        )

    def _determine_answer_correctness(
        self, case: Dict[str, Any], trace: Dict[str, Any]
    ) -> bool:
        if "answer_correct" in trace:
            return bool(trace["answer_correct"])

        expected = case.get("expected_answer")
        if expected is None:
            return False

        observed = trace.get("answer")
        if observed is None:
            return False

        return str(observed).strip().lower() == str(expected).strip().lower()

