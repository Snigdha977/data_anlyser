"""
state.py
--------
Holds all shared state for a single agent run: the original goal, the plan,
every tool call made, its result, and the LLM's reflection on it. This is
what lets later steps in the workflow "see" what earlier steps produced,
and what lets us print a full trace / build the final report.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class StepRecord:
    """One executed step in the workflow: what was planned, what tool ran,
    what it returned, and whether the agent judged the step successful."""
    step_number: int
    description: str
    tool_name: str
    tool_input: dict
    tool_output: Any
    reflection: Optional[str] = None
    success: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class AgentState:
    """Mutable context object threaded through the whole run."""
    goal: str
    data_path: Optional[str] = None
    plan: list[str] = field(default_factory=list)
    history: list[StepRecord] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)  # free-form working memory
    final_report: Optional[str] = None

    def add_step(self, record: StepRecord) -> None:
        self.history.append(record)

    def last_result(self) -> Any:
        return self.history[-1].tool_output if self.history else None

    def get_result(self, tool_name: str) -> Any:
        """Fetch the most recent output of a given tool, if any ran."""
        for record in reversed(self.history):
            if record.tool_name == tool_name:
                return record.tool_output
        return None

    def trace(self) -> str:
        """Human-readable trace of the whole run, used for debugging / demo."""
        lines = [f"GOAL: {self.goal}", f"PLAN: {' -> '.join(self.plan)}", ""]
        for r in self.history:
            status = "OK" if r.success else "FAILED"
            lines.append(f"[Step {r.step_number}] {r.tool_name} ({status})")
            lines.append(f"  input : {r.tool_input}")
            out_preview = str(r.tool_output)
            if len(out_preview) > 300:
                out_preview = out_preview[:300] + " ...(truncated)"
            lines.append(f"  output: {out_preview}")
            if r.reflection:
                lines.append(f"  reflection: {r.reflection}")
            lines.append("")
        return "\n".join(lines)
