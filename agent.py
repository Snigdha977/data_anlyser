"""
agent.py
--------
The orchestrator. Implements the Plan -> Act -> Observe -> Respond loop:

  1. PLAN     : ask the LLM to turn the user's goal into an ordered list of
                tool calls.
  2. ACT      : for each planned tool, ask the LLM what arguments to call it
                with (given everything learned so far), then actually run
                the tool.
  3. OBSERVE  : ask the LLM to reflect on whether the result was useful; if
                a step clearly fails, retry once with a corrected input
                before moving on (basic self-correction, not just a fixed
                pipeline).
  4. RESPOND  : ask the LLM to synthesize the full trace into a final report.

This file intentionally has no framework dependency (no LangChain/CrewAI) —
the loop is ~80 lines of plain Python so it's easy to read, explain, and
defend as "your own" logic for the contest.
"""

from __future__ import annotations

from llm_client import LLMClient
from state import AgentState, StepRecord
from tools import TOOL_FUNCTIONS, TOOL_SPECS


class FinSightAgent:
    def __init__(self, llm: LLMClient | None = None, max_retries: int = 1):
        self.llm = llm or LLMClient()
        self.max_retries = max_retries
        self.tool_specs_by_name = {spec["name"]: spec for spec in TOOL_SPECS}

    def run(self, goal: str, data_path: str, verbose: bool = True) -> AgentState:
        state = AgentState(goal=goal, data_path=data_path)

        # --- 1. PLAN ---------------------------------------------------
        available = list(TOOL_FUNCTIONS.keys())
        state.plan = self.llm.plan(goal, available)
        # always make sure the data is loaded first if the plan forgot it
        if "load_data" not in state.plan:
            state.plan.insert(0, "load_data")
        if verbose:
            print(f"\n[PLAN] {' -> '.join(state.plan)}\n")

        # --- 2 & 3. ACT + OBSERVE, one planned tool at a time -----------
        for i, tool_name in enumerate(state.plan, start=1):
            if tool_name not in TOOL_FUNCTIONS:
                continue  # LLM hallucinated a tool name; skip safely

            attempt = 0
            success = False
            while attempt <= self.max_retries and not success:
                attempt += 1
                tool_input = self._decide_input(state, tool_name)
                tool_input.setdefault("file_path", data_path)

                if verbose:
                    print(f"[STEP {i}] {tool_name}({tool_input})")

                try:
                    output = TOOL_FUNCTIONS[tool_name](**tool_input)
                    error = None
                except Exception as exc:  # tool crashed -> treat as failed step
                    output = {"error": str(exc)}
                    error = str(exc)

                ok, note = self.llm.reflect(goal, tool_name, output)
                success = ok and error is None

                record = StepRecord(
                    step_number=i,
                    description=f"attempt {attempt}",
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=output,
                    reflection=note,
                    success=success,
                )
                state.add_step(record)

                if verbose:
                    tag = "OK" if success else "RETRY" if attempt <= self.max_retries else "FAILED"
                    print(f"          -> {tag}  ({note})\n")

        # --- 4. RESPOND ---------------------------------------------------
        state.final_report = self.llm.write_report(goal, state.trace())
        return state

    # -- helpers -------------------------------------------------------

    def _decide_input(self, state: AgentState, tool_name: str) -> dict:
        schema = self.tool_specs_by_name[tool_name]["input_schema"]
        summary = state.trace() if state.history else "(no steps run yet)"
        proposed = self.llm.decide_tool_input(state.goal, tool_name, schema, summary)
        if not isinstance(proposed, dict):
            proposed = {}
        return proposed
