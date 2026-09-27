"""
llm_client.py
-------------
Thin wrapper around the Anthropic API. Centralizes every LLM call the agent
makes: planning, deciding the next tool call, reflecting on a result, and
writing the final report. Keeping this in one place makes it trivial to
swap models or providers later.
"""

from __future__ import annotations
import json
import os
from typing import Any, Optional

import anthropic

MODEL = "claude-sonnet-4-6"


class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY as an environment variable (see .env.example)."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    # -- 1. Planning ---------------------------------------------------

    def plan(self, goal: str, available_tools: list[str]) -> list[str]:
        """Ask the LLM to break the goal into an ordered list of tool names."""
        system = (
            "You are the planning module of a financial data analysis agent. "
            f"Available tools, in the order they would typically run: {available_tools}. "
            "Given the user's goal, output ONLY a JSON array of the tool names "
            "(a subset of the available tools, in the order they should run) "
            "needed to accomplish it. No prose, no markdown fences."
        )
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
        )
        text = resp.content[0].text.strip()
        return _safe_json_list(text, fallback=available_tools)

    # -- 2. Decide tool input for a given step --------------------------

    def decide_tool_input(
        self, goal: str, tool_name: str, tool_schema: dict, state_summary: str
    ) -> dict:
        """Ask the LLM for the JSON input arguments to call `tool_name` with,
        given everything learned so far."""
        system = (
            "You are the execution module of a financial data analysis agent. "
            "Given the goal, the tool about to be called, its input schema, and "
            "a summary of the run so far, output ONLY a JSON object with the "
            "arguments to call the tool with. No prose, no markdown fences."
        )
        user = (
            f"Goal: {goal}\n"
            f"Tool: {tool_name}\n"
            f"Schema: {json.dumps(tool_schema)}\n"
            f"Run so far:\n{state_summary}\n"
        )
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip()
        return _safe_json_obj(text, fallback={})

    # -- 3. Reflect on a step's result -----------------------------------

    def reflect(self, goal: str, tool_name: str, tool_output: Any) -> tuple[bool, str]:
        """Ask the LLM whether this step's result actually helps the goal,
        and for a one-line note. Returns (success, note)."""
        system = (
            "You are the reflection module of an agent. Given the goal, the "
            "tool that just ran, and its output, respond with ONLY a JSON "
            'object like {"success": true/false, "note": "one short sentence"}. '
            "success=false only if the output is clearly broken or empty."
        )
        user = f"Goal: {goal}\nTool: {tool_name}\nOutput: {json.dumps(tool_output)[:2000]}"
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=150,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip()
        parsed = _safe_json_obj(text, fallback={"success": True, "note": ""})
        return bool(parsed.get("success", True)), str(parsed.get("note", ""))

    # -- 4. Final report ---------------------------------------------------

    def write_report(self, goal: str, trace: str) -> str:
        """Synthesize the full run trace into a final, readable markdown report."""
        system = (
            "You are the reporting module of a financial data analysis agent. "
            "Given the user's original goal and a full trace of the steps the "
            "agent executed (with tool outputs), write a clear, well-structured "
            "markdown report that directly answers the goal. Use headings and "
            "bullet points. Do not just repeat the raw trace — synthesize it "
            "into insights and, where relevant, concrete recommendations."
        )
        user = f"Goal: {goal}\n\nFull trace:\n{trace}"
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Small parsing helpers — LLMs occasionally wrap JSON in prose/fences even
# when told not to, so we defensively strip and fall back gracefully.
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _safe_json_list(text: str, fallback: list[str]) -> list[str]:
    try:
        result = json.loads(_strip_fences(text))
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return fallback


def _safe_json_obj(text: str, fallback: dict) -> dict:
    try:
        result = json.loads(_strip_fences(text))
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return fallback
