"""ReAct (Reason + Act) agent loop for the agent tutorials extension.

This module implements the core ReAct pattern:
  Think -> Act (call a tool) -> Observe -> Repeat until done.

The agent is connected to an arbitrary set of named tools. Each tool is a
callable that accepts a plain-string input and returns a plain-string output.
The RAG retrieval functions from earlier tutorials are the canonical example of
a tool, but any callable with that signature works.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class AgentStep:
    """Single Thought-Action-Observation cycle inside a ReAct loop."""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """Final result produced by a completed ReAct loop run."""

    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)


def _build_system_prompt(tool_names: list[str]) -> str:
    tool_list = ", ".join(tool_names)
    return (
        "You are a helpful assistant that solves problems step by step.\n"
        "You have access to these tools: " + tool_list + ".\n"
        "\n"
        "At each turn output a JSON object with exactly these keys:\n"
        "  thought       - your reasoning about what to do next\n"
        "  action        - the name of the tool to call, or 'finish'\n"
        "  action_input  - the input to pass to the tool (plain string)\n"
        "\n"
        "When you have a final answer, set action to 'finish' and put the\n"
        "answer in action_input.\n"
        "\n"
        "Respond with valid JSON only. No extra text outside the JSON block."
    )


def _parse_agent_response(text: str) -> dict:
    """Extract the JSON object from an LLM response string."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def run_react_loop(
    question: str,
    tools: dict[str, object],
    model: str = "gpt-4.1-mini",
    max_steps: int = 5,
) -> AgentResult:
    """Run the ReAct agent loop for a given question.

    Args:
        question: The user question to answer.
        tools: Mapping of tool name to callable (str -> str).
        model: OpenAI chat model used for the agent.
        max_steps: Maximum number of Thought-Action-Observation cycles.

    Returns:
        AgentResult with the final answer and full step trace.
    """
    client = OpenAI()
    tool_names = list(tools.keys())
    messages = [
        {"role": "system", "content": _build_system_prompt(tool_names)},
        {"role": "user", "content": f"Question: {question}"},
    ]

    steps: list[AgentStep] = []

    for _ in range(max_steps):
        response = client.chat.completions.create(model=model, messages=messages)
        raw = response.choices[0].message.content or ""

        try:
            parsed = _parse_agent_response(raw)
        except (json.JSONDecodeError, ValueError):
            # Treat malformed output as a finish with the raw text as answer
            return AgentResult(question=question, answer=raw.strip(), steps=steps)

        thought = parsed.get("thought", "")
        action = parsed.get("action", "finish")
        action_input = parsed.get("action_input", "")

        if action == "finish":
            return AgentResult(question=question, answer=action_input, steps=steps)

        # Execute the chosen tool
        tool_fn = tools.get(action)
        if tool_fn is None:
            observation = f"Unknown tool '{action}'. Available: {tool_names}"
        else:
            try:
                observation = str(tool_fn(action_input))
            except Exception as exc:  # noqa: BLE001
                observation = f"Tool error: {exc}"

        step = AgentStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        steps.append(step)

        # Add the agent turn and tool result to the conversation
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {"role": "user", "content": f"Observation: {observation}\nContinue."}
        )

    # If max_steps reached without a finish, return last observation as answer
    last_obs = steps[-1].observation if steps else "No answer produced."
    return AgentResult(question=question, answer=last_obs, steps=steps)
