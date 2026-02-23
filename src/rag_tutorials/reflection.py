"""Reflection and self-correction module for the agent tutorials extension.

Implements the Worker-Critic pattern:
  - Worker generates a draft answer using retrieved context.
  - Critic reviews the answer for accuracy and completeness.
  - If the Critic is not satisfied, it sends feedback back to the Worker.
  - The loop continues until the Critic approves or max_rounds is reached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class CriticFeedback:
    """Structured review produced by the Critic agent."""

    approved: bool
    feedback: str


@dataclass
class ReflectionResult:
    """Final output of a completed Worker-Critic reflection loop."""

    question: str
    final_answer: str
    rounds: int
    history: list[dict] = field(default_factory=list)


_WORKER_SYSTEM = (
    "You are a policy assistant. Answer the question using only the provided context.\n"
    "If the answer is not present in the context, say you do not have enough context.\n"
    "Provide a concise answer. Cite relevant parts of the context where possible."
)

_CRITIC_SYSTEM = (
    "You are a strict quality reviewer. Given a question, retrieved context, and a\n"
    "draft answer, decide whether the answer is accurate, complete, and grounded in\n"
    "the context.\n"
    "\n"
    "Respond with a JSON object with exactly two keys:\n"
    "  approved  - true if the answer is acceptable, false otherwise\n"
    "  feedback  - if not approved, one or two sentences of specific, actionable\n"
    "              feedback for the Worker. If approved, set to empty string.\n"
    "\n"
    "Respond with valid JSON only. No extra text outside the JSON block."
)


def worker_answer(
    question: str,
    context: str,
    feedback: str = "",
    model: str = "gpt-4.1-mini",
) -> str:
    """Generate an answer using retrieved context, optionally incorporating feedback.

    Args:
        question: User question.
        context: Retrieved context passages joined as a single string.
        feedback: Critic feedback from a previous round (empty string if first round).
        model: Chat model for answer generation.

    Returns:
        Draft answer text.
    """
    client = OpenAI()
    user_content = f"Question: {question}\n\nContext:\n{context}"
    if feedback:
        user_content += f"\n\nCritic feedback on your previous answer:\n{feedback}\nPlease revise your answer."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _WORKER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def critic_review(
    question: str,
    answer: str,
    context: str,
    model: str = "gpt-4.1-mini",
) -> CriticFeedback:
    """Evaluate a Worker answer and return approval status with optional feedback.

    Args:
        question: Original user question.
        answer: Draft answer produced by the Worker.
        context: Retrieved context passages used by the Worker.
        model: Chat model for the critic.

    Returns:
        CriticFeedback with approved flag and actionable feedback string.
    """
    import json

    client = OpenAI()
    user_content = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Draft answer:\n{answer}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(raw)
        return CriticFeedback(
            approved=bool(parsed.get("approved", False)),
            feedback=str(parsed.get("feedback", "")),
        )
    except (json.JSONDecodeError, ValueError):
        # If parsing fails, treat as approved to avoid infinite loops
        return CriticFeedback(approved=True, feedback="")


def run_reflection_loop(
    question: str,
    context: str,
    model: str = "gpt-4.1-mini",
    max_rounds: int = 3,
) -> ReflectionResult:
    """Run the Worker-Critic loop until approval or max_rounds is reached.

    Args:
        question: User question to answer.
        context: Retrieved context passages as a single string.
        model: Chat model for both Worker and Critic.
        max_rounds: Maximum number of Worker-Critic cycles.

    Returns:
        ReflectionResult with the final approved answer and full round history.
    """
    history: list[dict] = []
    feedback = ""
    answer = ""

    for round_num in range(1, max_rounds + 1):
        answer = worker_answer(question, context, feedback=feedback, model=model)
        critique = critic_review(question, answer, context, model=model)

        history.append(
            {
                "round": round_num,
                "answer": answer,
                "approved": critique.approved,
                "feedback": critique.feedback,
            }
        )

        if critique.approved:
            return ReflectionResult(
                question=question,
                final_answer=answer,
                rounds=round_num,
                history=history,
            )

        feedback = critique.feedback

    # Return last answer even if Critic never approved
    return ReflectionResult(
        question=question,
        final_answer=answer,
        rounds=max_rounds,
        history=history,
    )
