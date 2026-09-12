from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from prompts import extract_answer


@dataclass
class AgentStep:
    """One tool call in the agent loop, paired with what came back."""

    order: int
    tool: str
    arguments: dict[str, Any]
    observation: str

    @property
    def miss(self) -> bool:
        """The tool found nothing — every such message starts "No …"/"Unknown …"."""
        return self.observation.startswith(("No ", "Unknown "))

    @property
    def num_paragraphs(self) -> int:
        return self.observation.count("[Paragraph ")

    @property
    def num_articles(self) -> int:
        """Article titles listed by a lookup/search step."""
        return 0 if self.miss else sum(
            1 for line in self.observation.splitlines() if line.startswith("- ")
        )

    def as_record(self) -> dict:
        return {
            "order": self.order,
            "tool": self.tool,
            "arguments": self.arguments,
            "miss": self.miss,
            "num_articles": self.num_articles,
            "num_paragraphs": self.num_paragraphs,
        }


@dataclass
class AgentRun:
    """Result of running the agent on one example."""

    prediction: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    messages: list[Any] = field(default_factory=list)  # kept only for tracing
    gate_below: int = 0
    forced: int = 0

    @classmethod
    def from_messages(cls, messages: list) -> "AgentRun":
        """Pair each tool call with its observation, and take the last non-tool"""

        calls = {tc["id"]: (tc["name"], tc.get("args", {}))
                 for m in messages if isinstance(m, AIMessage)
                 for tc in m.tool_calls or []}

        steps = []
        for m in messages:
            if isinstance(m, ToolMessage):
                name, args = calls.get(m.tool_call_id, (m.name, {}))
                observation_text = m.content if isinstance(m.content, str) else str(m.content)
                steps.append(AgentStep(len(steps) + 1, name, args, observation_text))

        prediction = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not m.tool_calls:
                raw_content = ""
                if isinstance(m.content, str):
                    raw_content = m.content
                elif isinstance(m.content, list):
                    raw_content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in m.content
                    )
                else:
                    raw_content = str(m.content)

                prediction = extract_answer(raw_content)

                if prediction is not None:
                    prediction = str(prediction).strip()

                break

        return cls(prediction=prediction, steps=steps)

    @property
    def tool_called(self) -> bool:
        return bool(self.steps)

    @property
    def paragraphs_read(self) -> int:
        return sum(s.num_paragraphs for s in self.steps)

    @property
    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]
