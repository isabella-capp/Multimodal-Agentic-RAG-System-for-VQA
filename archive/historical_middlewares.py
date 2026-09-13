"""Middlewares removed in ad8c6dc, kept because the paper cites what they did.

Recovered 2026-09-13 from the snapshot runs/20260906-173214-score-names, the
most recent job that still used them. Not imported by anything: this file
exists so the runs/ snapshots that hold the only other copies can be deleted.
See REPRODUCE.md, section on the Qwen2.5-VL history.
"""

def force_first_tool() -> Any:
    """Forces the LLM to call a tool on its very first conversational turn."""
    @wrap_model_call
    def force_first_middleware(request: Any, handler: Callable) -> Any:
        if not any(isinstance(m, ToolMessage) for m in reversed(request.messages)):
            request = request.override(tool_choice="required")
        return handler(request)

    return force_first_middleware


def remind_original_question(original_question: str, with_format: bool = True) -> Any:
    """Re-inject the question AND the answer format before the final answer.

    The format block sits at the end of the system prompt, tens of thousands of
    retrieved tokens back by the time the agent answers, and the agent ignores
    it: every C run so far averaged 7-18 words per answer against 2.5 for the
    same format in baseline B. That is not cosmetic — BEM pays for length, so B
    on the short format (0.359) and C at fifteen words (0.384) were never
    measured in the same regime. Repeating the format after each tool result is
    what puts them back in one.
    """
    @wrap_model_call
    def remind_question_middleware(request, handler):
        if request.messages and isinstance(request.messages[-1], ToolMessage):
            text = (f"Reminder: keep your final answer strictly focused on the "
                    f"original question: '{original_question}'")
            # With --final-pass the agent's own answer is thrown away and the
            # pipeline regenerates it, so telling the agent how to format an
            # answer only nudges it to stop retrieving and produce one.
            if with_format:
                text += f"\n\n{ANSWER_FORMAT}"
            reminder = SystemMessage(content=text)
            request = request.override(messages=request.messages + [reminder])
        return handler(request)

    return remind_question_middleware

def require_tool_before_answer(tool_name: str, max_calls: int = 3) -> Any:
    """Do not let the agent answer until it has called ``tool_name`` once.

    The same lever as `require_distinct_names`, for the same reason: the agent
    is told when a tool would help and does not act on it — a `verify` tool
    reported NOT CONFIRMED on 64.5% of examples and the agent changed course on
    0.2% of those. Forcing the call separates "the tool does not help" from "the
    agent does not use it", which are very different conclusions.
    """
    @wrap_model_call
    def middleware(request, handler):
        if should_force(request.messages, tool_name):
            return handler(request.override(tool_choice=tool_name))
        return handler(request)

    return middleware


