from __future__ import annotations

import time
from typing import Any

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message
from agent.prompts import PREVIEW_PROMPT, SYSTEM_PROMPT, UNIFIED_PROMPT
from agent.run import AgentRun
from agent.tools import build_tools
from prompts import RAG_PROMPT, extract_answer
from retrieval.bm25 import BM25Ranker
from retrieval.fusion import Ranking, rank_paragraphs


def open_text_gate(state: dict, tool_name: str, threshold: float) -> Any:
    """Force ``tool_name`` only where the pool looks like it lacks the answer."""
    @wrap_model_call
    def middleware(request, handler):
        score = state.get("top_score")
        if score is not None and score < threshold:
            state["gate_below"] = state.get("gate_below", 0) + 1
            if should_force(request.messages, tool_name):
                state["forced"] = state.get("forced", 0) + 1
                return handler(request.override(tool_choice=tool_name))
        return handler(request)

    return middleware


def should_force(messages, tool_name: str) -> bool:
    """True while the agent has read a tool result but never called ``tool_name``."""
    if not any(isinstance(m, ToolMessage) for m in messages):
        return False        # nothing read yet: too early to demand a second round
    called = sum(1 for m in messages if isinstance(m, AIMessage)
                 for tc in (m.tool_calls or []) if tc["name"] == tool_name)
    return called == 0


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time."""

    def __init__(self, llm, retriever, kb, reranker,
                 ranking: Ranking = Ranking(), top_k=20, max_iterations=8,
                 text_limit: int = 5, max_names: int = 4, lookup_limit: int = 3,
                 text_gate: float | None = None,
                 final_pass: bool = False, preview: int = 0,
                 tool_set: str = "minimal"):
        self.llm = llm
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.bm25 = BM25Ranker()
        self.ranking = ranking
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.text_limit = text_limit
        self.max_names = max_names
        self.lookup_limit = lookup_limit
        self.text_gate = text_gate
        self.final_pass = final_pass
        self.preview = preview
        self.tool_set = tool_set

    def _middleware(self, question: str, state: dict) -> list[Any]:
        """The one middleware left, and the only one that ever earned its place."""
        if self.text_gate is None:
            return []
        # the gate forces by name, so the name must exist in the installed set
        second = "search_paragraphs" if self.tool_set == "legacy" else "search"
        return [open_text_gate(state, second, self.text_gate)]

    def run(self, image_path: str, question: str) -> AgentRun:
        t0 = time.time()

        state: dict = {}   # per example: the tools write into it, the middleware reads

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return AgentRun(error=f"Image load error: {str(e)}")

        agent = create_agent(
            model=self.llm,
            tools=build_tools(self.retriever, self.kb, self.reranker, self.bm25,
                              image, top_k=self.top_k,
                              ranking=self.ranking,
                              text_limit=self.text_limit,
                              max_names=self.max_names,
                              lookup_limit=self.lookup_limit,
                              state=state, tool_set=self.tool_set,
                              preview=self.preview, question=question),
            # the prompt must name the tools that are actually installed
            system_prompt=(SYSTEM_PROMPT if self.tool_set == "legacy" else
                           PREVIEW_PROMPT if self.preview else UNIFIED_PROMPT),
            middleware=self._middleware(question, state),
        )

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path, question)]},
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(out["messages"])
            run.messages = out["messages"]
        except GraphRecursionError:
            run = AgentRun(error="recursion_limit")
        except Exception as e:
            run = AgentRun(error=str(e))

        if self.final_pass and not run.error:
            answer = self._answer_from_pool(image_path, question, state)
            if answer is not None:
                run.prediction = answer

        run.gate_below = state.get("gate_below", 0)
        run.forced = state.get("forced", 0)
        run.elapsed_seconds = round(time.time() - t0, 2)
        return run

    def _answer_from_pool(self, image_path: str, question: str, state: dict):
        """Answer from the pool the agent assembled, the way baseline B does."""
        candidates = state.get("candidates") or {}
        pooled = [p for c in candidates.values()
                  for p in self.kb.get_paragraphs_by_url(
                      wiki_url=c.wiki_url)]
        if not pooled:
            return None
        best = rank_paragraphs(question, pooled, strategy=self.ranking.final,
                               top_k=self.ranking.top_n, reranker=self.reranker,
                               bm25_ranker=self.bm25,
                               bm25_top_m=self.ranking.bm25_top_m,
                               rrf_k=self.ranking.rrf_k)
        if not best:
            return None
        prompt = RAG_PROMPT.format(context="\n\n".join(best), question=question)
        reply = self.llm.invoke([build_user_message(image_path, prompt)]).content
        return extract_answer(reply if isinstance(reply, str) else str(reply))
