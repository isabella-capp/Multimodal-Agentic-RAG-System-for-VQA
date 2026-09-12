"""Tool descriptions are model-facing: a change here invalidates every measurement."""
import json
import pathlib
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("faiss", MagicMock())

from agent.tools import build_tools          # noqa: E402
from retrieval.fusion import Ranking         # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "tool_contract.json"


def render() -> dict:
    out = {}
    for tool_set in ("minimal", "legacy"):
        tools = build_tools(kb=MagicMock(), retriever=MagicMock(), reranker=MagicMock(),
                            bm25=MagicMock(), ranking=Ranking(), image=MagicMock(),
                            question="q", tool_set=tool_set)
        out[tool_set] = [{"name": t.name, "description": t.description, "schema": t.args}
                         for t in tools]
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def main() -> int:
    want = json.loads(FIXTURE.read_text())
    got = render()
    if got == want:
        n = sum(len(v) for v in got.values())
        print(f"  tool contract unchanged ({n} tools)")
        return 0
    for ts in sorted(set(want) | set(got)):
        for a, b in zip(want.get(ts, []), got.get(ts, [])):
            if a != b:
                print(f"  CHANGED [{ts}] {a.get('name')}")
                if a.get("description") != b.get("description"):
                    print(f"    - {a.get('description')!r}")
                    print(f"    + {b.get('description')!r}")
                if a.get("schema") != b.get("schema"):
                    print(f"    schema: {a.get('schema')} -> {b.get('schema')}")
    print("  Tool text reaches the model. If this change is intended, rerun the "
          "affected experiments and refresh tests/fixtures/tool_contract.json.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
