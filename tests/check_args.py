"""Every `args.X` in the entrypoints must be a declared argument.

Written after a rename left `args.final_pass_mode` and `args.retrieval_mode`
behind: argparse still built, so `--help` exited 0 and looked like proof, and
both C jobs died on the dead attribute after loading the model and the index.
`--help` checks the parser, not the code that reads it.

    uv run python tests/check_args.py
"""

from __future__ import annotations

import ast
import sys

# entrypoint -> the files that declare its arguments (itself included)
ENTRYPOINTS = {
    "src/agent/run_inference.py": ["src/agent/run_inference.py"],
    "src/vlm/run_inference.py": ["src/vlm/arg_parser.py", "src/vlm/run_inference.py"],
}


def declared(path: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                names.add(str(arg.value)[2:].replace("-", "_"))
        for kw in node.keywords:
            if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                names.add(str(kw.value.value))
    return names


def used(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in ast.walk(ast.parse(open(path).read())):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"):
            out.setdefault(node.attr, node.lineno)
    return out


def check_call_kwargs(path: str) -> int:
    """Every keyword at a call site must exist in the callee's signature."""
    tree = ast.parse(open(path).read())
    sigs = {n.name: n.args for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    bad = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        a = sigs.get(node.func.id)
        if a is None:
            continue
        if a.kwarg:
            continue
        names = {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
        for kw in node.keywords:
            if kw.arg and kw.arg not in names:
                print(f"{path}:{node.lineno}: {node.func.id}() has no parameter "
                      f"{kw.arg!r}")
                bad += 1
    return bad


def main() -> int:
    failed = False
    for entry, sources in ENTRYPOINTS.items():
        names = set().union(*(declared(s) for s in sources))
        missing = {k: v for k, v in used(entry).items() if k not in names}
        print(f"{entry}: {len(names)} declared, {len(used(entry))} read"
              f"{'' if not missing else '  -- ' + str(len(missing)) + ' DEAD'}")
        for name, line in sorted(missing.items(), key=lambda kv: kv[1]):
            print(f"  {entry}:{line}  args.{name} is not a declared argument")
            failed = True
        for source in set(sources) | {entry}:
            if check_call_kwargs(source):
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
