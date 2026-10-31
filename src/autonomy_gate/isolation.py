"""Prove a module can't reach network or send code, by walking its imports.

``assert_isolated("myagent.policy")`` parses the module's source, follows every import
that stays inside the same top-level package, and fails if any module on the way imports
something on the forbidden list (network, processes, LLM SDKs, or modules you name, such
as your sender). The error shows the import chain.

It finds imports by parsing source, not by watching what a module does when it runs. It
locates modules with :func:`importlib.util.find_spec`, though, which imports the parent
package of each name it looks up, so the packages on the walk do get imported; point it
only at code you would import anyway. It follows imports written anywhere in a module,
including inside functions. It does not follow the implicit import of a parent package's
``__init__``, and it does not look inside the standard library or third-party packages:
anything it doesn't follow is judged by its name.
"""

from __future__ import annotations

import ast
import importlib.util
from collections import deque
from collections.abc import Iterable

NETWORK_AND_SEND = (
    "socket",
    "ssl",
    "http",
    "urllib",
    "urllib3",
    "smtplib",
    "ftplib",
    "imaplib",
    "poplib",
    "telnetlib",
    "xmlrpc",
    "asyncio",
    "subprocess",
    "multiprocessing",
    "ctypes",
    "requests",
    "httpx",
    "aiohttp",
    "websocket",
    "websockets",
    "grpc",
    "openai",
    "anthropic",
)


class IsolationError(AssertionError):
    """A module can reach something it must not."""


def _matches(name: str, forbidden: Iterable[str]) -> str | None:
    for f in forbidden:
        if name == f or name.startswith(f + "."):
            return f
    return None


def _spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None


def _imports(module: str) -> list[str]:
    spec = _spec(module)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return []
    with open(spec.origin, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=spec.origin)
    is_package = spec.submodule_search_locations is not None
    package = module if is_package else module.rpartition(".")[0]
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base_parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(base_parts)
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            names.append(target)
            for alias in node.names:
                sub = f"{target}.{alias.name}"
                if _spec(sub) is not None:
                    names.append(sub)
    return names


def reachable(module: str) -> dict[str, list[str]]:
    """Every module reachable from ``module`` inside its top-level package, with the chain
    of imports that reaches it. Imports that leave the package are listed but not followed."""
    root = module.split(".")[0]
    chains: dict[str, list[str]] = {module: [module]}
    queue = deque([module])
    while queue:
        current = queue.popleft()
        for name in _imports(current):
            if name in chains:
                continue
            chains[name] = [*chains[current], name]
            if name.split(".")[0] == root:
                queue.append(name)
    return chains


def assert_isolated(
    module: str, *, forbidden: Iterable[str] = NETWORK_AND_SEND, also: Iterable[str] = ()
) -> None:
    """Raise :class:`IsolationError` if ``module`` can reach a forbidden module."""
    if _spec(module) is None:
        raise IsolationError(f"can't find module {module!r}")
    banned = (*forbidden, *also)
    hits = []
    for name, chain in reachable(module).items():
        if name == module:
            continue
        hit = _matches(name, banned)
        if hit:
            hits.append(f"{' -> '.join(chain)} (forbidden: {hit})")
    if hits:
        raise IsolationError(f"{module} can reach:\n  " + "\n  ".join(sorted(hits)))
