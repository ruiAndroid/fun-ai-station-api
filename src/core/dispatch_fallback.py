from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

RunItem = Callable[[Dict[str, Any], int, int, List[Dict[str, Any]]], Awaitable[Dict[str, Any]]]
ShouldSkip = Callable[[Dict[str, Any], int, int], bool]


def _clean_agent_code(item: Dict[str, Any]) -> str:
    code = item.get("agent")
    return code.strip() if isinstance(code, str) else ""


def _clean_depends_on(depends_on: Any, *, known_agents: set, agent_code: str) -> List[str]:
    if not isinstance(depends_on, list):
        return []
    out: List[str] = []
    for raw in depends_on:
        if not isinstance(raw, str):
            continue
        dep = raw.strip()
        if not dep or dep == agent_code or dep not in known_agents:
            continue
        if dep in out:
            continue
        out.append(dep)
    return out


async def run_plan_with_dependencies(
    *,
    plan: Sequence[Dict[str, Any]],
    max_parallel: int,
    run_item: RunItem,
    should_skip: Optional[ShouldSkip] = None,
) -> List[Dict[str, Any]]:
    total = len(plan)
    if total == 0:
        return []

    max_parallel = max(1, int(max_parallel or 1))
    results: List[Optional[Dict[str, Any]]] = [None] * total

    known_agents = {_clean_agent_code(item) for item in plan if _clean_agent_code(item)}
    pending: List[int] = []
    completed_codes = set()

    for idx, item in enumerate(plan):
        code = _clean_agent_code(item)
        if not code:
            continue
        if should_skip and should_skip(item, idx, total):
            completed_codes.add(code)
            continue
        pending.append(idx)

    if not pending:
        return [r for r in results if r]

    while pending:
        ready: List[int] = []
        for idx in pending:
            item = plan[idx]
            code = _clean_agent_code(item)
            deps = _clean_depends_on(item.get("depends_on"), known_agents=known_agents, agent_code=code)
            if all(dep in completed_codes for dep in deps):
                ready.append(idx)

        if not ready:
            ready = [pending[0]]

        batch = ready[:max_parallel]
        prior_results = [r for r in results if r]
        tasks = [asyncio.create_task(run_item(plan[idx], idx, total, prior_results)) for idx in batch]
        done = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, res in zip(batch, done):
            if isinstance(res, Exception):
                raise res
            if res is None:
                continue
            results[idx] = res
            code = _clean_agent_code(res) or _clean_agent_code(plan[idx])
            if code:
                completed_codes.add(code)

        pending = [i for i in pending if i not in batch]

    return [r for r in results if r]
