from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import httpx

from src.core.config import get_settings


@dataclass(frozen=True)
class AgentLike:
    """
    Minimal shape needed for routing.
    Works with SQLAlchemy Agent model via attribute access.
    """

    code: str
    name: str
    handle: str
    description: str = ""


@dataclass(frozen=True)
class DispatchItem:
    agent_code: str
    agent_name: str
    text: str


def _agent_aliases(a: AgentLike) -> List[str]:
    aliases: List[str] = []
    if a.handle:
        aliases.append(a.handle)
    # also allow english handle form: @code
    if a.code:
        aliases.append(f"@{a.code}")
    # allow @ + display name
    if a.name:
        aliases.append(f"@{a.name}")
    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for x in aliases:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _find_mention_hits(text: str, agents: Sequence[AgentLike]) -> List[Tuple[int, int, AgentLike]]:
    hits: List[Tuple[int, int, AgentLike]] = []
    for a in agents:
        for token in _agent_aliases(a):
            start = 0
            while start < len(text):
                idx = text.find(token, start)
                if idx < 0:
                    break
                hits.append((idx, idx + len(token), a))
                start = idx + len(token)

    # sort by appearance; for same start prefer longer token
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # remove overlaps
    out: List[Tuple[int, int, AgentLike]] = []
    last_end = -1
    for s, e, a in hits:
        if s < last_end:
            continue
        out.append((s, e, a))
        last_end = e
    return out


def _remove_mentions(text: str, hits: List[Tuple[int, int, AgentLike]]) -> str:
    if not hits:
        return text.strip()
    out = []
    last = 0
    for s, e, _a in hits:
        out.append(text[last:s])
        last = e
    out.append(text[last:])
    cleaned = "".join(out)
    return " ".join(cleaned.split()).strip()


def _keyword_routes(text: str) -> List[Tuple[int, str]]:
    """
    Simple keyword-based routing. Returns (first_hit_index, agent_code).
    If no keyword hit, returns [].
    """
    rules = {
        "attendance": ["打卡", "签到", "签退", "下班", "请假", "加班", "考勤", "补卡"],
        "expense": ["报销", "发票", "费用", "差旅", "打车", "出差", "报账"],
        "admin": ["维修", "报修", "坏了", "领用", "领取", "申领", "门禁", "工位", "会议室", "快递"],
    }
    hits: List[Tuple[int, str]] = []
    for code, kws in rules.items():
        best: Optional[int] = None
        for kw in kws:
            idx = text.find(kw)
            if idx >= 0 and (best is None or idx < best):
                best = idx
        if best is not None:
            hits.append((best, code))
    hits.sort(key=lambda x: x[0])
    return hits


def _extract_json_obj(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # try to locate a JSON object in the string
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


async def _llm_route_agents(
    *,
    text: str,
    agents: Sequence[AgentLike],
    default_agent_code: str,
    trace_id: str,
) -> List[str]:
    """
    Ask an external LLM to choose an ordered list of agent codes to handle the message.
    Uses the same LLM config as fun-agent-service via GET {FUN_AGENT_SERVICE_URL}/config/llm.
    """
    settings = get_settings()
    service_base = (settings.FUN_AGENT_SERVICE_URL or "").rstrip("/")
    if not service_base:
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            cfg_resp = await client.get(
                f"{service_base}/config/llm",
                headers={"x-trace-id": trace_id} if trace_id else None,
            )
            cfg = cfg_resp.json() if cfg_resp.status_code < 400 else {}
        except Exception:
            cfg = {}

    base_url = str((cfg or {}).get("base_url") or "").rstrip("/")
    api_key = str((cfg or {}).get("api_key") or "")
    model = str((cfg or {}).get("model") or "gpt-4o-mini")
    timeout = int((cfg or {}).get("timeout") or 30)

    if not base_url or not api_key:
        return []

    allowed = {a.code for a in agents if a.code}
    if not allowed:
        return []

    # Build compact agent list for prompt
    agent_lines: List[str] = []
    for a in agents:
        desc = (a.description or "").strip()
        if len(desc) > 80:
            desc = desc[:79] + "…"
        agent_lines.append(f"- code={a.code} name={a.name} handle={a.handle} desc={desc}")

    system = (
        "你是一个路由器，负责根据用户输入选择应该调用哪些智能体，以及调用顺序。\n"
        "你只能从下面提供的智能体 code 中选择，输出严格 JSON。\n"
        '输出格式：{"agents":["code1","code2"],"reason":"..."}\n'
        "规则：\n"
        "- 如果用户明确 @了某个智能体，则无需路由（上游会处理）。\n"
        "- 如果用户包含多个意图，可以返回多个 agents，顺序代表处理顺序。\n"
        f"- 如果不确定，返回默认：{default_agent_code}\n"
        "可选智能体如下：\n"
        + "\n".join(agent_lines)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
    }

    url = f"{base_url}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
    if resp.status_code >= 400:
        return []
    try:
        data = resp.json()
    except Exception:
        return []

    content = (
        (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content")  # type: ignore[union-attr]
        if isinstance(data, dict)
        else None
    )
    if not isinstance(content, str):
        return []

    obj = _extract_json_obj(content)
    if not obj:
        return []
    raw_agents = obj.get("agents")
    if not isinstance(raw_agents, list):
        return []

    picked: List[str] = []
    for x in raw_agents:
        if not isinstance(x, str):
            continue
        code = x.strip()
        if not code or code not in allowed:
            continue
        if code in picked:
            continue
        picked.append(code)
        if len(picked) >= 3:
            break

    if not picked and default_agent_code in allowed:
        picked = [default_agent_code]
    return picked


async def build_dispatch_plan_llm(
    *,
    text: str,
    agents: Sequence[AgentLike],
    default_agent_code: str,
    trace_id: str,
) -> List[DispatchItem]:
    """
    LLM-first routing (explicit mentions still win).
    Falls back to keyword routing if LLM unavailable.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) explicit mentions
    mhits = _find_mention_hits(text, agents)
    if mhits:
        cleaned = _remove_mentions(text, mhits)
        items: List[DispatchItem] = []
        for i, (_s, e, a) in enumerate(mhits):
            next_s = mhits[i + 1][0] if i + 1 < len(mhits) else len(text)
            seg = text[e:next_s].strip()
            items.append(
                DispatchItem(
                    agent_code=a.code,
                    agent_name=a.name or a.code,
                    text=seg or cleaned or text,
                )
            )
        return items

    # 2) LLM routing
    try:
        codes = await _llm_route_agents(
            text=text, agents=agents, default_agent_code=default_agent_code, trace_id=trace_id
        )
    except Exception:
        codes = []
    if codes:
        out: List[DispatchItem] = []
        for code in codes:
            a = next((x for x in agents if x.code == code), None)
            out.append(DispatchItem(agent_code=code, agent_name=(a.name if a else code), text=text))
        return out

    # 3) keyword routing fallback
    kw = _keyword_routes(text)
    if kw:
        codes2: List[str] = []
        for _idx, code in kw:
            if code not in codes2:
                codes2.append(code)
        out2: List[DispatchItem] = []
        for code in codes2:
            a = next((x for x in agents if x.code == code), None)
            out2.append(DispatchItem(agent_code=code, agent_name=(a.name if a else code), text=text))
        return out2

    # 4) default fallback
    a = next((x for x in agents if x.code == default_agent_code), None)
    return [
        DispatchItem(
            agent_code=default_agent_code,
            agent_name=(a.name if a else default_agent_code),
            text=text,
        )
    ]


async def build_dispatch_plan_auto(
    *,
    text: str,
    agents: Sequence[AgentLike],
    default_agent_code: str,
    trace_id: str,
    mode: str,
) -> List[DispatchItem]:
    """
    Routing entrypoint with mode switch.
    """
    m = (mode or "hybrid").strip().lower()
    if m == "llm":
        # mentions > llm > default (no keyword fallback)
        text = (text or "").strip()
        if not text:
            return []
        mhits = _find_mention_hits(text, agents)
        if mhits:
            cleaned = _remove_mentions(text, mhits)
            items: List[DispatchItem] = []
            for i, (_s, e, a) in enumerate(mhits):
                next_s = mhits[i + 1][0] if i + 1 < len(mhits) else len(text)
                seg = text[e:next_s].strip()
                items.append(
                    DispatchItem(
                        agent_code=a.code,
                        agent_name=a.name or a.code,
                        text=seg or cleaned or text,
                    )
                )
            return items
        try:
            codes = await _llm_route_agents(
                text=text, agents=agents, default_agent_code=default_agent_code, trace_id=trace_id
            )
        except Exception:
            codes = []
        if codes:
            out: List[DispatchItem] = []
            for code in codes:
                a = next((x for x in agents if x.code == code), None)
                out.append(DispatchItem(agent_code=code, agent_name=(a.name if a else code), text=text))
            return out
        a = next((x for x in agents if x.code == default_agent_code), None)
        return [DispatchItem(agent_code=default_agent_code, agent_name=(a.name if a else default_agent_code), text=text)]

    if m == "keywords":
        # mentions > keywords > default
        text = (text or "").strip()
        if not text:
            return []
        mhits = _find_mention_hits(text, agents)
        if mhits:
            cleaned = _remove_mentions(text, mhits)
            items2: List[DispatchItem] = []
            for i, (_s, e, a) in enumerate(mhits):
                next_s = mhits[i + 1][0] if i + 1 < len(mhits) else len(text)
                seg = text[e:next_s].strip()
                items2.append(
                    DispatchItem(
                        agent_code=a.code,
                        agent_name=a.name or a.code,
                        text=seg or cleaned or text,
                    )
                )
            return items2
        kw = _keyword_routes(text)
        if kw:
            codes2: List[str] = []
            for _idx, code in kw:
                if code not in codes2:
                    codes2.append(code)
            out2: List[DispatchItem] = []
            for code in codes2:
                a = next((x for x in agents if x.code == code), None)
                out2.append(DispatchItem(agent_code=code, agent_name=(a.name if a else code), text=text))
            return out2
        a = next((x for x in agents if x.code == default_agent_code), None)
        return [DispatchItem(agent_code=default_agent_code, agent_name=(a.name if a else default_agent_code), text=text)]

    # hybrid (default)
    return await build_dispatch_plan_llm(
        text=text, agents=agents, default_agent_code=default_agent_code, trace_id=trace_id
    )


def build_dispatch_plan(
    *,
    text: str,
    agents: Sequence[AgentLike],
    default_agent_code: str,
) -> List[DispatchItem]:
    """
    Build sequential dispatch plan for a single user message.

    Priority:
    - If explicit @mentions exist: dispatch in mention order; each agent handles its segment.
    - Else: use keyword routing; dispatch in keyword order.
    - Else: fallback to default agent.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) explicit mentions
    mhits = _find_mention_hits(text, agents)
    if mhits:
        cleaned = _remove_mentions(text, mhits)
        items: List[DispatchItem] = []
        for i, (s, e, a) in enumerate(mhits):
            next_s = mhits[i + 1][0] if i + 1 < len(mhits) else len(text)
            seg = text[e:next_s].strip()
            items.append(
                DispatchItem(
                    agent_code=a.code,
                    agent_name=a.name or a.code,
                    text=seg or cleaned or text,
                )
            )
        return items

    # 2) keyword routing
    kw = _keyword_routes(text)
    if kw:
        codes: List[str] = []
        for _idx, code in kw:
            if code not in codes:
                codes.append(code)
        out: List[DispatchItem] = []
        for code in codes:
            a = next((x for x in agents if x.code == code), None)
            out.append(DispatchItem(agent_code=code, agent_name=(a.name if a else code), text=text))
        return out

    # 3) fallback
    a = next((x for x in agents if x.code == default_agent_code), None)
    return [
        DispatchItem(
            agent_code=default_agent_code,
            agent_name=(a.name if a else default_agent_code),
            text=text,
        )
    ]

