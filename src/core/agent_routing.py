from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def _keyword_rules() -> Dict[str, List[str]]:
    return {
        "attendance": [
            "打卡",
            "签到",
            "签退",
            "下班",
            "请假",
            "加班",
            "考勤",
            "补卡",
            "迟到",
            "早退",
            "人事",
            "HR",
        ],
        "expense": [
            "报销",
            "发票",
            "费用",
            "差旅",
            "打车",
            "出差",
            "报账",
            "财务",
            "付款",
            "对公",
        ],
        "admin": [
            "行政",
            "维修",
            "报修",
            "坏了",
            "领用",
            "领取",
            "申领",
            "门禁",
            "工位",
            "会议室",
            "快递",
            "保洁",
            "网络",
            "打印机",
            "电脑",
        ],
    }


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
    rules = _keyword_rules()
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


def _split_clauses(text: str) -> List[str]:
    """
    Split a user utterance into rough clauses for multi-intent dispatch.

    Heuristics:
    - Prefer natural separators: 。！？；\n
    - Then split by commas and common conjunctions (然后/另外/同时/以及/并且/顺便/再/还要)
    """
    s = (text or "").strip()
    if not s:
        return []

    # Normalize whitespace
    s = " ".join(s.split())

    # Stage 1: strong punctuation
    parts: List[str] = []
    for p in re.split(r"[。！？!?\n；;]+", s):
        p = p.strip(" ,，。；;")
        if p:
            parts.append(p)

    # Stage 2: weaker separators + conjunctions
    clauses: List[str] = []
    for p in parts:
        # Keep the conjunction token as a splitter only; don't remove semantic words inside phrases like “再次”.
        sub = re.split(r"(?:，|,|、|\s+(?:然后|另外|同时|以及|并且|顺便|再|还要)\s+)", p)
        for x in sub:
            x = x.strip(" ,，。；;")
            if x:
                clauses.append(x)

    # De-dup empty/super short noise but keep intent-bearing short clauses.
    out: List[str] = []
    for c in clauses:
        if not c:
            continue
        out.append(c)

    return out or [s]


def _score_clause_for_agent(clause: str, agent_code: str) -> int:
    rules = _keyword_rules()
    kws = rules.get(agent_code) or []
    if not kws:
        return 0
    score = 0
    for kw in kws:
        if kw and kw in clause:
            score += 3
    # mild boosts for common patterns
    if agent_code == "attendance" and ("迟到" in clause or "早退" in clause):
        score += 2
    if agent_code == "expense" and ("金额" in clause or "票" in clause):
        score += 1
    if agent_code == "admin" and ("联系" in clause or "坏" in clause or "修" in clause):
        score += 1
    return score


def _assign_clauses_to_agents(
    *,
    clauses: List[str],
    agent_codes_in_order: List[str],
    default_agent_code: str,
    agent_lookup: Dict[str, AgentLike],
) -> List[DispatchItem]:
    """
    Map each clause to one agent and merge per-agent clauses while preserving order of first appearance.
    """
    if not clauses:
        return []

    picked_order: List[str] = []
    buckets: Dict[str, List[str]] = {}

    allowed_set = set(agent_codes_in_order)
    if default_agent_code and default_agent_code not in allowed_set and default_agent_code in agent_lookup:
        agent_codes_in_order = agent_codes_in_order + [default_agent_code]
        allowed_set.add(default_agent_code)

    for clause in clauses:
        best_code: Optional[str] = None
        best_score = -1
        for code in agent_codes_in_order:
            score = _score_clause_for_agent(clause, code)
            if score > best_score:
                best_score = score
                best_code = code
            elif score == best_score and best_code is not None:
                # keep earlier agent in agent_codes_in_order
                continue

        if not best_code or best_score <= 0:
            best_code = default_agent_code or (agent_codes_in_order[0] if agent_codes_in_order else "")

        if not best_code:
            continue
        if best_code not in buckets:
            buckets[best_code] = []
        buckets[best_code].append(clause)
        if best_code not in picked_order:
            picked_order.append(best_code)

    items: List[DispatchItem] = []
    for code in picked_order:
        a = agent_lookup.get(code)
        text = "；".join([x.strip() for x in buckets.get(code, []) if x.strip()]).strip()
        if not text:
            continue
        items.append(
            DispatchItem(
                agent_code=code,
                agent_name=(a.name if a else code),
                text=text,
            )
        )

    return items


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


async def _llm_route_items(
    *,
    text: str,
    agents: Sequence[AgentLike],
    default_agent_code: str,
    trace_id: str,
) -> List[DispatchItem]:
    """
    Ask an external LLM to choose an ordered list of agent codes AND provide per-agent subtask text.
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
        "你是一个路由器，负责根据用户输入把任务拆分，并分配给合适的智能体，给出执行顺序。\n"
        "你只能从下面提供的智能体 code 中选择，输出严格 JSON。\n"
        '输出格式（推荐）：{"items":[{"agent":"code1","text":"子任务1"},{"agent":"code2","text":"子任务2"}],"reason":"..."}\n'
        '兼容格式：{"agents":["code1","code2"],"reason":"..."}（如果无法拆分子任务）\n'
        "规则：\n"
        "- 如果用户明确 @了某个智能体，则无需路由（上游会处理）。\n"
        "- 如果用户包含多个意图，必须尽量返回多个 items，并且每个 item.text 只包含该智能体需要处理的那一段，不要把整段原文复制给每个智能体。\n"
        "- item.text 尽量保持用户原话的关键信息，并补足该智能体所需的上下文（如：时间/金额/地点/联系人等），但不要替其他智能体做事。\n"
        "- items 最多 3 个。\n"
        f"- 如果不确定，返回默认：{default_agent_code}。\n"
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

    # Preferred: items with per-agent task text. We still re-segment using local heuristics
    # to avoid "each agent gets the whole original text" which hurts response quality.
    raw_items = obj.get("items")
    if isinstance(raw_items, list):
        codes_in_order: List[str] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            agent = it.get("agent") or it.get("code") or it.get("name")
            agent = agent.strip() if isinstance(agent, str) else ""
            if not agent or agent not in allowed:
                continue
            if agent in codes_in_order:
                continue
            codes_in_order.append(agent)
            if len(codes_in_order) >= 3:
                break
        if codes_in_order:
            agent_lookup = {a.code: a for a in agents if a.code}
            clauses = _split_clauses(text)
            # Allow local keyword hits even if the LLM missed some agents, but keep LLM order as tiebreaker.
            return _assign_clauses_to_agents(
                clauses=clauses,
                agent_codes_in_order=(codes_in_order + [c for c in allowed if c not in codes_in_order]),
                default_agent_code=default_agent_code,
                agent_lookup=agent_lookup,
            )

    # Fallback: only agents list; we'll segment with heuristics later.
    raw_agents = obj.get("agents")
    picked: List[str] = []
    if isinstance(raw_agents, list):
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

    agent_lookup = {a.code: a for a in agents if a.code}
    clauses = _split_clauses(text)
    return _assign_clauses_to_agents(
        clauses=clauses,
        agent_codes_in_order=picked,
        default_agent_code=default_agent_code,
        agent_lookup=agent_lookup,
    )


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
        items = await _llm_route_items(
            text=text, agents=agents, default_agent_code=default_agent_code, trace_id=trace_id
        )
    except Exception:
        items = []
    if items:
        return items

    # 3) keyword routing fallback
    kw = _keyword_routes(text)
    if kw:
        codes2: List[str] = []
        for _idx, code in kw:
            if code not in codes2:
                codes2.append(code)
        agent_lookup = {a.code: a for a in agents if a.code}
        clauses = _split_clauses(text)
        items2 = _assign_clauses_to_agents(
            clauses=clauses,
            agent_codes_in_order=codes2,
            default_agent_code=default_agent_code,
            agent_lookup=agent_lookup,
        )
        if items2:
            return items2

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
            items = await _llm_route_items(
                text=text, agents=agents, default_agent_code=default_agent_code, trace_id=trace_id
            )
        except Exception:
            items = []
        if items:
            return items
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
            agent_lookup = {a.code: a for a in agents if a.code}
            clauses = _split_clauses(text)
            out2 = _assign_clauses_to_agents(
                clauses=clauses,
                agent_codes_in_order=codes2,
                default_agent_code=default_agent_code,
                agent_lookup=agent_lookup,
            )
            if out2:
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
        agent_lookup = {a.code: a for a in agents if a.code}
        clauses = _split_clauses(text)
        out = _assign_clauses_to_agents(
            clauses=clauses,
            agent_codes_in_order=codes,
            default_agent_code=default_agent_code,
            agent_lookup=agent_lookup,
        )
        if out:
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

