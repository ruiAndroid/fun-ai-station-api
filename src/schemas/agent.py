from typing import List

from pydantic import BaseModel


class AgentOut(BaseModel):
    id: int
    code: str
    name: str
    handle: str
    description: str
    tags: List[str]
    capabilities: List[str]

