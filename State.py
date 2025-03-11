from pydantic import BaseModel
from typing import List

class State(BaseModel):
    narrative: str = ""
    inventory: List[str] = []
    location: str = ""
    score: int = 0
    actions: List[str] = []