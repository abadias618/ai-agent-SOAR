from pydantic import BaseModel

class State(BaseModel):
    next_action: str = 