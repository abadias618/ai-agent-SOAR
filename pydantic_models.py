from pydantic import BaseModel, Field
from typing import List

class GetAnswerAsList(BaseModel):
    '''Returns the response in the form of a list (For list prompts).'''
    response: List[str]

class GetAnswerAsString(BaseModel):
    '''Returns response as a single String'''
    response: str