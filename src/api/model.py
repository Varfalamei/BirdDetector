from pydantic import BaseModel
from typing import Dict, List, Literal, Optional


class PostGetRequestFromModel(BaseModel):
    message: str
