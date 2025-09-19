from pydantic import BaseModel
from typing import List, Dict, Optional 

class Question(BaseModel):
    id: str
    question: str
    answers: List[str]
    metadata: Optional[Dict]

class Document(BaseModel):
    id: str
    text: str
    questions: List[Question]
    metadata: Optional[Dict]


class Dataset(BaseModel):
    documents: List[Document]
