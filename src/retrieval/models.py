from dataclasses import dataclass
import typing
from enum import Enum
from typing import Any


class RetrievalStrategy(Enum):

    DENSE="dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class RetrievalRequest:
    
    query:str
    top_k:int
    strategy:RetrievalStrategy
    score_threshold: float|None=None
    filters:dict[str,Any] |None=None

@dataclass
class RetrievalBatchRequest:
    
    query:list[str]
    top_k:int
    strategy:RetrievalStrategy
    score_threshold: float|None=None
    filters:dict[str,Any] |None=None

@dataclass
class RetrievedDocument:
    document_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any]

    retrieval_rank: int
    final_rank: int

    distance: float | None = None
    reranker_score: float | None = None
    
@dataclass
class SearchCandidate:
    document_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]

    retrieval_rank: int | None = None
    final_rank: int | None = None

    distance: float | None = None
    reranker_score: float | None = None



@dataclass
class RetrievalResult:

    query:str
    strategy:RetrievalStrategy
    retrieved_documents:list[RetrievedDocument]
    retrieval_time:float

@dataclass
class RetrievalBatchResult:

    query:str
    strategy:RetrievalStrategy
    retrieved_documents:list[RetrievedDocument]
    retrieval_time:float


