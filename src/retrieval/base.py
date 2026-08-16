from pathlib import Path
import sys

# Setup path - get project root dynamically
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/retrieval/
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from src.retrieval.models import (RetrievalRequest, RetrievalResult, RetrievalBatchRequest,RetrievalBatchResult )

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, request: RetrievalRequest)-> RetrievalResult:

        """ retrive using request"""
        pass

    @abstractmethod
    def retrieve_batch(self, request: RetrievalBatchRequest)-> RetrievalBatchResult:

        """ retrive using request"""
        pass
