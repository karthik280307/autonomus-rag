from abc import ABC, abstractmethod
from pathlib import Path
import sys

# Setup path - get project root dynamically
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/query_rewriter/
sys.path.insert(0, str(project_root))

from src.query_rewriter.models import QueryRewriteRequest

class BaseQueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, request: QueryRewriteRequest):
        """
        Rewrite the input query.
        """
        pass

