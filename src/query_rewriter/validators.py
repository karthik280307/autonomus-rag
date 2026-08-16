from pathlib import Path
import sys

# Setup path - get project root dynamically
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/query_rewriter/
sys.path.insert(0, str(project_root))

from src.query_rewriter.models import QueryRewriteRequest, RewrittenQueryResult


class QueryValidator:
    """Validation utilities for query rewriting components."""
    
    @staticmethod
    def validate_query_request(request: QueryRewriteRequest) -> bool:
        """
        Validate a QueryRewriteRequest.
        
        Args:
            request: QueryRewriteRequest to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(request, QueryRewriteRequest):
            raise ValueError("Request must be a QueryRewriteRequest instance")
        
        if not request.query or not isinstance(request.query, str):
            raise ValueError("Query must be a non-empty string")
        
        if request.query.strip() == "":
            raise ValueError("Query cannot be empty or whitespace")
        
        if request.history is not None and not isinstance(request.history, list):
            raise ValueError("History must be a list or None")
        
        return True
    
    @staticmethod
    def validate_rewritten_result(result: RewrittenQueryResult) -> bool:
        """
        Validate a RewrittenQueryResult.
        
        Args:
            result: RewrittenQueryResult to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(result, RewrittenQueryResult):
            raise ValueError("Result must be a RewrittenQueryResult instance")
        
        if not result.original_query or not isinstance(result.original_query, str):
            raise ValueError("Original query must be a non-empty string")
        
        if not result.reformulated_query or not isinstance(result.reformulated_query, str):
            raise ValueError("Reformulated query must be a non-empty string")
        
        if result.expanded_queries is not None:
            if not isinstance(result.expanded_queries, list):
                raise ValueError("Expanded queries must be a list or None")
            if result.expanded_queries and not all(isinstance(q, str) for q in result.expanded_queries):
                raise ValueError("All expanded queries must be strings")
        
        if result.step_back_query is not None and not isinstance(result.step_back_query, str):
            raise ValueError("Step-back query must be a string or None")
        
        return True
    
    @staticmethod
    def validate_expanded_queries(queries: list[str]) -> bool:
        """
        Validate a list of expanded queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(queries, list):
            raise ValueError("Queries must be a list")
        
        if len(queries) == 0:
            raise ValueError("Queries list cannot be empty")
        
        if not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("All queries must be non-empty strings")
        
        return True
    
    @staticmethod
    def validate_step_back_query(query: str) -> bool:
        """
        Validate a step-back query.
        
        Args:
            query: Step-back query string
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(query, str):
            raise ValueError("Step-back query must be a string")
        
        if not query or not query.strip():
            raise ValueError("Step-back query cannot be empty")
        
        return True