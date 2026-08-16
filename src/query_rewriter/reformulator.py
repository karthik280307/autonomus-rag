from pathlib import Path
import sys
import logging

# Setup path - get project root dynamically
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/query_rewriter/
sys.path.insert(0, str(project_root))
from typing import Any

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    AIMessage
)
from langchain_core.language_models import BaseLLM

from src.query_rewriter.base import BaseQueryRewriter
from src.query_rewriter.models import (
    ChatMessage,
    QueryRewriteRequest,
    ReformulationResult
)

from src.query_rewriter.prompts import REFORMULATION_SYSTEM_PROMPT
from src.query_rewriter.utils import (llm_util, message_util)
from src.query_rewriter.validators import QueryValidator

logger = logging.getLogger(__name__)

class Reformulator(BaseQueryRewriter):
    """Reformulate user queries into clear, standalone search queries."""

    def __init__(self, llm: BaseLLM):
        """
        Initialize Reformulator.
        
        Args:
            llm: Language model instance
        """
        if llm is None:
            raise ValueError("LLM instance is required for Reformulator")
        self.llm = llm
        logger.info("Reformulator initialized")
    
    def _build_messages( self, request: QueryRewriteRequest ) ->list:

        messages: list[BaseMessage] =[ SystemMessage( content = REFORMULATION_SYSTEM_PROMPT )]

        messages.extend(message_util.build_messages(history=request.history))
        
        messages.append(
            HumanMessage(
                content=f"""
Latest User Query:

{request.query}

Rewrite the above query into a standalone search query.
"""
            )
        )

        return messages
    
    
    def _parse_response(self, response:AIMessage) ->str:
        if not isinstance(response.content, str):
            raise TypeError("Expected text response from the LLM.")

        return response.content.strip()
    
    def _validate(self, rewritten_query:str)->str:
        if not rewritten_query:
            raise ValueError(" Empty reformulated query")
        return rewritten_query
    
    def rewrite(self, request: QueryRewriteRequest)-> ReformulationResult:
        """Rewrite a query into a clear, standalone search query."""
        try:
            # Validate input
            QueryValidator.validate_query_request(request)
            
            messages=self._build_messages(request=request)
            response=llm_util.invoke_llm(llm=self.llm, messages=messages)

            rewritten_query=self._parse_response(response)
            rewritten_query=self._validate(rewritten_query)

            logger.info(f"Reformulation complete: '{request.query}' -> '{rewritten_query}'")
            return ReformulationResult( request.query, reformulated_query=rewritten_query)
        except Exception as e:
            logger.error(f"Error during reformulation: {e}", exc_info=True)
            raise