from pathlib import Path
import sys
import logging

# Setup path - get project root dynamically
parent_root = Path(__file__).parent.parent.parent  # Go up to project root from src/query_rewriter/
sys.path.insert(0, str(parent_root))
from src.query_rewriter.models import (QueryRewriteRequest, ExpansionResult)
from src.query_rewriter.base import BaseQueryRewriter
from src.query_rewriter.models import ChatMessage
from langchain_core.messages import(HumanMessage, BaseMessage, SystemMessage, AIMessage)
from langchain_core.language_models import BaseLLM
from src.query_rewriter.prompts import EXPANSION_SYSTEM_PROMPT

import re
from src.query_rewriter.utils import (llm_util, message_util)
from src.query_rewriter.validators import QueryValidator

logger = logging.getLogger(__name__)

class Expansion(BaseQueryRewriter):
    """Generate multiple expanded search queries for improved recall."""

    def __init__(self, llm: BaseLLM):
        """
        Initialize Expansion.
        
        Args:
            llm: Language model instance
        """
        if llm is None:
            raise ValueError("LLM instance is required for Expansion")
        self.llm = llm
        logger.info("Expansion initialized")

    def _build_messages(self, request: QueryRewriteRequest)-> list[BaseMessage]:
        messages: list[BaseMessage]= [SystemMessage(content= EXPANSION_SYSTEM_PROMPT )]

        messages.extend(message_util.build_messages(history=request.history))

        
        messages.append(
            HumanMessage(
    content=f"""
    Current Query:

    {request.query}

    Generate multiple expanded search queries.
    Return one query per line.
    """
    )
        )

        return messages


    def _parse(self, response) -> list[str]:
        return [
            re.sub(r"^(\d+\.\s*|[-*]\s*)", "", line).strip()
            for line in response.content.splitlines()
            if line.strip()
        ]
    
    def _validate(self, expanded_queries:list):
        if not expanded_queries:
            raise ValueError(" there were no expanded queries it is empty")
        return expanded_queries

    def rewrite(self, request: QueryRewriteRequest) -> ExpansionResult:
        """Generate expanded search queries."""
        try:
            # Validate input
            QueryValidator.validate_query_request(request)
            
            messages=self._build_messages(request=request)
            response=llm_util.invoke_llm( llm=self.llm, messages=messages)
            expanded_queries=self._parse(response)
            expanded_queries=self._validate(expanded_queries)

            logger.info(f"Query expansion complete: Generated {len(expanded_queries)} expanded queries")
            return ExpansionResult(original_query=request.query, expanded_queries=expanded_queries )
        except Exception as e:
            logger.error(f"Error during query expansion: {e}", exc_info=True)
            raise
        