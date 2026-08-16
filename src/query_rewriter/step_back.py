from pathlib import Path
import sys
import logging

# Setup path - get project root dynamically
parent_root = Path(__file__).parent.parent.parent  # Go up to project root from src/query_rewriter/
sys.path.insert(0, str(parent_root))

from src.query_rewriter.models import (QueryRewriteRequest, StepBackResult)
from src.query_rewriter.base import BaseQueryRewriter
from src.query_rewriter.models import ChatMessage
from langchain_core.messages import(HumanMessage, BaseMessage, SystemMessage, AIMessage)
from langchain_core.language_models import BaseLLM
from src.query_rewriter.prompts import STEP_BACK_SYSTEM_PROMPT

import re
from src.query_rewriter.utils import (llm_util, message_util)
from src.query_rewriter.validators import QueryValidator

logger = logging.getLogger(__name__)

class StepBack(BaseQueryRewriter):
    """Generate broader step-back queries for background knowledge retrieval."""

    def __init__(self, llm: BaseLLM):
        """
        Initialize StepBack.
        
        Args:
            llm: Language model instance
        """
        if llm is None:
            raise ValueError("LLM instance is required for StepBack")
        self.llm = llm
        logger.info("StepBack initialized")

    def _build_messages(self, request: QueryRewriteRequest)-> list[BaseMessage]:
        messages: list[BaseMessage]= [SystemMessage(content= STEP_BACK_SYSTEM_PROMPT )]

        messages.extend(message_util.build_messages(history=request.history))

        messages.append(
            HumanMessage(
    content=f"""
    Current Query:

    {request.query}

    Generate a single step-back query.
    """
    )
        )

        return messages

    def _parse(self, response) -> str:
        return response.content.strip()
    
    def _validate(self, step_back_query:str):
        if not step_back_query:
            raise ValueError(" there were no expanded queries it is empty")
        return step_back_query

    def rewrite(self, request: QueryRewriteRequest) -> StepBackResult:
        """Generate a step-back query."""
        try:
            # Validate input
            QueryValidator.validate_query_request(request)
            
            messages=self._build_messages(request=request)
            response=llm_util.invoke_llm( llm=self.llm, messages=messages)

            step_back_query=self._parse(response)
            step_back_query=self._validate(step_back_query)

            logger.info(f"Step-back query generation complete: '{step_back_query}'")
            return StepBackResult(original_query=request.query, step_back_query=step_back_query)
        except Exception as e:
            logger.error(f"Error during step-back query generation: {e}", exc_info=True)
            raise