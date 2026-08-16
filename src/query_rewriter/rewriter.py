from pathlib import Path
import sys
import logging

# Setup path - get project root dynamically
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/query_rewriter/
sys.path.insert(0, str(project_root))

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM

from src.query_rewriter.models import(
    QueryRewriteRequest, ChatMessage
)

from src.query_rewriter.models import (RewrittenQueryResult, StepBackResult, ExpansionResult, ReformulationResult)

from src.query_rewriter.reformulator import Reformulator
from src.query_rewriter.expansion import Expansion
from src.query_rewriter.step_back import StepBack

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Complete query rewriting pipeline combining reformulation, expansion, and step-back strategies."""

    def __init__(
        self, 
        reformulator: Reformulator = None, 
        expansion: Expansion = None, 
        step_back: StepBack = None,
        llm: BaseLLM = None
    ):
        """
        Initialize QueryRewriter with components.
        
        Args:
            reformulator: Reformulator instance (optional, will be created if not provided)
            expansion: Expansion instance (optional, will be created if not provided)
            step_back: StepBack instance (optional, will be created if not provided)
            llm: Language model instance (optional, defaults to Groq if available)
        """
        # Initialize LLM if not provided
        if llm is None:
            try:
                llm = ChatGroq(model="mixtral-8x7b-32k", temperature=0)
                logger.info("Initialized Groq LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM: {e}. Trying OpenAI...")
                try:
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    logger.info("Initialized OpenAI LLM")
                except Exception as e2:
                    logger.error(f"Failed to initialize any LLM: {e2}")
                    raise
        
        self.llm = llm
        
        # Initialize components if not provided
        self.reformulator = reformulator or Reformulator(llm=self.llm)
        self.expansion = expansion or Expansion(llm=self.llm)
        self.step_back = step_back or StepBack(llm=self.llm)

    def rewrite(self, request: QueryRewriteRequest) -> RewrittenQueryResult:
        """
        Rewrite a query using all three strategies.
        
        Args:
            request: QueryRewriteRequest with query and optional history
            
        Returns:
            RewrittenQueryResult with all rewritten query variations
        """
        logger.info(f"Rewriting query: {request.query}")
        
        try:
            # Step 1: Reformulate the query
            reformulator_result = self.reformulator.rewrite(request=request)
            logger.debug(f"Reformulated query: {reformulator_result.reformulated_query}")
            
            # Create new request with reformulated query for expansion and step-back
            new_request = QueryRewriteRequest(
                query=reformulator_result.reformulated_query, 
                history=request.history
            )
            
            # Step 2: Expand the reformulated query
            expansion_result = self.expansion.rewrite(request=new_request)
            logger.debug(f"Generated {len(expansion_result.expanded_queries)} expanded queries")
            
            # Step 3: Generate step-back query
            step_back_result = self.step_back.rewrite(request=new_request)
            logger.debug(f"Step-back query: {step_back_result.step_back_query}")

            return RewrittenQueryResult(
                original_query=request.query,
                reformulated_query=reformulator_result.reformulated_query,
                step_back_query=step_back_result.step_back_query,
                expanded_queries=expansion_result.expanded_queries
            )
        except Exception as e:
            logger.error(f"Error during query rewriting: {e}", exc_info=True)
            raise
        
        


