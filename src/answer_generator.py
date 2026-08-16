import logging
from typing import Optional, List

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from langchain_core.language_models import BaseLLM
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval.models import RetrievedDocument, SearchCandidate


logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate final answers from reranked documents using an LLM."""

    SYSTEM_PROMPT = """You are an expert at answering questions based on provided context.

Your task is to provide a clear, concise, and accurate answer to the user's question using ONLY the provided context.

Rules:
1. Answer based exclusively on the provided documents.
2. Be concise and direct.
3. If the answer is not in the provided context, say "I don't have enough information to answer this question."
4. Cite the relevant sources when appropriate.
5. Structure your answer clearly with proper formatting.
"""

    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize AnswerGenerator.
        
        Args:
            llm: Language model instance (optional, will try to initialize if not provided)
        """
        self.llm = llm
        
        # Try to initialize LLM if not provided
        if self.llm is None:
            self.llm = self._initialize_llm()
        
        if self.llm:
            logger.info("AnswerGenerator initialized successfully")
        else:
            logger.warning("No LLM available. Answer generation will be skipped.")

    def _initialize_llm(self) -> Optional[BaseLLM]:
        """
        Try to initialize an LLM with available credentials.
        
        Returns:
            LLM instance or None if no credentials available
        """
        # Try Groq first
        if ChatGroq:
            try:
                llm = ChatGroq(model="mixtral-8x7b-32k", temperature=0)
                logger.info("Initialized Groq LLM for answer generation")
                return llm
            except Exception as e:
                logger.debug(f"Failed to initialize Groq: {e}")
        
        # Try OpenAI as fallback
        if ChatOpenAI:
            try:
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                logger.info("Initialized OpenAI LLM for answer generation")
                return llm
            except Exception as e:
                logger.debug(f"Failed to initialize OpenAI: {e}")
        
        logger.warning("No LLM credentials available (GROQ_API_KEY or OPENAI_API_KEY)")
        return None

    def generate_answer(
        self,
        query: str,
        documents: List[SearchCandidate],
    ) -> str:
        """
        Generate an answer from query and retrieved documents.
        
        Args:
            query: User's original query
            documents: List of reranked SearchCandidate documents
            
        Returns:
            Generated answer as a string
        """
        if not self.llm:
            logger.warning("No LLM available for answer generation")
            return self._generate_fallback_answer(query, documents)
        
        if not documents:
            return "No relevant documents found to answer this question."
        
        try:
            logger.info(f"Generating answer for query: {query}")
            
            # Build context from documents
            context = self._build_context(documents)
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""Context:

{context}

User Question: {query}

Please provide a clear answer based on the provided context."""
                )
            ]
            
            # Call the LLM
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("Answer generation successful")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return self._generate_fallback_answer(query, documents)

    def _build_context(self, documents: List[SearchCandidate]) -> str:
        """
        Build context string from documents.
        
        Args:
            documents: List of SearchCandidate documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown Source") if doc.metadata else "Unknown Source"
            content = doc.text[:500] + "..." if len(doc.text) > 500 else doc.text
            score = f"{doc.reranker_score:.2f}" if doc.reranker_score else "N/A"
            
            context_parts.append(
                f"[Document {i}] (Score: {score}, Source: {source})\n{content}"
            )
        
        return "\n\n".join(context_parts)

    def _generate_fallback_answer(
        self,
        query: str,
        documents: List[SearchCandidate],
    ) -> str:
        """
        Generate a fallback answer without an LLM (summarize retrieved documents).
        
        Args:
            query: User's query
            documents: List of retrieved documents
            
        Returns:
            Simple fallback answer
        """
        logger.info("Generating fallback answer (no LLM available)")
        
        if not documents:
            return f"No documents found to answer: '{query}'"
        
        answer = f"Based on {len(documents)} retrieved documents:\n\n"
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown") if doc.metadata else "Unknown"
            content = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            score = f"{doc.reranker_score:.2f}" if doc.reranker_score else "N/A"
            
            answer += f"[{i}] (Relevance Score: {score}) {source}\n{content}\n\n"
        
        answer += f"\nNote: To get a full AI-generated answer, set up a Groq API key (GROQ_API_KEY) or OpenAI key (OPENAI_API_KEY)."
        
        return answer
