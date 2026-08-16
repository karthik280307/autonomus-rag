from pathlib import Path
import sys
import logging

# Setup path - get project root dynamically
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/retrieval/
sys.path.insert(0, str(project_root))

from sentence_transformers import CrossEncoder

from src.retrieval.models import (RetrievedDocument, SearchCandidate)

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        logger.info(f"Initializing CrossEncoderReranker with model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("CrossEncoderReranker initialized")

    def rerank(
        self,
        query: str,
        documents: list[SearchCandidate],
        top_k: int = None,
    ) -> list[SearchCandidate]:

        if not documents:
            logger.warning("No documents to rerank")
            return []

        logger.info(f"Reranking {len(documents)} documents for query: {query}")

        pairs = [
            (query, doc.text)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc.reranker_score = float(score)

        # Sort by reranker score in descending order
        documents_sorted = sorted(documents, key=lambda x: x.reranker_score, reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            documents_sorted = documents_sorted[:top_k]
        
        # Update final ranks
        for rank, doc in enumerate(documents_sorted, 1):
            doc.final_rank = rank

        logger.info(f"Reranking complete. Top document score: {documents_sorted[0].reranker_score:.4f}")
        return documents_sorted
