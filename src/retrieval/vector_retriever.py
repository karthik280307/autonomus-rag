import logging
import time
from typing import List
import numpy as np

from src.retrieval.base import BaseRetriever
from src.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDocument,
    SearchCandidate,
    RetrievalBatchRequest,
    RetrievalBatchResult,
)
from src.vector_db.vector_store import VectorStore
from src.embeddings.embedding_manager import EmbeddingManager


logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Dense vector retriever using embeddings for semantic search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        """
        Initialize VectorRetriever.
        
        Args:
            vector_store: VectorStore instance for database operations
            embedding_manager: EmbeddingManager instance for generating embeddings
        """
        if vector_store is None:
            raise ValueError("VectorStore instance is required")
        if embedding_manager is None:
            raise ValueError("EmbeddingManager instance is required")
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        logger.info("VectorRetriever initialized")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve documents using dense vector similarity.
        
        Args:
            request: RetrievalRequest with query and parameters
            
        Returns:
            RetrievalResult with retrieved documents
        """
        logger.info(f"Retrieving documents for query: {request.query}")
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([request.query])
            query_embedding = query_embedding[0]  # Get first (only) embedding
            
            # Query vector store
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=request.top_k
            )
            
            # Parse results into RetrievedDocument objects
            retrieved_documents = self._parse_results(results, request.top_k)
            
            # Apply score threshold filtering if specified
            if request.score_threshold is not None:
                retrieved_documents = [
                    doc for doc in retrieved_documents
                    if doc.distance is not None and doc.distance >= request.score_threshold
                ]
            
            retrieval_time = time.time() - start_time
            
            logger.info(
                f"Retrieved {len(retrieved_documents)} documents in {retrieval_time:.2f}s"
            )
            
            return RetrievalResult(
                query=request.query,
                strategy=request.strategy,
                retrieved_documents=retrieved_documents,
                retrieval_time=retrieval_time,
            )
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            raise

    def retrieve_batch(
        self, request: RetrievalBatchRequest
    ) -> RetrievalBatchResult:
        """
        Retrieve documents for multiple queries.
        
        Args:
            request: RetrievalBatchRequest with multiple queries
            
        Returns:
            RetrievalBatchResult with all retrieved documents
        """
        logger.info(f"Batch retrieving for {len(request.query)} queries")
        
        start_time = time.time()
        all_documents = []
        
        try:
            # Generate embeddings for all queries
            query_embeddings = self.embedding_manager.generate_embeddings(request.query)
            
            # Query vector store for each embedding
            for i, (query, embedding) in enumerate(zip(request.query, query_embeddings)):
                logger.debug(f"Processing batch query {i+1}/{len(request.query)}")
                
                results = self.vector_store.query(
                    query_embedding=embedding,
                    top_k=request.top_k
                )
                
                # Parse and add to results
                docs = self._parse_results(results, request.top_k, query=query)
                all_documents.extend(docs)
            
            retrieval_time = time.time() - start_time
            
            logger.info(
                f"Batch retrieval complete: Retrieved {len(all_documents)} "
                f"documents in {retrieval_time:.2f}s"
            )
            
            return RetrievalBatchResult(
                query=", ".join(request.query),
                strategy=request.strategy,
                retrieved_documents=all_documents,
                retrieval_time=retrieval_time,
            )
            
        except Exception as e:
            logger.error(f"Error during batch retrieval: {e}", exc_info=True)
            raise

    def _parse_results(
        self,
        results: dict,
        top_k: int,
        query: str = None,
    ) -> List[RetrievedDocument]:
        """
        Parse Chroma query results into RetrievedDocument objects.
        
        Args:
            results: Raw results from Chroma
            top_k: Number of results to process
            query: Optional query string for logging
            
        Returns:
            List of RetrievedDocument objects
        """
        documents = []
        
        try:
            # Extract components from Chroma results
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            documents_text = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            # Parse into RetrievedDocument objects
            for rank, (doc_id, distance, text, metadata) in enumerate(
                zip(ids, distances, documents_text, metadatas)
            ):
                # Extract chunk_id from document_id (format: doc_{uuid}_{index})
                chunk_id = f"{doc_id}_chunk"
                
                # Extract document_id from metadata or use first part of id
                document_id = metadata.get("source", doc_id.split("_")[0])
                
                retrieved_doc = RetrievedDocument(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content=text,
                    metadata=metadata,
                    retrieval_rank=rank + 1,
                    final_rank=rank + 1,
                    distance=float(distance),
                    reranker_score=None,
                )
                documents.append(retrieved_doc)
            
            logger.debug(f"Parsed {len(documents)} documents from retrieval results")
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing retrieval results: {e}", exc_info=True)
            raise

    def to_search_candidates(
        self, documents: List[RetrievedDocument]
    ) -> List[SearchCandidate]:
        """
        Convert RetrievedDocument objects to SearchCandidate objects for reranking.
        
        Args:
            documents: List of RetrievedDocument objects
            
        Returns:
            List of SearchCandidate objects
        """
        candidates = []
        
        for doc in documents:
            candidate = SearchCandidate(
                document_id=doc.document_id,
                chunk_id=doc.chunk_id,
                text=doc.content,
                metadata=doc.metadata,
                retrieval_rank=doc.retrieval_rank,
                final_rank=doc.final_rank,
                distance=doc.distance,
                reranker_score=doc.reranker_score,
            )
            candidates.append(candidate)
        
        logger.debug(f"Converted {len(candidates)} documents to search candidates")
        return candidates
