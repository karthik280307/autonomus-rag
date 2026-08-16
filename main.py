import sys
from pathlib import Path
import argparse
import logging

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.loaders.document_loader import Loader
from src.chunker.chunker import TextSplitter
from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_db.vector_store import VectorStore
from src.query_rewriter.rewriter import QueryRewriter
from src.query_rewriter.models import QueryRewriteRequest, ChatMessage
from src.retrieval.models import RetrievalStrategy
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.answer_generator import AnswerGenerator
from dotenv import load_dotenv

load_dotenv()
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete Autonomous RAG Pipeline.
    
    Orchestrates document loading, chunking, embedding generation,
    vector storage, query rewriting, and retrieval.
    """
    
    def __init__(
        self,
        pdf_directory: str = "./data/pdf_documents",
        vector_store_directory: str = "./vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            pdf_directory: Path to PDF documents
            vector_store_directory: Path for persistent vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of embedding model to use
        """
        logger.info("Initializing RAG Pipeline...")
        
        self.pdf_directory = pdf_directory
        self.vector_store_directory = vector_store_directory
        
        # Initialize components
        self.loader = Loader(path=pdf_directory)
        self.chunker = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.vector_store = VectorStore(persistent_directory=vector_store_directory)
        
        # Initialize query rewriting components
        try:
            self.query_rewriter = QueryRewriter()
            logger.info("Query rewriter initialized successfully")
        except Exception as e:
            logger.warning(f"Query rewriter not initialized: {e}. Will skip query rewriting.")
            self.query_rewriter = None
        
        # Initialize retrieval components
        try:
            self.vector_retriever = VectorRetriever(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager
            )
            logger.info("Vector retriever initialized successfully")
            
            self.reranker = CrossEncoderReranker()
            logger.info("Cross-encoder reranker initialized successfully")
        except Exception as e:
            logger.warning(f"Retrieval components not fully initialized: {e}")
            self.vector_retriever = None
            self.reranker = None
        
        # Initialize answer generator
        try:
            self.answer_generator = AnswerGenerator()
            logger.info("Answer generator initialized")
        except Exception as e:
            logger.warning(f"Answer generator not initialized: {e}")
            self.answer_generator = None
        
        logger.info("RAG Pipeline initialized successfully!")
    
    def ingest_documents(self) -> int:
        """
        Load, chunk, embed, and store documents in the vector DB.
        
        Returns:
            Number of documents stored
        """
        logger.info("Starting document ingestion pipeline...")
        
        # Step 1: Load documents
        logger.info(f"Loading documents from {self.pdf_directory}...")
        documents = self.loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found. Exiting ingestion.")
            return 0
        
        # Step 2: Chunk documents
        logger.info("Chunking documents...")
        chunked_documents = self.chunker.split(documents)
        logger.info(f"Created {len(chunked_documents)} chunks from documents")
        
        # Step 3: Generate embeddings
        logger.info("Generating embeddings for chunks...")
        chunk_texts = [doc.page_content for doc in chunked_documents]
        embeddings = self.embedding_manager.generate_embeddings(chunk_texts)
        
        # Step 4: Store in vector database
        logger.info("Storing chunks and embeddings in vector database...")
        self.vector_store.add_documents(chunked_documents, embeddings)
        logger.info(f"Successfully stored {len(chunked_documents)} chunks in vector store")
        
        return len(chunked_documents)
    
    def rewrite_query(self, query: str, chat_history: list[ChatMessage] = None) -> dict:
        """
        Rewrite a query using the query rewriting pipeline.
        
        Args:
            query: Original user query
            chat_history: Optional chat history for context
            
        Returns:
            Dictionary with rewritten query variations
        """
        if self.query_rewriter is None:
            logger.warning("Query rewriter not available")
            return {"original_query": query}
        
        logger.info(f"Rewriting query: {query}")
        
        request = QueryRewriteRequest(
            query=query,
            history=chat_history or []
        )
        
        try:
            result = self.query_rewriter.rewrite(request)
            logger.info("Query rewriting completed")
            
            return {
                "original_query": result.original_query,
                "reformulated_query": result.reformulated_query,
                "step_back_query": result.step_back_query,
                "expanded_queries": result.expanded_queries
            }
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return {"original_query": query}
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.DENSE,
        score_threshold: float = None,
        use_reranking: bool = False
    ) -> dict:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            strategy: Retrieval strategy (DENSE, SPARSE, HYBRID)
            score_threshold: Optional score threshold for filtering
            use_reranking: Whether to apply cross-encoder reranking
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        try:
            if self.vector_retriever is None:
                raise RuntimeError("Vector retriever not initialized")
            
            # Create retrieval request
            from src.retrieval.models import RetrievalRequest
            request = RetrievalRequest(
                query=query,
                top_k=top_k,
                strategy=strategy,
                score_threshold=score_threshold
            )
            
            # Retrieve documents
            retrieval_result = self.vector_retriever.retrieve(request)
            
            logger.info(f"Retrieved {len(retrieval_result.retrieved_documents)} documents")
            
            # Apply reranking if requested
            if use_reranking and self.reranker:
                logger.info("Applying cross-encoder reranking...")
                candidates = self.vector_retriever.to_search_candidates(
                    retrieval_result.retrieved_documents
                )
                ranked_candidates = self.reranker.rerank(
                    query=query,
                    documents=candidates,
                    top_k=top_k
                )
                
                # Convert back to result format
                return {
                    "query": query,
                    "strategy": strategy.value,
                    "top_k": top_k,
                    "retrieval_time": retrieval_result.retrieval_time,
                    "retrieved_documents": retrieval_result.retrieved_documents,
                    "ranked_documents": ranked_candidates,
                    "reranked": True
                }
            
            return {
                "query": query,
                "strategy": strategy.value,
                "top_k": top_k,
                "retrieval_time": retrieval_result.retrieval_time,
                "retrieved_documents": retrieval_result.retrieved_documents,
                "reranked": False
            }
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return {"query": query, "retrieved_documents": None, "error": str(e)}
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        rewrite_query: bool = True,
        chat_history: list[ChatMessage] = None,
        use_reranking: bool = False
    ) -> dict:
        """
        Complete RAG query pipeline: rewrite query and retrieve documents.
        
        Args:
            query: User query
            top_k: Number of results to return
            rewrite_query: Whether to apply query rewriting
            chat_history: Optional chat history for context
            use_reranking: Whether to apply cross-encoder reranking
            
        Returns:
            Dictionary with query results and rewritten queries
        """
        logger.info(f"Processing query: {query}")
        
        results = {
            "original_query": query,
            "query_rewrites": None,
            "retrieval_results": None
        }
        
        # Optionally rewrite the query
        if rewrite_query:
            results["query_rewrites"] = self.rewrite_query(query, chat_history)
        
        # Retrieve documents
        results["retrieval_results"] = self.retrieve(
            query, 
            top_k=top_k,
            use_reranking=use_reranking
        )
        
        # Generate answer from reranked documents
        results["answer"] = self.generate_answer(
            query,
            results["retrieval_results"]
        )
        
        return results
    
    def generate_answer(self, query: str, retrieval_results: dict) -> str:
        """
        Generate a final answer from retrieval results.
        
        Args:
            query: User query
            retrieval_results: Results from retrieve() method
            
        Returns:
            Generated answer string
        """
        if not self.answer_generator:
            logger.warning("Answer generator not available")
            return "Answer generation not available"
        
        if retrieval_results.get("error"):
            return f"Error during retrieval: {retrieval_results['error']}"
        
        # Get reranked or retrieved documents
        if retrieval_results.get("reranked") and retrieval_results.get("ranked_documents"):
            documents = retrieval_results["ranked_documents"]
        elif retrieval_results.get("retrieved_documents"):
            # Convert RetrievedDocument to SearchCandidate
            documents = self.vector_retriever.to_search_candidates(
                retrieval_results["retrieved_documents"]
            )
        else:
            return "No documents found to generate answer from"
        
        try:
            answer = self.answer_generator.generate_answer(query, documents)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"


def main():
    """
    Main entry point for the Autonomous RAG system.
    """
    parser = argparse.ArgumentParser(
        description="Autonomous RAG - Retrieval Augmented Generation System"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ingest", "query"],
        default="ingest",
        help="Operation mode: 'ingest' for document ingestion, 'query' for querying"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query string (for query mode)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./data/pdf_documents",
        help="Path to PDF documents directory"
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="./vector_store",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable cross-encoder reranking"
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Disable query rewriting"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG pipeline
        pipeline = RAGPipeline(
            pdf_directory=args.pdf_dir,
            vector_store_directory=args.vector_store_dir,
            chunk_size=args.chunk_size,
            embedding_model=args.embedding_model
        )
        
        if args.mode == "ingest":
            logger.info("=" * 60)
            logger.info("Document Ingestion Mode")
            logger.info("=" * 60)
            
            num_docs = pipeline.ingest_documents()
            logger.info(f"Ingestion complete! Stored {num_docs} document chunks.")
            
        elif args.mode == "query":
            if not args.query:
                logger.error("Query string required for query mode. Use --query to provide it.")
                return
            
            logger.info("=" * 60)
            logger.info("Query Mode")
            logger.info("=" * 60)
            
            results = pipeline.query(
                query=args.query,
                top_k=args.top_k,
                rewrite_query=not args.no_query_rewrite,
                use_reranking=not args.no_reranking
            )
            
            # Display results
            logger.info("\n" + "=" * 60)
            logger.info("RESULTS")
            logger.info("=" * 60)
            logger.info(f"Original Query: {results['original_query']}")
            
            if results['query_rewrites']:
                logger.info("\nQuery Rewrites:")
                logger.info(f"  - Reformulated: {results['query_rewrites'].get('reformulated_query', 'N/A')}")
                logger.info(f"  - Step Back: {results['query_rewrites'].get('step_back_query', 'N/A')}")
                if results['query_rewrites'].get('expanded_queries'):
                    logger.info(f"  - Expanded ({len(results['query_rewrites']['expanded_queries'])} variants): {results['query_rewrites']['expanded_queries']}")
            
            if results['retrieval_results']:
                ret_results = results['retrieval_results']
                
                if ret_results.get('error'):
                    logger.error(f"Retrieval Error: {ret_results['error']}")
                    return
                
                logger.info(f"\nRetrieval completed in {ret_results.get('retrieval_time', 0):.2f}s")
                
                # Display reranked results if available
                if ret_results.get('reranked') and ret_results.get('ranked_documents'):
                    logger.info(f"\nReranked Results ({len(ret_results['ranked_documents'])} documents):")
                    for i, doc in enumerate(ret_results['ranked_documents'], 1):
                        logger.info(f"\n  [{i}] Score: {doc.reranker_score:.4f} (Distance: {doc.distance:.4f})")
                        logger.info(f"      Metadata: {doc.metadata}")
                        logger.info(f"      Content: {doc.text[:200]}...")
                # Display original retrieved results
                elif ret_results.get('retrieved_documents'):
                    logger.info(f"\nRetrieved Documents ({len(ret_results['retrieved_documents'])} results):")
                    for i, doc in enumerate(ret_results['retrieved_documents'], 1):
                        logger.info(f"\n  [{i}] Distance: {doc.distance:.4f}")
                        logger.info(f"      Metadata: {doc.metadata}")
                        logger.info(f"      Content: {doc.content[:200]}...")
            
            # Display final answer
            if results.get('answer'):
                logger.info("\n" + "=" * 60)
                logger.info("FINAL ANSWER")
                logger.info("=" * 60)
                logger.info(results['answer'])
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
