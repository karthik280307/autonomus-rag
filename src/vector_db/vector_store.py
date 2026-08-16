import chromadb 
import os
import numpy as np
from langchain_core.documents import Document
import uuid
import logging

logger = logging.getLogger(__name__)

class VectorStore:

    def __init__(self, collections_name: str='documents', persistent_directory: str='../../vector_store'):
        
        self.collections_name=collections_name
        self.client=None
        self.persistent_directory=persistent_directory
        self.collection=None
        self._initialize_store()


    def _initialize_store(self):
        try:
            os.makedirs(self.persistent_directory, exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persistent_directory)
            self.collection=self.client.get_or_create_collection(self.collections_name)
            logger.info(f"Vector store initialized with collection: {self.collections_name}")
            logger.info(f"Collection has {self.collection.count()} documents")
        except Exception as e:
            logger.error(f'Failed to initialize vector store: {e}')
            raise
    
    def add_documents(self, documents:list[Document], embeddings:np.ndarray):
        
        if( len(documents) != len(embeddings)):
            raise ValueError("embeddings are not corresponding to the documents")
        
        ids=[]
        metadata_list=[]
        content_list=[]
        embedding_list=[]

        if len(documents) ==0 :
            raise ValueError(' documents are empty')
        
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        for i, (doc, embed) in enumerate(zip(documents, embeddings)):

            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"

            ids.append(doc_id)

            metadata=dict(doc.metadata)
            metadata["doc_index"]=i
            metadata["content_length"]=len(doc.page_content)
            metadata_list.append(metadata)

            content_list.append(doc.page_content)
            embedding_list.append(embed.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                documents=content_list,
                metadatas=metadata_list
            )
            logger.info(f"Successfully added {len(ids)} documents to vector store")
            logger.info(f"Total collection size: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def query(self, query_embedding:np.ndarray, top_k:int):
        if self.collection is None:
            raise ValueError("vector collection is not available")
        
        results=self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results

        