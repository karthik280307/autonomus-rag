from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
class TextSplitter():
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def split(self, documents):
        return self.text_splitter.split_documents(documents)
    