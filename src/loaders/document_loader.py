from langchain_community.document_loaders import DirectoryLoader   
from langchain_community.document_loaders import PyMuPDFLoader   


class Loader:

    def __init__(self, path:str="../data/pdf_documents"):
        self.loader=DirectoryLoader(
            path=path,
            loader_cls=PyMuPDFLoader,
            glob="**/*.pdf",
            show_progress=True
        )  

    def load(self):
        return self.loader.load()
        
