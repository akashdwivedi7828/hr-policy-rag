from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
import streamlit as st
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200, separators=["\n", "\n\n", " ", "."])

    def process_uploaded_file(self, uploaded_files):
        with open(f"temp_{uploaded_files.name}", "wb") as f:
            f.write(uploaded_files.getvalue())
        
        file_path = f"temp_{uploaded_files.name}"
        if(uploaded_files.name.endswith("pdf")):
            loader = PyPDFLoader(file_path)
        elif(uploaded_files.name.endswith("docx")):
            loader = Docx2txtLoader(file_path)
        else:
            st.error("Unsupported file")
            return []
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        return [chunk.page_content for chunk in chunks]