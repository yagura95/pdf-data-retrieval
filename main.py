import pypdf 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from pathlib import Path
import os
import sys

class PDFAnalyse():
    def __init__(self):
        self.llm = Ollama(
                model="mistral",
                temperature=0.2
                )
        
        self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
                )

        self.vector_store = None

    def load_pdf(self, file_path):
        pdf_reader = pypdf.PdfReader(file_path)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        return text

    def create_chunks(self, content, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
                )
        
        return text_splitter.split_text(content)
        
    def create_vector_store(self, chunks):
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
    
    def setup_qa_chain(self):

        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please process a document first.")

        retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}
                )

        return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
                )
    
    def process_pdf(self, file_path):
        content = self.load_pdf(file_path)
        chunks = self.create_chunks(content)
        self.create_vector_store(chunks)
    
    def save_vector_store(self, path):
        if self.vector_store is None:
            raise ValueError("No vector store to save. Please process a document first.")
        
        self.vector_store.save_local(path)

    def load_vector_store(self, path):
        self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
                )

    def query(self, question):
        qa_chain = self.setup_qa_chain()
        return qa_chain({ "query": question})

def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python3 main.py <file_path>")

    file = sys.argv[1]
    
    analyser = PDFAnalyse()
    analyser.process_pdf(file)

    try:
        analyser.load_vector_store(Path("stores") / file) 
    except Exception as e:
        analyser.save_vector_store(Path("stores") / file) 

    while True:
        question = input("Insert question:\n")

        if question == "exit":
            break 

        answer = analyser.query(question)

        print("\n\n\n")
        print("RESULT:\n")
        print(answer["result"])
        print("\n")

if __name__ == "__main__":
    main()



