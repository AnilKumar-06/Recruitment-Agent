import re
import PyPDF2
import io
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langhain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import json


class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore=None
        self.analysis_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weaknesses = []
        self.resume_strengths = []
        self.improvement_suggestions = {}


    def extracted_text_from_pdf(self, pdf_file):
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = PyPDF2.PyPdfReader(pdf_file_like)
            else:
                reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
        

    def extract_text_from_text(self, text_file):
        try:
            if hasattr(text_file, 'getvalue'):
                return text_file.getvalue().decode('utf-8')
            else:
                with open(text_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error extracting text from text file: {e}")
            return ""
        
    def extract_text_from_file(self, file):
        if hasattr(file, 'name'):
            file_extension = file.name.split('.').lower()
        else:
            file_extension = file.split('.')[-1].lower()

        if file_extension == 'pdf':
            return self.extracted_text_from_pdf(file)
        elif file_extension == 'txt':
            return self. extract_text_from_text(file)
        else:
            print(f"Unsupported file extension: {file_extension}")
            return ""
        
    def create_rag_vector_store(self, text);
        """Extract text from a text file"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    
    def create_vector_store(self, text):
        """Vectore store for skills Analysis"""
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_texts([text], embeddings)
        return vectorstore
    
    def analyze_skills(self, qa_chain, skill);
        """Analyse the skills in the resume"""
        query = f"On a scale of 0-10, how clearly does the candidate mention \
        proficiency in {skill}? Provide a numeric rating first, followed by reasoning"

        response = qa_chain.run(query)
        match = re.search(r"(\d{1,2})", response)
        score = int(match.group(1)) if match else 0

        reasoning = response.split('.'. 1)[1].strip() if '.' in response and \ 
                    len(response.split('.')) > 1 else ""
        
        return skill, min(score, 10), reasoning