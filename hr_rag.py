from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import google.generativeai as genai  # Gemini

class HRRAGSystem:
    def __init__(self):
        # Configure Pinecone
        self.pc = Pinecone(
            api_key=st.secrets["PINECONE_API_KEY"]
        )
        
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # use Streamlit secrets for consistency

        # Initialize vector store and embedding model
        self.index_name = "hr-assistant"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
        self.setup_pinecone()

    def setup_pinecone(self):
        """Setup Pinecone index"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return []

    def store_documents(self, texts: List[str], source: str):
        """Store documents in vector database"""
        embeddings = self.generate_embeddings(texts)

        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append({
                "id": f"{source}_{i}",
                "values": embedding,
                "metadata": {
                    "text": text,
                    "source": source,
                    "chunk_id": i
                }
            })

        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        return len(vectors)
    
    def generate_answer(self, query: str) -> Dict:
        """Generate answer using RAG and format it with Gemini 1.5 Flash"""
        # Step 1: Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]
        
        # Step 2: Search similar documents
        search_results = self.index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Step 3: Extract context
        sources = []
        context_texts = []
        for match in search_results["matches"]:
            if match["score"] > 0.3:
                sources.append(match["metadata"]["source"])
                context_texts.append(match["metadata"]["text"])
        
        if context_texts:
            context_str = "\n\n".join(context_texts[:2])  # top 2 chunks

            # Step 4: Create Gemini prompt
            prompt = f"""
                You are an AI HR Assistant. Use the following context from internal HR documents to answer the user's query clearly, concisely, and professionally.

                Context:
                {context_str}

                User Query:
                {query}

                Answer:
            """

            try:
                # Step 5: Call Gemini 1.5 Flash
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                formatted_answer = response.text
            except Exception as e:
                st.error(f"Gemini generation failed: {e}")
                formatted_answer = "There was an issue formatting the response with Gemini."
        else:
            formatted_answer = "I couldn't find relevant information in the HR documents for your question."

        return {
            "answer": formatted_answer,
            "sources": list(set(sources))
        }