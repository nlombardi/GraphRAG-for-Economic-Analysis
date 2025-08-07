from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings
import os
from typing import List, Dict

# Initialize your qdrant client
qdrant_client = QdrantClient(host=os.getenv('QDRANT_ENDPOINT'), port=6333, api_key=os.getenv('QDRANT_API_KEY'))
ollama_embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBEDDINGS_MODEL'))


class VectorStoreManager:
    def __init__(self, qdrant_client = qdrant_client, embeddings = ollama_embeddings):
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = "economic_documents"
        self.setup_collection()

    def setup_collection(self):
        """Initialize Qdrant collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Collection might already exist: {e}")

    def embed_documents(self, chunks: List[Dict]):
        """Embed document chunks and store in vector database"""
        points = []

        for i, chunk in enumerate(chunks):
            # Create enriched content for embedding
            enriched_content = f"""
            Title: {chunk['title']}
            Source: {chunk['source']}
            Content: {chunk['content']}
            """

            # Generate embedding
            embedding = self.embeddings.embed_query(enriched_content)

            # Create point for Qdrant
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    'chunk_id': chunk['chunk_id'],
                    'content': chunk['content'],
                    'title': chunk['title'],
                    'source': chunk['source'],
                    'date': chunk['date'],
                    'doc_type': chunk['doc_type']
                }
            )
            points.append(point)

        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Embedded and stored {len(points)} document chunks")
