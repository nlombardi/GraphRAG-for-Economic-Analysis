from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings
from google import genai
from google.genai.errors import ServerError
import os
from typing import List, Dict
from dotenv import load_dotenv
import time
from httpx import RemoteProtocolError

load_dotenv()

# Initialize the qdrant client
qdrant_client = QdrantClient(url=os.getenv('QDRANT_ENDPOINT'), port=6333, api_key=os.getenv('QDRANT_API_KEY'))
ollama_embeddings = OllamaEmbeddings(model=f"{os.getenv('OLLAMA_EMBEDDINGS_MODEL')}")
google_embeddings = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

class VectorStoreManager:
    def __init__(self, qdrant_client = qdrant_client, embeddings = "google"):
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = "economic_documents"
        self.setup_collection()

    def setup_collection(self):
        """Initialize Qdrant collection"""

        try:
            # We set the vector size to 3072 because that's what google embeddings returns, different embeddings models
            # may return different vector sizes and this would need to be adjusted to accommodate if using a local model.
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
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

            processed_embeddings = []
            tries = True
            num_tries = 0
            embedding = ''
            # Generate embedding
            if self.embeddings == "google":
                while tries:
                    try:
                        num_tries += 1
                        embedding = google_embeddings.models.embed_content(
                            model="gemini-embedding-001",
                            contents=enriched_content
                        )
                        tries = False
                    except (ServerError, RemoteProtocolError) as e:
                        if num_tries == 5:  # Move to next entry
                            tries = False
                            continue
                        print(f"Issue with Google Server: {e}")
                        time.sleep(30)  # Pause for 30 seconds before trying the embeddings again
                if embedding:
                    processed_embeddings = embedding.embeddings[0].values
            elif self.embeddings == "ollama":
                processed_embeddings = ollama_embeddings.embed_query(enriched_content)
            else:
                print(f"Embeddings model not available for embedding type: {self.embeddings}.")
                continue

            # Create point for Qdrant
            point = PointStruct(
                id=i,
                vector=processed_embeddings,
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

            # Progress tracking
            if i % 100 == 0 or i == len(chunks) - 1:  # Update every 5 chunks since there are fewer documents
                print(f"   Processed chunk {i + 1}/{len(chunks)}")

        with open(Path.cwd() / 'db' / 'points.p', 'wb') as f:
            pickle.dump(points, f)

        # Chunk the points to upsert into the Qdrant db to avoid issues with payload size
        for i in range(0, len(points), 10):
            points_to_upsert = points[i:i + 10]

            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_upsert
            )

        print(f"Embedded and stored {len(points)} document chunks")
        return None
