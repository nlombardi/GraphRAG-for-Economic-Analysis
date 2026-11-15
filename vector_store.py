from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings
from google import genai
from google.genai.errors import ServerError
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import time
from httpx import RemoteProtocolError
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
        self.max_retries = 5
        self.qdrant_batch_size = 50

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

    def embed_chunk(self, chunk_data: Tuple[Dict, int]):
        """Embed document chunks for vector storage"""
        chunk, i = chunk_data

        # Create enriched content for embedding
        enriched_content = f"""
        Title: {chunk['title']}
        Source: {chunk['source']}
        Content: {chunk['content']}
        """

        processed_embeddings = []

        for attempt in range(self.max_retries):
            embedding = ''
            # Generate embedding
            try:
                if self.embeddings == "google":
                    embedding = google_embeddings.models.embed_content(
                        model="gemini-embedding-001",
                        contents=enriched_content
                    )
                    if embedding:
                        processed_embeddings = embedding.embeddings[0].values
                    break

                elif self.embeddings == "ollama":
                    processed_embeddings = ollama_embeddings.embed_query(enriched_content)
                    break

            except (ServerError, RemoteProtocolError) as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed to embed chunk {i} after {self.max_retries} attempts: {e}")
                    return None
                time.sleep(30 * (attempt + 1))  # Exponential backoff

            except Exception as e:
                print(f"Unexpected error with chunk {i}: {e}")
                return None

        if not processed_embeddings:
            return None

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
        return point

    def embed_documents_parallel(self, chunks: List[Dict]):
        """Embed document chunks in parallel with proper error handling"""

        # Determine optimal number of workers
        points = []
        failed_chunks = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {
                executor.submit(self.embed_chunk,(chunk, i)): i
                for i, chunk in enumerate(chunks)
            }

            # Process results as they complete
            with tqdm(total=len(chunks), desc="Embedding chunks") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        if result:
                            points.append(result)
                        else:
                            failed_chunks.append(index)
                    except Exception as e:
                        failed_chunks.append(index)
                        print(f"Exception processing chunk {index}: {e}")
                    pbar.update(1)

        print(f"Successfully processed {len(points)} chunks, failed: {len(failed_chunks)}")

        # Save points incrementally to avoid memory issues
        self._save_and_upload_points(points)
        return points


    def _save_and_upload_points(self, points):
        """Save and upload points with chunking"""
        # Save to file
        with open(Path.cwd() / 'db' / 'points.p', 'wb') as f:
            pickle.dump(points, f)

        # Upload to Qdrant in batches
        batch_size = self.qdrant_batch_size

        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            points_to_upsert = points[i:i + batch_size]

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_to_upsert
                )
            except Exception as e:
                print(f"Error uploading batch {i // batch_size}: {e}")
                # Consider saving failed batches for retry

        print(f"Embedded and stored {len(points)} document chunks")

