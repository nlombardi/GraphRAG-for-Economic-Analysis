import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import spacy
# from langchain.llms import OpenAI
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
import re
import ollama
from get_data import WorldBankDataCollector

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load config
config = json.load(open('config.json'))
countries = config.get("COUNTRIES", {}).get("ISO2", [])
indicators = config.get("INDICATORS", {}).get("WB_CODE", [])

# Initialize components
nlp = spacy.load("en_core_web_sm")  # Used for extracting words from documents

# Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
embeddings = OllamaEmbeddings(model="dengcao/Qwen3-Embedding-4B:Q4_K_M",)

# LLM
# llm = OpenAI(temperature=0.1)  # Need API key
llm = OllamaLLM(model="gpt-oss:latest", temperature=0.1)  # We use a lower temperature to reduce hallucinations for factual economic analysis

# DB - Neo4j for graph storage and Qdrant for vector storage
neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
qdrant_client = QdrantClient(path=os.getenv("QDRANT_PATH"))


# Scrape the data
wb_collector = WorldBankDataCollector()
wdi_data = wb_collector.get_indicators(countries, indicators)

# Clean the data
wdi_data = wdi_data.dropna(subset=['value'])
print(f"Collected {len(wdi_data)} data points.")

# Print a preview
print("\nSample data:")
print(wdi_data.head())
print(wdi_data.describe())
