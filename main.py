import json
from get_data import WorldBankDataCollector
from get_docs import DocumentCollector
from make_graph import EconGraphBuilder
from vector_store import VectorStoreManager
from graph_rag import EconGraphRag

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load config
config = json.load(open('config.json'))
countries = config.get("COUNTRIES", {}).get("ISO2", [])
indicators = config.get("INDICATORS", {}).get("WB_CODE", [])

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

# Get document chunks
doc_collector = DocumentCollector()
all_chunks = doc_collector.main()
print("\nSample chunk:")
sample_chunk = all_chunks[0]
for key, value in sample_chunk.items():
    if key == 'content':
        print(f"{key}: {value[:100]}...")  # Truncate content for display
    else:
        print(f"{key}: {value}")

# Build the knowledge graph
graph_builder = EconGraphBuilder()
graph_builder.create_structured_nodes(wdi_data)
graph_builder.create_document_nodes(all_chunks)

# Setup vector store and push document chunks to the vector db
vector_manager = VectorStoreManager()
vector_manager.embed_documents(all_chunks)

# Initialize GraphRAG system
graph_rag = EconGraphRag()

# Test questions
test_questions = [
    "What was Thailand's GDP growth in 2021, and what factors contributed to it?",
    "What is the middle classes debt situation in Thailand? Is it a problem?",
    "What economic challenges are facing South East Asian countries according to recent reports?",
    "What countries in ASEAN had the highest government debt levels and what were the underlying causes?"
]

for question in test_questions:
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 60}")

    answer = graph_rag.answer_question(question)
    print(f"\nANSWER:\n{answer}")
    print(f"{'=' * 60}")
