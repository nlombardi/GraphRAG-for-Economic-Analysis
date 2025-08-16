<<<<<<< README.md
![image](https://datasen.net/wp-content/uploads/2024/03/data_sense-e1710604295341.png)
=======
![image](https://datasen.net/wp-content/uploads/2024/03/data_sense-e1710604295341.png)
>>>>>>> README.md

# GraphRAG for Economic Analysis

This project demonstrates a Retrieval-Augmented Generation (RAG) system using a Knowledge Graph to analyze economic data and documents. It leverages structured data from the World Bank and unstructured data from PDF reports to answer complex economic questions.

For a detailed walkthrough of this project, please refer to the tutorial on our blog: [How to Use GraphRAG for Economic Data Analysis](https://datasen.net/2025/08/16/how-to-use-graphrag-for-economic-data-analysis-tutorial/).


## Features

-   **Data Ingestion**: Collects economic indicators from the World Bank and processes PDF documents.
-   **Knowledge Graph**: Builds a Neo4j graph database to represent relationships between countries, economic indicators, and documents.
-   **Vector Store**: Uses Qdrant to store document embeddings for semantic search.
-   **Hybrid RAG Pipeline**: Combines graph-based retrieval with semantic search to provide context to a Large Language Model (LLM).
-   **Modular Architecture**: Code is organized into modules for data collection, graph construction, vector storage, and the RAG pipeline.

## Prerequisites

-   Python 3.8+
-   A Neo4j instance (local or on [Aura](https://neo4j.com/cloud/aura/))
-   A Qdrant instance (local or on [Qdrant Cloud](https://qdrant.tech/cloud/))
-   A Google Cloud Platform (GCP) account with the Vision AI and Generative Language APIs enabled.
-   (Optional) [Ollama](https://ollama.com/) for running local LLMs.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up a Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```
pandas
requests
python-dotenv
pdfplumber
urllib3
langchain
langchain-ollama
qdrant-client
neo4j
spacy
google-generativeai
google-cloud-vision
httpx
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model

Download the English language model for spaCy:

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory and add the following variables. Replace the placeholder values with your actual credentials.

```
# Neo4j Credentials
NEO4J_URL="bolt://your-neo4j-host:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-neo4j-password"

# Qdrant Configuration
QDRANT_ENDPOINT="http://localhost"
QDRANT_API_KEY="your-qdrant-api-key" # Optional, depends on your setup

# Google API Key
GOOGLE_API_KEY="your-google-api-key"

# Ollama Configuration (Optional)
OLLAMA_MODEL="mistral"
OLLAMA_EMBEDDINGS_MODEL="nomic-embed-text"
OLLAMA_BASE_URL="http://localhost:11434"
```

### 6. Create Configuration File

Create a `config.json` file in the root directory. This file specifies the countries and indicators to be analyzed.

```json
{
  "COUNTRIES": {
    "NAME": ["Thailand", "Vietnam", "Malaysia", "Singapore", "Indonesia"],
    "ISO2": ["TH", "VN", "MY", "SG", "ID"]
  },
  "INDICATORS": {
    "WB_CODE": [
      "NY.GDP.MKTP.CD",
      "NY.GDP.MKTP.KD.ZG",
      "FP.CPI.TOTL.ZG",
      "SL.UEM.TOTL.ZS",
      "GC.DOD.TOTL.GD.ZS"
    ]
  }
}
```

### 7. Set Up Google Cloud CLI

The project uses Google Cloud services for OCR and for the LLM. You need to install and configure the `gcloud` CLI.

**1. Install the gcloud CLI**

Follow the official instructions to install the Google Cloud CLI for your operating system: [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install).

**2. Initialize the gcloud CLI**

Run the `gcloud init` command and follow the on-screen instructions to authenticate your account and select a default project.

```bash
gcloud init
```

**3. Enable APIs**

You need to enable the **Cloud Vision API** and the **Generative Language API** for your GCP project.

```bash
gcloud services enable vision.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

**4. Application Default Credentials**

The Vision API client uses Application Default Credentials. Log in with your user account:

```bash
gcloud auth application-default login
```

This will store credentials that the client library can automatically find.

### 8. Add PDF Documents

This project is designed to process PDF documents located in the `docs` directory. Create a `docs` folder in the root of the project. Inside the `docs` folder, you can create subdirectories for different sources (e.g., `docs/IMF`, `docs/WorldBank`). Place your PDF files into these subdirectories. The `get_docs.py` script will automatically scan these folders.

## Running the Application

To run the entire pipeline, from data collection to answering the test questions, execute the `main.py` script:

```bash
python main.py
```

The script will print the progress at each stage and finally output the questions and the generated answers. The `main.py` script contains a list of test questions that are used to demonstrate the system's capabilities. You can modify this list to ask your own questions.
