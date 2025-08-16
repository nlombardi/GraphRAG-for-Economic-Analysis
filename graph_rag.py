from langchain_ollama import OllamaEmbeddings, OllamaLLM
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from typing import List, Dict
import spacy
import re
from google import genai

load_dotenv()

# Define neo4j driver
neo4j_driver = GraphDatabase.driver(
    os.getenv('NEO4J_URL'),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# Define the qdrant client
qdrant_client = QdrantClient(url=os.getenv('QDRANT_ENDPOINT'), port=6333, api_key=os.getenv('QDRANT_API_KEY'))

# Google client
google_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the embeddings model
ollama_embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBEDDINGS_MODEL'), base_url=os.getenv('OLLAMA_BASE_URL'))

# Initialize the LLM
ollama_llm = OllamaLLM(model=os.getenv('OLLAMA_MODEL'), base_url=os.getenv('OLLAMA_BASE_URL'), temperature=0.1)


class EconGraphRag:
    """
    Combines the knowledge graph with the vector search to conduct the graphRAG
    """
    def __init__(self, neo4j_driver = neo4j_driver, qdrant_client = qdrant_client,
                 provider = "google", llm_type = "google", collection_name = "economic_documents"):
        self.driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.embeddings = google_client if provider == "google" else ollama_embeddings
        self.provider = provider
        self.llm = google_client if llm_type == "google" else ollama_llm
        self.collection_name = collection_name

    def extract_query_entities(self, question: str) -> Dict[str, List[str]]:
        """Extract entities and concepts from user question"""
        nlp = spacy.load("en_core_web_sm")
        query = nlp(question)

        entities = {
            'countries': [],
            'indicator_codes': [],
            'indicator_categories': [],
            'years': [],
            'concepts': []
        }

        # Extract entities
        for ent in query.ents:
            if ent.label_ == "GPE":
                entities['countries'].append(ent.text.lower())
            elif ent.label_ == "DATE":
                # Simple year extraction
                year_match = re.search(r'\b(19|20)\d{2}\b', ent.text)
                if year_match:
                    entities['years'].append(year_match.group())


        # World Bank WDI Indicator Code Mappings
        wdi_indicators = {
            # GDP and Growth indicators
            'gdp': 'NY.GDP.MKTP.CD',  # GDP (current US$)
            'gdp growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'gdp per capita': 'NY.GDP.PCAP.CD',  # GDP per capita (current US$)
            'gdp per capita growth': 'NY.GDP.PCAP.KD.ZG',  # GDP per capita growth (annual %)
            'gross domestic product': 'NY.GDP.MKTP.CD',
            'economic growth': 'NY.GDP.MKTP.KD.ZG',
            'national income': 'NY.GDP.MKTP.CD',

            # Inflation indicators
            'inflation': 'FP.CPI.TOTL.ZG',  # Inflation, consumer prices (annual %)
            'inflation rate': 'FP.CPI.TOTL.ZG',
            'consumer price index': 'FP.CPI.TOTL',  # Consumer price index (2010 = 100)
            'cpi': 'FP.CPI.TOTL.ZG',
            'price level': 'FP.CPI.TOTL',

            # Employment indicators
            'unemployment': 'SL.UEM.TOTL.ZS',  # Unemployment, total (% of total labor force)
            'unemployment rate': 'SL.UEM.TOTL.ZS',
            'employment rate': 'SL.EMP.TOTL.SP.ZS',  # Employment to population ratio
            'labor force': 'SL.TLF.TOTL.IN',  # Labor force, total
            'employment': 'SL.EMP.TOTL.SP.ZS',

            # Trade indicators
            'exports': 'NE.EXP.GNFS.ZS',  # Exports of goods and services (% of GDP)
            'imports': 'NE.IMP.GNFS.ZS',  # Imports of goods and services (% of GDP)
            'trade': 'NE.TRD.GNFS.ZS',  # Trade (% of GDP)
            'trade balance': 'NE.RSB.GNFS.ZS',  # External balance on goods and services (% of GDP)
            'current account': 'BN.CAB.XOKA.GD.ZS',  # Current account balance (% of GDP)

            # Fiscal indicators
            'government debt': 'GC.DOD.TOTL.GD.ZS',  # Central government debt, total (% of GDP)
            'debt': 'GC.DOD.TOTL.GD.ZS',
            'public debt': 'GC.DOD.TOTL.GD.ZS',
            'fiscal balance': 'GC.BAL.CASH.GD.ZS',  # Cash surplus/deficit (% of GDP)
            'government expenditure': 'GC.XPN.TOTL.GD.ZS',  # Expense (% of GDP)
            'government revenue': 'GC.REV.XGRT.GD.ZS',  # Revenue, excluding grants (% of GDP)

            # Population and Demographics
            'population': 'SP.POP.TOTL',  # Population, total
            'population growth': 'SP.POP.GROW',  # Population growth (annual %)
            'birth rate': 'SP.DYN.CBRT.IN',  # Birth rate, crude (per 1,000 people)
            'death rate': 'SP.DYN.CDRT.IN',  # Death rate, crude (per 1,000 people)

            # Education
            'literacy': 'SE.ADT.LITR.ZS',  # Literacy rate, adult total (% of people ages 15 and above)
            'school enrollment': 'SE.PRM.NENR',  # School enrollment, primary (% net)
            'education spending': 'SE.XPD.TOTL.GD.ZS',  # Government expenditure on education, total (% of GDP)

            # Health
            'life expectancy': 'SP.DYN.LE00.IN',  # Life expectancy at birth, total (years)
            'infant mortality': 'SP.DYN.IMRT.IN',  # Mortality rate, infant (per 1,000 live births)
            'health spending': 'SH.XPD.CHEX.GD.ZS',  # Current health expenditure (% of GDP)

            # Energy and Environment
            'energy consumption': 'EG.USE.PCAP.KG.OE',  # Energy use (kg of oil equivalent per capita)
            'electricity': 'EG.USE.ELEC.KH.PC',  # Electric power consumption (kWh per capita)
            'co2 emissions': 'EN.ATM.CO2E.PC',  # CO2 emissions (metric tons per capita)
            'renewable energy': 'EG.FEC.RNEW.ZS',  # Renewable energy consumption (% of total final energy consumption)

            # Investment and Finance
            'foreign direct investment': 'BX.KLT.DINV.WD.GD.ZS',  # Foreign direct investment, net inflows (% of GDP)
            'fdi': 'BX.KLT.DINV.WD.GD.ZS',
            'gross capital formation': 'NE.GDI.TOTL.ZS',  # Gross capital formation (% of GDP)
            'investment': 'NE.GDI.TOTL.ZS'
        }

        # Category mappings for broader searches
        indicator_categories = {
            'growth': 'Growth',
            'economic growth': 'Growth',
            'inflation': 'Inflation',
            'prices': 'Inflation',
            'employment': 'Employment',
            'unemployment': 'Employment',
            'jobs': 'Employment',
            'trade': 'Trade',
            'exports': 'Trade',
            'imports': 'Trade',
            'fiscal': 'Fiscal',
            'debt': 'Fiscal',
            'government': 'Fiscal',
            'budget': 'Fiscal'
        }

        question_lower = question.lower()

        # Check for specific indicator code matches
        for indicator, code in wdi_indicators.items():
            if indicator in question_lower:
                entities['indicator_codes'].append(code)
                entities['concepts'].append(indicator)

        # Check for category matches
        for indicator, category in indicator_categories.items():
            if indicator in question_lower:
                entities['indicator_categories'].append(category)
                if indicator not in entities['concepts']:
                    entities['concepts'].append(indicator)

        # Extract country names and codes that might not be caught by NER
        common_countries = {
            'us': 'united states', 'usa': 'united states', 'america': 'united states',
            'uk': 'united kingdom', 'britain': 'united kingdom',
            'china': 'china', 'prc': 'china',
            'india': 'india',
            'germany': 'germany',
            'france': 'france',
            'japan': 'japan',
            'brazil': 'brazil',
            'russia': 'russia',
            'canada': 'canada',
            'australia': 'australia',
            'south korea': 'south korea', 'korea': 'south korea',
            'mexico': 'mexico',
            'italy': 'italy',
            'spain': 'spain',
            'netherlands': 'netherlands',
            'switzerland': 'switzerland',
            'sweden': 'sweden',
            'norway': 'norway',
            'denmark': 'denmark'
        }

        for country_variant, country_name in common_countries.items():
            if country_variant in question_lower and country_name not in entities['countries']:
                entities['countries'].append(country_name)

        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        return entities

    def query_graph(self, entities: Dict[str, List[str]]) -> List[dict]:
        """Query the knowledge graph based on extracted entities, order by the year desc"""
        results = []

        with self.driver.session() as session:
            # Query for country data points
            if entities['countries']:
                for country in entities['countries']:
                    query = """
                        MATCH (c:Country)-[:HAS_DATA_POINT]->(dp:DataPoint)-[:MEASURE]->(i:Indicator)
                        MATCH (dp)-[:FOR_YEAR]->(y:Year)
                        WHERE toLower(c.name) CONTAINS $country or toLower(c.code) = $country
                    """
                    params = {'country': country}

                    # Filter by indicator codes if specified
                    if entities['indicator_codes']:
                        query += " AND i.code IN $indicator_codes"
                        params['indicator_codes'] = entities['indicator_codes']
                    elif entities['indicator_categories']:
                        query += " AND i.category IN $indicator_categories"
                        params['indicator_categories'] = entities['indicator_categories']

                    if entities['years']:
                        query += " AND y.value in $years"
                        params['years'] = [int(y) for y in entities['years']]

                    query += """
                        RETURN c.name as country, i.name as indicator, i.code as indicator_code,
                           i.category as category, y.value as year, dp.value as value, dp.last_updated as last_updated
                        ORDER BY dp.last_updated, y.value DESC
                        LIMIT 20
                    """

                    result = session.run(query, params)
                    results.extend([dict(record) for record in result])

            # Query for related documents
            if entities['countries'] or entities['concepts']:
                doc_query = """
                    MATCH (d:Document)
                    WHERE 
                    """
                conditions = []
                params = {}

                if entities['countries']:
                    conditions.append("""
                            EXISTS {
                                MATCH (d)-[:MENTIONS]->(c:Country)
                                WHERE ANY(country IN $countries WHERE toLower(c.name) CONTAINS country)
                            }
                        """)
                    params['countries'] = entities['countries']

                if entities['concepts']:
                    conditions.append("""
                            EXISTS {
                                MATCH (d)-[:DISCUSSES]->(ec:EconomicConcept)
                                WHERE ec.name IN $concepts
                            }
                        """)
                    params['concepts'] = entities['concepts']

                doc_query += " OR ".join(conditions)
                doc_query += """
                    RETURN d.chunk_id as chunk_id, d.title as title, 
                           d.source as source, d.content as content
                    LIMIT 10
                    """

                if conditions:
                    doc_result = session.run(doc_query, params)
                    doc_results = [dict(record) for record in doc_result]
                    results.extend(doc_results)

            # Query for indicators without specific countries
            if entities['indicator_codes'] or entities['indicator_categories']:
                query = """
                        MATCH (c:Country)-[:HAS_DATA_POINT]->(dp:DataPoint)-[:MEASURE]->(i:Indicator)
                        MATCH (dp)-[:FOR_YEAR]->(y:Year)
                        WHERE 1=1
                        """
                params = {}

                # Filter by specific indicator codes
                if entities['indicator_codes']:
                    query += " AND i.code IN $indicator_codes"
                    params['indicator_codes'] = entities['indicator_codes']
                # Filter by categories if no specific codes
                elif entities['indicator_categories']:
                    query += " AND i.category IN $indicator_categories"
                    params['indicator_categories'] = entities['indicator_categories']

                if entities['years']:
                    query += " AND y.value IN $years"
                    params['years'] = [int(y) for y in entities['years']]

                query += """
                                RETURN c.name as country, i.name as indicator, i.code as indicator_code,
                                       i.category as category, y.value as year, dp.value as value, dp.unit as unit
                                ORDER BY y.value DESC, c.name ASC
                                LIMIT 50
                            """

                result = session.run(query, params)
                results.extend([dict(record) for record in result])

        return results

    def semantic_search(self, question: str, limit: int = 10) -> List[Dict]:
        """Perform semantic search on document vectors"""
        if self.provider == "google":
            query_embedding = self.embeddings.models.embed_content(
                    model="gemini-embedding-001",
                    contents=question
                )
            if query_embedding:
                processed_embeddings = query_embedding.embeddings[0].values
            else:
                print("Embeddings not found.")
                return None
        else:
            processed_embeddings = self.embeddings.embed_query(question)

        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=processed_embeddings,
            limit=limit
        )

        return [
            {
                'content': result.payload['content'],
                'title': result.payload['title'],
                'source': result.payload['source'],
                'score': result.score
            }
            for result in search_results.points
        ]

    def generate_answer(self, question: str, graph_results: List[Dict],
                        semantic_results: List[Dict]) -> str:
        """Generate final answer using LLM"""

        # Prepare context
        context_parts = []

        # Add graph-derived structured data
        if graph_results:
            context_parts.append("STRUCTURED DATA FROM KNOWLEDGE GRAPH:")
            for result in graph_results[:10]:  # Limit context size
                if 'value' in result:  # Numeric data
                    context_parts.append(
                        f"- {result.get('country', 'N/A')}: {result.get('indicator', 'N/A')} "
                        f"in {result.get('year', 'N/A')} was {result.get('value', 'N/A')}"
                    )
                else:  # Document data
                    context_parts.append(f"- {result.get('title', 'N/A')}: {result.get('content', 'N/A')[:200]}...")

        # Add semantic search results
        if semantic_results:
            context_parts.append("\nRELEVANT DOCUMENT EXCERPTS:")
            for result in semantic_results:
                context_parts.append(
                    f"- From '{result['title']}' ({result['source']}): {result['content'][:500]}..."
                )

        context = "\n".join(context_parts)

        prompt = f"""
            Based on the following structured economic data and document excerpts, 
            please provide a comprehensive answer to the question: "{question}"
    
            Available Context:
            {context}
    
            Please provide a detailed answer that:
            1. Uses specific data points when available
            2. Explains relationships between different economic indicators
            3. References sources appropriately
            4. Acknowledges any limitations in the available data
    
            Answer:
        """

        if self.provider == "google":
            response = self.llm.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
        else:
            response = self.llm(prompt)
        return response

    def answer_question(self, question: str) -> str:
        """Main method to answer economic questions"""
        print(f"Processing question: {question}")

        # Extract entities from question
        entities = self.extract_query_entities(question)
        print(f"Extracted entities: {entities}")

        # Query knowledge graph
        graph_results = self.query_graph(entities)
        print(f"Found {len(graph_results)} graph results")

        # Perform semantic search
        semantic_results = self.semantic_search(question)
        print(f"Found {len(semantic_results)} semantic results")

        # Generate answer
        answer = self.generate_answer(question, graph_results, semantic_results)
        return answer
