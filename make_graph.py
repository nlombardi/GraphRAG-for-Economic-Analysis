from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import pandas as pd
import spacy
from typing import List, Tuple, Dict
import json

load_dotenv()
config = json.load(open('config.json'))

# Here we are using a free-tier neo4j instance on neo4j aura
neo4j_driver = GraphDatabase.driver(
    os.getenv('NEO4J_URL'),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

class EconGraphBuilder:
    """
    Constructs and populates the economic knowledge graph.

    Class handles:
    1) setting up graph constraints for data integrity.
    2) Creating nodes from structured WDI data.
    3) Extracting entities from unstructured documents.
    4) Building relationships between all entities.
    """

    def __init__(self, neo4j_driver = neo4j_driver):
        self.driver = neo4j_driver
        self.setup_constraints()

    def setup_constraints(self):
        """
        Create unique constraints to prevent duplicate entities.

        Constraints ensure data integrity and improve query performance by creating indexes on frequently accessed properties.
        """
        constraints = [
            # Each country has a unique code (BR, AR, etc.)
            "CREATE CONSTRAINT country_code IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE",

            # Each indicator has a unique code (NY.GDP.MKTP.KD.ZG, etc.)
            "CREATE CONSTRAINT indicator_code IF NOT EXISTS FOR (i:Indicator) REQUIRE i.code IS UNIQUE",

            # Each year is unique
            "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",

            # Each document chunk has unique ID
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.chunk_id IS UNIQUE"
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✓ Created constraint: {constraint.split('(')[1].split(')')[0]}")
                except Exception as e:
                    print(f"⚠ Constraint might already exist: {e}")

    def _categorize_indicator(self, indicator_code: str) -> str:
        """Categorize indicators for better organization"""
        if any(code in indicator_code for code in ('GDP', 'MKTP')):
            return 'Growth'
        elif any(code in indicator_code for code in ('CPI', 'INF')):
            return 'Inflation'
        elif any(code in indicator_code for code in ('UEM', 'EMP', 'TLF')):
            return 'Employment'
        elif any(code in indicator_code for code in ('TRD', 'EXP', 'IMP', 'RSB', 'CAB')):
            return 'Trade'
        elif any(code in indicator_code for code in ('DOD', 'DEBT', 'BAL', 'XPN', 'REV')):
            return 'Fiscal'
        elif any(code in indicator_code for code in ('POP', 'CBRT', 'CDRT')):
            return 'Population'
        elif any(code in indicator_code for code in ('ADT', 'PRM', 'SE.XPD')):
            return 'Education'
        elif any(code in indicator_code for code in ('LE00', 'IMRT', 'CHEX')):
            return 'Health'
        elif any(code in indicator_code for code in ('USE', 'ATM', 'FEC')):
            return 'Environment'
        elif any(code in indicator_code for code in ('DINV', 'GDI')):
            return 'Investment'
        else:
            return 'Other'

    def create_structured_nodes(self, wdi_data: pd.DataFrame):
        """
        Create nodes and relationships from World Bank WDI data.

        This method transforms tabular data into a graph structure:
        Country -> DataPoint -> Indicator
                     |
                     v
                   Year

        """
        print("Creating structured data nodes...")

        with self.driver.session() as session:

            # 1) Create country nodes
            countries = wdi_data[['country_code', 'country_name']].drop_duplicates()
            print(f"Creating {len(countries)} country nodes...")
            for _, row in countries.iterrows():
                session.run(
                    """
                        MERGE (c:Country {code: $code})
                        SET c.name = $name
                    """,
                    code=row['country_code'],
                    name=row['country_name']
                )

            # 2) Create indicator nodes
            indicators = wdi_data[['indicator_code', 'indicator_name']].drop_duplicates()
            print(f"Creating {len(indicators)} indicator nodes...")
            for _, row in indicators.iterrows():
                session.run(
                    """
                        MERGE (i:Indicator {code: $code})
                        SET i.name = $name, i.category = $category
                    """,
                    code=row['indicator_code'],
                    name=row['indicator_name'],
                    category=self._categorize_indicator(row['indicator_code'])
                )

            # 3) Create Year Nodes
            years = wdi_data['year'].dropna().unique()
            print(f"Creating {len(years)} year nodes...")
            for year in years:
                session.run(
                    """
                        MERGE (y:Year {value: $year})
                    """,
                    year=int(year)
                )

            # 4) Create DataPoint nodes and relationships
            # -- filter out null values for clean data
            valid_data = wdi_data.dropna(subset=['value'])
            print(f"Creating {len(valid_data)} data points with relationships...")

            for counter, (_, row) in enumerate(valid_data.iterrows()):
                try:
                    # Create DataPoint with explicit property setting
                    session.run(
                        """
                        MATCH (c:Country {code: $country_code})
                        MATCH (i:Indicator {code: $indicator_code})
                        MATCH (y:Year {value: $year})

                        MERGE (dp:DataPoint {
                            country_code: $country_code,
                            indicator_code: $indicator_code,
                            year: $year
                        })
                        SET dp.value = $value,
                            dp.last_updated = datetime()

                        MERGE (c)-[:HAS_DATA_POINT]->(dp)
                        MERGE (dp)-[:MEASURES]->(i)
                        MERGE (dp)-[:FOR_YEAR]->(y)
                        """,
                        country_code=row['country_code'],
                        indicator_code=row['indicator_code'],
                        year=int(row['year']),
                        value=float(row['value'])
                    )
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    print(f"Row data: {dict(row)}")
                    break

                    # Progress indicator for large datasets
                if counter % 100 == 0:
                    print(f"   Processed {counter}/{len(valid_data)} data points...")

            # Final progress update
            print(f"   Processed {len(valid_data)}/{len(valid_data)} data points...")

        print("✓ Structured data nodes created successfully")

    def extract_entities_from_text(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract countries, economic terms, and events from text"""
        nlp = spacy.load("en_core_web_sm")  # Used for extracting words from documents
        doc = nlp(text)

        countries = []
        economic_terms = []
        events = []

        # Define some known countries to aid in detection including the countries for our analysis plus other major economies
        known_countries = set(config.get("COUNTRIES", {}).get("NAME", []) +
                              ['United States', 'United Kingdom', 'China', 'Germany', 'Japan', 'US', 'UK', 'France',
                               'Australia', 'Russia'])

        # Predefined economic terms to look for extraction
        economic_keywords = {
            'gdp', 'inflation', 'unemployment', 'fiscal', 'monetary', 'debt', 'growth', 'recession', 'stimulus',
            'reform', 'trade', 'exports', 'imports', 'deficit', 'surplus', 'policy', 'central bank', 'interest rates'
        }

        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geopolitical entities
                if ent.text.lower() in known_countries:
                    countries.append(ent.text)
            elif ent.label_ in ["ORG", "EVENT"]:
                events.append(ent.text)

        # Extract economic terms
        for token in doc:
            if token.text.lower() in economic_keywords:
                economic_terms.append(token.text.lower())

        return list(set(countries)), list(set(economic_terms)), list(set(events))

    def create_document_nodes(self, chunks: List[Dict]):
        """Create document nodes and extract entities"""
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks):
                print("Processing document chunks into nodes...")
                # Create document chunk node
                session.run(
                    """
                        MERGE (d:Document {doc_id: $doc_id})
                        SET d.title = $title, 
                            d.content = $content,
                            d.chunk_id = $chunk_id
                    """,
                    **chunk
                )

                # Extract and link entities
                countries, economic_terms, events = self.extract_entities_from_text(chunk["content"])

                # Link countries
                for country in countries:
                    session.run(
                        """
                            MATCH (d:Document {chunk_id: $chunk_id})
                            MATCH (c:Country {name: $country})
                            MERGE (d)-[:MENTIONS]->(c)
                        """,
                        chunk_id=chunk["chunk_id"],
                        country=country
                    )

                # Create economic concept nodes
                for term in economic_terms:
                    session.run(
                        """
                            MATCH (d:Document {chunk_id: $chunk_id})
                            MERGE (ec:EconomicConcept {name: $term})
                            MERGE (d)-[:DISCUSSES]->(ec)
                        """,
                        chunk_id=chunk["chunk_id"],
                        term=term
                    )

                # Create event nodes
                for event in events:
                    session.run(
                        """
                            MATCH (d:Document {chunk_id: $chunk_id})
                            MERGE (e:Event {name: $event})
                            MERGE (d)-[:DESCRIBES]->(e)
                        """,
                        chunk_id=chunk["chunk_id"],
                        event=event
                    )

                # Progress tracking
                if i % 100 == 0 or i == len(chunks) - 1:  # Update every 5 chunks since there are fewer documents
                    entities_found = len(countries) + len(economic_terms) + len(events)
                    print(f"   Processed chunk {i + 1}/{len(chunks)} - Found {entities_found} entities")

        print("Document nodes and entity relationships created successfully")
