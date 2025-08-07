from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

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

    def __init__(self, neo4j_driver):
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

