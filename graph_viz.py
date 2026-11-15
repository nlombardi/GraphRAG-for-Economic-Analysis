# Visualize graph
from graphdatascience import GraphDataScience
import os
from neo4j_viz.gds import from_gds

gds = GraphDataScience(os.getenv('NEO4J_URL'), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")), database='neo4j')
G = gds.graph.load_cora()
# Similarity algorithms
gds.nodeSimilarity.mutate(
    G, mutateRelationshipType="SIMILAR", mutateProperty="similarity"
)
gds.pageRank.mutate(G, mutateProperty="pagerank")
gds.louvain.mutate(G, mutateProperty="componentId")

VG = from_gds(
    gds,
    G,
    size_property="pagerank",
    additional_node_properties=["componentId", "subject"],
    max_node_count=100,
)