from py2neo import Graph, Node, Relationship
import networkx as nx

G = nx.read_gexf("ontology_kg.gexf")

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

def shorten(name):
    # If IRI contains # → take fragment
    if "#" in name:
        return name.split("#")[-1]
    return name.replace("/", "_").replace(":", "_")


# Create nodes
for node_id, data in G.nodes(data=True):

    node = Node(
        "OntologyNode",
        id=node_id,
        label=shorten(node_id)
    )
    graph.merge(node, "OntologyNode", "id")

# Create edges
for u, v, data in G.edges(data=True):
    relation = data.get("relation", "RELATED_TO")
    rel_type = shorten(relation)

    node_u = graph.nodes.match("OntologyNode", id=u).first()
    node_v = graph.nodes.match("OntologyNode", id=v).first()

    if node_u and node_v:
        graph.create(Relationship(node_u, rel_type, node_v))

print("ONTOLOGY imported into Neo4j successfully!")
