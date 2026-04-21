from py2neo import Graph, Node, Relationship
import networkx as nx

# Load food graph
G = nx.read_gexf("food_kg.gexf")

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# OPTIONAL: clear DB
graph.run("MATCH (n) DETACH DELETE n")

def clean_label(name):
    """Ensure Neo4j-safe labels."""
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


# 1) Create ALL food nodes
for node_id, data in G.nodes(data=True):

    # Float nodes are nutrient values → skip creating nodes
    try:
        float(node_id)
        continue
    except:
        pass

    node = Node(
        "FoodNode",
        id=node_id,
        label=clean_label(node_id)
    )
    graph.merge(node, "FoodNode", "id")


# 2) Process edges → convert numeric targets into PROPERTIES
for u, v, data in G.edges(data=True):
    relation = data.get("relation") or data.get("0") or data.get("value")

    # If v is numeric → set as property
    try:
        num_value = float(v)
        # Example relation: hasNutrient_carb_g → property = carb_g
        prop_name = relation.replace("hasNutrient_", "")
        node_u = graph.nodes.match("FoodNode", id=u).first()
        if node_u:
            node_u[prop_name] = num_value
            graph.push(node_u)
        continue
    except:
        pass

    # Otherwise: create normal relationship
    node_u = graph.nodes.match("FoodNode", id=u).first()
    node_v = graph.nodes.match("FoodNode", id=v).first()

    if node_u and node_v:
        rel_type = clean_label(relation)
        rel = Relationship(node_u, rel_type, node_v)
        graph.create(rel)

print("FOOD GRAPH imported into Neo4j successfully!")
