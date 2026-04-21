import networkx as nx
import matplotlib.pyplot as plt

# Load your GEXF ontology graph
G = nx.read_gexf("ontology_kg.gexf")

# Layout for visualization
pos = nx.spring_layout(G, k=0.15, iterations=50)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=80, node_color="skyblue")

# Draw edges
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=10)

# Draw only short labels for readability
labels = {n: n.split("#")[-1] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=6)

plt.axis("off")
plt.show()
