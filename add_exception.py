from py2neo import Graph
import pandas as pd
import re

# ---------------- CONFIG ----------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"   # <-- change this
EXCEL_PATH = "indb_filled_ayurveda.xlsx"


# ---------------- CONNECT ----------------
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------- HELPERS ----------------
def clean_name(s: str) -> str:
    """Ensures Excel name matches Neo4j label format."""
    s = str(s).strip()
    return (
        s.replace(" ", "_")
         .replace("-", "_")
         .replace("(", "")
         .replace(")", "")
    )


def get_food(label):
    """Returns FoodNode or None."""
    return graph.evaluate(
        "MATCH (f:FoodNode {label:$l}) RETURN f",
        parameters={"l": label}
    )


def merge_node(value):
    """Creates a FoodNode for exception values if needed."""
    return graph.evaluate(
        "MERGE (n:FoodNode {label:$v}) RETURN n",
        parameters={"v": value}
    )


# ---------------- CLEANUP ----------------
def cleanup_existing_exceptions():
    """Safely deletes all existing :hasException relationships (but keeps the nodes)."""
    result = graph.run("""
        MATCH (f:FoodNode)-[r:hasException]->(x:FoodNode)
        DETACH DELETE r
        RETURN count(r) AS deleted_count
    """).data()
    
    deleted_count = result[0]['deleted_count'] if result else 0
    print(f"Cleaned up {deleted_count} existing :hasException relationships.")


# ---------------- MAIN UPDATE ----------------
def update_exceptions_only():
    df = pd.read_excel(EXCEL_PATH)

    updated = 0
    missing = 0

    for _, row in df.iterrows():

        # Clean food name to match Neo4j label
        food_original = row["Food Name"]
        food_label = clean_name(food_original)

        food_node = get_food(food_label)
        if not food_node:
            missing += 1
            continue

        # Extract Notes/Exceptions as a single full string (no splitting)
        exception_text = str(row.get("Notes/Exceptions", "")).strip()
        if not exception_text:
            continue

        # Create :hasException relationship with the full text as a single node
        merge_node(exception_text)
        graph.run("""
            MATCH (f:FoodNode {label:$food}), (x:FoodNode {label:$val})
            MERGE (f)-[:hasException]->(x)
        """, food=food_label, val=exception_text)

        updated += 1

    print("--------------------------------------------------")
    print(f"Updated {updated} foods with full Exceptions text.")
    print(f"Missing foods in Neo4j (name mismatch): {missing}")
    print("--------------------------------------------------")


# ---------------- RUN ----------------
if __name__ == "__main__":
    cleanup_existing_exceptions()  # Run this first to wipe old exceptions
    update_exceptions_only()       # Then add the new ones