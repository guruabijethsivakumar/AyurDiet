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


def split_multi(text):
    """Splits comma, slash, AND, ampersand correctly."""
    if pd.isna(text):
        return []

    text = str(text).strip()
    if not text:
        return []

    # Normalize separators to comma
    text = re.sub(r"[\/&]| and ", ",", text, flags=re.IGNORECASE)

    return [v.strip() for v in text.split(",") if v.strip()]


def get_food(label):
    """Returns FoodNode or None."""
    return graph.evaluate(
        "MATCH (f:FoodNode {label:$l}) RETURN f",
        parameters={"l": label}
    )


def merge_node(value):
    """Creates a FoodNode for virya/vipaka/exception values if needed."""
    return graph.evaluate(
        "MERGE (n:FoodNode {label:$v}) RETURN n",
        parameters={"v": value}
    )


# ---------------- MAIN UPDATE ----------------
def update_ayurvedic_properties():
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

        # Extract columns
        viryas = split_multi(row.get("Virya (hot/cold)", None))
        vipakas = split_multi(row.get("Vipaka", None))
        exceptions = split_multi(row.get("Notes/Exceptions", None))

        # Create :hasVirya relationships
        for v in viryas:
            merge_node(v)
            graph.run("""
                MATCH (f:FoodNode {label:$food}), (x:FoodNode {label:$val})
                MERGE (f)-[:hasVirya]->(x)
            """, food=food_label, val=v)

        # Create :hasVipaka relationships
        for vp in vipakas:
            merge_node(vp)
            graph.run("""
                MATCH (f:FoodNode {label:$food}), (x:FoodNode {label:$val})
                MERGE (f)-[:hasVipaka]->(x)
            """, food=food_label, val=vp)

        # Create :hasException relationships
        for exc in exceptions:
            merge_node(exc)
            graph.run("""
                MATCH (f:FoodNode {label:$food}), (x:FoodNode {label:$val})
                MERGE (f)-[:hasException]->(x)
            """, food=food_label, val=exc)

        updated += 1

    print("--------------------------------------------------")
    print(f"Updated {updated} foods with Virya/Vipaka/Exceptions.")
    print(f"Missing foods in Neo4j (name mismatch): {missing}")
    print("--------------------------------------------------")


# ---------------- RUN ----------------
if __name__ == "__main__":
    update_ayurvedic_properties()
