import os
import pandas as pd
import networkx as nx
from rdflib import Graph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ======================================================
# CONFIG
# ======================================================
ONTOLOGY_PATH = "untitled-ontology-3"    # RDF/XML from Protégé
EXCEL_PATH = "indb_filled_ayurveda.xlsx"  # Your updated food file

ONTO_VECTOR_PATH = "ontology_vector_db"
FOOD_VECTOR_PATH = "food_vector_db"
COMBINED_VECTOR_PATH = "combined_vector_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ======================================================
# 1. LOAD ONTOLOGY (RDF/XML)
# ======================================================
def load_ontology_rdf(path=ONTOLOGY_PATH):
    g = Graph()
    g.parse(path, format="xml")
    print("Ontology loaded:", len(g), "triples")
    return g


# ======================================================
# 2. EXTRACT TRIPLES FROM ONTOLOGY
# ======================================================
def rdf_to_triples(g):
    triples = []
    for s, p, o in g:
        triples.append((str(s), str(p), str(o)))
    print("Extracted ontology triples:", len(triples))
    return triples


# ======================================================
# 3. BUILD ONTOLOGY KNOWLEDGE GRAPH (NetworkX)
# ======================================================
def build_ontology_kg(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, relation=p)
    nx.write_gexf(G, "ontology_kg.gexf")
    print("Ontology KG saved as ontology_kg.gexf")
    return G


# ======================================================
# 4. ONTOLOGY VECTOR DB (FAISS)
# ======================================================
def ontology_vector_db(triples, out_path=ONTO_VECTOR_PATH):
    texts = [f"{s} {p} {o}" for s, p, o in triples]
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.from_texts(texts, embedder)
    store.save_local(out_path)
    print("Ontology vector DB saved at:", out_path)
    return store


# ======================================================
# Detect nutrient columns
# ======================================================
def detect_nutrient_columns(df):
    nutrient_cols = []
    for col in df.columns:
        if any(x in col.lower() for x in ["calorie", "energy_kcal", "fat", "protein",
                                          "iron", "carb", "vitamin", "calcium",
                                          "fiber", "sodium", "zinc"]):
            nutrient_cols.append(col)
    return nutrient_cols


# ======================================================
# 5. LOAD FOOD EXCEL
# ======================================================
def load_food_excel(path=EXCEL_PATH):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    print("Loaded food rows:", len(df))
    return df


# ======================================================
# 6. CONVERT FOOD ROWS TO TRIPLES (UPDATED)
# ======================================================
def food_to_triples(df):
    triples = []
    nutrient_cols = detect_nutrient_columns(df)

    for _, row in df.iterrows():
        food = str(row["Food Name"]).strip().replace(" ", "_")

        # --- Rasa ---
        if pd.notna(row.get("Rasa (list)", None)):
            for rasa in str(row["Rasa (list)"]).split(","):
                triples.append((food, "hasRasa", rasa.strip()))

        # --- Virya ---
        if pd.notna(row.get("Virya (hot/cold)", None)):
            triples.append((food, "hasVirya", str(row["Virya (hot/cold)"]).strip()))

        # --- Vipaka ---
        if pd.notna(row.get("Vipaka", None)):
            triples.append((food, "hasVipaka", str(row["Vipaka"]).strip()))

        # --- Gunas ---
        if pd.notna(row.get("Gunas (list)", None)):
            for guna in str(row["Gunas (list)"]).split(","):
                triples.append((food, "hasGuna", guna.strip()))

        # --- Pacifies Dosha ---
        if pd.notna(row.get("PacifiesDosha (list)", None)):
            for d in str(row["PacifiesDosha (list)"]).split(","):
                triples.append((food, "pacifiesDosha", d.strip()))

        # --- Aggravates Dosha ---
        if pd.notna(row.get("AggravatesDosha (list)", None)):
            for d in str(row["AggravatesDosha (list)"]).split(","):
                triples.append((food, "aggravatesDosha", d.strip()))

        # --- NEW: Region Availability ---
        if pd.notna(row.get("Region Availability", None)):
            for region in str(row["Region Availability"]).split(","):
                triples.append((food, "availableInRegion", region.strip().replace(" ", "_")))

        # --- NEW: Season Suitability ---
        if pd.notna(row.get("Season Suitability", None)):
            for season in str(row["Season Suitability"]).split(","):
                triples.append((food, "suitableForSeason", season.strip().replace(" ", "_")))

        # --- NEW: Allergen Info ---
        if pd.notna(row.get("Allergen Info", None)):
            for allergen in str(row["Allergen Info"]).split(","):
                triples.append((food, "hasAllergen", allergen.strip().replace(" ", "_")))

        # --- Nutrients (dynamic detection) ---
        for col in nutrient_cols:
            if pd.notna(row[col]):
                value = row[col]
                predicate = "hasNutrient_" + col.replace(" ", "_")
                triples.append((food, predicate, str(value)))

    print("Extracted food triples:", len(triples))
    return triples


# ======================================================
# 7. BUILD FOOD KNOWLEDGE GRAPH
# ======================================================
def build_food_kg(food_triples):
    G = nx.DiGraph()
    for s, p, o in food_triples:
        G.add_edge(s, o, relation=p)
    nx.write_gexf(G, "food_kg.gexf")
    print("Food KG saved as food_kg.gexf")
    return G


# ======================================================
# 8. FOOD VECTOR DB (FAISS)
# ======================================================
def food_vector_db(food_triples, out_path=FOOD_VECTOR_PATH):
    texts = [f"{s} {p} {o}" for s, p, o in food_triples]
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.from_texts(texts, embedder)
    store.save_local(out_path)
    print("Food vector DB saved at:", out_path)
    return store


# ======================================================
# 9. BUILD COMBINED VECTOR STORE
# ======================================================
def combined_vector_db(onto_triples, food_triples):
    texts = [f"{s} {p} {o}" for s, p, o in onto_triples + food_triples]
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.from_texts(texts, embedder)
    store.save_local(COMBINED_VECTOR_PATH)
    print("Combined vector DB created at:", COMBINED_VECTOR_PATH)


# ======================================================
# RUN EVERYTHING
# ======================================================
if __name__ == "__main__":

    # ---- Ontology ----
    g = load_ontology_rdf()
    onto_triples = rdf_to_triples(g)
    onto_kg = build_ontology_kg(onto_triples)
    ontology_vector_db(onto_triples)

    # ---- Food Excel ----
    df = load_food_excel()
    food_triples = food_to_triples(df)
    food_kg = build_food_kg(food_triples)
    food_vector_db(food_triples)

    # ---- Combined Vector DB ----
    combined_vector_db(onto_triples, food_triples)

    print("\nALL PROCESSES COMPLETED SUCCESSFULLY!")
