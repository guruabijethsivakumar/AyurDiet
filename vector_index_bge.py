import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import os

EXCEL_PATH = "indb_filled_ayurveda.xlsx"
OUTPUT_INDEX = "food_vector_db_bge"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"



def j(v):
    """Join list or return None."""
    if isinstance(v, list):
        return ", ".join([str(x) for x in v if str(x).strip() != ""])
    return str(v) if v else "None"



def load_excel_and_convert_to_text_blocks():
    print("Loading Excel:", EXCEL_PATH)
    df = pd.read_excel(EXCEL_PATH)

    text_blocks = []

    for _, row in df.iterrows():
        food = str(row.get("Food Name", "")).strip()
        if not food:
            continue

        # --------------------------------------------
        # Extract Ayurvedic properties
        # --------------------------------------------
        dosha = j(row.get("PacifiesDosha (list)"))
        rasa = j(row.get("Rasa (list)"))
        guna = j(row.get("Gunas (list)"))
        virya = j(row.get("Virya (hot/cold)"))
        vipaka = j(row.get("Vipaka"))

        # Exception FULL string (no splitting)
        exception = row.get("Notes/Exceptions", "")
        if pd.isna(exception):
            exception = "None"

        # Regional + Seasonal applicability
        region = j(row.get("Region Availability"))
        season = j(row.get("Season Suitability"))

        # --------------------------------------------
        # Extract nutrients (100g values)
        # --------------------------------------------
        calories = row.get("Calories")
        carbs = row.get("carb_g")
        protein = row.get("Protein")
        fat = row.get("Fat")
        iron = row.get("iron_mg")
        calcium = row.get("calcium_mg")
        sodium = row.get("sodium_mg")
        zinc = row.get("zinc_mg")

        # --------------------------------------------
        # Build text block
        # --------------------------------------------
        text = f"""
Food: {food}
dosha: {dosha}
Ayurvedic Properties:
  Rasa: {rasa}
  Guna: {guna}
  Virya: {virya}
  Vipaka: {vipaka}
  Exceptions: {exception}

Availability:
  Region: {region}
  Season: {season}

Nutritional Values (approx per 100g):
  Calories: {calories}
  Carbs: {carbs}
  Protein: {protein}
  Fat: {fat}
  Iron: {iron}
  Calcium: {calcium}
  Sodium: {sodium}
  Zinc: {zinc}
"""
        text_blocks.append(text)

    print(f"Total foods converted into text blocks: {len(text_blocks)}")
    return text_blocks



def build_faiss_food_index():
    print("\n=== Building Food Vector Database (FAISS) ===\n")

    # Load and format all text blocks
    texts = load_excel_and_convert_to_text_blocks()

    if len(texts) == 0:
        raise ValueError("ERROR: No foods were found in the Excel file!")

    # Split if needed
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    docs = splitter.create_documents(texts)

    # Build embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Build FAISS index
    store = FAISS.from_documents(docs, embeddings)

    # Save
    store.save_local(OUTPUT_INDEX)

    print(f"\nFAISS Index successfully saved at: {OUTPUT_INDEX}/")
    print(f"Total vector documents stored: {len(docs)}")
    print("\nYou can now perform vector retrieval using:")
    print("  FAISS.load_local('food_vector_db_bge', embeddings, allow_dangerous_deserialization=True)")


if __name__ == "__main__":
    build_faiss_food_index()