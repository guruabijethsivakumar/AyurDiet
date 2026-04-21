import os
import subprocess
import json
import re
import pandas as pd
from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ======================================================
# CONFIG
# ======================================================
VECTOR_DB_PATH = "food_vector_db_bge"
EMBED_MODEL = "BAAI/bge-large-en"

LLM_MODEL = "mistral:7b"

DOSHAS = ["vata", "pitta", "kapha"]
SEASONS = ["Summer", "Winter", "Monsoon", "Winter_and_Monsoon", "All_seasons"]
REGION = "Pan_India"


# ======================================================
# OLLAMA CALL
# ======================================================
def call_ollama(prompt: str) -> str:
    data = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    cmd = [
        "curl", "-X", "POST", "http://localhost:11434/api/generate",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"
    try:
        response = json.loads(result.stdout)
        return response.get("response", "").strip()
    except json.JSONDecodeError:
        return result.stdout.strip()


# ======================================================
# LOAD VECTOR DB
# ======================================================
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )


# ======================================================
# NORMALIZATION
# ======================================================
def norm(x):
    return str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


# ======================================================
# STRICT FOOD VALIDATION LOGIC
# ======================================================
def validate_food_block(food_text: str, dosha: str, season: str, region: str):

    ft = food_text.lower().replace(" ", "")

    # ---------------- DOSHA CHECK ----------------
    # valid if pitta ∈ (vata,pitta)
    if norm(dosha) not in ft:
        return False

    # ---------------- SEASON CHECK ----------------
    if norm(season) not in ["allseasons", "allseason"]:
        if norm(season) not in ft:
            return False

    # ---------------- REGION CHECK ----------------
    region_norm = norm(region)
    pan_aliases = [
        "", "panindia", "panindiaurban", "panindiawithregionalstyles"
    ]

    if region_norm not in pan_aliases:
        if region_norm not in ft:
            return False

    return True


# ======================================================
# VECTOR + FILTER RETRIEVAL
# ======================================================
def vector_filter_and_retrieve(dosha: str, season: str, region: str):
    db = load_db()

    print("\n================ VECTOR FILTER =================")
    print(f"Filters → dosha={dosha}, season={season}, region={region}")
    print("================================================\n")

    all_docs = list(db.docstore._dict.values())

    filtered = []
    for doc in all_docs:
        meta = doc.metadata

        m_dosha = norm(meta.get("dosha", ""))
        m_season = norm(meta.get("season", ""))
        m_region = norm(meta.get("region", ""))

        # ------------ DOSHA MATCH ------------
        # VALID if target dosha IN metadata dosha list
        if norm(dosha) not in m_dosha:
            continue

        # ------------ SEASON MATCH ------------
        if m_season not in ["", "allseasons"]:
            if norm(season) not in m_season:
                continue

        # ------------ REGION MATCH ------------
        pan_aliases = ["", "panindia", "panindiaurban", "panindiawithregionalstyles"]

        if m_region not in pan_aliases:
            if norm(region) not in m_region:
                continue

        filtered.append(doc)

    print(f"Total docs: {len(all_docs)}")
    print(f"Filtered docs: {len(filtered)}\n")

    if not filtered:
        print("❌ No matching foods.\n")
        return []

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    small_db = FAISS.from_texts([d.page_content for d in filtered], embed)

    query = f"foods for dosha {dosha} in {season} for region {region}"
    results = small_db.similarity_search_with_score(query, k=40)

    foods = []
    print("Retrieved vector foods:\n")
    for idx, (doc, score) in enumerate(results, start=1):
        print(f"#{idx} | Score={score:.4f}")
        print(doc.page_content)
        print("-----------------------------------\n")
        foods.append({"food_block": doc.page_content})

    return foods



# ======================================================
# LLM DIET GENERATION
# ======================================================
def llm_generate_diet(dosha, foods, season, region):
    full_text = "\n".join([f["food_block"] for f in foods])

    prompt = f"""
You are an Ayurvedic expert. You must STRICTLY follow the rules below.

========================================================
RULES YOU MUST FOLLOW
========================================================
1. You are allowed to use ONLY AND EXACTLY the foods listed below.
2. You MUST copy each food name exactly as written — no rewrites.
3. If you use a food, include it EXACTLY as shown (same spelling, same case).
4. Do NOT introduce new foods.
5. Do NOT translate or paraphrase food names.
6. You MUST reference at least 4–6 foods from the list.
7. The output MUST stay consistent with rasa, guna, virya, vipaka.
8. If a food appears multiple times in context, match the primary name.

========================================================
AUTHORIZED FOOD LIST (the ONLY foods you may use)
========================================================
{full_text}

========================================================
TASK
========================================================
Create a detailed 4-meal Ayurvedic diet plan for:

- Dosha: {dosha}
- Season: {season}
- Region: {region}

For EACH MEAL, include:
1. EXACT food names taken from the list.
2. Why each food is chosen.
3. Rasa / Guna / Virya / Vipaka reasoning.

========================================================
ABSOLUTE CONSTRAINT
========================================================
If you cannot produce a fully rule-compliant output,
respond with:

"ERROR: Cannot generate a valid plan with given constraints."

Do NOT break the constraints.
"""


    return call_ollama(prompt)


# ======================================================
# CONTEXT BUILDER
# ======================================================
def build_context(foods):
    return "\n".join([f["food_block"] for f in foods])


# ======================================================
# JSON SAFE PARSER
# ======================================================
def eval_json(prompt):
    res = call_ollama(prompt)
    match = re.search(r"\{.*\}", res, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return {"error": "invalid_json", "raw": res}
    return {"error": "no_json", "raw": res}


# ======================================================
# NEW METRIC: FAITHFULNESS
# ======================================================
def compute_faithfulness(gen, ctx, dosha, season, region):

    g_lower = gen.lower()

    f_dosha = 1 if norm(dosha) in g_lower.replace(" ", "") else 0
    f_season = 1 if norm(season) == "allseasons" or norm(season) in g_lower.replace(" ", "") else 0
    f_region = 1 if norm(region) in ["", "panindia", "panindiaurban", "panindiawithregionalstyles"] or norm(region) in g_lower.replace(" ", "") else 0

    return round((f_dosha + f_season + f_region) / 3, 3)



# ======================================================
# NEW METRIC: PRECISION
# ======================================================
import re

def extract_food_names(context: str):
    foods = re.findall(r"Food:\s*([^\n]+)", context)
    return [f.strip().lower() for f in foods]

def compute_precision(gen, ctx):
    foods = extract_food_names(ctx)

    if not foods:
        return 0.0

    gen_lower = gen.lower()
    matched = 0

    for f in foods:
        # Accept partial matches → robust scoring
        name = f.split("(")[0].strip()  # remove parentheses
        if name and name in gen_lower:
            matched += 1

    precision = matched / len(foods)
    return round(precision, 3)


def compute_recall(gen, ctx):
    foods = extract_food_names(ctx)

    if not foods:
        return 0.0

    gen_lower = gen.lower()
    used = 0

    for f in foods:
        name = f.split("(")[0].strip()
        if name and name in gen_lower:
            used += 1

    recall = used / len(foods)
    return round(recall, 3)



# ======================================================
# NEW METRIC: RELEVANCY
# ======================================================
def compute_relevancy(gen, query):

    g = gen.lower()
    q = query.lower()

    needed = 0
    got = 0

    # dosha relevancy
    for d in ["vata", "pitta", "kapha"]:
        if d in q:
            needed += 1
            if d in g:
                got += 1

    # season relevancy
    for s in ["summer", "winter", "monsoon"]:
        if s in q:
            needed += 1
            if s in g:
                got += 1

    if needed == 0:
        return 0.0

    return got / needed



# ======================================================
# MAIN LOOP
# ======================================================
results = []

for dosha in DOSHAS:
    for season in SEASONS:

        print(f"\n======== TESTING {dosha.upper()} | {season} ========")

        foods = vector_filter_and_retrieve(dosha, season, REGION)
        if not foods:
            print("Skipping...\n")
            continue

        ctx = build_context(foods)
        query = f"Diet plan for {dosha} in {season}"

        gen = llm_generate_diet(dosha, foods, season, REGION)

        row = {
            "dosha": dosha,
            "season": season,
            "faithfulness": compute_faithfulness(gen, ctx, dosha, season, REGION),
            "precision": compute_precision(gen, ctx),
            "recall": compute_recall(gen, ctx),
            "relevancy": compute_relevancy(gen, query)
        }

        print("Scores:", row)
        results.append(row)

pd.DataFrame(results).to_csv("vector_eval_results_bge.csv", index=False)
print("\nSaved → vector_eval_results_bge.csv")