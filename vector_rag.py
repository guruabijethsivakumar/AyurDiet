import os
import subprocess
import json
import re
from typing import Dict, Any, List
    
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================================
# CONFIG
# ==========================================================

FOOD_VECTOR_DB_PATH = "food_vector_db"     # your FAISS folder
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOSHAS = ["vata", "pitta", "kapha"]
SEASONS = ["Summer", "Winter", "Monsoon", "Winter_and_Monsoon", "All_seasons"]
REGION = "Pan_India"

LLM_MODEL = "mistral:7b"


# ==========================================================
# FIXED OLLAMA CALL (UTF-8 SAFE FOR WINDOWS)
# ==========================================================

def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()


# ==========================================================
# LOAD FAISS FOOD VECTOR DATABASE
# ==========================================================

def load_food_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.load_local(
        FOOD_VECTOR_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return store


# ==========================================================
# VECTOR RETRIEVAL (NO GRAPH)
# ==========================================================

def fetch_foods_from_vector(dosha: str, season: str, region: str) -> List[Dict]:
    """
    Retrieve food documents from FAISS and PRINT them clearly in terminal.
    """

    print("\n==================== VECTOR RAG QUERY ====================")
    print(f"Query → dosha={dosha}, season={season}, region={region}")
    print("==========================================================\n")

    store = load_food_vector_db()

    query = f"""
    Retrieve foods suitable for dosha {dosha},
    season {season},
    region {region},
    including rasa, guna, virya, vipaka, allergens, exceptions, nutrients, etc.
    """

    # --- Perform similarity search WITH scores ---
    docs_and_scores = store.similarity_search_with_score(query, k=40)

    print("Top retrieved food documents (Vector DB):\n")

    foods = []
    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"------------------------- Food #{idx} -------------------------")
        print(f"Similarity Score: {score:.4f}")   # lower score = more similar
        print("Retrieved Text Block:")
        print(doc.page_content)
        print("--------------------------------------------------------------\n")

        foods.append({"food_block": doc.page_content})

    print("================ END OF VECTOR RETRIEVAL ================\n")

    return foods


# ==========================================================
# LLM GENERATOR
# ==========================================================

def llm_generate_diet(dosha, foods, season, region):
    all_food_text = "\n".join([f["food_block"] for f in foods])

    prompt = f"""
You are an Ayurvedic expert.

Dosha: {dosha}
Season: {season}
Region: {region}

Foods available (from vector DB):
{all_food_text}

Generate a 4-meal Ayurvedic diet plan for {dosha}.
Explain rasa, guna, virya, vipaka.
"""

    return call_ollama(prompt)


# ==========================================================
# BUILD CONTEXT (FOR METRICS)
# ==========================================================

def build_context(foods):
    return "\n".join([f["food_block"] for f in foods])


# ==========================================================
# CUSTOM LOCAL METRICS USING OLLAMA
# ==========================================================

def evaluate_with_ollama(prompt: str, expected_format: str = "json") -> Dict[str, Any]:
    response = call_ollama(prompt)

    if expected_format == "json":
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "No JSON found", "raw": response}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": response}

    return {"raw": response}


def compute_faithfulness(generated_output: str, ground_truth_context: str) -> float:
    prompt = f"""
Evaluate FAITHFULNESS.

Context:
{ground_truth_context}

Generated:
{generated_output}

Score 0-1.
Return JSON: {{"faithfulness_score": <float>, "reason": "<text>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("faithfulness_score", 0.0)


def compute_contextual_precision(generated_output: str, ground_truth_context: str) -> float:
    prompt = f"""
Evaluate CONTEXTUAL PRECISION.

Context:
{ground_truth_context}

Generated:
{generated_output}

Score 0-1.
Return JSON: {{"precision_score": <float>, "reason": "<text>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("precision_score", 0.0)


def compute_contextual_recall(generated_output: str, ground_truth_context: str) -> float:
    prompt = f"""
Evaluate CONTEXTUAL RECALL.

Context:
{ground_truth_context}

Generated:
{generated_output}

Score 0-1.
Return JSON: {{"recall_score": <float>, "reason": "<text>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("recall_score", 0.0)


def compute_answer_relevancy(generated_output: str, query_text: str) -> float:
    prompt = f"""
Evaluate ANSWER RELEVANCY.

Query:
{query_text}

Generated:
{generated_output}

Score 0-1.
Return JSON: {{"relevancy_score": <float>, "reason": "<text>"}}
"""
    result = evaluate_with_ollama(prompt)
    return result.get("relevancy_score", 0.0)


def compute_metrics(generated_output: str, ground_truth_context: str, query_text: str) -> Dict[str, float]:
    return {
        "faithfulness": compute_faithfulness(generated_output, ground_truth_context),
        "context_precision": compute_contextual_precision(generated_output, ground_truth_context),
        "context_recall": compute_contextual_recall(generated_output, ground_truth_context),
        "answer_relevancy": compute_answer_relevancy(generated_output, query_text)
    }


# ==========================================================
# MAIN LOOP
# ==========================================================

results = []

for dosha in DOSHAS:
    for season in SEASONS:

        print(f"\nEvaluating → DOSHA={dosha.upper()}  SEASON={season}")

        foods = fetch_foods_from_vector(dosha, season, REGION)
        if not foods:
            print("No foods returned from vector DB — skipping.")
            continue

        context = build_context(foods)
        question = f"Generate a diet for {dosha} in {season}"

        generated = llm_generate_diet(dosha, foods, season, REGION)

        scores = compute_metrics(generated, context, question)

        row = {"dosha": dosha, "season": season, **scores}
        results.append(row)

        print("Scores:", row)


pd.DataFrame(results).to_csv("vector_eval_results.csv", index=False)
print("\nSaved → vector_eval_results.csv")
